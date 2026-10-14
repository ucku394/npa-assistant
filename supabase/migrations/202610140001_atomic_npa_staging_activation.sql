-- Atomic activation of a staged NPA revision.
-- The loader inserts staged chunks with status='inactive'. Search RPCs that
-- filter for active chunks cannot return them before this function commits.
-- The delete + renumber + activation below run in one PostgreSQL transaction.

CREATE OR REPLACE FUNCTION public.activate_npa_staging_document(
    p_document_id uuid,
    p_staging_doc_name text,
    p_doc_name text,
    p_expected_count integer
)
RETURNS integer
LANGUAGE plpgsql
SECURITY INVOKER
SET search_path = ''
AS $function$
DECLARE
    v_document_found boolean;
    v_staged_count integer;
    v_updated_count integer;
    v_final_count integer;
BEGIN
    IF p_document_id IS NULL
       OR NULLIF(btrim(p_staging_doc_name), '') IS NULL
       OR NULLIF(btrim(p_doc_name), '') IS NULL
       OR p_expected_count IS NULL
       OR p_expected_count < 1 THEN
        RAISE EXCEPTION 'Invalid arguments for atomic NPA activation';
    END IF;

    IF p_staging_doc_name = p_doc_name
       OR left(p_staging_doc_name, length('__NPA_STAGING__')) <> '__NPA_STAGING__' THEN
        RAISE EXCEPTION 'Invalid staging document name: %', p_staging_doc_name;
    END IF;

    -- Serialize concurrent replacements of the same document.
    SELECT true
      INTO v_document_found
      FROM public.npa_documents d
     WHERE d.id = p_document_id
       AND d.doc_name = p_doc_name
     FOR UPDATE;

    IF NOT COALESCE(v_document_found, false) THEN
        RAISE EXCEPTION
            'Document record not found or name mismatch: document_id=%, doc_name=%',
            p_document_id, p_doc_name;
    END IF;

    SELECT count(*)::integer
      INTO v_staged_count
      FROM public.npa_chunks c
     WHERE c.document_id = p_document_id
       AND c.doc_name = p_staging_doc_name
       AND c.status = 'inactive';

    IF v_staged_count <> p_expected_count THEN
        RAISE EXCEPTION
            'Staging count mismatch: expected %, found %',
            p_expected_count, v_staged_count;
    END IF;

    -- Do not let an accidentally reused staging label delete unrelated rows.
    IF EXISTS (
        SELECT 1
          FROM public.npa_chunks c
         WHERE c.doc_name = p_staging_doc_name
           AND c.document_id IS DISTINCT FROM p_document_id
    ) THEN
        RAISE EXCEPTION 'Staging name is associated with another document';
    END IF;

    -- Delete old canonical and any legacy rows attached to this document.
    -- If any later statement fails, PostgreSQL rolls this deletion back.
    DELETE FROM public.npa_chunks c
     WHERE c.document_id = p_document_id
       AND c.doc_name IS DISTINCT FROM p_staging_doc_name;

    -- Move staging chunk numbers to a disjoint temporary range first. This
    -- avoids collisions with UNIQUE(document_id, chunk_no) while renumbering.
    UPDATE public.npa_chunks c
       SET chunk_no = -c.chunk_no
     WHERE c.document_id = p_document_id
       AND c.doc_name = p_staging_doc_name
       AND c.status = 'inactive';

    WITH ordered AS (
        SELECT c.id,
               row_number() OVER (ORDER BY -c.chunk_no, c.id)::integer AS new_chunk_no
          FROM public.npa_chunks c
         WHERE c.document_id = p_document_id
           AND c.doc_name = p_staging_doc_name
           AND c.status = 'inactive'
    )
    UPDATE public.npa_chunks c
       SET chunk_no = ordered.new_chunk_no,
           doc_name = p_doc_name,
           status = 'active'
      FROM ordered
     WHERE c.id = ordered.id;

    GET DIAGNOSTICS v_updated_count = ROW_COUNT;

    IF v_updated_count <> p_expected_count THEN
        RAISE EXCEPTION
            'Activation update count mismatch: expected %, updated %',
            p_expected_count, v_updated_count;
    END IF;

    SELECT count(*)::integer
      INTO v_final_count
      FROM public.npa_chunks c
     WHERE c.document_id = p_document_id
       AND c.doc_name = p_doc_name
       AND c.status = 'active';

    IF v_final_count <> p_expected_count THEN
        RAISE EXCEPTION
            'Final active count mismatch: expected %, found %',
            p_expected_count, v_final_count;
    END IF;

    RETURN v_final_count;
END;
$function$;

REVOKE ALL ON FUNCTION public.activate_npa_staging_document(uuid, text, text, integer) FROM PUBLIC;
REVOKE ALL ON FUNCTION public.activate_npa_staging_document(uuid, text, text, integer) FROM anon;
REVOKE ALL ON FUNCTION public.activate_npa_staging_document(uuid, text, text, integer) FROM authenticated;
GRANT EXECUTE ON FUNCTION public.activate_npa_staging_document(uuid, text, text, integer) TO service_role;


-- Legacy semantic RPCs are still used by compatibility/fallback paths.
-- Keep inactive staging chunks invisible through every supported search RPC.
CREATE OR REPLACE FUNCTION public.match_npa_chunks(
    query_embedding vector,
    match_threshold double precision,
    match_count integer
)
RETURNS TABLE(doc_name text, doc_type text, point_num text, content text, similarity double precision)
LANGUAGE sql
STABLE
SET search_path = ''
AS $function$
    SELECT
        n.doc_name,
        n.doc_type,
        n.point_num,
        n.content,
        1 - (n.embedding OPERATOR(public.<=>) query_embedding) AS similarity
    FROM public.npa_chunks AS n
    WHERE
        n.embedding IS NOT NULL
        AND COALESCE(n.status, 'active') = 'active'
        AND 1 - (n.embedding OPERATOR(public.<=>) query_embedding) >= match_threshold
    ORDER BY n.embedding OPERATOR(public.<=>) query_embedding
    LIMIT match_count;
$function$;

CREATE OR REPLACE FUNCTION public.match_npa_chunks_v2(
    query_embedding vector,
    match_threshold double precision,
    match_count integer,
    legal_domain_filter text
)
RETURNS TABLE(doc_name text, doc_type text, point_num text, content text, legal_domain text, similarity double precision)
LANGUAGE sql
STABLE
SET search_path = ''
AS $function$
    SELECT
        n.doc_name,
        n.doc_type,
        n.point_num,
        n.content,
        n.legal_domain,
        1 - (n.embedding OPERATOR(public.<=>) query_embedding) AS similarity
    FROM public.npa_chunks AS n
    WHERE
        n.embedding IS NOT NULL
        AND COALESCE(n.status, 'active') = 'active'
        AND n.legal_domain IN (legal_domain_filter, 'general')
        AND 1 - (n.embedding OPERATOR(public.<=>) query_embedding) >= match_threshold
    ORDER BY n.embedding OPERATOR(public.<=>) query_embedding
    LIMIT match_count;
$function$;

CREATE OR REPLACE FUNCTION public.match_npa_chunks_v3(
    query_embedding vector,
    match_threshold double precision,
    match_count integer,
    legal_domain_filter text,
    topic_filter text DEFAULT NULL
)
RETURNS TABLE(doc_name text, doc_type text, point_num text, content text, legal_domain text, topic text, similarity double precision)
LANGUAGE sql
STABLE
SET search_path = ''
AS $function$
    SELECT
        n.doc_name,
        n.doc_type,
        n.point_num,
        n.content,
        n.legal_domain,
        n.topic,
        1 - (n.embedding OPERATOR(public.<=>) query_embedding) AS similarity
    FROM public.npa_chunks AS n
    WHERE
        n.embedding IS NOT NULL
        AND COALESCE(n.status, 'active') = 'active'
        AND (n.legal_domain = legal_domain_filter OR n.legal_domain = 'general')
        AND (
            topic_filter IS NULL
            OR n.topic = topic_filter
            OR n.topic = 'general'
            OR n.topic IS NULL
        )
        AND 1 - (n.embedding OPERATOR(public.<=>) query_embedding) >= match_threshold
    ORDER BY n.embedding OPERATOR(public.<=>) query_embedding
    LIMIT match_count;
$function$;
