-- 003_fix_pgvector_operator.sql
-- Fix: explicit qualification of pgvector cosine-distance operator.
--
-- Root cause:
-- Security hardening in 002 set search_path='' on search functions.
-- PostgreSQL then could not resolve the unqualified <=> operator even though
-- both operands were public.vector:
--   operator does not exist: public.vector <=> public.vector
--
-- The pgvector extension and its <=> operator are installed in public.
-- Keep search_path empty and qualify the operator explicitly.

create or replace function public.semantic_search_npa_chunks(
    query_embedding vector(384),
    match_count integer default 40,
    domain_filter text default null,
    topic_filter text default null
)
returns table(
    id uuid,
    document_id uuid,
    doc_name text,
    doc_number text,
    doc_type text,
    issuer text,
    point_num text,
    content text,
    legal_domain text,
    topic text,
    document_priority integer,
    semantic_rank integer,
    similarity double precision
)
language sql
stable
set search_path = ''
as $function$
    select
        c.id,
        c.document_id,
        d.doc_name,
        d.doc_number,
        d.doc_type,
        d.issuer,
        c.point_num,
        c.content,
        c.legal_domain,
        c.topic,
        d.priority as document_priority,
        row_number() over (
            order by c.embedding OPERATOR(public.<=>) query_embedding
        )::integer as semantic_rank,
        (1 - (c.embedding OPERATOR(public.<=>) query_embedding))::double precision
            as similarity
    from public.npa_chunks c
    join public.npa_documents d
      on d.id = c.document_id
    where
        c.embedding is not null
        and coalesce(c.status, 'active') = 'active'
        and d.status = 'active'
        and (
            domain_filter is null
            or c.legal_domain = domain_filter
            or c.legal_domain = 'general'
        )
        and (
            topic_filter is null
            or c.topic = topic_filter
            or c.topic = 'general'
            or c.topic is null
        )
    order by c.embedding OPERATOR(public.<=>) query_embedding
    limit greatest(match_count, 1);
$function$;

-- hybrid_search_npa_chunks calls semantic_search_npa_chunks, so no rewrite
-- of the hybrid function itself is required.
