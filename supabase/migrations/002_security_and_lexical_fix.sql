-- 002_security_and_lexical_fix
-- Security hardening + PGroonga relevance ranking for the NPA hybrid search.
--
-- Applied to production Supabase project:
-- whegmwebreuicdipwdgu
--
-- This migration:
--   1. Enables RLS on npa_documents and keeps direct Data API access denied.
--   2. Adds restrictive deny policies to npa_chunks/npa_documents.
--   3. Fixes mutable search_path warnings on all NPA RPC/trigger functions.
--   4. Replaces UUID-based lexical ranking with real PGroonga score ranking.
--
-- Existing data, embeddings, indexes and RPC names are preserved.

begin;

-- ------------------------------------------------------------
-- 1. RLS: keep NPA tables non-readable through direct Data API
-- ------------------------------------------------------------
alter table public.npa_documents enable row level security;
alter table public.npa_chunks enable row level security;

drop policy if exists "deny_direct_npa_documents_access" on public.npa_documents;
create policy "deny_direct_npa_documents_access"
on public.npa_documents
as restrictive
for all
to anon, authenticated
using (false)
with check (false);

drop policy if exists "deny_direct_npa_chunks_access" on public.npa_chunks;
create policy "deny_direct_npa_chunks_access"
on public.npa_chunks
as restrictive
for all
to anon, authenticated
using (false)
with check (false);

-- Defense in depth: the bot should use the server-side/service role
-- through the RPC layer rather than direct table access.
revoke all on table public.npa_documents from anon, authenticated;
revoke all on table public.npa_chunks from anon, authenticated;

-- ------------------------------------------------------------
-- 2. Fix mutable search_path warnings
-- ------------------------------------------------------------
alter function public.match_npa_chunks(vector, double precision, integer)
    set search_path = '';

alter function public.match_npa_chunks_v2(vector, double precision, integer, text)
    set search_path = '';

alter function public.match_npa_chunks_v3(vector, double precision, integer, text, text)
    set search_path = '';

alter function public.npa_chunks_refresh_search_text()
    set search_path = '';

alter function public.npa_documents_refresh_search_text()
    set search_path = '';

alter function public.semantic_search_npa_chunks(vector, integer, text, text)
    set search_path = '';

alter function public.lexical_search_npa_chunks(text, integer, text, text)
    set search_path = '';

alter function public.hybrid_search_npa_chunks(vector, text, integer, text, text, integer, integer, integer)
    set search_path = '';

-- ------------------------------------------------------------
-- 3. Fix lexical ranking: use real PGroonga relevance score
--
-- Previously lexical_rank was based on UUID ordering:
--     row_number() over (order by c.id)
--
-- That made the lexical side of RRF effectively arbitrary.
-- PGroonga exposes pgroonga_score(tableoid, ctid), which ranks
-- matching records by search precision.
-- ------------------------------------------------------------
create or replace function public.lexical_search_npa_chunks(
    search_query text,
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
    lexical_rank integer
)
language sql
stable
set search_path = ''
as $function$
    with matched as (
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
            public.pgroonga_score(c.tableoid, c.ctid)::double precision as lexical_score
        from public.npa_chunks c
        join public.npa_documents d
          on d.id = c.document_id
        where
            coalesce(c.status, 'active') = 'active'
            and d.status = 'active'
            and nullif(trim(search_query), '') is not null
            and c.search_text OPERATOR(public.&@~) search_query
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
    ),
    ranked as (
        select
            m.*,
            row_number() over (
                order by m.lexical_score desc, m.id
            )::integer as lexical_rank
        from matched m
    )
    select
        r.id,
        r.document_id,
        r.doc_name,
        r.doc_number,
        r.doc_type,
        r.issuer,
        r.point_num,
        r.content,
        r.legal_domain,
        r.topic,
        r.document_priority,
        r.lexical_rank
    from ranked r
    order by r.lexical_rank
    limit greatest(match_count, 1);
$function$;

commit;
