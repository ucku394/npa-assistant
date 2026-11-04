-- 001_npa_hybrid_search.sql
-- Safe additive migration for the existing npa_chunks table.
-- Keeps all legacy columns and legacy RPCs intact.

begin;

create extension if not exists vector;
create extension if not exists pgroonga;
create extension if not exists pg_trgm;

create table if not exists public.npa_documents (
    id uuid primary key default gen_random_uuid(),
    doc_name text not null,
    short_name text,
    doc_number text,
    doc_type text,
    issuer text,
    legal_domain text not null default 'general',
    status text not null default 'active',
    effective_from date,
    effective_to date,
    source_url text,
    priority integer not null default 50,
    search_text text,
    metadata jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

create unique index if not exists npa_documents_doc_name_uidx
on public.npa_documents (doc_name);
create index if not exists npa_documents_domain_idx on public.npa_documents (legal_domain);
create index if not exists npa_documents_status_idx on public.npa_documents (status);
create index if not exists npa_documents_number_idx on public.npa_documents (doc_number);
create index if not exists npa_documents_doc_name_trgm_idx
on public.npa_documents using gin (doc_name gin_trgm_ops);
create index if not exists npa_documents_doc_number_trgm_idx
on public.npa_documents using gin (doc_number gin_trgm_ops);

alter table public.npa_chunks
    add column if not exists document_id uuid,
    add column if not exists chunk_no integer,
    add column if not exists point_path text,
    add column if not exists content_normalized text,
    add column if not exists search_text text,
    add column if not exists metadata jsonb not null default '{}'::jsonb,
    add column if not exists updated_at timestamptz not null default now();

insert into public.npa_documents (
    doc_name, doc_type, issuer, legal_domain, status, source_url,
    priority, search_text
)
select
    doc_name,
    max(doc_type),
    max(issuer),
    coalesce(max(nullif(legal_domain, '')), 'general'),
    'active',
    max(source_url),
    case
        when doc_name ilike 'Трудовой кодекс%' then 100
        when doc_name ilike '%Правила по обеспечению СИЗ%' then 100
        when doc_name ilike '%№209%' then 100
        when doc_name ilike '%№253%' then 100
        when doc_name ilike '%№74%' then 100
        when doc_name ilike '%№175%' then 100
        when doc_name ilike '%№6%' then 100
        when doc_name ilike '%№30%' then 100
        when doc_name ilike '%№81%' then 100
        when doc_name = 'requirements' then 10
        else 70
    end,
    concat_ws(' ', doc_name, max(doc_type), max(issuer),
              coalesce(max(legal_domain), 'general'))
from public.npa_chunks
group by doc_name
on conflict (doc_name) do update
set doc_type = excluded.doc_type,
    issuer = excluded.issuer,
    legal_domain = excluded.legal_domain,
    source_url = excluded.source_url,
    updated_at = now();

update public.npa_chunks c
set document_id = d.id
from public.npa_documents d
where c.document_id is null
  and c.doc_name = d.doc_name;

with numbered as (
    select id,
           row_number() over (
               partition by doc_name
               order by point_num nulls last, id
           )::integer as rn
    from public.npa_chunks
)
update public.npa_chunks c
set chunk_no = n.rn
from numbered n
where c.id = n.id
  and c.chunk_no is null;

update public.npa_chunks
set legal_domain = coalesce(nullif(legal_domain, ''), 'general'),
    content_normalized = lower(
        regexp_replace(coalesce(content, ''), '[[:space:]]+', ' ', 'g')
    ),
    point_path = coalesce(point_path, point_num)
where legal_domain is null
   or legal_domain = ''
   or content_normalized is null
   or point_path is null;

update public.npa_chunks c
set search_text = concat_ws(
    ' ', c.doc_name, c.doc_type, c.issuer, c.point_num,
    c.legal_domain, c.topic, c.content_normalized, c.content
)
where c.search_text is null;

alter table public.npa_chunks
    alter column document_id set not null,
    alter column chunk_no set not null;

alter table public.npa_chunks
    drop constraint if exists npa_chunks_document_chunk_unique;
alter table public.npa_chunks
    add constraint npa_chunks_document_chunk_unique
    unique (document_id, chunk_no);

alter table public.npa_chunks
    drop constraint if exists npa_chunks_document_fk;
alter table public.npa_chunks
    add constraint npa_chunks_document_fk
    foreign key (document_id)
    references public.npa_documents(id)
    on delete cascade;

create index if not exists npa_chunks_document_id_idx on public.npa_chunks (document_id);
create index if not exists npa_chunks_domain_idx on public.npa_chunks (legal_domain);
create index if not exists npa_chunks_topic_idx on public.npa_chunks (topic);
create index if not exists npa_chunks_status_idx on public.npa_chunks (status);
create index if not exists npa_chunks_point_num_trgm_idx
on public.npa_chunks using gin (point_num gin_trgm_ops);

create index if not exists npa_chunks_embedding_hnsw_idx
on public.npa_chunks using hnsw (embedding vector_cosine_ops)
with (m = 16, ef_construction = 64);

create index if not exists npa_chunks_search_text_pgroonga_idx
on public.npa_chunks using pgroonga (search_text);

create or replace function public.npa_chunks_refresh_search_text()
returns trigger
language plpgsql
as $$
declare d public.npa_documents;
begin
    select * into d from public.npa_documents where id = new.document_id;
    new.content_normalized :=
        lower(regexp_replace(coalesce(new.content, ''), '[[:space:]]+', ' ', 'g'));
    new.point_path := coalesce(new.point_path, new.point_num);
    new.search_text := concat_ws(
        ' ', d.doc_name, d.short_name, d.doc_number, d.doc_type,
        d.issuer, new.point_num, new.point_path, new.legal_domain,
        new.topic, new.content_normalized, new.content
    );
    new.updated_at := now();
    return new;
end;
$$;

drop trigger if exists trg_npa_chunks_refresh_search_text on public.npa_chunks;
create trigger trg_npa_chunks_refresh_search_text
before insert or update of document_id, point_num, point_path, content,
legal_domain, topic, doc_name, doc_type, issuer
on public.npa_chunks
for each row execute function public.npa_chunks_refresh_search_text();

create or replace function public.npa_documents_refresh_search_text()
returns trigger
language plpgsql
as $$
begin
    new.search_text := concat_ws(
        ' ', new.doc_name, new.short_name, new.doc_number,
        new.doc_type, new.issuer, new.legal_domain
    );
    new.updated_at := now();
    return new;
end;
$$;

drop trigger if exists trg_npa_documents_refresh_search_text on public.npa_documents;
create trigger trg_npa_documents_refresh_search_text
before insert or update of doc_name, short_name, doc_number, doc_type,
issuer, legal_domain
on public.npa_documents
for each row execute function public.npa_documents_refresh_search_text();

update public.npa_documents
set search_text = concat_ws(
    ' ', doc_name, short_name, doc_number, doc_type, issuer, legal_domain
);

create or replace function public.semantic_search_npa_chunks(
    query_embedding vector(384),
    match_count integer default 40,
    domain_filter text default null,
    topic_filter text default null
)
returns table (
    id uuid, document_id uuid, doc_name text, doc_number text,
    doc_type text, issuer text, point_num text, content text,
    legal_domain text, topic text, document_priority integer,
    semantic_rank integer, similarity double precision
)
language sql stable
as $$
    select c.id, c.document_id, d.doc_name, d.doc_number, d.doc_type,
           d.issuer, c.point_num, c.content, c.legal_domain, c.topic,
           d.priority,
           row_number() over (order by c.embedding <=> query_embedding)::integer,
           (1 - (c.embedding <=> query_embedding))::double precision
    from public.npa_chunks c
    join public.npa_documents d on d.id = c.document_id
    where c.embedding is not null
      and coalesce(c.status, 'active') = 'active'
      and d.status = 'active'
      and (domain_filter is null or c.legal_domain = domain_filter
           or c.legal_domain = 'general')
      and (topic_filter is null or c.topic = topic_filter
           or c.topic = 'general' or c.topic is null)
    order by c.embedding <=> query_embedding
    limit greatest(match_count, 1);
$$;

create or replace function public.lexical_search_npa_chunks(
    search_query text,
    match_count integer default 40,
    domain_filter text default null,
    topic_filter text default null
)
returns table (
    id uuid, document_id uuid, doc_name text, doc_number text,
    doc_type text, issuer text, point_num text, content text,
    legal_domain text, topic text, document_priority integer,
    lexical_rank integer
)
language sql stable
as $$
    select c.id, c.document_id, d.doc_name, d.doc_number, d.doc_type,
           d.issuer, c.point_num, c.content, c.legal_domain, c.topic,
           d.priority,
           row_number() over (order by c.id)::integer
    from public.npa_chunks c
    join public.npa_documents d on d.id = c.document_id
    where coalesce(c.status, 'active') = 'active'
      and d.status = 'active'
      and nullif(trim(search_query), '') is not null
      and c.search_text &@~ search_query
      and (domain_filter is null or c.legal_domain = domain_filter
           or c.legal_domain = 'general')
      and (topic_filter is null or c.topic = topic_filter
           or c.topic = 'general' or c.topic is null)
    limit greatest(match_count, 1);
$$;

create or replace function public.hybrid_search_npa_chunks(
    query_embedding vector(384),
    search_query text,
    match_count integer default 50,
    domain_filter text default null,
    topic_filter text default null,
    semantic_limit integer default 40,
    lexical_limit integer default 40,
    rrf_k integer default 60
)
returns table (
    id uuid, document_id uuid, doc_name text, doc_number text,
    doc_type text, issuer text, point_num text, content text,
    legal_domain text, topic text, document_priority integer,
    semantic_rank integer, lexical_rank integer,
    semantic_score double precision, rrf_score double precision,
    final_score double precision
)
language sql stable
as $$
with semantic as (
    select * from public.semantic_search_npa_chunks(
        query_embedding, greatest(semantic_limit,1), domain_filter, topic_filter
    )
),
lexical as (
    select * from public.lexical_search_npa_chunks(
        search_query, greatest(lexical_limit,1), domain_filter, topic_filter
    )
),
combined as (
    select
        coalesce(s.id,l.id) id,
        coalesce(s.document_id,l.document_id) document_id,
        coalesce(s.doc_name,l.doc_name) doc_name,
        coalesce(s.doc_number,l.doc_number) doc_number,
        coalesce(s.doc_type,l.doc_type) doc_type,
        coalesce(s.issuer,l.issuer) issuer,
        coalesce(s.point_num,l.point_num) point_num,
        coalesce(s.content,l.content) content,
        coalesce(s.legal_domain,l.legal_domain) legal_domain,
        coalesce(s.topic,l.topic) topic,
        coalesce(s.document_priority,l.document_priority,50) document_priority,
        s.semantic_rank, l.lexical_rank,
        coalesce(s.similarity,0)::double precision semantic_score,
        (
            coalesce(1.0/(rrf_k+s.semantic_rank),0.0) +
            coalesce(1.0/(rrf_k+l.lexical_rank),0.0)
        )::double precision rrf_score
    from semantic s
    full outer join lexical l on l.id=s.id
)
select id, document_id, doc_name, doc_number, doc_type, issuer, point_num,
       content, legal_domain, topic, document_priority, semantic_rank,
       lexical_rank, semantic_score, rrf_score,
       (
          rrf_score * 0.60 +
          semantic_score * 0.25 +
          (least(greatest(document_priority,0),100)/100.0) * 0.10 +
          case when domain_filter is not null
                    and legal_domain = domain_filter then 0.05 else 0.0 end
       )::double precision final_score
from combined
order by final_score desc, semantic_score desc
limit greatest(match_count,1);
$$;

commit;
