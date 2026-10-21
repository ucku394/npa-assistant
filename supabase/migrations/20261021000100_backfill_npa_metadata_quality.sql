-- Backfill explicit metadata-quality and legal-source verification state.
-- Intentionally does not invent official URLs, issuing authorities, or validity dates.
update public.npa_documents d
set metadata = coalesce(d.metadata, '{}'::jsonb) || jsonb_build_object(
  'metadata_schema_version', 1,
  'metadata_quality', case when nullif(btrim(d.source_url), '') is null then 'incomplete' else 'review_required' end,
  'source_url_status', case when nullif(btrim(d.source_url), '') is null then 'missing' else 'present_unverified' end,
  'validity_status', case when d.effective_from is null and d.effective_to is null then 'not_verified' else 'partially_recorded' end,
  'metadata_review_required', (nullif(btrim(d.source_url), '') is null or d.effective_from is null),
  'metadata_review_reason', trim(both '; ' from concat(
      case when nullif(btrim(d.source_url), '') is null then 'source_url_missing;' else '' end,
      case when d.effective_from is null then 'effective_from_missing;' else '' end,
      case when d.issuer is null or btrim(d.issuer) = '' then 'issuer_missing;' else '' end,
      case when d.doc_number is null or btrim(d.doc_number) = '' then 'doc_number_missing;' else '' end
  )),
  'review_policy', 'Do not infer legal validity or official source URL; verify against Belarus official primary source'
),
updated_at = now()
where d.doc_name <> 'requirements';

update public.npa_chunks c
set metadata = coalesce(c.metadata, '{}'::jsonb) || jsonb_build_object(
  'metadata_schema_version', 1,
  'document_id', d.id::text,
  'document_name', d.doc_name,
  'document_number', d.doc_number,
  'legal_domain', d.legal_domain,
  'source_url_status', case when nullif(btrim(coalesce(c.source_url, d.source_url, '')), '') is null then 'missing' else 'present_unverified' end,
  'validity_status', case when d.effective_from is null and d.effective_to is null then 'not_verified' else 'partially_recorded' end,
  'metadata_review_required', (nullif(btrim(coalesce(c.source_url, d.source_url, '')), '') is null or d.effective_from is null),
  'review_policy', 'Source URL and validity must be confirmed from official Belarus primary source'
),
updated_at = now()
from public.npa_documents d
where c.document_id = d.id;
