"""Read-only quality audit for the Belarusian NPA RAG corpus.

Usage:
    python audit_npa_corpus.py
    python audit_npa_corpus.py --output npa_corpus_audit.json

Reads Supabase in pages and writes a local JSON report. It never changes rows
and deliberately does not guess legal metadata, effective dates, or official URLs.
"""
import argparse
import json
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from supabase import create_client


def fetch_all(table, columns, page_size=1000):
    rows = []
    start = 0
    while True:
        response = (
            client.table(table)
            .select(columns)
            .range(start, start + page_size - 1)
            .execute()
        )
        batch = response.data or []
        rows.extend(batch)
        if len(batch) < page_size:
            break
        start += page_size
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="npa_corpus_audit.json")
    args = parser.parse_args()

    load_dotenv()
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    if not url or not key:
        raise SystemExit("Set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY in the environment or .env file.")

    global client
    client = create_client(url, key)

    chunks = fetch_all(
        "npa_chunks",
        "id,doc_name,document_id,topic,legal_domain,status,source_url,metadata,embedding,search_text",
    )
    documents = fetch_all(
        "npa_documents",
        "id,doc_name,doc_number,doc_type,issuer,legal_domain,status,source_url,effective_from,effective_to,metadata",
    )

    chunks_by_doc = Counter(str(row.get("doc_name") or "<missing doc_name>") for row in chunks)
    chunk_count_by_document_id = Counter(str(row.get("document_id")) for row in chunks if row.get("document_id"))
    topics_by_doc = defaultdict(Counter)
    domains_by_doc = defaultdict(Counter)
    for row in chunks:
        doc = str(row.get("doc_name") or "<missing doc_name>")
        topics_by_doc[doc][str(row.get("topic") or "<missing>")] += 1
        domains_by_doc[doc][str(row.get("legal_domain") or "<missing>")] += 1

    topic_counts = Counter(str(row.get("topic") or "<missing>") for row in chunks)
    domain_counts = Counter(str(row.get("legal_domain") or "<missing>") for row in chunks)
    status_counts = Counter(str(row.get("status") or "<missing>") for row in chunks)

    chunk_ids = {str(row.get("document_id")) for row in chunks if row.get("document_id")}
    doc_ids = {str(row.get("id")) for row in documents if row.get("id")}
    missing_doc_links = sum(not row.get("document_id") for row in chunks)
    orphan_chunks = sum(
        bool(row.get("document_id")) and str(row.get("document_id")) not in doc_ids
        for row in chunks
    )

    document_rows = []
    for doc in documents:
        name = str(doc.get("doc_name") or "<missing doc_name>")
        missing = [
            field for field in ("doc_number", "issuer", "source_url", "effective_from")
            if not doc.get(field)
        ]
        count = chunk_count_by_document_id.get(str(doc.get("id")), 0)
        document_rows.append({
            "doc_name": name,
            "document_id": doc.get("id"),
            "doc_number": doc.get("doc_number"),
            "issuer": doc.get("issuer"),
            "legal_domain": doc.get("legal_domain"),
            "status": doc.get("status"),
            "effective_from": doc.get("effective_from"),
            "effective_to": doc.get("effective_to"),
            "source_url": doc.get("source_url"),
            "chunk_count_by_document_id": count,
            "missing_metadata_fields": missing,
            "topic_distribution": dict(topics_by_doc.get(name, {})),
            "domain_distribution": dict(domains_by_doc.get(name, {})),
        })

    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "read_only": True,
        "summary": {
            "chunk_count": len(chunks),
            "document_count": len(documents),
            "chunks_missing_topic": sum(not row.get("topic") for row in chunks),
            "chunks_missing_legal_domain": sum(not row.get("legal_domain") for row in chunks),
            "chunks_missing_source_url": sum(not row.get("source_url") for row in chunks),
            "chunks_missing_metadata": sum(not row.get("metadata") for row in chunks),
            "chunks_missing_embedding": sum(not row.get("embedding") for row in chunks),
            "chunks_missing_search_text": sum(not row.get("search_text") for row in chunks),
            "chunks_missing_document_id": missing_doc_links,
            "orphan_chunks": orphan_chunks,
            "documents_without_chunks_by_document_id": sum(
                chunk_count_by_document_id.get(str(doc.get("id")), 0) == 0
                for doc in documents
            ),
            "documents_missing_source_url": sum(not doc.get("source_url") for doc in documents),
            "documents_missing_effective_from": sum(not doc.get("effective_from") for doc in documents),
            "documents_missing_issuer": sum(not doc.get("issuer") for doc in documents),
            "documents_missing_doc_number": sum(not doc.get("doc_number") for doc in documents),
        },
        "topic_distribution": dict(topic_counts),
        "legal_domain_distribution": dict(domain_counts),
        "chunk_status_distribution": dict(status_counts),
        "documents": sorted(document_rows, key=lambda row: (row["chunk_count_by_document_id"], row["doc_name"])),
    }
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(f"Full report: {args.output}")


if __name__ == "__main__":
    main()
