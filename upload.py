"""
NPA document indexer.

This version uses exactly the same local embedding model as bot.py:
intfloat/multilingual-e5-base, 768 dimensions.

Before the first run after migrating from Gemini embeddings, set:
    REINDEX_ALL=true
This removes old vectors for the documents being indexed and recreates them
with the E5 model. Do NOT mix Gemini and E5 vectors in the same vector index.
"""

import hashlib
import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

from dotenv import load_dotenv
from docx import Document
from supabase import create_client

from embedding import EMBEDDING_DIM, EMBEDDING_MODEL, get_document_embeddings


load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL", "").strip()
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip()

BATCH_SIZE = int(os.getenv("UPLOAD_BATCH_SIZE", "32"))
REINDEX_ALL = os.getenv("REINDEX_ALL", "false").strip().lower() in {
    "1", "true", "yes", "y", "on"
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


# Keep the original mapping here. Keys are normalized filename stems.
# Add/change entries to match the actual names of your source files.
DOC_NAME_MAP: Dict[str, str] = {
    # Examples:
    # "175": "Название НПА",
    # "some_file_name": "Название НПА",
}


def normalize_filename(value: str) -> str:
    value = Path(value).stem.strip()
    value = re.sub(r"\s+", " ", value)
    return value.casefold()


def resolve_doc_name(file_path: Path) -> str:
    stem = normalize_filename(file_path.name)

    normalized_map = {
        normalize_filename(key): value.strip()
        for key, value in DOC_NAME_MAP.items()
        if str(value).strip()
    }

    if stem in normalized_map:
        return normalized_map[stem]

    # If there is no mapping, use the readable filename itself instead of
    # silently skipping a valid document.
    fallback = re.sub(r"\s+", " ", file_path.stem).strip()
    logger.warning(
        "No DOC_NAME_MAP entry for %r. Using filename as doc_name: %r",
        file_path.stem,
        fallback,
    )
    return fallback


def read_docx(path: Path) -> str:
    doc = Document(path)
    parts: List[str] = []

    for paragraph in doc.paragraphs:
        text = re.sub(r"\s+", " ", paragraph.text).strip()
        if text:
            parts.append(text)

    # NPA files sometimes contain important requirements in tables.
    for table in doc.tables:
        for row in table.rows:
            cells = []
            for cell in row.cells:
                text = re.sub(r"\s+", " ", cell.text).strip()
                cells.append(text)
            row_text = " | ".join(x for x in cells if x)
            if row_text:
                parts.append(row_text)

    return "\n".join(parts)


def read_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig", errors="replace")


def read_source(path: Path) -> str:
    if path.suffix.lower() == ".docx":
        return read_docx(path)
    if path.suffix.lower() == ".txt":
        return read_txt(path)
    raise ValueError(f"Unsupported file type: {path.suffix}")


_ARTICLE_RE = re.compile(r"^\s*Статья\s+([\d.]+)\b", re.IGNORECASE)
_POINT_RE = re.compile(r"^\s*(?:Пункт|П\.?)\s*([\d.]+)\b", re.IGNORECASE)
_NUMBER_RE = re.compile(r"^\s*(\d+(?:\.\d+)*)[.)]?\s+")


def split_text_into_chunks(text: str) -> List[Tuple[str, str]]:
    """
    Split mainly on explicit NPA article/point markers.

    Returns (point_num, content).
    """
    lines = [re.sub(r"[ \t]+", " ", x).strip() for x in text.splitlines()]
    lines = [x for x in lines if x]

    chunks: List[Tuple[str, str]] = []
    current_point = "Без номера"
    current: List[str] = []

    def flush():
        nonlocal current
        if not current:
            return
        content = "\n".join(current).strip()
        if content:
            chunks.append((current_point, content))
        current = []

    for line in lines:
        m = _ARTICLE_RE.match(line)
        if m:
            flush()
            current_point = f"Статья {m.group(1)}"
            current.append(line)
            continue

        m = _POINT_RE.match(line)
        if m:
            flush()
            current_point = m.group(1)
            current.append(line)
            continue

        m = _NUMBER_RE.match(line)
        if m:
            # Only treat numbered lines as boundaries when they are reasonably
            # short headings/points. Long numbered prose remains in the chunk.
            if len(line) <= 300:
                flush()
                current_point = m.group(1)
                current.append(line)
                continue

        current.append(line)

    flush()

    # Safety fallback for very unusual documents.
    if not chunks and text.strip():
        return [("Без номера", text.strip())]

    return chunks


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def get_existing_content_hashes(supabase, doc_name: str) -> set:
    """
    Uses pagination so large documents do not depend on one oversized response.
    """
    hashes = set()
    start = 0
    page_size = 1000

    while True:
        response = (
            supabase.table("npa_chunks")
            .select("content")
            .eq("doc_name", doc_name)
            .range(start, start + page_size - 1)
            .execute()
        )
        rows = response.data or []
        for row in rows:
            content = row.get("content")
            if content:
                hashes.add(sha256_text(content))

        if len(rows) < page_size:
            break
        start += page_size

    return hashes


def delete_document_vectors(supabase, doc_name: str) -> None:
    supabase.table("npa_chunks").delete().eq("doc_name", doc_name).execute()
    logger.info("Deleted old vectors for %s", doc_name)


def insert_rows(supabase, rows: List[dict]) -> None:
    for start in range(0, len(rows), BATCH_SIZE):
        batch = rows[start:start + BATCH_SIZE]
        supabase.table("npa_chunks").insert(batch).execute()
        logger.info("Inserted %d/%d chunks", min(start + BATCH_SIZE, len(rows)), len(rows))


def process_file(supabase, path: Path) -> Tuple[int, int]:
    doc_name = resolve_doc_name(path)
    logger.info("Processing %s -> %s", path.name, doc_name)

    if REINDEX_ALL:
        delete_document_vectors(supabase, doc_name)

    source = read_source(path)
    if not source.strip():
        logger.warning("Empty source: %s", path)
        return 0, 0

    chunks = split_text_into_chunks(source)
    if not chunks:
        logger.warning("No chunks produced: %s", path)
        return 0, 0

    existing = set() if REINDEX_ALL else get_existing_content_hashes(supabase, doc_name)

    pending = []
    skipped = 0

    for point_num, content in chunks:
        content_hash = sha256_text(content)
        if content_hash in existing:
            skipped += 1
            continue
        pending.append((point_num, content, content_hash))

    if not pending:
        logger.info("Nothing new for %s; skipped=%d", doc_name, skipped)
        return 0, skipped

    uploaded = 0

    for start in range(0, len(pending), BATCH_SIZE):
        part = pending[start:start + BATCH_SIZE]
        texts = [item[1] for item in part]
        embeddings = get_document_embeddings(texts)

        if len(embeddings) != len(part):
            raise RuntimeError(
                f"Embedding count mismatch for {path.name}: "
                f"{len(embeddings)} != {len(part)}"
            )

        rows = []
        for (point_num, content, _), embedding in zip(part, embeddings):
            if len(embedding) != EMBEDDING_DIM:
                raise RuntimeError(
                    f"Wrong vector dimension for {path.name}: "
                    f"{len(embedding)} != {EMBEDDING_DIM}"
                )
            rows.append(
                {
                    "doc_name": doc_name,
                    "doc_type": "НПА",
                    "point_num": point_num,
                    "content": content,
                    "embedding": embedding,
                }
            )

        insert_rows(supabase, rows)
        uploaded += len(rows)

    logger.info(
        "Done: %s | uploaded=%d | skipped=%d | model=%s",
        doc_name, uploaded, skipped, EMBEDDING_MODEL
    )
    return uploaded, skipped


def validate_config():
    missing = []
    if not SUPABASE_URL:
        missing.append("SUPABASE_URL")
    if not SUPABASE_SERVICE_ROLE_KEY:
        missing.append("SUPABASE_SERVICE_ROLE_KEY")
    if missing:
        raise RuntimeError("Missing environment variables: " + ", ".join(missing))


def main():
    validate_config()

    supabase = create_client(
        SUPABASE_URL,
        SUPABASE_SERVICE_ROLE_KEY,
    )

    base_dir = Path(__file__).resolve().parent
    files = sorted(
        p for p in base_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".docx", ".txt"}
    )

    if not files:
        logger.warning("No .docx/.txt files found in %s", base_dir)
        return

    logger.info(
        "Indexing %d files with %s (%d dims), REINDEX_ALL=%s",
        len(files), EMBEDDING_MODEL, EMBEDDING_DIM, REINDEX_ALL
    )

    total_uploaded = 0
    total_skipped = 0

    for path in files:
        try:
            uploaded, skipped = process_file(supabase, path)
            total_uploaded += uploaded
            total_skipped += skipped
        except Exception:
            logger.exception("Failed to process %s", path)

    logger.info(
        "Finished. uploaded=%d skipped=%d",
        total_uploaded, total_skipped
    )


if __name__ == "__main__":
    main()
