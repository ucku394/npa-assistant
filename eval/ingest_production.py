"""Normalize anonymised production questions into evaluation JSONL."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output")
    args = parser.parse_args()
    rows = []
    for number, line in enumerate(Path(args.input).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        raw = json.loads(line)
        question = str(raw.get("question", "")).strip()
        if not question:
            continue
        rows.append({
            "id": raw.get("id") or f"prod-{number:04d}",
            "question": question,
            "expected_documents": raw.get("expected_documents", []),
            "expected_points": raw.get("expected_points", []),
            "forbidden_documents": raw.get("forbidden_documents", []),
            "forbidden_phrases": raw.get("forbidden_phrases", []),
            "tags": raw.get("tags", []),
        })
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(
        "\n".join(json.dumps(x, ensure_ascii=False) for x in rows) + "\n",
        encoding="utf-8",
    )
    print(f"Prepared {len(rows)} production questions: {args.output}")

if __name__ == "__main__":
    main()
