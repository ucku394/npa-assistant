# RAG evaluation

## Run

```bash
python evaluation/rag_eval.py evaluation/questions.json
```

The runner sends each case through `ChatService.process_text`. It reports topic classification accuracy, expected-document and expected-point recall, optional source-ID recall, Evidence Gate pass rate, verification-metadata coverage, and per-category results. Failures include request IDs and expected-versus-retrieved documents/points to support diagnosis.

## Dataset rules

Each case must have a unique `id` and non-empty `question`. Recommended fields:

- `category`: stable area used for category-level reporting.
- `jurisdiction`: use `BY` for Belarus-law cases; comparative questions must be explicitly labeled.
- `expected_topic`: expected classifier topic.
- `expected_documents`: recognizable document identifiers/titles expected in retrieved evidence.
- `expected_points`: exact normalized point labels, only where confirmed from the source.
- `expected_source_ids`: optional stable chunk/source identifiers.
- `reference_verified`: true only after a reviewer checks the expectation against an official Belarusian legal source.
- `reference_url` and `verified_on`: required provenance for verified cases.

Do not mark an expected document or point as verified merely because it was returned by the current RAG pipeline. That would make the test circular. Every official-reference expectation should be checked independently against the authoritative legal publication and applicable version/date.

## Current baseline

The original 20 cases are retained as a regression baseline. Their expected topics/documents/points are legacy expectations and are explicitly marked as pending official-source review; they are not counted as independently verified legal ground truth. Build the 200–300-case suite by adding reviewed cases across occupational safety, fire safety, industrial safety, medical examinations, training/briefings, PPE, work at height, equipment-specific requirements, accident investigation, and document validity. Include paraphrases and hard negatives, but do not inflate the set with unverified question variants.
