from __future__ import annotations

def recall_at_k(hit_ranks, k):
    return 1.0 if any(rank <= k for rank in hit_ranks) else 0.0

def reciprocal_rank(hit_ranks):
    return 1.0 / min(hit_ranks) if hit_ranks else 0.0

def precision_at_k(hit_ranks, k):
    return sum(1 for rank in hit_ranks if rank <= k) / max(k, 1)

def mean(values):
    values = list(values)
    return sum(values) / len(values) if values else 0.0
