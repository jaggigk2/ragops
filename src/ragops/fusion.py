"""Hybrid fusion via Reciprocal Rank Fusion (RRF).

Dense and sparse scores live on incomparable scales, so weighting them is
a tuning trap. RRF sidesteps it by fusing the *rankings*: a document both
legs rank highly floats to the top; a document only one leg loves still
survives into the pool.
"""

from collections import defaultdict


def rrf_fuse(rankings: list[list[str]], k: int = 60) -> list[str]:
    """Fuse ranked id lists. Standard RRF: score += 1 / (k + rank)."""
    scores: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] += 1.0 / (k + rank)
    return sorted(scores, key=lambda d: scores[d], reverse=True)
