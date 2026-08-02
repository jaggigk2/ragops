"""Sparse retrieval (BM25) — the terminology rung.

Term frequency, inverse document frequency, length normalization. No
semantics at all — which is exactly why it nails the error codes and
acronyms that embeddings fumble. Dense and sparse are not competitors;
they have complementary blind spots.
"""

import re

from ragops.ingest import Passage

_TOKEN = re.compile(r"[a-z0-9][a-z0-9\-]*")


def tokenize(text: str) -> list[str]:
    """Lowercase word tokenizer that keeps hyphenated codes (flx-401) intact."""
    return _TOKEN.findall(text.lower())


class SparseRetriever:
    def __init__(self, passages: list[Passage]):
        from rank_bm25 import BM25Okapi

        self.passages = passages
        self.bm25 = BM25Okapi([tokenize(p.content) for p in passages])

    def search(self, query: str, k: int) -> list[str]:
        """Return passage ids, best first."""
        scores = self.bm25.get_scores(tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [self.passages[i].id for i in ranked[:k]]
