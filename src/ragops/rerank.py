"""Cross-encoder reranking — the emphasis rung.

A cross-encoder is NOT embedding-based: it reads the query and the passage
together and scores the pair jointly. That joint attention makes it more
accurate than any similarity comparison — and slow per pair, which is
exactly why it only ever sees the small fused candidate pool, never the
corpus. The cost profile is a design constraint, not a flaw.
"""

from ragops.ingest import Passage


class Reranker:
    def __init__(self, model_name: str):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, pool: list[Passage]) -> list[Passage]:
        """Reorder the candidate pool, best first."""
        if not pool:
            return pool
        scores = self.model.predict([(query, p.content) for p in pool])
        return [p for _, p in sorted(zip(scores, pool), key=lambda t: t[0], reverse=True)]
