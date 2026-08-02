"""Dense (semantic) retrieval — the MVP rung.

Catches paraphrase and meaning. Weak on exactly what enterprise queries are
full of: acronyms, error codes, IDs — see sparse.py for the complement.
"""

import numpy as np

from ragops.ingest import Passage


class DenseRetriever:
    def __init__(self, passages: list[Passage], model_name: str):
        # heavy imports kept local so the module imports without the deps
        import faiss
        from sentence_transformers import SentenceTransformer

        self.passages = passages
        self.model = SentenceTransformer(model_name)
        embeddings = self.model.encode(
            [p.content for p in passages],
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        embeddings = np.asarray(embeddings, dtype="float32")
        self.index = faiss.IndexFlatIP(embeddings.shape[1])  # cosine via normalized IP
        self.index.add(embeddings)

    def search(self, query: str, k: int) -> list[str]:
        """Return passage ids, best first."""
        q = self.model.encode([query], normalize_embeddings=True)
        q = np.asarray(q, dtype="float32")
        _, idx = self.index.search(q, min(k, len(self.passages)))
        return [self.passages[i].id for i in idx[0] if i != -1]
