"""Runtime assembly.

Build order and runtime order are not the same: query rewriting is built
last on the ladder but runs FIRST here. The retrieval portion keeps its
relative order (dense/sparse legs -> fusion -> rerank).

Every search returns a Trace alongside the results — the per-stage record
that the quality gates in gates.py consume. A pipeline you cannot trace
is a pipeline you cannot operate.
"""

import time
from dataclasses import dataclass, field

from ragops.config import PipelineConfig
from ragops.ingest import Passage


@dataclass
class Trace:
    """What each stage saw and produced, with per-stage latency in ms."""

    original_query: str
    effective_query: str
    dense_ids: list[str] = field(default_factory=list)
    sparse_ids: list[str] = field(default_factory=list)
    fused_ids: list[str] = field(default_factory=list)
    reranked_ids: list[str] = field(default_factory=list)
    final_ids: list[str] = field(default_factory=list)
    timings_ms: dict[str, float] = field(default_factory=dict)


class RagPipeline:
    def __init__(self, passages: list[Passage], config: PipelineConfig):
        config.validate()
        self.config = config
        self.passages = {p.id: p for p in passages}

        from ragops.dense import DenseRetriever

        self.dense = DenseRetriever(passages, config.embedding_model)

        self.sparse = None
        if config.use_sparse:
            from ragops.sparse import SparseRetriever

            self.sparse = SparseRetriever(passages)

        self.reranker = None
        if config.use_reranker:
            from ragops.rerank import Reranker

            self.reranker = Reranker(config.cross_encoder_model)

        self.rewriter = None
        if config.rewrite_mode != "off":
            from ragops.rewrite import QueryRewriter

            self.rewriter = QueryRewriter(config.rewrite_llm, config.rewrite_mode)

    def search(self, query: str) -> tuple[list[Passage], Trace]:
        cfg = self.config
        trace = Trace(original_query=query, effective_query=query)

        # Stage: query rewriting — built last, runs first
        if self.rewriter is not None:
            t0 = time.perf_counter()
            trace.effective_query = self.rewriter.rewrite(query)
            trace.timings_ms["rewrite"] = (time.perf_counter() - t0) * 1000

        q = trace.effective_query

        # Stage: dense leg
        t0 = time.perf_counter()
        trace.dense_ids = self.dense.search(q, cfg.leg_k)
        trace.timings_ms["dense"] = (time.perf_counter() - t0) * 1000

        # Stage: sparse leg
        if self.sparse is not None:
            t0 = time.perf_counter()
            trace.sparse_ids = self.sparse.search(q, cfg.leg_k)
            trace.timings_ms["sparse"] = (time.perf_counter() - t0) * 1000

        # Stage: fusion (falls back to the dense ranking when disabled)
        if cfg.use_hybrid_fusion and self.sparse is not None:
            from ragops.fusion import rrf_fuse

            t0 = time.perf_counter()
            trace.fused_ids = rrf_fuse([trace.dense_ids, trace.sparse_ids], k=cfg.rrf_k)
            trace.timings_ms["fusion"] = (time.perf_counter() - t0) * 1000
        else:
            trace.fused_ids = list(trace.dense_ids)

        # Stage: reranking — only ever the small fused pool, never the corpus
        pool_ids = trace.fused_ids[: cfg.rerank_pool_size]
        if self.reranker is not None:
            t0 = time.perf_counter()
            pool = [self.passages[i] for i in pool_ids]
            reranked = self.reranker.rerank(trace.effective_query, pool)
            trace.reranked_ids = [p.id for p in reranked]
            trace.timings_ms["rerank"] = (time.perf_counter() - t0) * 1000
            trace.final_ids = trace.reranked_ids[: cfg.top_k]
        else:
            trace.final_ids = pool_ids[: cfg.top_k]

        return [self.passages[i] for i in trace.final_ids], trace
