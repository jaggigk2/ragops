"""Pipeline configuration — every stage is a toggleable flag.

The point of the flags: each stage can be A/B'd against the pipeline that
existed before it. Turn one flag on, rerun the gates, and you can see exactly
what that stage bought you (or cost you).
"""

from dataclasses import dataclass


@dataclass
class PipelineConfig:
    # --- retrieval stages (build order, bottom of the ladder upward) ---
    use_sparse: bool = False          # BM25 leg — catches acronyms/IDs dense misses
    use_hybrid_fusion: bool = False   # RRF over dense + sparse rankings
    use_reranker: bool = False        # cross-encoder over the fused pool
    rewrite_mode: str = "off"         # "off" | "expand" | "hyde"

    # --- sizes ---
    top_k: int = 5                    # results returned to the caller
    leg_k: int = 20                   # candidates each retrieval leg contributes
    rerank_pool_size: int = 10        # cross-encoder never sees more than this

    # --- models ---
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rewrite_llm: str = "gpt-4o-mini"  # any chat-completions model works

    # --- fusion ---
    rrf_k: int = 60                   # standard RRF constant

    def validate(self) -> None:
        if self.rewrite_mode not in ("off", "expand", "hyde"):
            raise ValueError(f"unknown rewrite_mode: {self.rewrite_mode!r}")
        if self.use_hybrid_fusion and not self.use_sparse:
            raise ValueError("hybrid fusion needs the sparse leg enabled")
        if self.rerank_pool_size < self.top_k:
            raise ValueError("rerank_pool_size must be >= top_k")
