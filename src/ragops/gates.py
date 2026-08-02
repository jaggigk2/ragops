"""Per-stage quality gates — the RagOps part.

A demo has one quality bar: does the answer look right? Production needs a
gate per stage, because "the answers got worse" is not a debuggable signal.
Each gate here owns one stage of the pipeline:

    dense / sparse legs -> recall@k
    fusion              -> precision@k
    reranking           -> precision lift over fusion (+ latency, from Trace)
    query rewriting     -> retrieval delta + intent-drift check

Answer-level metrics (faithfulness / relevance / accuracy) belong at the
final generation gate, which sits outside this retrieval pipeline — they
are not measurable after a retrieval-only stage.
"""

from dataclasses import dataclass

from ragops.pipeline import RagPipeline, Trace


@dataclass(frozen=True)
class EvalQuery:
    query: str
    relevant_ids: frozenset[str]


def recall_at_k(retrieved: list[str], relevant: frozenset[str], k: int) -> float:
    """Of the relevant passages, how many made the top k of this stage?"""
    if not relevant:
        return 0.0
    return len(set(retrieved[:k]) & relevant) / len(relevant)


def precision_at_k(retrieved: list[str], relevant: frozenset[str], k: int) -> float:
    """Of this stage's top k, how many are actually relevant?"""
    top = retrieved[:k]
    if not top:
        return 0.0
    return len(set(top) & relevant) / len(top)


def precision_lift(reranked: list[str], fused: list[str], relevant: frozenset[str], k: int) -> float:
    """What the reranker bought over fusion order. Negative = it hurt."""
    return precision_at_k(reranked, relevant, k) - precision_at_k(fused, relevant, k)


def retrieval_delta(with_rewrite: Trace, without_rewrite: Trace, relevant: frozenset[str], k: int) -> float:
    """What rewriting bought, measured at the final stage. Negative = it hurt."""
    return precision_at_k(with_rewrite.final_ids, relevant, k) - precision_at_k(
        without_rewrite.final_ids, relevant, k
    )


def intent_drift(original_query: str, rewritten_query: str) -> float:
    """Crude drift check: fraction of original content tokens the rewrite lost.

    0.0 = every original term survived; 1.0 = nothing survived. High drift is
    not automatically wrong (that's what expansion does) — it is a signal to
    LOOK, which is what a monitoring metric is for. Swap in an embedding
    similarity for a stronger check.
    """
    from ragops.sparse import tokenize

    original = set(tokenize(original_query))
    if not original:
        return 0.0
    kept = original & set(tokenize(rewritten_query))
    return 1.0 - len(kept) / len(original)


def run_gates(pipeline: RagPipeline, eval_set: list[EvalQuery], k: int = 5) -> dict[str, float]:
    """Run every applicable gate over an eval set. Returns stage -> mean metric.

    Read the report as a debugging map: if dense recall dropped, your
    embedding model or index changed; recall fine but fused precision down,
    look at fusion; precision fine but lift vanished, look at the reranker.
    """
    report: dict[str, list[float]] = {}

    def add(stage: str, value: float) -> None:
        report.setdefault(stage, []).append(value)

    for item in eval_set:
        _, trace = pipeline.search(item.query)
        add("dense_recall@k", recall_at_k(trace.dense_ids, item.relevant_ids, k))
        if trace.sparse_ids:
            add("sparse_recall@k", recall_at_k(trace.sparse_ids, item.relevant_ids, k))
        add("fused_precision@k", precision_at_k(trace.fused_ids, item.relevant_ids, k))
        if trace.reranked_ids:
            add("rerank_lift", precision_lift(trace.reranked_ids, trace.fused_ids, item.relevant_ids, k))
        if trace.effective_query != trace.original_query:
            add("rewrite_intent_drift", intent_drift(trace.original_query, trace.effective_query))
        for stage, ms in trace.timings_ms.items():
            add(f"latency_ms_{stage}", ms)

    return {stage: sum(values) / len(values) for stage, values in report.items()}
