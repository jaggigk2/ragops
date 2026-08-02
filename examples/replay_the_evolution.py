"""Replay the evolution: run the same eval set at each rung of the ladder.

Each config below turns on one more stage. Watch the gate report change —
that per-stage visibility is the whole point.

Usage:
    pip install -r requirements.txt
    python examples/replay_the_evolution.py

Query rewriting is off by default (it needs an OPENAI_API_KEY); flip
rewrite_mode to "expand" or "hyde" to add the last rung.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ragops.config import PipelineConfig
from ragops.gates import EvalQuery, run_gates
from ragops.ingest import load_corpus
from ragops.pipeline import RagPipeline

RUNGS = [
    ("1. dense only (the MVP)", PipelineConfig()),
    ("2. + sparse (BM25)", PipelineConfig(use_sparse=True)),
    ("3. + hybrid fusion (RRF)", PipelineConfig(use_sparse=True, use_hybrid_fusion=True)),
    (
        "4. + cross-encoder rerank",
        PipelineConfig(use_sparse=True, use_hybrid_fusion=True, use_reranker=True),
    ),
    # ("5. + query rewriting", PipelineConfig(use_sparse=True, use_hybrid_fusion=True,
    #                                         use_reranker=True, rewrite_mode="expand")),
]


def main() -> None:
    passages = load_corpus(ROOT / "data" / "kb.jsonl")
    eval_set = [
        EvalQuery(query=row["query"], relevant_ids=frozenset(row["relevant_ids"]))
        for row in map(json.loads, (ROOT / "data" / "eval.jsonl").read_text(encoding="utf-8").splitlines())
        if row
    ]

    for label, config in RUNGS:
        pipeline = RagPipeline(passages, config)
        report = run_gates(pipeline, eval_set, k=config.top_k)
        print(f"\n=== {label} ===")
        for stage, value in sorted(report.items()):
            print(f"  {stage:24s} {value:8.3f}")


if __name__ == "__main__":
    main()
