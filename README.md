# RagOps — a technique for turning a simple question into a great one, verified stage by stage

An answer is only as good as the question behind it. This repo is the reference implementation behind the essay [*Great Answers Don't Start With Great Models. They Start With Great Questions.*](https://medium.com/@jaggi.gk/great-answers-dont-start-with-great-models-they-start-with-great-questions-a32916471aae) — four moves that elevate a simple question into a great one, each with its own mechanism and its own check:

| Move | Mechanism | Module | Check |
|---|---|---|---|
| **Intent** — what you actually meant | Query rewriting (expand / HyDE) | `rewrite.py` | retrieval delta + intent-drift check |
| **Meaning** — beyond your exact words | Dense retrieval | `dense.py` | recall@k |
| **Terminology** — the domain's own term | Sparse retrieval (BM25) + fusion (RRF) | `sparse.py`, `fusion.py` | recall@k / precision@k |
| **Emphasis** — what matters most | Cross-encoder reranking | `rerank.py` | precision lift over fusion + latency |

Every move is a toggleable flag, so you can replay the elevation yourself and watch each check change — see `examples/replay_the_evolution.py`.

## You ask simple. The system builds great.

This isn't a cleverer prompt — a prompt is one attempt at phrasing. It's a mechanism: your simple question goes in, and Intent, Meaning, Terminology, and Emphasis do the elevation on your behalf, before anything answers you. One subtlety the build teaches: query rewriting (Intent) is built **last** in the ladder but runs **first** at runtime — nothing downstream can search an untransformed query. See `pipeline.py`.

## A gate per move, not one score at the end

A demo has one quality bar: *does the answer look right?* This technique needs a check per move, because "the answer got worse" doesn't say which move failed. `gates.py` gives each move a metric it owns; every `search()` returns a `Trace` — what each move saw, produced, and how long it took. A pipeline you cannot trace is a technique you cannot verify.

Answer-level metrics (faithfulness / relevance / accuracy) belong at the final generation gate, outside this retrieval pipeline — they aren't measurable after a retrieval-only move.

**Proof, not a demo:** running the included eval set (`data/eval.jsonl`, 8 queries) shows fused precision genuinely dip after adding Terminology (0.250 → 0.225) before Emphasis recovers it exactly (+0.025, back to 0.250) — a real regression a single end-to-end score would hide. Small sample, illustrative rather than a benchmark, but the numbers are unedited output from this code.

## Quickstart

```bash
pip install -r requirements.txt
python examples/replay_the_evolution.py
```

The sample corpus (`data/kb.jsonl`) is a support knowledge base for a fictional data-sync product — deliberately full of the things that expose dense retrieval's blind spots: error codes (`FLX-401`), acronyms (`SCIM`, `MAR`), and doc-speak nobody types verbatim.

Query rewriting needs an `OPENAI_API_KEY` (any chat-completions model); everything else runs fully local.

## Layout

```
src/ragops/
├── config.py     # every move a toggleable flag
├── ingest.py     # corpus loading
├── dense.py      # FAISS semantic leg          (Meaning)
├── sparse.py     # BM25 leg                    (Terminology)
├── fusion.py     # RRF                         (Terminology)
├── rerank.py     # cross-encoder               (Emphasis)
├── rewrite.py    # expansion + HyDE            (Intent)
├── pipeline.py   # runtime assembly + Trace
└── gates.py      # per-move quality gates — the RagOps part
```

## On the name

"RagOps" is also the subject of a 2025 survey — [*RAGOps: Operating and Managing Retrieval-Augmented Generation Pipelines*](https://arxiv.org/abs/2506.03401) — which maps this territory at research depth. This repo converged on the same name independently, from a working build rather than the literature: one pipeline, one gate per move, small enough to read in an evening.

MIT licensed.
