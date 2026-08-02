# RagOps — a retrieval pipeline that evolves, with a quality gate per stage

Strip away the vector store and the reranker, and RAG is one thing: **structuring context so the model understands you.** That is communication, and the pipeline stages map onto what every effective communicator does — **Intent** (query rewriting), **Meaning** (dense retrieval), **Terminology** (sparse retrieval + fusion), **Emphasis** (reranking).

This repo is the reference implementation behind the essay: *[How to Speak to an LLM Effectively — ARTICLE LINK]*. Every stage is a toggleable flag, so you can replay the evolution yourself and watch the per-stage gates change.

## Let the failures pick your stages

A RAG should not be architected in full on day one — it should evolve, each stage added when a specific failure signal appears:

| Failure signal | Stage you add | Module |
|---|---|---|
| — (start here) | Dense retrieval — the MVP | `dense.py` |
| Acronyms & error codes miss | Sparse retrieval (BM25) | `sparse.py` |
| Two incomparable score scales | Reciprocal Rank Fusion | `fusion.py` |
| Right docs, wrong order | Cross-encoder reranking | `rerank.py` |
| Users don't speak document | Query rewriting (expand / HyDE) | `rewrite.py` |

One subtlety the build teaches you: query rewriting is built **last** but runs **first** — nothing downstream can search an untransformed query. See `pipeline.py`.

## A gate per stage

A demo has one quality bar: *does the answer look right?* Production needs a gate per stage, because "the answers got worse" is not a debuggable signal. `gates.py` gives each stage a metric it can own:

| Stage | Gate |
|---|---|
| Dense / sparse legs | recall@k |
| Hybrid fusion | precision@k |
| Reranking | precision lift over fusion + latency |
| Query rewriting | retrieval delta + intent-drift check |

Answer-level metrics (faithfulness / relevance / accuracy) belong at the final generation gate, outside this retrieval pipeline — they are not measurable after a retrieval-only stage.

Every `search()` returns a `Trace` — what each stage saw, produced, and how long it took. The gates consume traces. A pipeline you cannot trace is a pipeline you cannot operate.

## Quickstart

```bash
pip install -r requirements.txt
python examples/replay_the_evolution.py
```

The sample corpus (`data/kb.jsonl`) is a support knowledge base for a fictional data-sync product — deliberately full of the things that expose dense retrieval's blind spots: error codes (`FLX-401`), acronyms (`SCIM`, `MAR`), and doc-speak nobody types verbatim. `data/eval.jsonl` holds the labeled queries the gates run against.

Query rewriting needs an `OPENAI_API_KEY` (any chat-completions model); everything else runs fully local.

## Layout

```
src/ragops/
├── config.py     # every stage a toggleable flag
├── ingest.py     # corpus loading
├── dense.py      # FAISS semantic leg          (Meaning)
├── sparse.py     # BM25 leg                    (Terminology)
├── fusion.py     # RRF                         (Terminology)
├── rerank.py     # cross-encoder               (Emphasis)
├── rewrite.py    # expansion + HyDE            (Intent)
├── pipeline.py   # runtime assembly + Trace
└── gates.py      # per-stage quality gates — the RagOps part
```

## On the name

"RagOps" is also the subject of a 2025 survey — [*RAGOps: Operating and Managing Retrieval-Augmented Generation Pipelines*](https://arxiv.org/abs/2506.03401) — which maps this territory at research depth. This repo is the practitioner's version: one working pipeline, one gate per stage, small enough to read in an evening.

MIT licensed.
