"""Query rewriting — the intent rung.

Users speak colloquially; documents speak doc-speak. Two bridges:

- expand: rewrite the query toward document vocabulary
- hyde:   have the LLM draft a *hypothetical answer* and search with that,
          because an answer lives in the same vocabulary space as the docs

Built last on the ladder — but it runs FIRST at runtime, since nothing
downstream can search an untransformed query.
"""

_EXPAND_PROMPT = (
    "Rewrite this support-search query using the formal vocabulary a product "
    "knowledge base would use. Keep it one sentence. Preserve any error codes "
    "or identifiers exactly as written.\n\nQuery: {query}"
)

_HYDE_PROMPT = (
    "Write a short, plausible knowledge-base paragraph (2-3 sentences) that "
    "would answer this query. Do not say you are unsure; just write the "
    "hypothetical passage.\n\nQuery: {query}"
)


class QueryRewriter:
    def __init__(self, model: str, mode: str):
        if mode not in ("expand", "hyde"):
            raise ValueError(f"unsupported rewrite mode: {mode!r}")
        from openai import OpenAI

        self.client = OpenAI()
        self.model = model
        self.mode = mode

    def rewrite(self, query: str) -> str:
        prompt = (_EXPAND_PROMPT if self.mode == "expand" else _HYDE_PROMPT).format(query=query)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        text = (response.choices[0].message.content or "").strip()
        return text or query  # never let a bad rewrite blank the query
