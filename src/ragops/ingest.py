"""Corpus loading.

The sample corpus is a support knowledge base for a fictional data-sync
product ("Fluxline"). It is deliberately full of the things that expose
dense retrieval's blind spots: error codes (FLX-401), acronyms (SCIM, MAR),
and formal doc-speak that users never type verbatim.
"""

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Passage:
    id: str
    title: str
    text: str

    @property
    def content(self) -> str:
        return f"{self.title}\n{self.text}"


def load_corpus(path: str | Path) -> list[Passage]:
    """Load a JSONL corpus with id / title / text fields."""
    passages = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            passages.append(Passage(id=row["id"], title=row["title"], text=row["text"]))
    if not passages:
        raise ValueError(f"no passages found in {path}")
    return passages
