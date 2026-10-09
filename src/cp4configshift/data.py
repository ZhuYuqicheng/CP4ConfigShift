"""Examples: the unit of data every stage works on.

A dataset is a JSONL file with one example per line::

    {"id": "mmlu-0", "question": "Find the degree for the given field extension Q(sqrt(2), sqrt(3), sqrt(18)) over Q.",
     "choices": ["0", "4", "2", "6"], "answer": 1, "meta": {"subject": "abstract algebra"}}

* Multiple-choice: ``choices`` lists the options and ``answer`` is the index
  of the correct one.
* Open-ended: ``choices`` is omitted and ``answer`` is a list of acceptable
  answer strings (aliases).

``meta`` is free-form. Its string fields can be used as prompt placeholders
(e.g. ``{subject}``), and a matcher may read extra gold information from it
(e.g. a reference passage).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class Example:
    id: str
    question: str
    answer: int | list[str]
    choices: list[str] | None = None
    meta: dict = field(default_factory=dict)

    @property
    def is_multiple_choice(self) -> bool:
        return self.choices is not None


def load_examples(path: str | Path) -> list[Example]:
    with open(path) as f:
        examples = [Example(**json.loads(line)) for line in f if line.strip()]
    _validate(examples)
    return examples


def save_examples(examples: list[Example], path: str | Path) -> None:
    _validate(examples)
    with open(path, "w") as f:
        for ex in examples:
            f.write(json.dumps(asdict(ex)) + "\n")


def _validate(examples: list[Example]) -> None:
    """Catch the mistakes that would otherwise surface much later as silent
    misalignment: duplicate ids and answers of the wrong type."""
    ids = [ex.id for ex in examples]
    if len(set(ids)) != len(ids):
        raise ValueError("example ids must be unique")
    for ex in examples:
        if ex.is_multiple_choice:
            if not isinstance(ex.answer, int) or not 0 <= ex.answer < len(ex.choices):
                raise ValueError(f"{ex.id}: multiple-choice answer must be an index into choices")
        elif not (isinstance(ex.answer, list) and ex.answer):
            raise ValueError(f"{ex.id}: open-ended answer must be a non-empty list of strings")
