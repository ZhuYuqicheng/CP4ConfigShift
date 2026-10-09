"""ScoreTable: nonconformity scores for every candidate answer of every example.

This is the single interface between scoring and conformal prediction, for
both task types:

* multiple-choice with logits — the K options are the candidates;
* sampling-based scores — the distinct sampled answers are the candidates
  (so the number of candidates varies per example).

Candidates of example ``i`` live at flat positions ``offsets[i]:offsets[i+1]``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class ScoreTable:
    ids: list[str]           # (n,) example ids
    candidates: list[str]    # (P,) candidate answers, flat
    scores: np.ndarray       # (P,) nonconformity scores; lower = more conforming
    correct: np.ndarray      # (P,) True where the candidate matches the gold answer
    offsets: np.ndarray      # (n + 1,) segment boundaries into the flat arrays
    info: dict = field(default_factory=dict)  # score name, parameters, run config

    def __post_init__(self) -> None:
        self.scores = np.asarray(self.scores, dtype=np.float64)
        self.correct = np.asarray(self.correct, dtype=bool)
        self.offsets = np.asarray(self.offsets, dtype=np.int64)
        self._example_of = np.repeat(np.arange(len(self.ids)), np.diff(self.offsets))

    def __len__(self) -> int:
        return len(self.ids)

    def calibration_scores(self) -> np.ndarray:
        """Score of the best-scoring correct candidate per example; +inf when no
        candidate is correct (e.g. the model never sampled the gold answer).

        An example is covered at threshold t exactly when this score is <= t.
        """
        r = np.full(len(self), np.inf)
        np.minimum.at(r, self._example_of[self.correct], self.scores[self.correct])
        return r

    def set_sizes(self, threshold: float) -> np.ndarray:
        """Number of candidates with score <= threshold, per example."""
        return np.bincount(self._example_of, weights=self.scores <= threshold,
                           minlength=len(self)).astype(np.int64)

    def prediction_sets(self, threshold: float) -> list[list[str]]:
        """Candidates with score <= threshold, per example."""
        keep = self.scores <= threshold
        return [
            [self.candidates[p] for p in range(lo, hi) if keep[p]]
            for lo, hi in zip(self.offsets[:-1], self.offsets[1:])
        ]

    def save(self, path: str | Path) -> None:
        np.savez_compressed(
            path, ids=np.asarray(self.ids), candidates=np.asarray(self.candidates),
            scores=self.scores, correct=self.correct, offsets=self.offsets,
            info=np.asarray(json.dumps(self.info)),
        )

    @classmethod
    def load(cls, path: str | Path) -> ScoreTable:
        with np.load(path) as z:
            return cls(
                ids=z["ids"].tolist(), candidates=z["candidates"].tolist(),
                scores=z["scores"], correct=z["correct"], offsets=z["offsets"],
                info=json.loads(str(z["info"])),
            )
