"""Evaluating split CP under configuration shift.

Each configuration (prompt template, temperature, model variant, ...) yields
a ScoreTable over the same examples. For every pair (calibration config,
test config) and every random calibration/test split, we calibrate on the
calibration examples scored under the first config and measure coverage and
set size on the test examples scored under the second. Diagonal cells
(same config) are the i.i.d. reference; off-diagonal cells measure shift.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cp4configshift.conformal import conformal_threshold
from cp4configshift.table import ScoreTable


def make_splits(
    n: int, n_cal: int, n_test: int, n_trials: int,
    seed: int = 0, groups: list | np.ndarray | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Disjoint (calibration, test) index arrays, one pair per trial.

    With ``groups`` (e.g. the subject of each example), ``n_cal`` and
    ``n_test`` are drawn from every group separately.
    """
    rng = np.random.default_rng(seed)
    if groups is None:
        if n_cal + n_test > n:
            raise ValueError(f"n_cal + n_test = {n_cal + n_test} exceeds {n} examples")
        splits = []
        for _ in range(n_trials):
            perm = rng.permutation(n)
            splits.append((np.sort(perm[:n_cal]), np.sort(perm[n_cal:n_cal + n_test])))
        return splits

    groups = np.asarray(groups)
    members = [np.flatnonzero(groups == g) for g in sorted(set(groups.tolist()))]
    if min(len(m) for m in members) < n_cal + n_test:
        raise ValueError(f"every group needs at least n_cal + n_test = {n_cal + n_test} examples")
    splits = []
    for _ in range(n_trials):
        cal, test = [], []
        for m in members:
            perm = rng.permutation(len(m))
            cal.append(m[perm[:n_cal]])
            test.append(m[perm[n_cal:n_cal + n_test]])
        splits.append((np.concatenate(cal), np.concatenate(test)))
    return splits


@dataclass
class ShiftResult:
    configs: list[str]
    alpha: float
    coverage: np.ndarray   # (C, C, trials), indexed [calibration config, test config, trial]
    set_size: np.ndarray   # (C, C, trials)
    n_cal: np.ndarray      # (C, C, trials) calibration examples actually used
    n_test: np.ndarray     # (C, C, trials) test examples actually used

    def records(self) -> list[dict]:
        """One row per (calibration config, test config, trial), e.g. for pandas."""
        C, _, R = self.coverage.shape
        return [
            {"cal_config": self.configs[i], "test_config": self.configs[j], "trial": t,
             "coverage": self.coverage[i, j, t], "set_size": self.set_size[i, j, t],
             "n_cal": int(self.n_cal[i, j, t]), "n_test": int(self.n_test[i, j, t])}
            for i in range(C) for j in range(C) for t in range(R)
        ]

    def summary(self, tau: float | None = None) -> dict:
        """Mean coverage, mean set size, and undercoverage rate (fraction of
        runs with coverage < tau, default tau = 1 - alpha), for i.i.d. cells
        (same config) and shift cells (different configs)."""
        tau = 1 - self.alpha if tau is None else tau
        diag = np.eye(len(self.configs), dtype=bool)
        out = {}
        for name, mask in (("iid", diag), ("shift", ~diag)):
            cov = self.coverage[mask].ravel()
            size = self.set_size[mask].ravel()
            ok = ~np.isnan(cov)
            out[name] = {
                "coverage": float(cov[ok].mean()),
                "set_size": float(size[ok].mean()),
                "undercoverage_rate": float((cov[ok] < tau).mean()),
            }
        return out


def evaluate_shift(
    tables: dict[str, ScoreTable],
    alpha: float,
    splits: list[tuple[np.ndarray, np.ndarray]],
    answerable_only: bool = False,
) -> ShiftResult:
    """Coverage and set size for every (calibration config, test config, split).

    ``tables`` maps a config name to its ScoreTable; all tables must cover the
    same example ids. Split indices refer to the example order of the first
    table.

    With ``answerable_only``, each cell keeps only examples that have a correct
    candidate under both of its configs. This matters for sampling-based scores
    on open-ended tasks, where the model may never produce the gold answer.
    """
    names = list(tables)
    ids = tables[names[0]].ids
    order, r = {}, {}
    for name, t in tables.items():
        pos = {id_: p for p, id_ in enumerate(t.ids)}
        if len(t.ids) != len(ids) or any(id_ not in pos for id_ in ids):
            raise ValueError(f"config {name!r} does not cover the same examples as {names[0]!r}")
        order[name] = np.array([pos[id_] for id_ in ids])
        r[name] = t.calibration_scores()[order[name]]

    C, R = len(names), len(splits)
    coverage = np.full((C, C, R), np.nan)
    set_size = np.full((C, C, R), np.nan)
    n_cal = np.zeros((C, C, R), dtype=np.int64)
    n_test = np.zeros((C, C, R), dtype=np.int64)
    for t, (cal, test) in enumerate(splits):
        for i, a in enumerate(names):
            for j, b in enumerate(names):
                c, s = cal, test
                if answerable_only:
                    both = np.isfinite(r[a]) & np.isfinite(r[b])
                    c, s = cal[both[cal]], test[both[test]]
                n_cal[i, j, t], n_test[i, j, t] = len(c), len(s)
                if len(c) == 0 or len(s) == 0:
                    continue
                lam = conformal_threshold(r[a][c], alpha)
                coverage[i, j, t] = (r[b][s] <= lam).mean()
                set_size[i, j, t] = tables[b].set_sizes(lam)[order[b][s]].mean()
    return ShiftResult(names, alpha, coverage, set_size, n_cal, n_test)
