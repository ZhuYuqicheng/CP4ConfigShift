"""Split conformal prediction.
"""

from __future__ import annotations

import numpy as np

from cp4configshift.table import ScoreTable


def conformal_threshold(cal_scores: np.ndarray, alpha: float) -> float:
    s = np.asarray(cal_scores, dtype=np.float64)
    q = min(np.ceil((len(s) + 1) * (1 - alpha)) / len(s), 1.0)
    return float(np.quantile(s, q, method="higher"))


def calibrate(table: ScoreTable, alpha: float, idx: np.ndarray | None = None) -> float:
    """Threshold from the examples ``idx`` of ``table`` (all examples if None)."""
    r = table.calibration_scores()
    return conformal_threshold(r if idx is None else r[idx], alpha)


def evaluate(table: ScoreTable, threshold: float, idx: np.ndarray | None = None) -> dict:
    """Empirical coverage and mean set size on the examples ``idx``."""
    covered = table.calibration_scores() <= threshold
    sizes = table.set_sizes(threshold)
    if idx is not None:
        covered, sizes = covered[idx], sizes[idx]
    return {"coverage": float(covered.mean()), "set_size": float(sizes.mean())}
