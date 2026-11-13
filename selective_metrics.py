"""Selective-prediction and calibration metrics for the abstention experiment.

Pure numpy, no sklearn or scipy, so it runs in any Colab/Kaggle image and in CI.

The metric that carries the thesis is `ece_by_group`: the prediction is that scalar
confidence is *selectively* miscalibrated — fine on cleanly answerable questions and
bad on `context_dependent` / `underspecified` ones. An aggregate ECE averages that
away, which is exactly why it must be reported per label.

Conventions
-----------
confidence : float array in [0, 1]; higher means the system is more willing to answer.
correct    : bool/0-1 array; whether the emitted answer was right.
             Items the system abstained on are *excluded* from `correct` — pass only
             answered items to calibration functions. Selective metrics take the full
             set and do their own thresholding.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "average_ranks",
    "auroc",
    "expected_calibration_error",
    "ece_by_group",
    "risk_coverage_curve",
    "aurc",
    "risk_at_coverage",
    "bootstrap_ci",
]


def _as_float(a) -> np.ndarray:
    return np.asarray(a, dtype=float).ravel()


# np.trapezoid landed in numpy 2.0; Colab images still ship 1.26 with np.trapz.
_trapz = getattr(np, "trapezoid", None) or np.trapz


def average_ranks(x) -> np.ndarray:
    """Ranks starting at 1, with ties sharing their average rank.

    Ties matter here: verbalised confidences cluster hard on round numbers like
    0.9, and a rank function that broke ties arbitrarily would inflate AUROC.
    """
    x = _as_float(x)
    n = x.size
    order = np.argsort(x, kind="mergesort")
    sx = x[order]
    ranks = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sx[j + 1] == sx[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def auroc(scores, positive) -> float:
    """AUROC via the Mann-Whitney U statistic. Returns nan if either class is empty.

    `positive` is the class the score should rank *higher*. For hallucination
    detection, pass positive=1 for items the model got right, so a good confidence
    signal scores above 0.5.
    """
    scores = _as_float(scores)
    pos = np.asarray(positive).ravel().astype(bool)
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = average_ranks(scores)
    r_pos = ranks[pos].sum()
    return float((r_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def expected_calibration_error(confidence, correct, n_bins: int = 10) -> float:
    """Equal-width binned ECE: sum over bins of |accuracy - mean confidence|, weighted.

    Returns nan for an empty input rather than 0.0 — an empty slice is missing data,
    not perfect calibration, and reporting 0.0 would flatter a thin per-label cell.
    """
    confidence = _as_float(confidence)
    correct = _as_float(correct)
    if confidence.size == 0:
        return float("nan")
    if confidence.size != correct.size:
        raise ValueError(f"length mismatch: {confidence.size} vs {correct.size}")

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # digitize on the interior edges so values land in 0..n_bins-1, and 1.0 lands in
    # the last bin rather than overflowing past it.
    idx = np.digitize(confidence, edges[1:-1], right=False)
    total = 0.0
    for b in range(n_bins):
        m = idx == b
        k = int(m.sum())
        if k == 0:
            continue
        total += (k / confidence.size) * abs(correct[m].mean() - confidence[m].mean())
    return float(total)


def ece_by_group(confidence, correct, groups, n_bins: int = 10) -> dict[str, float]:
    """Per-group ECE plus the aggregate, as {group: ece, ..., '__overall__': ece}.

    This is the decisive table in the write-up. Report it whole: quoting only the
    aggregate hides precisely the effect the experiment is looking for.
    """
    confidence = _as_float(confidence)
    correct = _as_float(correct)
    groups = np.asarray(groups).ravel()
    if not (confidence.size == correct.size == groups.size):
        raise ValueError("confidence, correct and groups must be the same length")

    out: dict[str, float] = {}
    for g in sorted(set(groups.tolist())):
        m = groups == g
        out[str(g)] = expected_calibration_error(confidence[m], correct[m], n_bins)
    out["__overall__"] = expected_calibration_error(confidence, correct, n_bins)
    return out


def risk_coverage_curve(confidence, correct) -> tuple[np.ndarray, np.ndarray]:
    """Risk (error rate) as a function of coverage, answering most-confident first.

    Returns (coverage, risk) with one point per item, coverage running 1/n .. 1.
    """
    confidence = _as_float(confidence)
    correct = _as_float(correct)
    if confidence.size == 0:
        return np.empty(0), np.empty(0)
    if confidence.size != correct.size:
        raise ValueError(f"length mismatch: {confidence.size} vs {correct.size}")

    order = np.argsort(-confidence, kind="mergesort")
    errors = 1.0 - correct[order]
    n = confidence.size
    k = np.arange(1, n + 1, dtype=float)
    return k / n, np.cumsum(errors) / k


def aurc(confidence, correct) -> float:
    """Area under the risk-coverage curve. Lower is better; threshold-free."""
    cov, risk = risk_coverage_curve(confidence, correct)
    if cov.size == 0:
        return float("nan")
    return float(_trapz(risk, cov) / (cov[-1] - cov[0])) if cov.size > 1 else float(risk[0])


def risk_at_coverage(confidence, correct, coverage: float) -> float:
    """Error rate when answering the most-confident `coverage` fraction.

    This is the headline number: hallucination rate at 80% coverage is
    risk_at_coverage(conf, correct, 0.8).
    """
    if not 0.0 < coverage <= 1.0:
        raise ValueError("coverage must be in (0, 1]")
    cov, risk = risk_coverage_curve(confidence, correct)
    if cov.size == 0:
        return float("nan")
    # first point at or above the requested coverage
    i = int(np.searchsorted(cov, coverage, side="left"))
    return float(risk[min(i, risk.size - 1)])


def bootstrap_ci(
    fn,
    *arrays,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int | None = 0,
) -> tuple[float, float, float]:
    """Percentile bootstrap CI for any metric above. Returns (point, lo, hi).

    With ~1,500 items an unqualified "we improved X" will not survive a judge asking
    about variance, so every headline number in the write-up should carry one of these.
    """
    arrays = [np.asarray(a).ravel() for a in arrays]
    n = arrays[0].size
    if any(a.size != n for a in arrays):
        raise ValueError("all arrays must be the same length")
    point = float(fn(*arrays))
    if n == 0:
        return point, float("nan"), float("nan")

    rng = np.random.default_rng(seed)
    stats = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        stats[b] = fn(*[a[idx] for a in arrays])
    good = stats[~np.isnan(stats)]
    if good.size == 0:
        return point, float("nan"), float("nan")
    lo, hi = np.quantile(good, [alpha / 2, 1 - alpha / 2])
    return point, float(lo), float(hi)
