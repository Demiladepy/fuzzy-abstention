"""Known-answer tests for selective_metrics. Run: python test_selective_metrics.py

No pytest needed, so it works in a bare Colab cell too.
"""

from __future__ import annotations

import numpy as np

from selective_metrics import (
    auroc,
    aurc,
    bootstrap_ci,
    ece_by_group,
    expected_calibration_error,
    risk_at_coverage,
    risk_coverage_curve,
)

failures: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        print(f"[PASS] {name}")
    else:
        print(f"[FAIL] {name} — {detail}")
        failures.append(name)


def close(a: float, b: float, tol: float = 1e-9) -> bool:
    return bool(abs(a - b) <= tol)


def auroc_bruteforce(scores, positive) -> float:
    """O(n^2) reference: P(score_pos > score_neg) + 0.5 * P(tie)."""
    scores = np.asarray(scores, dtype=float)
    pos = np.asarray(positive).astype(bool)
    p, n = scores[pos], scores[~pos]
    if p.size == 0 or n.size == 0:
        return float("nan")
    wins = ties = 0
    for a in p:
        for b in n:
            if a > b:
                wins += 1
            elif a == b:
                ties += 1
    return (wins + 0.5 * ties) / (p.size * n.size)


# ---------------------------------------------------------------- AUROC

check("auroc: perfect separation", close(auroc([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]), 1.0))
check("auroc: inverted", close(auroc([0.9, 0.8, 0.2, 0.1], [0, 0, 1, 1]), 0.0))
check("auroc: all tied is 0.5", close(auroc([0.5] * 6, [0, 1, 0, 1, 0, 1]), 0.5))
check("auroc: single class is nan", np.isnan(auroc([0.1, 0.9], [1, 1])))

rng = np.random.default_rng(7)
worst = 0.0
for _ in range(200):
    k = int(rng.integers(4, 40))
    # round to 1dp so ties actually occur — the case naive rank code gets wrong
    s = np.round(rng.random(k), 1)
    y = rng.integers(0, 2, k)
    if y.sum() in (0, k):
        continue
    worst = max(worst, abs(auroc(s, y) - auroc_bruteforce(s, y)))
check("auroc: matches brute force with ties", worst < 1e-12, f"max diff {worst:.2e}")

# ---------------------------------------------------------------- ECE

# Confidence exactly equals accuracy in every bin -> ECE 0.
conf = np.array([0.05] * 20 + [0.95] * 20)
corr = np.array([0] * 19 + [1] + [1] * 19 + [0])
check("ece: calibrated is ~0", expected_calibration_error(conf, corr) < 0.01)

# Always claims 1.0, right half the time -> ECE 0.5.
check(
    "ece: confidently half-wrong is ~0.5",
    close(expected_calibration_error(np.ones(100), np.tile([1, 0], 50)), 0.5, 1e-9),
)
check("ece: empty is nan", np.isnan(expected_calibration_error([], [])))

# The methodological claim from PLAN.md, as a test: a severely miscalibrated
# minority group can hide inside a healthy aggregate.
n_easy = 900
easy_conf = np.concatenate([np.full(n_easy // 2, 0.05), np.full(n_easy // 2, 0.95)])
easy_corr = np.concatenate([np.zeros(n_easy // 2), np.ones(n_easy // 2)])
n_hard = 100
hard_conf = np.full(n_hard, 0.95)       # confident...
hard_corr = np.zeros(n_hard)            # ...and always wrong
groups = np.array(["answerable_known"] * n_easy + ["context_dependent"] * n_hard)
res = ece_by_group(
    np.concatenate([easy_conf, hard_conf]),
    np.concatenate([easy_corr, hard_corr]),
    groups,
)
check(
    "ece_by_group: catches a group the aggregate hides",
    res["__overall__"] < 0.10 and res["context_dependent"] > 0.90,
    f"overall={res['__overall__']:.3f} context_dependent={res['context_dependent']:.3f}",
)
check("ece_by_group: reports every group plus overall", set(res) == {
    "answerable_known", "context_dependent", "__overall__"})

# ---------------------------------------------------------------- risk / coverage

# Confidence ranks all correct items above all wrong ones.
c = np.array([0.9, 0.8, 0.7, 0.2, 0.1])
y = np.array([1, 1, 1, 0, 0])
cov, risk = risk_coverage_curve(c, y)
check("risk_coverage: coverage ends at 1", close(cov[-1], 1.0))
check("risk_coverage: no risk while only correct answered", close(risk[2], 0.0))
check("risk_coverage: full coverage risk is base error", close(risk[-1], 0.4))
check("risk_at_coverage: 60% is clean", close(risk_at_coverage(c, y, 0.6), 0.0))
check("risk_at_coverage: 100% is base error", close(risk_at_coverage(c, y, 1.0), 0.4))
check("aurc: perfect ranking beats inverted", aurc(c, y) < aurc(-c, y))

try:
    risk_at_coverage(c, y, 0.0)
    check("risk_at_coverage: rejects coverage 0", False, "no error raised")
except ValueError:
    check("risk_at_coverage: rejects coverage 0", True)

try:
    expected_calibration_error([0.1, 0.2], [1])
    check("ece: rejects length mismatch", False, "no error raised")
except ValueError:
    check("ece: rejects length mismatch", True)

# ---------------------------------------------------------------- bootstrap

big_c = rng.random(400)
big_y = (rng.random(400) < big_c).astype(int)  # genuinely informative confidence
point, lo, hi = bootstrap_ci(auroc, big_c, big_y, n_boot=400, seed=1)
check("bootstrap: interval brackets the point", lo <= point <= hi, f"{lo:.3f} {point:.3f} {hi:.3f}")
check("bootstrap: detects real signal", lo > 0.5, f"lo={lo:.3f}")

point2, lo2, hi2 = bootstrap_ci(auroc, rng.random(400), rng.integers(0, 2, 400),
                                n_boot=400, seed=1)
check("bootstrap: noise interval straddles 0.5", lo2 < 0.5 < hi2, f"{lo2:.3f} {hi2:.3f}")

print("\n" + "-" * 60)
if failures:
    print(f"{len(failures)} FAILED: {', '.join(failures)}")
    raise SystemExit(1)
print("All checks passed.")
