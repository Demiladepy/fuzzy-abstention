"""Stage 3: the "before" numbers — how well scalar confidence tracks correctness.

    py -3.12 analyze.py --graded runs_dl/runs/default/graded.jsonl

Two scalar baselines, the family the thesis argues against:
    verbalised   the model's own 0-100 self-report, already a probability
    seq_prob     exp(mean answer logprob), a sequence-likelihood proxy in [0,1]

The decisive output is ECE **per label**. The prediction is that scalar confidence is
selectively miscalibrated: tolerable where the question has one clean answer, bad where
it is context-dependent or underspecified. An aggregate number averages that away.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

# Windows consoles default to cp1252 and raise UnicodeEncodeError on the first
# non-ASCII character printed — the same trap that kills the `hf` CLI mid-log.
# Reconfiguring here is more robust than policing every print statement.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from selective_metrics import (
    auroc,
    aurc,
    bootstrap_ci,
    ece_by_group,
    expected_calibration_error,
    risk_at_coverage,
)

LABEL_ORDER = [
    "answerable_known",
    "answerable_unknown",
    "unanswerable",
    "context_dependent",
    "underspecified",
]


def load(path: str) -> list[dict]:
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def signals(rows: list[dict]) -> dict[str, np.ndarray]:
    verb = np.array([r.get("verbalised", float("nan")) for r in rows], dtype=float)
    lp = np.array([r.get("logprob_mean", float("nan")) for r in rows], dtype=float)
    # exp(mean logprob) is the per-token geometric mean likelihood: a number in [0,1]
    # that is comparable across answers of different lengths.
    seq = np.exp(np.clip(lp, -20, 0))
    return {"verbalised": verb, "seq_prob": seq}


def report(path: str, n_bins: int) -> dict:
    rows = load(path)
    correct = np.array([bool(r["correct"]) for r in rows], dtype=float)
    labels = np.array([r["label"] for r in rows])
    sig = signals(rows)

    print(f"\n{'=' * 72}\n{path}   n={len(rows)}   overall accuracy {correct.mean():.1%}")

    out: dict = {"path": path, "n": len(rows), "accuracy": float(correct.mean())}

    # --- behaviour census -------------------------------------------------
    beh: dict[str, int] = {}
    for r in rows:
        beh[r["behaviour"]] = beh.get(r["behaviour"], 0) + 1
    print(f"\nbehaviours: {dict(sorted(beh.items(), key=lambda kv: -kv[1]))}")
    out["behaviours"] = beh

    # --- discrimination ---------------------------------------------------
    print(f"\n{'signal':<12} {'AUROC':>18} {'risk@80%cov':>12} {'AURC':>8}  (lower risk/AURC better)")
    out["signals"] = {}
    for name, s in sig.items():
        ok = ~np.isnan(s)
        a, lo, hi = bootstrap_ci(auroc, s[ok], correct[ok], n_boot=2000, seed=0)
        r80 = risk_at_coverage(s[ok], correct[ok], 0.8)
        ar = aurc(s[ok], correct[ok])
        print(f"{name:<12} {a:>6.3f} [{lo:.3f},{hi:.3f}] {r80:>12.1%} {ar:>8.3f}")
        out["signals"][name] = {
            "auroc": a, "auroc_lo": lo, "auroc_hi": hi,
            "risk_at_80_coverage": r80, "aurc": ar, "n_valid": int(ok.sum()),
        }

    # --- the decisive table: calibration per label ------------------------
    print(f"\nExpected Calibration Error by label ({n_bins} bins; lower is better)")
    header = f"{'label':<20} {'n':>4} {'acc':>7}"
    for name in sig:
        header += f" {name:>12}"
    print(header)
    out["ece_by_label"] = {}
    for name, s in sig.items():
        ok = ~np.isnan(s)
        out["ece_by_label"][name] = ece_by_group(s[ok], correct[ok], labels[ok], n_bins=n_bins)
    for lab in LABEL_ORDER:
        m = labels == lab
        if not m.any():
            continue
        line = f"{lab:<20} {int(m.sum()):>4} {correct[m].mean():>7.1%}"
        for name in sig:
            v = out["ece_by_label"][name].get(lab, float("nan"))
            line += f" {v:>12.3f}" if not math.isnan(v) else f" {'n/a':>12}"
        print(line)
    line = f"{'OVERALL':<20} {len(rows):>4} {correct.mean():>7.1%}"
    for name in sig:
        line += f" {out['ece_by_label'][name]['__overall__']:>12.3f}"
    print(line)

    # --- mean confidence by label: is the signal even moving? -------------
    print(f"\nMean confidence by label (does the signal distinguish the cells at all?)")
    hdr = f"{'label':<20} {'acc':>7}"
    for name in sig:
        hdr += f" {name:>12}"
    print(hdr)
    out["mean_conf_by_label"] = {}
    for lab in LABEL_ORDER:
        m = labels == lab
        if not m.any():
            continue
        line = f"{lab:<20} {correct[m].mean():>7.1%}"
        out["mean_conf_by_label"][lab] = {}
        for name, s in sig.items():
            v = float(np.nanmean(s[m]))
            out["mean_conf_by_label"][lab][name] = v
            line += f" {v:>12.3f}"
        print(line)

    # --- minimal pairs: the tightest comparison we have -------------------
    pairs: dict[str, list[int]] = {}
    for i, r in enumerate(rows):
        if r.get("pair_id"):
            pairs.setdefault(r["pair_id"], []).append(i)
    complete = {p: ix for p, ix in pairs.items() if len(ix) == 2}
    if complete:
        print(f"\nMinimal pairs (n={len(complete)}): same question, context supplied or withheld")
        for name, s in sig.items():
            deltas, flips = [], 0
            for ix in complete.values():
                a, b = ix
                amb = a if rows[a]["label"] in {"context_dependent", "underspecified"} else b
                oth = b if amb == a else a
                if np.isnan(s[amb]) or np.isnan(s[oth]):
                    continue
                deltas.append(abs(s[amb] - s[oth]))
                if correct[amb] != correct[oth]:
                    flips += 1
            if deltas:
                d = float(np.mean(deltas))
                print(f"  {name:<12} mean abs confidence gap across a pair = {d:.3f}"
                      f"   (correctness differs in {flips}/{len(deltas)} pairs)")
                out.setdefault("pairs", {})[name] = {
                    "mean_abs_delta": d, "n": len(deltas), "correctness_flips": flips,
                }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--graded", nargs="+", default=["runs_dl/runs/default/graded.jsonl"])
    ap.add_argument("--bins", type=int, default=5,
                    help="5 for thin per-label cells; 10 inflates sparse bins")
    ap.add_argument("--out", default="runs_dl/analysis.json")
    args = ap.parse_args()

    results = [report(p, args.bins) for p in args.graded]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
