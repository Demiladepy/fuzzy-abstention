"""Stage 4: cross-validated probe, and the policy comparison judges asked for.

    py -3.12 cv_probe.py --run runs_dl/runs/default

Two things the single 60/20/20 split could not give us:

1. **Out-of-fold predictions for all 210 items**, so per-label recall rests on the full
   cell (33-63 items) instead of 6-12, and the headline accuracy carries a bootstrap
   interval instead of being one draw.

2. **No layer leakage.** 37 layers is a lot of choices; picking the layer on data that
   later becomes test is the easiest way to manufacture a good number. Here the layer is
   chosen inside each training fold, on an inner split, and the outer fold is untouched
   until scoring. Folds disagreeing about the layer is itself worth knowing.

Then the policy table. Lexsi's reply asked specifically for what the intervention costs:
over-abstention, and accuracy on `answerable_known`. Scalar baselines are compared **at
matched coverage** — thresholded to answer as often as the probe policy does — because
comparing an abstention method at a different operating point is comparing nothing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from selective_metrics import auroc, bootstrap_ci, ece_by_group
from train_probe import LABELS, fit_logreg, macro_f1, predict_proba, stratified_split

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ANSWER_LABEL = "answerable_known"
ASK_LABELS = {"context_dependent", "underspecified"}


def k_folds(y: np.ndarray, k: int, seed: int = 0) -> list[np.ndarray]:
    """Stratified folds, so every fold sees every class."""
    rng = np.random.default_rng(seed)
    folds: list[list[int]] = [[] for _ in range(k)]
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        rng.shuffle(idx)
        for i, j in enumerate(idx):
            folds[i % k].append(int(j))
    return [np.array(sorted(f)) for f in folds]


def run_cv(hidden: np.ndarray, labels: list[str], k: int, seed: int) -> dict:
    y = np.array([LABELS.index(l) for l in labels])
    n, n_layers = hidden.shape[0], hidden.shape[1]
    folds = k_folds(y, k, seed)

    oof_proba = np.zeros((n, len(LABELS)))
    chosen_layers = []

    for f, test_idx in enumerate(folds):
        train_idx = np.setdiff1d(np.arange(n), test_idx)
        # Layer chosen on an inner split of the TRAINING fold only.
        inner_tr, inner_va, _ = stratified_split(y[train_idx], fracs=(0.75, 0.25, 0.0),
                                                 seed=seed + f)
        tr, va = train_idx[inner_tr], train_idx[inner_va]

        best, best_score = 0, -1.0
        for layer in range(n_layers):
            x = hidden[:, layer, :].astype(np.float32)
            m = fit_logreg(x[tr], y[tr], len(LABELS), seed=seed)
            score = macro_f1(y[va], predict_proba(m, x[va]).argmax(1), len(LABELS))
            if score > best_score:
                best, best_score = layer, score
        chosen_layers.append(best)

        x = hidden[:, best, :].astype(np.float32)
        final = fit_logreg(x[train_idx], y[train_idx], len(LABELS), seed=seed)
        oof_proba[test_idx] = predict_proba(final, x[test_idx])
        print(f"  fold {f + 1}/{k}: layer {best:>2}  inner macro-F1 {best_score:.3f}  "
              f"n_test={len(test_idx)}", flush=True)

    pred = oof_proba.argmax(1)
    return {
        "oof_proba": oof_proba, "pred": pred, "y": y,
        "chosen_layers": chosen_layers,
        "accuracy": float((pred == y).mean()),
        "macro_f1": macro_f1(y, pred, len(LABELS)),
    }


def policy_table(rows: list[dict], oof_proba: np.ndarray) -> list[dict]:
    """Compare abstention policies at matched coverage."""
    labels = np.array([r["label"] for r in rows])
    correct = np.array([bool(r["correct"]) for r in rows])
    is_known = labels == ANSWER_LABEL
    # Answering directly is the right move only on answerable_known, and only when the
    # content was right. Everywhere else a direct answer is the failure.
    answer_ok = is_known & correct

    verb = np.array([r.get("verbalised", np.nan) for r in rows], dtype=float)
    seq = np.exp(np.clip(np.array([r.get("logprob_mean", np.nan) for r in rows],
                                  dtype=float), -20, 0))

    p_answer = oof_proba[:, LABELS.index(ANSWER_LABEL)]
    probe_answers = oof_proba.argmax(1) == LABELS.index(ANSWER_LABEL)
    probe_asks = np.isin(oof_proba.argmax(1), [LABELS.index(l) for l in ASK_LABELS])
    target_coverage = float(probe_answers.mean())

    def row(name: str, answers: np.ndarray, asks: np.ndarray | None = None) -> dict:
        n_ans = int(answers.sum())
        err = float(1 - answer_ok[answers].mean()) if n_ans else float("nan")
        over = float((~answers[is_known]).mean())
        return {
            "policy": name,
            "coverage": float(answers.mean()),
            "error_when_answering": err,
            "over_abstention_on_known": over,
            "known_accuracy_retained": float(answer_ok[is_known & answers].sum()
                                             / max(int(answer_ok[is_known].sum()), 1)),
            "asks": float(asks.mean()) if asks is not None else None,
            "n_answered": n_ans,
        }

    out = [
        row("always answer", np.ones(len(rows), dtype=bool)),
        row("probe (5-way)", probe_answers, probe_asks),
    ]
    # Scalars thresholded to the probe's coverage — the only fair comparison.
    for name, s in (("verbalised", verb), ("seq_prob", seq), ("probe P(answerable_known)", p_answer)):
        ok = ~np.isnan(s)
        thr = np.quantile(s[ok], 1 - target_coverage)
        out.append(row(f"{name} @matched cov", (s >= thr) & ok))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs_dl/runs/default")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bins", type=int, default=5)
    args = ap.parse_args()

    run = Path(args.run)
    hidden = np.load(run / "hidden.npy")
    rows = [json.loads(l) for l in open(run / "graded.jsonl", encoding="utf-8") if l.strip()]
    labels = [r["label"] for r in rows]
    assert len(rows) == hidden.shape[0], f"{len(rows)} rows vs {hidden.shape[0]} activations"

    print(f"{run}  n={len(rows)}  layers={hidden.shape[1]}  {args.folds}-fold CV\n")
    cv = run_cv(hidden, labels, args.folds, args.seed)

    y, pred = cv["y"], cv["pred"]
    acc, lo, hi = bootstrap_ci(lambda a, b: float((a == b).mean()), pred, y,
                               n_boot=2000, seed=0)
    chance = 1.0 / len(LABELS)
    print(f"\nout-of-fold accuracy {acc:.3f} [{lo:.3f}, {hi:.3f}]   "
          f"macro-F1 {cv['macro_f1']:.3f}   chance {chance:.2f}")
    print(f"layers chosen per fold: {cv['chosen_layers']}")

    print(f"\n{'label':<20} {'n':>4} {'recall':>8} {'precision':>10}")
    per_label = {}
    for c, name in enumerate(LABELS):
        m = y == c
        rec = float((pred[m] == c).mean()) if m.any() else float("nan")
        sel = pred == c
        prec = float((y[sel] == c).mean()) if sel.any() else float("nan")
        per_label[name] = {"n": int(m.sum()), "recall": rec, "precision": prec}
        print(f"{name:<20} {int(m.sum()):>4} {rec:>8.3f} {prec:>10.3f}")

    # Does the probe's own confidence beat the model's self-report?
    correct = np.array([bool(r["correct"]) for r in rows], dtype=float)
    p_known = cv["oof_proba"][:, LABELS.index(ANSWER_LABEL)]
    verb = np.array([r.get("verbalised", np.nan) for r in rows], dtype=float)
    ok = ~np.isnan(verb)
    a_p, lo_p, hi_p = bootstrap_ci(auroc, p_known, correct, n_boot=2000, seed=0)
    a_v, lo_v, hi_v = bootstrap_ci(auroc, verb[ok], correct[ok], n_boot=2000, seed=0)
    print(f"\nAUROC for predicting correctness")
    print(f"  probe P(answerable_known)  {a_p:.3f} [{lo_p:.3f}, {hi_p:.3f}]")
    print(f"  model's verbalised conf    {a_v:.3f} [{lo_v:.3f}, {hi_v:.3f}]")

    ece = ece_by_group(p_known, correct, np.array(labels), n_bins=args.bins)
    print(f"\nprobe-signal ECE by label ({args.bins} bins)")
    for name in LABELS:
        if name in ece:
            print(f"  {name:<20} {ece[name]:.3f}")
    print(f"  {'OVERALL':<20} {ece['__overall__']:.3f}")

    print(f"\nPolicy comparison at matched coverage")
    tbl = policy_table(rows, cv["oof_proba"])
    print(f"{'policy':<30} {'cov':>6} {'err|ans':>8} {'over-abst':>10} {'asks':>6}")
    for r in tbl:
        asks = f"{r['asks']:.1%}" if r["asks"] is not None else "-"
        print(f"{r['policy']:<30} {r['coverage']:>6.1%} {r['error_when_answering']:>8.1%} "
              f"{r['over_abstention_on_known']:>10.1%} {asks:>6}")

    out = {
        "run": str(run), "folds": args.folds,
        "oof_accuracy": acc, "oof_accuracy_ci": [lo, hi], "macro_f1": cv["macro_f1"],
        "chosen_layers": cv["chosen_layers"], "per_label": per_label,
        "auroc_probe": [a_p, lo_p, hi_p], "auroc_verbalised": [a_v, lo_v, hi_v],
        "probe_ece_by_label": ece, "policy_table": tbl,
    }
    dest = run / "cv_results.json"
    with open(dest, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)

    # Per-item out-of-fold record, for the demo and for anyone auditing a single row.
    # These are genuinely held-out predictions: each item was scored by a probe that
    # never saw it, so the demo shows honest numbers rather than fitted ones.
    seq = np.exp(np.clip(np.array([r.get("logprob_mean", np.nan) for r in rows],
                                  dtype=float), -20, 0))
    per_item = []
    for i, r in enumerate(rows):
        per_item.append({
            "id": r["id"], "question": r["question"], "label": r["label"],
            "pair_id": r.get("pair_id"), "gold": r.get("gold"),
            "answer": r.get("answer", ""), "behaviour": r["behaviour"],
            "correct": bool(r["correct"]), "hallucinated": bool(r.get("hallucinated", False)),
            "verbalised": r.get("verbalised"),
            "seq_prob": None if np.isnan(seq[i]) else round(float(seq[i]), 4),
            "probe": {LABELS[c]: round(float(cv["oof_proba"][i, c]), 4)
                      for c in range(len(LABELS))},
            "probe_pred": LABELS[int(cv["pred"][i])],
        })
    oof_dest = run / "oof_items.json"
    with open(oof_dest, "w", encoding="utf-8") as fh:
        json.dump(per_item, fh, indent=2, ensure_ascii=False)
    print(f"\nwrote {dest}\nwrote {oof_dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
