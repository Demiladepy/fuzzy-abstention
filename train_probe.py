"""Stage 2: train the context-conditional probe and sweep layers.

Runs on CPU — the activations are small even though the model is not, which is why
only extraction needs a GPU.

    python train_probe.py --hidden out/hidden.npy --gens out/generations.jsonl
    python train_probe.py --self-test        # synthetic data, no model needed

The probe predicts the *epistemic kind* of the question (5 classes) from the
prompt's residual stream, not whether a specific answer is true. Its scalar for
selective prediction is P(answerable), i.e. the mass on `answerable_known` +
`answerable_unknown` — which is what gets compared against the logprob and
verbalised-confidence baselines.

Split discipline: the test split is carved out first and never touched by the layer
sweep. Choosing a layer on test and then reporting test numbers would be the easiest
way to produce an impressive and meaningless result.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

LABELS = [
    "answerable_known",
    "answerable_unknown",
    "unanswerable",
    "context_dependent",
    "underspecified",
]
ANSWERABLE = {"answerable_known", "answerable_unknown"}


def fit_logreg(
    x: np.ndarray,
    y: np.ndarray,
    n_classes: int,
    *,
    epochs: int = 300,
    lr: float = 0.05,
    weight_decay: float = 1e-3,
    seed: int = 0,
) -> torch.nn.Module:
    """Multinomial logistic regression, full-batch LBFGS-free, pure torch.

    Features are standardised inside this function using train statistics only; the
    returned module carries them so val/test use the same transform rather than
    their own, which would leak distribution information across the split.
    """
    torch.manual_seed(seed)
    mu = x.mean(0, keepdims=True)
    sd = x.std(0, keepdims=True) + 1e-6

    xt = torch.tensor((x - mu) / sd, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.long)

    lin = torch.nn.Linear(x.shape[1], n_classes)
    opt = torch.optim.Adam(lin.parameters(), lr=lr, weight_decay=weight_decay)
    lossf = torch.nn.CrossEntropyLoss()
    for _ in range(epochs):
        opt.zero_grad()
        loss = lossf(lin(xt), yt)
        loss.backward()
        opt.step()

    lin.eval()
    model = torch.nn.Module()
    model.lin = lin
    model.register_buffer("mu", torch.tensor(mu, dtype=torch.float32))
    model.register_buffer("sd", torch.tensor(sd, dtype=torch.float32))
    return model


def predict_proba(model: torch.nn.Module, x: np.ndarray) -> np.ndarray:
    xt = (torch.tensor(x, dtype=torch.float32) - model.mu) / model.sd
    with torch.no_grad():
        return model.lin(xt).softmax(-1).numpy()


def stratified_split(
    labels: np.ndarray, fracs=(0.6, 0.2, 0.2), seed: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stratify per class so thin cells appear in every split.

    With ~33 items in the smallest class, a random split can leave a class absent
    from val or test entirely, which silently breaks per-label reporting.
    """
    rng = np.random.default_rng(seed)
    tr, va, te = [], [], []
    for c in np.unique(labels):
        idx = np.flatnonzero(labels == c)
        rng.shuffle(idx)
        n = len(idx)
        n_tr = int(round(fracs[0] * n))
        n_va = int(round(fracs[1] * n))
        # Guarantee at least one item in val and test when the class allows it.
        if n >= 3:
            n_tr = min(n_tr, n - 2)
            n_va = max(1, min(n_va, n - n_tr - 1))
        tr += list(idx[:n_tr])
        va += list(idx[n_tr : n_tr + n_va])
        te += list(idx[n_tr + n_va :])
    return np.array(tr), np.array(va), np.array(te)


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    f1s = []
    for c in range(n_classes):
        tp = int(((y_pred == c) & (y_true == c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        fn = int(((y_pred != c) & (y_true == c)).sum())
        if tp == 0 and (fp or fn):
            f1s.append(0.0)
        elif tp == 0:
            continue
        else:
            prec, rec = tp / (tp + fp), tp / (tp + fn)
            f1s.append(2 * prec * rec / (prec + rec))
    return float(np.mean(f1s)) if f1s else 0.0


def run(hidden: np.ndarray, labels: list[str], seed: int = 0) -> dict:
    y = np.array([LABELS.index(l) for l in labels])
    tr, va, te = stratified_split(y, seed=seed)
    n_layers = hidden.shape[1]

    sweep = []
    for layer in range(n_layers):
        x = hidden[:, layer, :].astype(np.float32)
        m = fit_logreg(x[tr], y[tr], len(LABELS), seed=seed)
        pv = predict_proba(m, x[va])
        sweep.append(
            {
                "layer": layer,
                "val_acc": float((pv.argmax(1) == y[va]).mean()),
                "val_macro_f1": macro_f1(y[va], pv.argmax(1), len(LABELS)),
            }
        )

    best = max(sweep, key=lambda r: r["val_macro_f1"])
    layer = best["layer"]

    # Refit on train+val for the final model, then touch test exactly once.
    x = hidden[:, layer, :].astype(np.float32)
    trva = np.concatenate([tr, va])
    final = fit_logreg(x[trva], y[trva], len(LABELS), seed=seed)
    pt = predict_proba(final, x[te])

    answerable_cols = [LABELS.index(l) for l in ANSWERABLE]
    p_answerable = pt[:, answerable_cols].sum(1)

    per_label = {}
    for c, name in enumerate(LABELS):
        m = y[te] == c
        per_label[name] = {
            "n": int(m.sum()),
            "recall": float((pt[m].argmax(1) == c).mean()) if m.sum() else float("nan"),
        }

    return {
        "n_items": int(hidden.shape[0]),
        "n_layers": n_layers,
        "hidden_size": int(hidden.shape[2]),
        "split_sizes": {"train": len(tr), "val": len(va), "test": len(te)},
        "chosen_layer": layer,
        "chosen_layer_val_macro_f1": best["val_macro_f1"],
        "test_acc": float((pt.argmax(1) == y[te]).mean()),
        "test_macro_f1": macro_f1(y[te], pt.argmax(1), len(LABELS)),
        "per_label_test": per_label,
        "sweep": sweep,
        "p_answerable_test": p_answerable.tolist(),
        "test_labels": [LABELS[i] for i in y[te]],
    }


def self_test() -> int:
    """Synthetic activations with planted class structure, plus a null control.

    The point is to prove the pipeline recovers signal that is really there and does
    NOT invent signal that isn't — a probe that scores well on noise means the split
    or the metric is wrong.
    """
    rng = np.random.default_rng(0)
    n_per, n_layers, dim = 40, 6, 32
    labels, rows = [], []
    # Signal lives only in layer 3; every other layer is noise.
    centres = rng.normal(size=(len(LABELS), dim)) * 3.0
    for ci, name in enumerate(LABELS):
        for _ in range(n_per):
            h = rng.normal(size=(n_layers, dim)) * 1.0
            h[3] = centres[ci] + rng.normal(size=dim) * 1.0
            rows.append(h)
            labels.append(name)
    hidden = np.stack(rows).astype(np.float16)

    res = run(hidden, labels)
    ok = True

    if res["chosen_layer"] != 3:
        print(f"[FAIL] sweep picked layer {res['chosen_layer']}, expected 3")
        ok = False
    else:
        print("[PASS] layer sweep found the planted layer (3)")

    if res["test_acc"] < 0.8:
        print(f"[FAIL] test acc {res['test_acc']:.2f} too low on separable data")
        ok = False
    else:
        print(f"[PASS] recovers planted signal (test acc {res['test_acc']:.2f})")

    if any(v["n"] == 0 for v in res["per_label_test"].values()):
        print("[FAIL] a class is missing from the test split")
        ok = False
    else:
        print("[PASS] every class present in the test split")

    # Null control: labels shuffled, so no layer carries real signal. Averaged over
    # several seeds — with ~40 test items the single-draw standard error is about
    # 0.06, so one run can land 2 sigma high by luck and a one-shot check is then
    # either flaky or toothless. The mean is what should sit at chance.
    chance = 1.0 / len(LABELS)
    null_accs = []
    for s in range(5):
        shuffled = list(labels)
        np.random.default_rng(100 + s).shuffle(shuffled)
        null_accs.append(run(hidden, shuffled, seed=s)["test_acc"])
    null_mean = float(np.mean(null_accs))
    spread = ", ".join(f"{a:.2f}" for a in null_accs)
    if null_mean > chance + 0.08:
        print(f"[FAIL] null control mean {null_mean:.2f} above chance {chance:.2f} ({spread})")
        print("       a probe scoring on shuffled labels means the split or metric leaks")
        ok = False
    else:
        print(f"[PASS] null control at chance: mean {null_mean:.2f} vs {chance:.2f} ({spread})")

    print("\n" + ("All checks passed." if ok else "FAILURES above."))
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", default="out/hidden.npy")
    ap.add_argument("--gens", default="out/generations.jsonl")
    ap.add_argument("--out", default="out/probe_results.json")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        return self_test()

    hidden = np.load(args.hidden)
    labels = []
    with open(args.gens, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                labels.append(json.loads(line)["label"])
    if len(labels) != hidden.shape[0]:
        print(f"mismatch: {len(labels)} rows vs {hidden.shape[0]} activations")
        return 1

    res = run(hidden, labels)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2)

    print(f"layer {res['chosen_layer']} of {res['n_layers']}")
    print(f"test acc {res['test_acc']:.3f}  macro-F1 {res['test_macro_f1']:.3f}")
    for name, v in res["per_label_test"].items():
        print(f"  {name:<20} n={v['n']:<4} recall={v['recall']:.3f}")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
