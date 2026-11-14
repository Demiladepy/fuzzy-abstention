"""Cross-lingual transfer: does an English-trained probe read epistemic kind in Yoruba?

    py -3.12 transfer_eval.py

The careful bit. The multilingual items are **translations of the same questions**, so a
probe trained on all 210 English items has already seen their content — only the surface
language is new. Testing on those would measure "did it memorise this question",
dressed up as cross-lingual transfer.

So the headline number trains on the English items whose source questions are **not** in
the multilingual subset. The probe has then never seen these questions in any language.
The leaky variant is computed too, and reported as what it is: an upper bound.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from selective_metrics import bootstrap_ci
from train_probe import LABELS, fit_logreg, macro_f1, predict_proba

EN = Path("runs_dl/runs/default")
ML = Path("runs_dl/runs/multilingual")
LAYER = 20
LANG_NAME = {"yo": "Yoruba", "ha": "Hausa", "sw": "Swahili"}


def load(run: Path):
    rows = [json.loads(l) for l in open(run / "generations.jsonl", encoding="utf-8")
            if l.strip()]
    return rows, np.load(run / "hidden.npy")


def acc_ci(pred: np.ndarray, y: np.ndarray):
    return bootstrap_ci(lambda a, b: float((a == b).mean()), pred, y, n_boot=2000, seed=0)


def main() -> int:
    if not (ML / "generations.jsonl").exists():
        print(f"missing {ML} — run the multilingual job first")
        return 1

    en_rows, en_hid = load(EN)
    ml_rows, ml_hid = load(ML)
    print(f"english {len(en_rows)} | multilingual {len(ml_rows)} "
          f"({len(set(r['lang'] for r in ml_rows))} languages)\n")

    src_ids = {r["src_id"] for r in ml_rows}
    y_ml = np.array([LABELS.index(r["label"]) for r in ml_rows])
    x_ml = ml_hid[:, LAYER, :].astype(np.float32)

    results: dict = {"layer": LAYER, "n_multilingual": len(ml_rows)}

    for mode, keep in (("clean (source questions held out)",
                        [i for i, r in enumerate(en_rows) if r["id"] not in src_ids]),
                       ("leaky upper bound (all English)", list(range(len(en_rows))))):
        x_en = en_hid[keep][:, LAYER, :].astype(np.float32)
        y_en = np.array([LABELS.index(en_rows[i]["label"]) for i in keep])
        probe = fit_logreg(x_en, y_en, len(LABELS), seed=0)

        pred = predict_proba(probe, x_ml).argmax(1)
        a, lo, hi = acc_ci(pred, y_ml)
        print(f"--- probe trained on {len(keep)} English items, {mode} ---")
        print(f"  all languages   acc {a:.3f} [{lo:.3f}, {hi:.3f}]  "
              f"macro-F1 {macro_f1(y_ml, pred, len(LABELS)):.3f}")

        per_lang = {}
        for code in sorted({r["lang"] for r in ml_rows}):
            m = np.array([r["lang"] == code for r in ml_rows])
            aa, ll, hh = acc_ci(pred[m], y_ml[m])
            per_lang[code] = {"n": int(m.sum()), "acc": aa, "ci": [ll, hh]}
            print(f"  {LANG_NAME.get(code, code):<10} n={int(m.sum()):<4} "
                  f"acc {aa:.3f} [{ll:.3f}, {hh:.3f}]")

        # English control for the same probe, on items it did not train on.
        held = [i for i, r in enumerate(en_rows) if i not in set(keep)]
        if held:
            xe = en_hid[held][:, LAYER, :].astype(np.float32)
            ye = np.array([LABELS.index(en_rows[i]["label"]) for i in held])
            pe = predict_proba(probe, xe).argmax(1)
            ae, le, he = acc_ci(pe, ye)
            print(f"  {'English*':<10} n={len(held):<4} acc {ae:.3f} [{le:.3f}, {he:.3f}]"
                  f"   <- same probe, held-out English, the transfer gap is against this")
            per_lang["en_heldout"] = {"n": len(held), "acc": ae, "ci": [le, he]}

        results[mode] = {"n_train": len(keep), "overall": [a, lo, hi],
                         "macro_f1": macro_f1(y_ml, pred, len(LABELS)),
                         "per_language": per_lang}
        print()

    # Does the scalar baseline degrade too? If it degrades further, the policy gap widens.
    verb = np.array([r.get("verbalised", np.nan) for r in ml_rows], dtype=float)
    ok = ~np.isnan(verb)
    if ok.any():
        results["verbalised_mean_by_lang"] = {}
        print("Model's stated confidence, by language (English mean was 0.904):")
        for code in sorted({r["lang"] for r in ml_rows}):
            m = np.array([r["lang"] == code for r in ml_rows]) & ok
            if m.any():
                v = float(np.mean(verb[m]))
                results["verbalised_mean_by_lang"][code] = v
                print(f"  {LANG_NAME.get(code, code):<10} {v:.3f}")

    dest = Path("runs_dl/transfer_results.json")
    with open(dest, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nwrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
