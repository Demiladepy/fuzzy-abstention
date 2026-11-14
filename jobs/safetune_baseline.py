# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "torch",
#   "transformers>=5.0",
#   "huggingface_hub>=1.0",
#   "numpy",
#   "accelerate",
#   "safetune",
# ]
# ///
"""SafeTune's LinearProbeGuardTrainer as a real, numbered baseline.

    hf jobs uv run --flavor t4-small --secrets HF_TOKEN jobs/safetune_baseline.py

Why this is the right comparison: SafeTune ships a probe guard that is **one layer, one
threshold, a binary verdict** — precisely the representation this project argues is the
wrong shape for abstention. Beating weak scalars is unconvincing; beating the
organisers' own probe guard is not. Until now we had only cited it, which is not the
same as running it.

Mapping our labels onto its contrastive interface:
    harmless -> `answerable_known`      questions that SHOULD be answered directly
    harmful  -> `unanswerable` + `answerable_unknown`
                questions where a direct answer is a fabrication

That is a faithful use of a binary guard, and it deliberately gives it the easier job:
it never has to represent `context_dependent` or `underspecified`, because it cannot.

The API is undocumented, so this script introspects what `calibrate` returns and prints
it before trying to score. A run that fails to score still teaches us the shape for a
few cents, which beats guessing.
"""

from __future__ import annotations

import json
import os
import time
import traceback

import numpy as np
import torch
from huggingface_hub import HfApi, hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "CohereLabs/tiny-aya-global"
DATASET = "demiladepy/fuzzy-abstention-items"
ANSWER_OK = "answerable_known"
NO_ANSWER = {"unanswerable", "answerable_unknown"}


def describe(name: str, obj) -> dict:
    attrs = [a for a in dir(obj) if not a.startswith("_")][:40]
    info = {"name": name, "type": type(obj).__name__, "attrs": attrs}
    print(f"  {name}: {type(obj).__name__}", flush=True)
    print(f"    attrs: {attrs}", flush=True)
    return info


def main() -> int:
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("HF_TOKEN missing")
        return 1

    path = hf_hub_download(DATASET, "items.jsonl", repo_type="dataset", token=token)
    items = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    harmless = [i["question"] for i in items if i["label"] == ANSWER_OK]
    harmful = [i["question"] for i in items if i["label"] in NO_ANSWER]
    print(f"{len(items)} items | harmless(answer-ok)={len(harmless)} "
          f"harmful(should-not-answer)={len(harmful)}", flush=True)

    cuda = torch.cuda.is_available()
    dtype = torch.bfloat16 if cuda else torch.float32
    print(f"cuda={cuda} dtype={dtype}", flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL, token=token)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, token=token, dtype=dtype, device_map="auto" if cuda else None
    )
    model.eval()

    from safetune.runner.steer import LinearProbeGuardTrainer

    # SafeTune's Tiny Aya notes put refusal structure near layer 24 of 36; our own
    # cross-validated sweep landed at 14-25. Give the guard the documented layer so the
    # comparison is not handicapped by a bad layer choice.
    out: dict = {"model": MODEL, "layer": 24, "n_harmless": len(harmless),
                 "n_harmful": len(harmful), "cuda": cuda, "protocol": "5-fold out-of-fold"}
    t0 = time.time()

    # --- 5-fold, calibrate on train, score held-out ------------------------------
    # The first version of this script calibrated on ALL 210 items and then scored
    # those same items, which produced AUROC ~0.9999 — pure memorisation. Our own
    # probe is reported out-of-fold at 90.5%, so an in-sample baseline would be a
    # rigged comparison in the baseline's favour. Same folds, same discipline.
    contrast = [i for i in items if i["label"] == ANSWER_OK or i["label"] in NO_ANSWER]
    rng = np.random.default_rng(0)
    pos_idx = [k for k, i in enumerate(contrast) if i["label"] == ANSWER_OK]
    neg_idx = [k for k, i in enumerate(contrast) if i["label"] in NO_ANSWER]
    for g in (pos_idx, neg_idx):
        rng.shuffle(g)
    folds = [[] for _ in range(5)]
    for g in (pos_idx, neg_idx):            # stratify so every fold has both classes
        for n, k in enumerate(g):
            folds[n % 5].append(k)

    oof: dict[int, float] = {}
    try:
        for f, test_ix in enumerate(folds):
            tr = [k for k in range(len(contrast)) if k not in set(test_ix)]
            tr_harmless = [contrast[k]["question"] for k in tr
                           if contrast[k]["label"] == ANSWER_OK]
            tr_harmful = [contrast[k]["question"] for k in tr
                          if contrast[k]["label"] in NO_ANSWER]
            t = LinearProbeGuardTrainer(model, tok, layer=24, threshold=0.5)
            w, _ = t.calibrate(harmful=tr_harmful, harmless=tr_harmless)
            qs = [contrast[k]["question"] for k in test_ix]
            sc = list(w.score(qs))
            for k, v in zip(test_ix, sc):
                oof[k] = float(v if not isinstance(v, (list, tuple)) else v[0])
            print(f"  fold {f+1}/5: calibrated on {len(tr_harmless)}+{len(tr_harmful)}, "
                  f"scored {len(qs)} held-out", flush=True)

        s = np.array([oof[k] for k in range(len(contrast))], dtype=float)
        pos = np.array([contrast[k]["label"] == ANSWER_OK for k in range(len(contrast))])
        order = np.argsort(s)
        ranks = np.empty(len(s), float)
        ranks[order] = np.arange(1, len(s) + 1)
        n1, n0 = int(pos.sum()), int((~pos).sum())
        raw = float((ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
        # The guard's score points in the "refuse" direction, so the answer-ok AUROC
        # is its complement. Report both so the orientation is unambiguous.
        out["guard_auroc_oof_raw"] = raw
        out["guard_auroc_oof_answer_ok"] = max(raw, 1 - raw)
        out["guard_scores_oof"] = {contrast[k]["id"]: round(oof[k], 5) for k in oof}
        out["n_contrast"] = len(contrast)
        print(f"\n  OUT-OF-FOLD guard AUROC (answer-ok vs should-not-answer): "
              f"{out['guard_auroc_oof_answer_ok']:.4f}  (raw {raw:.4f})", flush=True)
        out["cv_ok"] = True
    except Exception as exc:  # noqa: BLE001
        out["cv_ok"] = False
        out["cv_error"] = f"{type(exc).__name__}: {exc}"
        print("CV failed:", out["cv_error"], flush=True)
        traceback.print_exc(limit=3)

    # --- in-sample run, kept only as the contrast that shows why CV was needed ----
    try:
        trainer = LinearProbeGuardTrainer(model, tok, layer=24, threshold=0.5)
        print("calibrating (in-sample, for contrast)...", flush=True)
        res = trainer.calibrate(harmful=harmful, harmless=harmless)
        print("calibrate returned:", type(res).__name__, flush=True)
        if isinstance(res, tuple):
            out["calibrate_returns"] = [describe(f"ret[{i}]", r) for i, r in enumerate(res)]
        else:
            out["calibrate_returns"] = [describe("ret", res)]
        out["trainer_attrs"] = [a for a in dir(trainer) if not a.startswith("_")]
        out["calibrated"] = True
    except Exception as exc:  # noqa: BLE001
        out["calibrated"] = False
        out["error"] = f"{type(exc).__name__}: {exc}"
        print("calibrate FAILED:", out["error"], flush=True)
        traceback.print_exc(limit=4)

    # The scoring API lives on the object `calibrate` RETURNS (a LinearProbeGuardModel
    # exposing score / predict / guard_batch), not on the trainer. Try the plausible
    # call shapes and record which one worked, so the write-up can state exactly what
    # was invoked rather than "we used SafeTune".
    scores = None
    if out.get("calibrated"):
        wrapped = res[0] if isinstance(res, tuple) else res
        questions = [i["question"] for i in items]
        attempts = [
            ("score(list)", lambda: wrapped.score(questions)),
            ("guard_batch(list)", lambda: wrapped.guard_batch(questions)),
            ("predict(list)", lambda: wrapped.predict(questions)),
            ("score(str) each", lambda: [wrapped.score(q) for q in questions]),
        ]
        for name, fn in attempts:
            try:
                raw = fn()
                vals = [float(x) if not isinstance(x, (list, tuple)) else float(x[0])
                        for x in (raw if isinstance(raw, (list, tuple)) else list(raw))]
                if len(vals) == len(items):
                    scores, out["score_fn"] = vals, name
                    print(f"  scored {len(vals)} items via {name}", flush=True)
                    break
                print(f"  {name} returned {len(vals)} values, expected {len(items)}",
                      flush=True)
            except Exception as exc:  # noqa: BLE001
                print(f"  {name} failed: {type(exc).__name__}: "
                      f"{str(exc).splitlines()[0][:90]}", flush=True)
        if scores is None:
            out["score_error"] = "no call shape produced one score per item"

    if scores is not None:
        labels = [i["label"] for i in items]
        s = np.array(scores, dtype=float)
        pos = np.array([l == ANSWER_OK for l in labels])
        # AUROC for separating answer-ok from should-not-answer.
        order = np.argsort(s)
        ranks = np.empty(len(s), float)
        ranks[order] = np.arange(1, len(s) + 1)
        n1, n0 = int(pos.sum()), int((~pos).sum())
        out["guard_auroc_insample"] = float(
            (ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else None
        out["scores_insample"] = [round(float(x), 5) for x in s]
        ins = out["guard_auroc_insample"]
        print(f"  IN-SAMPLE guard AUROC: raw {ins:.4f} -> answer-ok "
              f"{max(ins, 1 - ins):.4f}  (memorised; not a fair baseline)", flush=True)

    out["seconds"] = round(time.time() - t0, 1)
    os.makedirs("out", exist_ok=True)
    with open("out/safetune_guard.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, default=str)
    HfApi(token=token).upload_file(
        path_or_fileobj="out/safetune_guard.json",
        path_in_repo="analysis/safetune_guard.json",
        repo_id=DATASET, repo_type="dataset",
    )
    print(f"\nuploaded analysis/safetune_guard.json ({out['seconds']}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
