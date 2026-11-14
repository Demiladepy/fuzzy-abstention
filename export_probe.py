"""Fit the final probe on all 210 items and export it as a HF model repo.

    py -3.12 export_probe.py

The shipped probe is fitted on everything, because that is the artifact you would
deploy. The *reported* accuracy (90.5% [86.2, 94.3]) comes from the cross-validated
out-of-fold run, never from this fit — a number from a probe evaluated on its own
training data would be meaningless, and the model card says so plainly.

Layer 20 is the mode of the five cross-validation folds [14, 20, 25, 20, 18].
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from train_probe import LABELS, fit_logreg, predict_proba

RUN = Path("runs_dl/runs/default")
LAYER = 20
OUT = Path("probe_artifact")


def main() -> int:
    hidden = np.load(RUN / "hidden.npy")
    rows = [json.loads(l) for l in open(RUN / "graded.jsonl", encoding="utf-8") if l.strip()]
    y = np.array([LABELS.index(r["label"]) for r in rows])
    x = hidden[:, LAYER, :].astype(np.float32)

    model = fit_logreg(x, y, len(LABELS), seed=0)
    train_acc = float((predict_proba(model, x).argmax(1) == y).mean())

    OUT.mkdir(exist_ok=True)
    torch.save(
        {
            "weight": model.lin.weight.detach().cpu(),
            "bias": model.lin.bias.detach().cpu(),
            "mu": model.mu.cpu(),
            "sd": model.sd.cpu(),
            "labels": LABELS,
            "layer": LAYER,
        },
        OUT / "probe.pt",
    )
    cfg = {
        "base_model": "CohereLabs/tiny-aya-global",
        "layer": LAYER,
        "hidden_size": int(hidden.shape[2]),
        "labels": LABELS,
        "input": "last-token residual stream of the prompt, before generation",
        "standardisation": "per-feature, statistics stored in the checkpoint",
        "fit_on": f"all {len(rows)} items",
        "train_accuracy_not_a_result": train_acc,
        "reported_accuracy_out_of_fold": 0.905,
        "reported_accuracy_ci": [0.862, 0.943],
        "cv_layers_per_fold": [14, 20, 25, 20, 18],
    }
    (OUT / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    card = f"""---
license: cc-by-4.0
base_model: CohereLabs/tiny-aya-global
tags:
  - probe
  - calibration
  - abstention
  - ai-safety
  - lexsi-hackathon
---

# Fuzzy Abstention probe for Tiny Aya

A 5-way linear probe over the layer-{LAYER} residual stream of
[`CohereLabs/tiny-aya-global`](https://huggingface.co/CohereLabs/tiny-aya-global),
predicting the **epistemic kind** of a question from the prompt alone, *before the model
generates anything*.

| class | meaning | correct action |
|---|---|---|
| `answerable_known` | one answer, well represented | answer |
| `answerable_unknown` | one answer, long-tail | abstain |
| `unanswerable` | no answer exists | reject the premise |
| `context_dependent` | varies by jurisdiction/calendar/units/region/time | ask which context |
| `underspecified` | ambiguous referent | ask what is meant |

## Performance

**Out-of-fold accuracy 90.5% [86.2%, 94.3%]**, macro-F1 0.906, 5-fold cross-validated
with the layer chosen inside each training fold. Chance is 20%.

> The checkpoint here is fitted on all 210 items, because that is the deployable
> artifact. Its training accuracy ({train_acc:.3f}) is **not a result** and must not be
> quoted as one. Every performance number above comes from the cross-validated run.

Replicates at 90.5% [86.2%, 94.3%] under a minimal system preamble.

## Why it exists

The model's own stated confidence is calibrated to ECE 0.018 where a question has one
clean answer and 0.904 where it does not, while spanning only 0.865–0.955 across cells
whose accuracy spans 0%–93.7%. Its AUROC for predicting its own correctness is
0.557 [0.487, 0.625] — chance is inside the interval. The epistemic kind *is* present in
the activations; the self-report does not carry it.

**Honest caveat:** collapsed to a single scalar, this probe is no better than that
self-report (AUROC 0.556 vs 0.557). It wins by being multi-class and driving a
three-way policy, not by being a better number.

## Usage

```python
import torch
ckpt = torch.load("probe.pt")
# h: [hidden_size] last-token residual stream at layer 20 of the prompt
z = (h - ckpt["mu"]) / ckpt["sd"]
probs = torch.softmax(z @ ckpt["weight"].T + ckpt["bias"], dim=-1)
kind = ckpt["labels"][int(probs.argmax())]
```

## Links

- Dataset: [demiladepy/fuzzy-abstention-items](https://huggingface.co/datasets/demiladepy/fuzzy-abstention-items)
- Demo: [demiladepy/fuzzy-abstention](https://huggingface.co/spaces/demiladepy/fuzzy-abstention)
- Code: [github.com/Demiladepy/fuzzy-abstention](https://github.com/Demiladepy/fuzzy-abstention)

Built for the Lexsi AI Safety and Alignment Hackathon, Track 2b, problem statement 13.
"""
    (OUT / "README.md").write_text(card, encoding="utf-8")

    print(f"layer {LAYER}, {len(rows)} items, train acc {train_acc:.3f} (not a result)")
    print(f"wrote {OUT}/probe.pt, config.json, README.md")

    from huggingface_hub import HfApi, create_repo

    api = HfApi()
    repo = f"{api.whoami()['name']}/fuzzy-abstention-probe"
    create_repo(repo, repo_type="model", private=False, exist_ok=True)
    api.upload_folder(folder_path=str(OUT), repo_id=repo, repo_type="model")
    print(f"uploaded https://huggingface.co/{repo}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
