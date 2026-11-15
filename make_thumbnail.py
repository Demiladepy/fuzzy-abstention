"""Blog thumbnail: one chart, one claim, legible at card size.

    py -3.12 make_thumbnail.py

1200x630, the standard social and HF article card ratio. The test is whether the point
survives being shrunk to a thumbnail in a list of two hundred submissions, so there is
one idea on it and the numbers are large.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path("figures")
INK, MUTED, ACC, CONF = "#11131a", "#6b7280", "#2563eb", "#e07a3f"
LABELS = ["answerable_known", "underspecified", "context_dependent",
          "unanswerable", "answerable_unknown"]
SHORT = ["known", "under-\nspecified", "context-\ndependent", "unanswerable",
         "long-tail"]


def main() -> int:
    rows = [json.loads(l) for l in
            open("runs_dl/runs/default/graded.jsonl", encoding="utf-8") if l.strip()]
    ana = json.load(open("runs_dl/analysis.json", encoding="utf-8"))[0]
    acc, conf = [], []
    for l in LABELS:
        xs = [r for r in rows if r["label"] == l]
        acc.append(sum(r["correct"] for r in xs) / len(xs))
        conf.append(ana["mean_conf_by_label"][l]["verbalised"])

    fig = plt.figure(figsize=(12, 6.3), dpi=100)
    fig.patch.set_facecolor("white")

    fig.text(0.055, 0.90, "Nine points of confidence",
             fontsize=40, fontweight="bold", color=INK, va="top")
    fig.text(0.055, 0.785, "over ninety-four points of accuracy",
             fontsize=40, fontweight="bold", color=CONF, va="top")
    fig.text(0.055, 0.675,
             "A language model's stated confidence barely moves while its accuracy collapses.",
             fontsize=15.5, color=MUTED, va="top")

    ax = fig.add_axes([0.055, 0.155, 0.60, 0.42])
    x = np.arange(len(LABELS))
    ax.bar(x - 0.21, acc, 0.42, color=ACC)
    ax.bar(x + 0.21, conf, 0.42, color=CONF)
    for i, (a, c) in enumerate(zip(acc, conf)):
        ax.text(i - 0.21, a + 0.03, f"{a:.0%}", ha="center", fontsize=11.5,
                fontweight="bold", color=ACC)
        ax.text(i + 0.21, c + 0.03, f"{c:.2f}", ha="center", fontsize=11.5,
                fontweight="bold", color=CONF)
    ax.set_xticks(x, SHORT, fontsize=10.5, color=MUTED)
    ax.set_yticks([])
    ax.set_ylim(0, 1.2)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#d4d8de")
    ax.tick_params(axis="x", length=0)

    # Legend as plain text, so nothing competes with the headline.
    fig.text(0.055, 0.605, "accuracy", fontsize=12.5, color=ACC, fontweight="bold")
    fig.text(0.152, 0.605, "stated confidence", fontsize=12.5, color=CONF,
             fontweight="bold")

    panel = fig.add_axes([0.70, 0.155, 0.25, 0.42])
    panel.axis("off")
    panel.add_patch(plt.Rectangle((0, 0), 1, 1, transform=panel.transAxes,
                                  facecolor="#f4f6f9", edgecolor="#e1e5ea", lw=1.2))
    panel.text(0.5, 0.88, "a probe on layer 20", ha="center", fontsize=12.5,
               color=MUTED, transform=panel.transAxes)
    panel.text(0.5, 0.60, "90.5%", ha="center", fontsize=42, fontweight="bold",
               color=ACC, transform=panel.transAxes)
    panel.text(0.5, 0.42, "reads the question's kind\nbefore the model answers",
               ha="center", fontsize=11.5, color=INK, transform=panel.transAxes,
               linespacing=1.5)
    panel.text(0.5, 0.17, "chance is 20%", ha="center", fontsize=11,
               color=MUTED, style="italic", transform=panel.transAxes)

    fig.text(0.055, 0.065, "CohereLabs/tiny-aya-global   ·   210 items   ·   "
                           "CuratorKIT → SafeTune → AuditKIT",
             fontsize=11.5, color=MUTED)
    fig.text(0.945, 0.065, "Lexsi Hackathon · Track 2b", fontsize=11.5,
             color=MUTED, ha="right")

    fig.savefig(OUT / "thumbnail.png", dpi=100, facecolor="white")
    plt.close(fig)
    p = OUT / "thumbnail.png"
    print(f"  {p}  {p.stat().st_size // 1024} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
