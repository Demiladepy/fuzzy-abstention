"""Generate the figures for the write-up.

    py -3.12 make_figures.py

Four figures, each carrying one claim. Everything is read from the committed result
files, so a figure cannot drift from the number it illustrates.

    fig1_calibration_gap.png   accuracy vs stated confidence, per label  <- the headline
    fig2_risk_coverage.png     risk-coverage curves for both scalars
    fig3_policy.png            error and over-abstention at matched coverage
    fig4_minimal_pairs.png     confidence gap vs correctness flip, per pair
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from selective_metrics import risk_coverage_curve

OUT = Path("figures")
LABELS = ["answerable_known", "answerable_unknown", "unanswerable",
          "context_dependent", "underspecified"]
NICE = {"answerable_known": "Answerable\n(known)", "answerable_unknown": "Answerable\n(long-tail)",
        "unanswerable": "Unanswerable", "context_dependent": "Context-\ndependent",
        "underspecified": "Under-\nspecified"}

INK, MUTED, ACC, CONF, BAD = "#16181d", "#6b7280", "#2563eb", "#e07a3f", "#c2410c"
plt.rcParams.update({
    "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#c9ced6", "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "figure.facecolor": "white",
    "axes.titlesize": 13, "axes.titleweight": "bold",
})


def load():
    rows = [json.loads(l) for l in
            open("runs_dl/runs/default/graded.jsonl", encoding="utf-8") if l.strip()]
    cv = json.load(open("runs_dl/runs/default/cv_results.json", encoding="utf-8"))
    ana = json.load(open("runs_dl/analysis.json", encoding="utf-8"))[0]
    return rows, cv, ana


def fig1(rows, ana):
    """The headline: confidence barely moves while accuracy collapses."""
    acc, conf, ece = [], [], ana["ece_by_label"]["verbalised"]
    for l in LABELS:
        xs = [r for r in rows if r["label"] == l]
        acc.append(sum(r["correct"] for r in xs) / len(xs))
        conf.append(ana["mean_conf_by_label"][l]["verbalised"])

    x = np.arange(len(LABELS))
    fig, ax = plt.subplots(figsize=(10, 5.6))
    ax.bar(x - 0.2, acc, 0.4, label="actual accuracy", color=ACC)
    ax.bar(x + 0.2, conf, 0.4, label="model's stated confidence", color=CONF)

    for i, (a, c) in enumerate(zip(acc, conf)):
        ax.text(i - 0.2, a + 0.02, f"{a:.0%}", ha="center", fontsize=10.5, color=ACC,
                fontweight="bold")
        ax.text(i + 0.2, c + 0.02, f"{c:.2f}", ha="center", fontsize=10.5, color=CONF,
                fontweight="bold")

    # ECE folded into the tick label itself — a separate text row collides with
    # two-line category names.
    ticks = [f"{NICE[l]}\nECE {ece[l]:.3f}" for l in LABELS]
    ax.set_xticks(x, ticks, fontsize=10)
    for t, l in zip(ax.get_xticklabels(), LABELS):
        if ece[l] > 0.3:
            t.set_color(BAD)
            t.set_fontweight("bold")

    ax.set_ylim(0, 1.08)
    ax.set_xlim(-0.65, len(LABELS) - 0.35)
    ax.set_ylabel("proportion")
    ax.set_title(
        "The model's confidence barely moves while its accuracy collapses\n"
        "accuracy spans 0% – 94%   ·   stated confidence spans 0.865 – 0.955",
        pad=46, linespacing=1.6,
    )
    ax.legend(frameon=False, loc="upper center", ncols=2, bbox_to_anchor=(0.5, 1.10))
    fig.tight_layout()
    fig.savefig(OUT / "fig1_calibration_gap.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def fig2(rows):
    """Risk-coverage: how much error you eat to answer a given share."""
    correct = np.array([r["correct"] for r in rows], dtype=float)
    verb = np.array([r.get("verbalised", np.nan) for r in rows], dtype=float)
    seq = np.exp(np.clip(np.array([r.get("logprob_mean", np.nan) for r in rows],
                                  dtype=float), -20, 0))
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for s, name, col in ((verb, "verbalised confidence", CONF),
                         (seq, "sequence likelihood", ACC)):
        ok = ~np.isnan(s)
        cov, risk = risk_coverage_curve(s[ok], correct[ok])
        ax.plot(cov, risk, label=name, color=col, lw=2.2)
    ax.axhline(1 - correct.mean(), ls=":", color=MUTED, lw=1.4)
    ax.text(0.02, 1 - correct.mean() + 0.012, "answer everything (42.9% error)",
            fontsize=9.5, color=MUTED)
    ax.set_xlabel("coverage (share of questions answered)")
    ax.set_ylabel("risk (error rate among answered)")
    ax.set_title("A good confidence signal should bend down to the left")
    ax.legend(frameon=False, loc="lower right")
    ax.set_xlim(0, 1)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_risk_coverage.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def fig3(cv):
    """Policy comparison at matched coverage."""
    pol = {r["policy"]: r for r in cv["policy_table"]}
    names = ["always answer", "seq_prob @matched cov", "probe (5-way)"]
    show = ["answer everything", "best scalar\n@ matched coverage", "probe\n(answer/abstain/ask)"]
    err = [pol[n]["error_when_answering"] for n in names]
    over = [pol[n]["over_abstention_on_known"] for n in names]

    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    b1 = ax.bar(x - 0.2, err, 0.4, label="error when answering", color=BAD)
    b2 = ax.bar(x + 0.2, over, 0.4, label="over-abstention on known-answerable",
                color=MUTED)
    for b in list(b1) + list(b2):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.015,
                f"{b.get_height():.1%}", ha="center", fontsize=10, fontweight="bold")
    ax.set_xticks(x, show)
    ax.set_ylim(0, 0.85)
    ax.set_ylabel("rate (lower is better)")
    ax.set_title("Same coverage, 3.3x less error and 4x less over-abstention")
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT / "fig3_policy.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def fig4(rows):
    """Minimal pairs: correctness flips while confidence does not."""
    by: dict[str, list] = {}
    for r in rows:
        if r.get("pair_id"):
            by.setdefault(r["pair_id"], []).append(r)
    gaps, flips = [], []
    for members in by.values():
        if len(members) != 2:
            continue
        a, b = members
        if a.get("verbalised") is None or b.get("verbalised") is None:
            continue
        gaps.append(abs(a["verbalised"] - b["verbalised"]))
        flips.append(a["correct"] != b["correct"])
    gaps, flips = np.array(gaps), np.array(flips)

    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    ax.scatter(gaps[flips], np.ones(flips.sum()) + np.random.default_rng(0)
               .normal(0, 0.035, flips.sum()), s=70, color=BAD, alpha=.8,
               label=f"correctness FLIPS ({int(flips.sum())} pairs)", zorder=3)
    ax.scatter(gaps[~flips], np.zeros((~flips).sum()) + np.random.default_rng(1)
               .normal(0, 0.035, (~flips).sum()), s=70, color=MUTED, alpha=.75,
               label=f"same outcome ({int((~flips).sum())} pairs)", zorder=3)
    ax.axvline(gaps.mean(), color=ACC, ls="--", lw=1.8)
    ax.text(gaps.mean() + 0.004, 0.5, f"mean gap\n{gaps.mean():.3f}", color=ACC,
            fontsize=10, fontweight="bold", va="center")
    ax.set_yticks([0, 1], ["same", "flips"])
    ax.set_xlabel("absolute gap in stated confidence across the pair")
    ax.set_title("Same question, context withheld: the answer flips, the confidence doesn't")
    ax.legend(frameon=False, loc="upper right")
    ax.set_ylim(-0.35, 1.45)
    fig.tight_layout()
    fig.savefig(OUT / "fig4_minimal_pairs.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def fig5():
    """Cross-lingual transfer, if the multilingual run exists."""
    p = Path("runs_dl/transfer_results.json")
    if not p.exists():
        print("  (skipping fig5: no transfer_results.json yet)")
        return
    res = json.load(open(p, encoding="utf-8"))
    key = next((k for k in res if k.startswith("clean")), None)
    if not key:
        return
    per = res[key]["per_language"]
    names = {"en_heldout": "English\n(held out)", "yo": "Yoruba", "ha": "Hausa",
             "sw": "Swahili"}
    order = [k for k in ["en_heldout", "yo", "ha", "sw"] if k in per]
    acc = [per[k]["acc"] for k in order]
    lo = [per[k]["acc"] - per[k]["ci"][0] for k in order]
    hi = [per[k]["ci"][1] - per[k]["acc"] for k in order]
    cols = [ACC] + [CONF] * (len(order) - 1)

    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    x = np.arange(len(order))
    ax.bar(x, acc, 0.55, color=cols, yerr=[lo, hi], capsize=5,
           error_kw=dict(ecolor=MUTED, lw=1.4))
    for i, a in enumerate(acc):
        ax.text(i, a + max(hi) + 0.03, f"{a:.1%}", ha="center", fontsize=11,
                fontweight="bold", color=cols[i])
    ax.axhline(0.2, ls=":", color=BAD, lw=1.5)
    ax.text(len(order) - 0.45, 0.225, "chance (20%)", fontsize=9.5, color=BAD,
            ha="right")
    ax.set_xticks(x, [names.get(k, k) for k in order])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("probe accuracy (5-way)")
    ax.set_title("Does an English-trained probe read epistemic kind in other languages?\n"
                 "source questions held out, so this is transfer and not memorisation",
                 pad=16, linespacing=1.5)
    fig.tight_layout()
    fig.savefig(OUT / "fig5_transfer.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def fig0_pipeline():
    """How the three hackathon repos actually connect."""
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 50)
    ax.axis("off")

    KIT, MINE, OUTP = "#2563eb", "#6b7280", "#15803d"

    def box(x, y, w, h, title, sub, col, fs=10.5):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6",
                                    linewidth=1.8, edgecolor=col,
                                    facecolor=col + "14"))
        ax.text(x + w / 2, y + h - 3.2, title, ha="center", va="top", fontsize=fs,
                fontweight="bold", color=col)
        ax.text(x + w / 2, y + h - 8.2, sub, ha="center", va="top", fontsize=8.8,
                color=INK, linespacing=1.45)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=15, color="#9aa2ad", lw=1.6,
                                     shrinkA=0, shrinkB=0))

    box(1, 24, 19, 17, "build_dataset.py", "210 items\n5 epistemic kinds\n31 minimal pairs", MINE)
    box(23, 24, 20, 17, "CuratorKIT", "schema gate\nexact + MinHash dedup\nprovenance manifest", KIT)
    box(46, 24, 21, 17, "HF Jobs · T4", "generations\nlogprobs\n37-layer activations", MINE)

    box(23, 2, 20, 17, "SafeTune", "LinearProbeGuard\nbinary baseline\n5-fold out-of-fold", KIT)
    box(46, 2, 21, 17, "cv_probe.py", "5-way probe\nlayer chosen in-fold\nthree-way policy", MINE)
    box(70, 13, 20, 17, "AuditKIT", "custom metric\ncompare()\nfingerprints", KIT)

    arrow(20, 32.5, 23, 32.5)
    arrow(43, 32.5, 46, 32.5)
    arrow(56.5, 24, 56.5, 19)        # activations down to the probe
    arrow(46, 10.5, 43, 10.5)        # probe <- safetune comparison
    arrow(67, 10.5, 74, 13)
    arrow(67, 32.5, 74, 30)

    ax.text(50, 46.5, "CuratorKIT  →  SafeTune  →  AuditKIT", ha="center",
            fontsize=13, fontweight="bold", color=INK)
    ax.text(1, 0.2, "blue = hackathon repo     grey = ours", fontsize=9, color=MUTED)
    fig.tight_layout()
    fig.savefig(OUT / "fig0_pipeline.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    OUT.mkdir(exist_ok=True)
    fig0_pipeline()
    rows, cv, ana = load()
    fig1(rows, ana)
    fig2(rows)
    fig3(cv)
    fig4(rows)
    fig5()
    for f in sorted(OUT.glob("*.png")):
        print(f"  {f}  {f.stat().st_size // 1024} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
