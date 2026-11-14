"""Score the runs through AuditKIT, and compare the two preamble conditions.

    py -3.12 audit_eval.py

AuditKIT is the evaluation spine here rather than a logo on a diagram. It scores the
`answerable_known` subset against gold with its own metrics, and `ak.compare()` produces
the before/after comparison with a provenance fingerprint over model + seed + tasks +
config, so a result can be re-identified rather than just re-quoted.

The generations already exist, so the "model" handed to AuditKIT is a callable that
replays the stored answers. That is the supported `list[str] -> list[str]` interface and
it keeps scoring deterministic and free — re-generating on a GPU to score text we
already have would cost money and change the numbers.

Only `answerable_known` goes through gold-matching metrics. The other four labels are
graded on *behaviour* (abstain / reject premise / ask), which is not a string-match task;
those stay with `grade.py`, and mixing them into an exact-match average would produce a
number that means nothing.
"""

from __future__ import annotations

import json
from pathlib import Path

import auditkit as ak
from auditkit.metric import Direction, Metric, Score

from grade import gold_matches


@ak.METRICS.register("gold_containment")
class GoldContainment(Metric):
    """1.0 if the gold answer appears in the output.

    AuditKIT's `contains` metric takes a fixed substring at construction, so it checks
    one constant across every sample and cannot do per-sample gold. `exact_match`
    requires the whole output to equal the gold, which free-form answers never do:
    gold "Nairobi" against "The capital of Kenya is Nairobi."

    This reuses `grade.gold_matches` verbatim, so AuditKIT's score and our grader's
    accuracy are the same measurement and can be compared directly. Numeric golds match
    on word boundaries, so gold "9" does not match inside "1963".
    """

    name = "gold_containment"
    direction = Direction.MAXIMIZE
    required_fields = frozenset({"target"})

    def score(self, sample, output: str, context=None) -> Score:
        hit = gold_matches(output or "", str(sample.target or ""))
        return Score(name=self.name, value=1.0 if hit else 0.0, kind=self.kind)


RUNS = {"default": "runs_dl/runs/default", "minimal": "runs_dl/runs/minimal"}

# `contains` is the metric that fits free-form answers: the model says "The capital of
# Kenya is Nairobi." against a gold of "Nairobi". `exact_match` is kept deliberately
# alongside it, because the gap between the two is itself a result worth showing —
# off-the-shelf exact matching scores this model at 1.6% where it is really 93.7%.
METRICS = ["gold_containment", "exact_match", "f1_score"]


def load(run: str) -> list[dict]:
    return [json.loads(l) for l in
            open(Path(run) / "graded.jsonl", encoding="utf-8") if l.strip()]


def evaluate_known(rows: list[dict], tag: str):
    """AuditKIT over the gold-answerable subset."""
    known = [r for r in rows if r["label"] == "answerable_known" and r.get("gold")]
    samples = [ak.Sample(input=r["question"], target=r["gold"], id=r["id"]) for r in known]
    answers = {r["question"]: r.get("answer", "") for r in known}

    def replay(prompts: list[str]) -> list[str]:
        return [answers.get(p, "") for p in prompts]

    # `cache=True` is the default and it bit us: both conditions hashed to the same
    # fingerprint (same samples, same callable identity), so the second evaluation was
    # served from cache and returned the first one's numbers byte-for-byte — while 52
    # of the 63 answers actually differ. A comparison built on that is vacuous.
    replay.__name__ = f"replay_{tag}"
    result = ak.evaluate(samples, replay, METRICS, cache=False, tags=[f"preamble:{tag}"])
    print(f"\n--- AuditKIT: {tag} ({len(samples)} answerable_known) ---")
    print(result.summary())
    return result


def main() -> int:
    results = {}
    for tag, run in RUNS.items():
        if not (Path(run) / "graded.jsonl").exists():
            print(f"skip {tag}: no graded.jsonl")
            continue
        results[tag] = evaluate_known(load(run), tag)

    out: dict = {}
    for tag, r in results.items():
        out[tag] = {
            "headline": {k: float(v) for k, v in dict(r.headline).items()},
            "fingerprint": getattr(r, "fingerprint", None),
            "n": len(getattr(r, "predictions", []) or []),
        }
        print(f"\n{tag}: headline={out[tag]['headline']} "
              f"fingerprint={str(out[tag]['fingerprint'])[:16]}")

    if len(results) == 2:
        a, b = results["default"], results["minimal"]
        try:
            cmp = ak.compare(a, b)
            print("\n--- AuditKIT compare: default vs minimal preamble ---")
            print(cmp.summary())
            out["comparison"] = str(cmp.summary())
        except Exception as exc:  # noqa: BLE001
            out["comparison_error"] = f"{type(exc).__name__}: {exc}"
            print("compare failed:", out["comparison_error"])

    dest = Path("runs_dl/auditkit_results.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"\nwrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
