"""Schema and invariant checks for the abstention dataset.

Run on every regeneration — a mislabelled `context_dependent` item is the single
easiest way to invalidate the headline result, since that group is where the whole
thesis is measured.

Usage:
    python validate_dataset.py data/seed_items.jsonl
"""

from __future__ import annotations

import collections
import json
import sys

LABELS = {
    "answerable_known",
    "answerable_unknown",
    "unanswerable",
    "context_dependent",
    "underspecified",
}
NEEDS_CLARIFIER = {"context_dependent", "underspecified"}
FIELDS = {
    "id", "lang", "label", "question", "gold", "answer_exists",
    "context_axis", "clarifying_question", "defect", "pair_id",
}
DEFECTS = {"false_premise", "nonexistent_entity", "category_error", "future_event"}


def validate(path: str) -> int:
    errors: list[str] = []
    rows: list[dict] = []

    with open(path, encoding="utf-8") as fh:
        for n, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                errors.append(f"line {n}: invalid JSON — {exc}")

    seen_ids: set[str] = set()
    seen_questions: dict[str, str] = {}
    pairs: dict[str, list[str]] = collections.defaultdict(list)

    for r in rows:
        rid = r.get("id", "<no id>")

        missing = FIELDS - set(r)
        extra = set(r) - FIELDS
        if missing:
            errors.append(f"{rid}: missing fields {sorted(missing)}")
        if extra:
            errors.append(f"{rid}: unexpected fields {sorted(extra)}")

        if rid in seen_ids:
            errors.append(f"{rid}: duplicate id")
        seen_ids.add(rid)

        q = (r.get("question") or "").strip().lower()
        if not q:
            errors.append(f"{rid}: empty question")
        elif q in seen_questions:
            errors.append(f"{rid}: duplicate question, same text as {seen_questions[q]}")
        else:
            seen_questions[q] = rid

        label = r.get("label")
        if label not in LABELS:
            errors.append(f"{rid}: bad label {label!r}")
            continue

        # Per-label invariants.
        if label == "answerable_known":
            if not r.get("gold"):
                errors.append(f"{rid}: answerable_known needs a gold answer")
            if r.get("answer_exists") is not True:
                errors.append(f"{rid}: answerable_known must have answer_exists true")
        if label == "unanswerable":
            if r.get("answer_exists") is not False:
                errors.append(f"{rid}: unanswerable must have answer_exists false")
            if r.get("defect") not in DEFECTS:
                errors.append(f"{rid}: unanswerable needs a defect from {sorted(DEFECTS)}")
            if r.get("gold"):
                errors.append(f"{rid}: unanswerable must not carry a gold answer")
        else:
            if r.get("defect") is not None:
                errors.append(f"{rid}: only unanswerable items may set defect")

        if label in NEEDS_CLARIFIER:
            if not r.get("clarifying_question"):
                errors.append(f"{rid}: {label} needs a clarifying_question")
            if r.get("gold"):
                errors.append(f"{rid}: {label} must not carry a single gold answer")
        elif r.get("clarifying_question"):
            errors.append(f"{rid}: clarifying_question only belongs on {sorted(NEEDS_CLARIFIER)}")

        if label == "context_dependent" and not r.get("context_axis"):
            errors.append(f"{rid}: context_dependent needs a context_axis")
        if label != "context_dependent" and r.get("context_axis"):
            errors.append(f"{rid}: only context_dependent may set context_axis")

        if r.get("pair_id"):
            pairs[r["pair_id"]].append(rid)

    # A minimal pair is exactly two items whose epistemic kind differs *because* one
    # supplies context the other withholds. So: exactly one member must be the
    # ambiguous side (context_dependent / underspecified), and the other must be
    # answerable — either `answerable_known` where a verified gold exists, or
    # `answerable_unknown` where the answer is now unique but we make no claim the
    # model knows it. Requiring `answerable_known` would force us to assert golds we
    # have not checked, which is a worse failure than a looser invariant.
    by_id = {r.get("id"): r for r in rows}
    for pid, members in sorted(pairs.items()):
        if len(members) != 2:
            errors.append(f"pair {pid!r}: has {len(members)} members ({members}), expected 2")
            continue
        labels = [by_id[m]["label"] for m in members]
        ambiguous = [l for l in labels if l in NEEDS_CLARIFIER]
        answerable = [l for l in labels if l.startswith("answerable_")]
        if len(ambiguous) != 1:
            errors.append(
                f"pair {pid!r}: needs exactly one context_dependent/underspecified "
                f"member, got {labels}"
            )
        if len(answerable) != 1:
            errors.append(f"pair {pid!r}: needs exactly one answerable_* member, got {labels}")

    counts = collections.Counter(r.get("label") for r in rows)
    print(f"{len(rows)} items from {path}\n")
    for label in sorted(LABELS):
        print(f"  {label:<20} {counts.get(label, 0):>4}")
    print(f"\n  minimal pairs       {len(pairs):>4}")
    langs = collections.Counter(r.get("lang") for r in rows)
    print(f"  languages           {dict(langs)}")

    print("\n" + "-" * 60)
    if errors:
        print(f"{len(errors)} problem(s):\n")
        for e in errors:
            print(f"  - {e}")
        return 1
    print("Schema and invariants OK.")
    return 0


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "data/seed_items.jsonl"
    sys.exit(validate(target))
