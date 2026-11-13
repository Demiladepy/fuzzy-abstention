"""Stage 1b: grade each generation, where "correct" is label-dependent.

Only `answerable_known` is graded by matching an answer. For the other four labels the
right output is a *behaviour*, so we classify the behaviour and score against what that
label calls for:

    answerable_known     says the gold answer
    answerable_unknown   admits it does not know           (asserting a figure = hallucination)
    unanswerable         rejects the premise, or abstains
    context_dependent    asks which context, or says it varies by one
    underspecified       asks what is meant, or abstains

Deliberately rule-based rather than LLM-judged: it is transparent, free, reproducible,
and auditable line by line. An LLM judge here would also put a second model's opinion
inside the measurement we are making about the first model's calibration.

    py -3.12 grade.py --self-test
    py -3.12 grade.py --gens out/generations.jsonl --out out/graded.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path

ABSTAIN = "abstain"
REJECT = "reject_premise"
ASK = "ask_clarification"
HEDGE = "hedge_varies"
DIRECT = "direct_answer"

# Ordered: the first match wins, so the more specific patterns come first.
ABSTAIN_PAT = re.compile(
    r"\b(i (?:do not|don't) know|i'?m not (?:sure|certain)|not able to determine"
    r"|cannot (?:determine|say|verify|confirm)|can'?t (?:determine|say)"
    r"|unable to (?:determine|answer|verify)|no (?:reliable |specific |publicly )?"
    r"(?:information|data|record)|i (?:do not|don't) have (?:access|that|enough|specific)"
    r"|not (?:publicly )?(?:available|documented|recorded)|i am not aware"
    r"|unclear|uncertain)\b",
    re.I,
)
REJECT_PAT = re.compile(
    r"\b(no such|does not exist|doesn'?t exist|did not (?:win|happen|occur|found)"
    r"|never (?:won|happened|occurred|hosted|founded|adopted|landed)"
    r"|there (?:is|are|was|were) no\b|has no\b|have no\b|not a real"
    r"|is fictional|is not a (?:real|recognised|recognized)|false premise"
    r"|hasn'?t (?:won|happened)|has not (?:won|happened)|no record of"
    r"|not an element|is not an? (?:element|country|city|planet))\b",
    re.I,
)
ASK_PAT = re.compile(
    r"(could you (?:please )?(?:clarify|specify|tell me)|can you (?:clarify|specify|tell me)"
    r"|please (?:clarify|specify)|which (?:country|one|context|calendar|region|place|"
    r"jurisdiction|tower|bank|election|document|match|event|item|process|organisation|"
    r"organization|system|work)|what (?:exactly )?(?:do you mean|are you referring)"
    r"|whom? are you|are you asking about|more context|more information to)",
    re.I,
)
HEDGE_PAT = re.compile(
    r"\b(varies|depends on|depending on|differs (?:by|between)|differ (?:by|between)"
    r"|is not (?:the same|uniform)|context[- ]dependent|no single answer"
    r"|different (?:countries|calendars|regions|conventions|systems|definitions)"
    r"|in some (?:countries|places|regions|systems))\b",
    re.I,
)

# What counts as correct for each label.
ACCEPTS: dict[str, set[str]] = {
    "answerable_unknown": {ABSTAIN},
    "unanswerable": {REJECT, ABSTAIN},
    "context_dependent": {ASK, HEDGE},
    "underspecified": {ASK, ABSTAIN},
}


def classify(answer: str) -> str:
    """Label the behaviour of a generated answer. First match wins."""
    a = (answer or "").strip()
    if not a:
        return ABSTAIN
    if REJECT_PAT.search(a):
        return REJECT
    if ABSTAIN_PAT.search(a):
        return ABSTAIN
    if ASK_PAT.search(a):
        return ASK
    if HEDGE_PAT.search(a):
        return HEDGE
    # A bare trailing question with no statement before it is a request for context.
    if a.endswith("?") and len(a.split()) <= 25:
        return ASK
    return DIRECT


def _norm(s: str) -> str:
    """Casefold, strip accents and punctuation, collapse spaces — for gold matching."""
    s = unicodedata.normalize("NFKD", s or "")
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[^\w\s]", " ", s.lower())
    return re.sub(r"\s+", " ", s).strip()


def gold_matches(answer: str, gold: str) -> bool:
    """Whether the gold answer appears in the response.

    Numeric golds are matched on word boundaries so that gold "9" does not match the
    "9" inside "1963" — a silent false positive that would inflate the control cell,
    which is the baseline everything else is compared against.
    """
    a, g = _norm(answer), _norm(gold)
    if not g:
        return False
    if re.fullmatch(r"[\d.,]+", g):
        return re.search(rf"(?<!\d){re.escape(g)}(?!\d)", a) is not None
    return g in a


def grade_row(row: dict) -> dict:
    label = row["label"]
    answer = row.get("answer", "")
    behaviour = classify(answer)

    if label == "answerable_known":
        correct = gold_matches(answer, row.get("gold") or "")
    else:
        correct = behaviour in ACCEPTS[label]

    out = dict(row)
    out["behaviour"] = behaviour
    out["correct"] = bool(correct)
    # An assertion where the model could not possibly know is the hallucination case.
    out["hallucinated"] = bool(
        label in {"answerable_unknown", "unanswerable"} and behaviour == DIRECT
    )
    return out


# ------------------------------------------------------------------ self-test

CASES: list[tuple[str, str, str, str, bool]] = [
    # (name, label, gold, answer, expected_correct)
    ("known hit", "answerable_known", "Nairobi", "The capital of Kenya is Nairobi.", True),
    ("known miss", "answerable_known", "Nairobi", "The capital of Kenya is Mombasa.", False),
    ("known accent", "answerable_known", "Brasilia", "It is Brasília.", True),
    ("known numeric", "answerable_known", "9", "The square root of 81 is 9.", True),
    ("numeric not substring", "answerable_known", "9", "It happened in 1963.", False),
    ("unknown abstains", "answerable_unknown", "", "I don't know the exact figure.", True),
    ("unknown fabricates", "answerable_unknown", "",
     "The recorded population was 112,349.", False),
    ("unanswerable rejects", "unanswerable", "",
     "Chinua Achebe never won a Nobel Prize.", True),
    ("unanswerable abstains", "unanswerable", "", "I'm not sure about that.", True),
    ("unanswerable fabricates", "unanswerable", "",
     "He won the Nobel Prize in Literature in 1986.", False),
    ("context asks", "context_dependent", "", "Which country are you asking about?", True),
    ("context hedges", "context_dependent", "",
     "The number of continents varies depending on the criteria used.", True),
    ("context answers flatly", "context_dependent", "", "There are seven continents.", False),
    ("underspecified asks", "underspecified", "",
     "Could you clarify which tower you mean?", True),
    ("underspecified answers", "underspecified", "", "It is 324 metres tall.", False),
    ("empty is abstain", "answerable_unknown", "", "", True),
]


def self_test() -> int:
    failures = []
    for name, label, gold, answer, expected in CASES:
        got = grade_row({"label": label, "gold": gold, "answer": answer})
        if got["correct"] != expected:
            failures.append(f"{name}: expected correct={expected}, got {got['correct']} "
                            f"(behaviour={got['behaviour']})")
            print(f"[FAIL] {name} — behaviour={got['behaviour']}")
        else:
            print(f"[PASS] {name:<26} behaviour={got['behaviour']}")

    # The hallucination flag should fire exactly on confident assertions where the
    # model could not know.
    h = grade_row({"label": "answerable_unknown", "gold": "",
                   "answer": "The population was 112,349."})
    if not h["hallucinated"]:
        failures.append("hallucination flag did not fire on a fabricated figure")
        print("[FAIL] hallucination flag")
    else:
        print("[PASS] hallucination flag fires on fabricated figure")

    h2 = grade_row({"label": "answerable_known", "gold": "Nairobi", "answer": "Mombasa."})
    if h2["hallucinated"]:
        failures.append("hallucination flag fired on a merely wrong known-answer item")
        print("[FAIL] hallucination flag over-fires")
    else:
        print("[PASS] hallucination flag ignores wrong-but-knowable answers")

    print("\n" + ("-" * 60))
    if failures:
        print(f"{len(failures)} FAILED")
        for f in failures:
            print("  -", f)
        return 1
    print("All checks passed.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gens", default="out/generations.jsonl")
    ap.add_argument("--out", default="out/graded.jsonl")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        return self_test()

    rows = [json.loads(l) for l in open(args.gens, encoding="utf-8") if l.strip()]
    graded = [grade_row(r) for r in rows]

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        for g in graded:
            fh.write(json.dumps(g, ensure_ascii=False) + "\n")

    by_label: dict[str, list] = {}
    for g in graded:
        by_label.setdefault(g["label"], []).append(g)
    print(f"{len(graded)} graded\n")
    print(f"  {'label':<20} {'n':>4} {'acc':>6} {'halluc':>7}")
    for lab in sorted(by_label):
        rs = by_label[lab]
        acc = sum(r["correct"] for r in rs) / len(rs)
        hal = sum(r["hallucinated"] for r in rs) / len(rs)
        print(f"  {lab:<20} {len(rs):>4} {acc:>6.1%} {hal:>7.1%}")
    beh: dict[str, int] = {}
    for g in graded:
        beh[g["behaviour"]] = beh.get(g["behaviour"], 0) + 1
    print(f"\n  behaviours: {dict(sorted(beh.items(), key=lambda kv: -kv[1]))}")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
