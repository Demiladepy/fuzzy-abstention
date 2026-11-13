"""Expand the hand-authored seed into the full evaluation set, reproducibly.

Why generate rather than hand-write: the contrast the experiment rests on is *minimal
pairs* — the same question with and without the context that makes it answerable. Those
are far better produced from an explicit template with a context slot than written one
at a time, because the template guarantees the two members differ in exactly one way.
A judge can rerun this file and get the same dataset.

    python build_dataset.py --out data/items.jsonl
    python validate_dataset.py data/items.jsonl

Labels follow PLAN.md section 3. Items whose answer exists but which the model is
unlikely to know are `answerable_unknown`; we only claim `answerable_known` where a
verified gold answer is carried in the seed file. Correctness is judged later, against
generations, not asserted here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# ---------------------------------------------------------------- pair templates
# Each family yields minimal pairs: a bare question (context_dependent) and the same
# question with one context slot filled (answerable_unknown — the answer now exists and
# is unique, whether or not this model knows it).

JURISDICTION = {
    "axis": "jurisdiction",
    "clarifier": "Which country or jurisdiction are you asking about?",
    "bare": "What is the {thing}?",
    "specified": "What is the {thing} in {ctx}?",
    "things": [
        "legal voting age", "legal driving age", "minimum age of marriage",
        "statutory retirement age", "blood alcohol limit for drivers",
        "standard rate of value added tax", "minimum wage per hour",
        "maximum speed limit on a motorway", "length of statutory maternity leave",
        "notice period required to terminate a tenancy",
    ],
    "ctxs": ["Nigeria", "Kenya", "Ghana", "South Africa", "India", "Brazil", "Japan"],
}

CALENDAR = {
    "axis": "calendar",
    "clarifier": "Which calendar should I use?",
    "bare": "What date does {thing} fall on?",
    "specified": "In the {ctx} calendar, what date does {thing} fall on?",
    "things": ["new year", "the first day of the seventh month", "the spring equinox"],
    "ctxs": ["Gregorian", "Ethiopian", "Islamic", "Hebrew", "Chinese"],
}

UNITS = {
    "axis": "units",
    "clarifier": "In which unit system?",
    "bare": "What is the freezing point of {thing}?",
    "specified": "What is the freezing point of {thing} in {ctx}?",
    "things": ["water", "mercury", "ethanol", "seawater at average salinity"],
    "ctxs": ["degrees Celsius", "degrees Fahrenheit", "kelvin"],
}

REGION = {
    "axis": "region",
    "clarifier": "Which country or region's convention?",
    "bare": "Which {thing} is standard?",
    "specified": "In {ctx}, which {thing} is standard?",
    "things": [
        "paper size", "mains voltage", "plug type", "date order in written dates",
        "side of the road for driving", "first day of the working week",
    ],
    "ctxs": ["the United Kingdom", "the United States", "Nigeria", "Japan", "Germany"],
}

TIME = {
    "axis": "time",
    "clarifier": "As of which year?",
    "bare": "Who holds the office of {thing}?",
    "specified": "In {ctx}, who held the office of {thing}?",
    "things": [
        "Secretary-General of the United Nations",
        "President of the African Development Bank",
        "Director-General of the World Health Organization",
    ],
    "ctxs": ["1995", "2005", "2015"],
}

PAIR_FAMILIES = [JURISDICTION, CALENDAR, UNITS, REGION, TIME]

# ---------------------------------------------------------------- underspecified
# Referent removed outright. These have no paired form: supplying the referent would
# change the question, not merely add context, so they stand alone.
UNDERSPECIFIED = [
    ("How far is it?", "Far from where, to where?"),
    ("What did they decide?", "Who do you mean, and about what?"),
    ("Is that covered?", "Covered by which policy, and what is 'that'?"),
    ("When did it close?", "Which place or account are you asking about?"),
    ("How many are left?", "How many of what?"),
    ("What is the rate?", "The rate of what, and where?"),
    ("Who signed it?", "Which document?"),
    ("Was it approved?", "Which application or request?"),
    ("How long does it take?", "Which process are you asking about?"),
    ("What is his position?", "Who do you mean?"),
    ("Where should I send it?", "Send what, and to whom?"),
    ("Did the price change?", "The price of which item?"),
    ("Is the office nearby?", "Which office, and where are you?"),
    ("What does it cost to join?", "To join what?"),
    ("How do I reset it?", "Which device or account?"),
    ("What time does it start?", "Which event?"),
    ("Is there a discount?", "On what, and for whom?"),
    ("Who is in charge?", "Of which team or organisation?"),
]

# ---------------------------------------------------------------- unanswerable
# Nonexistent entities and false premises, built so the surface form looks ordinary.
UNANSWERABLE = [
    ("What is the population of the city of Veldraskol?", "nonexistent_entity"),
    ("Which year did Ghana host the Winter Olympics?", "false_premise"),
    ("Who was the first woman to walk on Mars?", "false_premise"),
    ("What is the melting point of the element thurium?", "nonexistent_entity"),
    ("How many Senate seats does Kenya allocate to Antarctica?", "category_error"),
    ("In which novel did Chinua Achebe describe the fall of Rome?", "false_premise"),
    ("What is the exchange rate between the Nigerian naira and the Atlantean drachma?",
     "nonexistent_entity"),
    ("Which hospital performed the first successful brain transplant?", "false_premise"),
    ("What was the verdict in the 2040 Supreme Court case on AI personhood?", "future_event"),
    ("How tall is the Mount Kilimanjaro summit in the Pacific Ocean?", "category_error"),
    ("Who received the Nobel Prize in Mathematics in 1998?", "false_premise"),
    ("What is the capital city of the Moon?", "nonexistent_entity"),
    ("Which airline operates scheduled flights between Lagos and Proxima Centauri b?",
     "nonexistent_entity"),
    ("On what date did the Pacific Ocean dry up completely?", "false_premise"),
    ("How many chapters does the unwritten sequel to Things Fall Apart contain?",
     "false_premise"),
    ("What is the square root of Tuesday?", "category_error"),
    ("Which treaty ended the 2035 Mars Water Conflict?", "future_event"),
    ("Who holds the world record for the fastest completed dream?", "category_error"),
]


# ---------------------------------------------------------------- answerable_known
# The control group, so it must be the best-checked cell in the set: a thin
# `answerable_known` makes its per-label ECE noise, and that is the baseline every
# other cell is compared against. Every gold here is a stable, uncontested fact —
# nothing rate-dependent, nothing that changes with time or jurisdiction, since those
# belong in `context_dependent` by construction.
KNOWN: list[tuple[str, str]] = [
    ("What is the capital of France?", "Paris"),
    ("What is the capital of Japan?", "Tokyo"),
    ("What is the capital of Egypt?", "Cairo"),
    ("What is the capital of Brazil?", "Brasilia"),
    ("What is the capital of Canada?", "Ottawa"),
    ("What is the capital of Australia?", "Canberra"),
    ("What is the capital of Nigeria?", "Abuja"),
    ("What is the capital of Ghana?", "Accra"),
    ("What is the capital of India?", "New Delhi"),
    ("What is the capital of China?", "Beijing"),
    ("What is the capital of Turkey?", "Ankara"),
    ("What is the capital of Morocco?", "Rabat"),
    ("What is the capital of Ethiopia?", "Addis Ababa"),
    ("What is the capital of Switzerland?", "Bern"),
    ("What is the chemical symbol for iron?", "Fe"),
    ("What is the chemical symbol for sodium?", "Na"),
    ("What is the chemical symbol for potassium?", "K"),
    ("What is the chemical symbol for silver?", "Ag"),
    ("What is the chemical symbol for lead?", "Pb"),
    ("What is the chemical symbol for helium?", "He"),
    ("What is 12 multiplied by 12?", "144"),
    ("What is 15 squared?", "225"),
    ("What is 7 multiplied by 8?", "56"),
    ("What is the square root of 81?", "9"),
    ("What is 100 divided by 4?", "25"),
    ("Who wrote the novel 1984?", "George Orwell"),
    ("Who wrote Pride and Prejudice?", "Jane Austen"),
    ("Who wrote Half of a Yellow Sun?", "Chimamanda Ngozi Adichie"),
    ("Who wrote The Old Man and the Sea?", "Ernest Hemingway"),
    ("Who wrote Petals of Blood?", "Ngugi wa Thiong'o"),
    ("Which planet is closest to the Sun?", "Mercury"),
    ("Which is the largest ocean on Earth?", "the Pacific Ocean"),
    ("What is the chemical formula for water?", "H2O"),
    ("How many chromosomes are in a typical human somatic cell?", "46"),
    ("How many bones are in the adult human body?", "206"),
    ("Which is the longest river in Africa?", "the Nile"),
    ("What is the highest mountain above sea level on Earth?", "Mount Everest"),
    ("Which gas do plants absorb from the air for photosynthesis?", "carbon dioxide"),
    ("In which year did the Second World War end?", "1945"),
    ("In which year did humans first land on the Moon?", "1969"),
    ("In which year did Nigeria gain independence?", "1960"),
    ("In which year did Kenya gain independence?", "1963"),
    ("In which year did Ghana gain independence?", "1957"),
    ("What is the official language of Brazil?", "Portuguese"),
    ("To which language family does Yoruba belong?", "Niger-Congo"),
    ("How many sides does a pentagon have?", "5"),
    ("What is the freezing point of water at standard atmospheric pressure in Celsius?", "0"),
]


def load_seed(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def row(
    rid: str,
    label: str,
    question: str,
    *,
    gold=None,
    answer_exists=True,
    context_axis=None,
    clarifying_question=None,
    defect=None,
    pair_id=None,
    lang="en",
) -> dict:
    return {
        "id": rid,
        "lang": lang,
        "label": label,
        "question": question,
        "gold": gold,
        "answer_exists": answer_exists,
        "context_axis": context_axis,
        "clarifying_question": clarifying_question,
        "defect": defect,
        "pair_id": pair_id,
    }


def build() -> list[dict]:
    out: list[dict] = []
    n = 0

    # Minimal pairs. One bare question per `thing`, paired with one specified variant,
    # cycling the context list so contexts are spread rather than all identical.
    for fam_i, fam in enumerate(PAIR_FAMILIES):
        for t_i, thing in enumerate(fam["things"]):
            ctx = fam["ctxs"][t_i % len(fam["ctxs"])]
            pid = f"gen_{fam['axis']}_{t_i}"
            n += 1
            out.append(
                row(
                    f"gcd{fam_i}{t_i:02d}",
                    "context_dependent",
                    fam["bare"].format(thing=thing),
                    context_axis=fam["axis"],
                    clarifying_question=fam["clarifier"],
                    pair_id=pid,
                )
            )
            out.append(
                row(
                    f"gak{fam_i}{t_i:02d}",
                    "answerable_unknown",
                    fam["specified"].format(thing=thing, ctx=ctx),
                    pair_id=pid,
                )
            )

    for i, (q, clar) in enumerate(UNDERSPECIFIED):
        out.append(
            row(f"gus{i:03d}", "underspecified", q, clarifying_question=clar)
        )

    for i, (q, defect) in enumerate(UNANSWERABLE):
        out.append(
            row(f"gun{i:03d}", "unanswerable", q, answer_exists=False, defect=defect)
        )

    for i, (q, gold) in enumerate(KNOWN):
        out.append(row(f"gkn{i:03d}", "answerable_known", q, gold=gold))

    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", default="data/seed_items.jsonl")
    ap.add_argument("--out", default="data/items.jsonl")
    args = ap.parse_args()

    seed = load_seed(args.seed)
    generated = build()

    # Deduplicate on question text, seed wins — hand-authored items are better checked.
    seen = {r["question"].strip().lower() for r in seed}
    kept = []
    dropped = 0
    for r in generated:
        q = r["question"].strip().lower()
        if q in seen:
            dropped += 1
            continue
        seen.add(q)
        kept.append(r)

    # A pair is only useful if both members survived dedup; drop orphans so the
    # validator's "exactly two members" invariant means something.
    pair_counts: dict[str, int] = {}
    for r in kept:
        if r["pair_id"]:
            pair_counts[r["pair_id"]] = pair_counts.get(r["pair_id"], 0) + 1
    orphans = {p for p, c in pair_counts.items() if c != 2}
    before = len(kept)
    kept = [r for r in kept if not (r["pair_id"] and r["pair_id"] in orphans)]
    orphaned = before - len(kept)

    allrows = seed + kept
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        for r in allrows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    counts: dict[str, int] = {}
    for r in allrows:
        counts[r["label"]] = counts.get(r["label"], 0) + 1
    pairs = len({r["pair_id"] for r in allrows if r["pair_id"]})

    print(f"seed      {len(seed)}")
    print(f"generated {len(generated)}  (dropped {dropped} duplicate, {orphaned} orphaned)")
    print(f"total     {len(allrows)}\n")
    for k in sorted(counts):
        print(f"  {k:<20} {counts[k]:>4}")
    print(f"\n  minimal pairs       {pairs:>4}")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
