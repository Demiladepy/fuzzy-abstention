"""Stage 0: run the item set through CuratorKIT's gates.

This is not a box-ticking use of the repo. Template generation has one obvious
weakness — the items it produces are formulaic, so many are near-duplicates of each
other in everything but a slot value. A reviewer would spot that immediately, and
exact-match dedup does not catch it. CuratorKIT's `DiversityGate` and near-duplicate
dedup are the right instruments, so the curation stage earns its place by fixing a
real defect rather than by appearing in the pipeline diagram.

    py -3.12 curate.py --dry-run          # gate plan, no work
    py -3.12 curate.py                    # writes curated output + HF README

Gates used, and why:
  SchemaGate      required fields present, lengths sane
  dedup=minhash   near-duplicate questions, not just exact matches
  DiversityGate   prunes formulaic clones left by template expansion

HallucinationGate and RewardGate are deliberately NOT used: both need an LLM judge,
and our items are questions whose answers are evaluated later against the model under
test. Scoring them with a second LLM here would conflate curation with measurement.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from curatorkit import Curator, CuratorConfig


def build_config(items: str, out: str, *, diversity: float | None, dedup: str) -> CuratorConfig:
    return CuratorConfig(
        dataset=items,
        # `format` is the DATA SCHEMA (alpaca / sharegpt / instruction / preference),
        # not the file type. Passing "jsonl" declares a schema that does not exist and
        # every row is rejected as `invalid_forced_format:jsonl` — 212 in, 0 out. Leave
        # it on auto-detection.
        format="auto",
        name="fuzzy_abstention_items",
        output_dir=out,
        # CuratorKIT normalises into a canonical schema (instruction / input / output /
        # chosen / rejected / label / responses); anything unrecognised is preserved
        # under `metadata`. Direction is {our_column: canonical_column}.
        #
        # `label` MUST be renamed. Our epistemic label collides with the canonical
        # `label` field and is silently coerced to None — which would null out the one
        # column every per-label metric depends on, with no error anywhere. Renaming it
        # to a non-canonical name routes it safely into metadata instead.
        field_mapping={
            "question": "instruction",
            "gold": "output",
            "label": "epistemic_label",
        },
        # --- schema ---
        schema_gate=True,
        schema_required_fields=["instruction"],
        min_tokens=3,          # "Did he win?" is a legitimate underspecified item
        max_tokens=256,
        # --- dedup: near-duplicates, not just exact ---
        dedup=dedup,
        minhash_threshold=0.85,
        minhash_ngram=3,
        # --- diversity: the point of this stage ---
        diversity_threshold=diversity,
        # --- splits, reproducible ---
        output_split={"train": 0.6, "validation": 0.2, "test": 0.2},
        output_split_seed=42,
        export_formats=["alpaca"],
        write_hf_readme=True,
        clean=True,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", default="data/items.jsonl")
    ap.add_argument("--out", default="output/curated")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--diversity", type=float, default=None,
                    help="DiversityGate threshold; omit to skip (needs embeddings)")
    ap.add_argument("--dedup", default="minhash", choices=["exact", "minhash", "none"])
    args = ap.parse_args()

    if not Path(args.items).exists():
        print(f"missing {args.items} — run build_dataset.py first")
        return 1

    # rejected.jsonl is APPEND-ONLY across runs, and manifest.json's rejected_count
    # accumulates with it — a second run reports thousands of rejections from a
    # 212-row input. Clear the directory so the numbers mean what they say.
    outdir = Path(args.out)
    if outdir.exists() and not args.dry_run:
        import shutil

        shutil.rmtree(outdir)
        print(f"cleared stale {outdir}")

    n_before = sum(1 for line in open(args.items, encoding="utf-8") if line.strip())
    cfg = build_config(args.items, args.out, diversity=args.diversity, dedup=args.dedup)
    curator = Curator(cfg)

    if args.dry_run:
        plan = curator.dry_run()
        print(json.dumps(plan, indent=2, default=str)[:3000])
        return 0

    result = curator.run()

    # CuratorKIT is used here as a *verdict source*, not as the data transformer. Its
    # exporters emit training formats (alpaca / sharegpt / messages / dpo), all of
    # which flatten to instruction/input/output and drop `metadata` — and our
    # epistemic label lives in metadata. So we take which questions survived and
    # re-apply that to our own rows, keeping our schema.
    survivors = {
        (getattr(s, "instruction", "") or "").strip().lower() for s in (result.passed or [])
    }
    orig = [json.loads(l) for l in open(args.items, encoding="utf-8") if l.strip()]
    kept = [o for o in orig if o["question"].strip().lower() in survivors]
    dropped = [o for o in orig if o["question"].strip().lower() not in survivors]

    print(f"\nin  {n_before}")
    print(f"out {len(kept)}  (removed {len(dropped)})")
    for o in dropped:
        print(f"  dropped {o['id']:<8} {o['label']:<20} {o['question'][:46]!r}")

    # A dedup pass that removes one half of a minimal pair leaves an orphan, which
    # breaks the pair invariant and quietly weakens the central comparison. Drop the
    # partner too and say so, rather than shipping a half-pair.
    kept_pairs: dict[str, int] = {}
    for o in kept:
        if o.get("pair_id"):
            kept_pairs[o["pair_id"]] = kept_pairs.get(o["pair_id"], 0) + 1
    orphan_pairs = {p for p, c in kept_pairs.items() if c != 2}
    if orphan_pairs:
        before = len(kept)
        kept = [o for o in kept if o.get("pair_id") not in orphan_pairs]
        print(f"\ndropped {before - len(kept)} orphaned pair member(s) from "
              f"{len(orphan_pairs)} broken pair(s): {sorted(orphan_pairs)}")

    out_items = Path(args.items).with_name("items_curated.jsonl")
    with open(out_items, "w", encoding="utf-8") as fh:
        for o in kept:
            fh.write(json.dumps(o, ensure_ascii=False) + "\n")

    counts: dict[str, int] = {}
    for o in kept:
        counts[o["label"]] = counts.get(o["label"], 0) + 1
    print()
    for k in sorted(counts):
        print(f"  {k:<20} {counts[k]:>4}")
    pairs = len({o["pair_id"] for o in kept if o.get("pair_id")})
    print(f"  {'minimal pairs':<20} {pairs:>4}")

    print(f"\nwrote {out_items}")
    print(f"provenance artifacts under {args.out} (manifest.json, checksums.txt, dataset_card.md)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
