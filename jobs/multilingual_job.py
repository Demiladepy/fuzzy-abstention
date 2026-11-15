# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch",
#   "transformers>=5.0",
#   "huggingface_hub>=1.0",
#   "numpy",
#   "accelerate",
# ]
# ///
"""Cross-lingual slice: does the English-trained probe transfer?

    hf jobs uv run --flavor t4-small --secrets HF_TOKEN jobs/multilingual_job.py -- --smoke
    hf jobs uv run --flavor t4-small --secrets HF_TOKEN jobs/multilingual_job.py

Tiny Aya covers 70+ languages and we measured it in English only. That is the obvious
hole in a submission built on a multilingual model, and the interesting question sits
right in it: **the epistemic kind of a question is a property of the question, not of
the language it is asked in.** If the probe has learned the concept it should transfer;
if it has learned English surface form it should collapse.

Three low-resource languages from Tiny Aya's coverage: Yoruba, Hausa, Swahili.

Translation is done by the model itself. That is a real limitation — a model
translating its own evaluation set — and it is disclosed rather than hidden. We keep
every translation in the output so a speaker can audit them, and we skip any item whose
translation comes back empty or unchanged from the English.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
from huggingface_hub import HfApi, hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "CohereLabs/tiny-aya-global"
DATASET = "demiladepy/fuzzy-abstention-items"
LANGS = {"yo": "Yoruba", "ha": "Hausa", "sw": "Swahili"}
LABELS = ["answerable_known", "answerable_unknown", "unanswerable",
          "context_dependent", "underspecified"]
VERBALISE = ("How likely is your previous answer to be correct? "
             "Reply with only a number from 0 to 100.")


def stratified(items: list[dict], per_label: int) -> list[dict]:
    out, seen = [], {}
    for it in items:
        c = seen.get(it["label"], 0)
        if c < per_label:
            out.append(it)
            seen[it["label"]] = c + 1
    return out


def gen(model, tok, msgs, device, max_new: int):
    enc = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                  return_tensors="pt", return_dict=True).to(device)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--per-label", type=int, default=21)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    # One language per job. The three-language run died at 29m55s to a 30-minute job
    # timeout, and because the upload happens at the end, all 220 completed items were
    # lost. Each language is ~13 minutes, so splitting keeps every run clear of the
    # limit and means one failure no longer destroys the others.
    ap.add_argument("--langs", default=None,
                    help="comma-separated codes, e.g. 'yo'. Default: all three.")
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        print("HF_TOKEN missing")
        return 1

    path = hf_hub_download(DATASET, "items.jsonl", repo_type="dataset", token=token)
    allitems = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    per = 1 if args.smoke else args.per_label
    items = stratified(allitems, per)
    if args.smoke:
        langs = {"yo": "Yoruba"}
    elif args.langs:
        langs = {c: LANGS[c] for c in args.langs.split(",") if c in LANGS}
    else:
        langs = LANGS
    print(f"{len(items)} items x {len(langs)} languages = {len(items)*len(langs)} runs",
          flush=True)

    cuda = torch.cuda.is_available()
    dtype = torch.bfloat16 if cuda else torch.float32
    tok = AutoTokenizer.from_pretrained(MODEL, token=token)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, token=token, dtype=dtype, device_map="auto" if cuda else None)
    model.eval()
    device = next(model.parameters()).device
    n_layers, hidden_size = model.config.num_hidden_layers, model.config.hidden_size
    print(f"cuda={cuda} layers={n_layers} hidden={hidden_size}", flush=True)

    rows: list[dict] = []
    hid: list[np.ndarray] = []
    t0 = time.time()

    for code, lang in langs.items():
        skipped = 0
        for i, item in enumerate(items):
            q_en = item["question"]
            translated = gen(model, tok, [{"role": "user", "content":
                f"Translate the following question into {lang}. "
                f"Reply with only the translation, nothing else.\n\n{q_en}"}],
                device, 96)
            # Guard against the two failure modes that would silently corrupt the slice.
            if not translated or translated.strip().lower() == q_en.strip().lower():
                skipped += 1
                continue

            enc = tok.apply_chat_template([{"role": "user", "content": translated}],
                                          add_generation_prompt=True,
                                          return_tensors="pt", return_dict=True).to(device)
            with torch.no_grad():
                pre = model(**enc, output_hidden_states=True)
            hid.append(np.stack([h[0, -1].float().cpu().numpy().astype(np.float16)
                                 for h in pre.hidden_states]))

            answer = gen(model, tok, [{"role": "user", "content": translated}],
                         device, args.max_new_tokens)
            vtext = gen(model, tok, [
                {"role": "user", "content": translated},
                {"role": "assistant", "content": answer or "(no answer)"},
                {"role": "user", "content": VERBALISE}], device, 8)
            digits = "".join(c for c in vtext if c.isdigit())[:3]

            rows.append({
                "id": f"{item['id']}_{code}", "src_id": item["id"],
                "label": item["label"], "lang": code,
                "question_en": q_en, "question": translated,
                "answer": answer, "gold": item.get("gold"),
                "verbalised": min(int(digits), 100) / 100.0 if digits else float("nan"),
            })
            if (len(rows)) % 20 == 0:
                r = (time.time() - t0) / len(rows)
                print(f"  {len(rows)} done  {r:.2f}s/item", flush=True)
        print(f"  {lang}: skipped {skipped} untranslated", flush=True)

    hidden = np.stack(hid) if hid else np.zeros((0, n_layers + 1, hidden_size), np.float16)
    os.makedirs("out", exist_ok=True)
    with open("out/generations.jsonl", "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    np.save("out/hidden.npy", hidden)
    meta = {"model": MODEL, "languages": list(langs), "n": len(rows),
            "per_label": per, "n_layers": n_layers, "hidden_size": hidden_size,
            "seconds": round(time.time() - t0, 1), "smoke": args.smoke,
            "translation": "model-generated by the model under test; see limitations"}
    with open("out/meta.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)

    tag = "multilingual_smoke" if args.smoke else "multilingual_" + "_".join(langs)
    HfApi(token=token).upload_folder(folder_path="out", path_in_repo=f"runs/{tag}",
                                     repo_id=DATASET, repo_type="dataset")
    print(f"\nuploaded runs/{tag}  hidden={hidden.shape}", flush=True)
    print(json.dumps(meta, indent=2), flush=True)
    for r in rows[:3]:
        print(f"  [{r['lang']}/{r['label']}] {r['question'][:60]!r}", flush=True)
        print(f"     -> {r['answer'][:70]!r}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
