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
"""Stage 1 on HF Jobs: generate answers, record scalar baselines, dump activations.

Launch (always smoke first — 8 items, a few cents, catches template/shape bugs):

    hf jobs uv run --flavor t4-small --secrets HF_TOKEN \
        jobs/extract_job.py -- --smoke

    hf jobs uv run --flavor t4-small --secrets HF_TOKEN \
        jobs/extract_job.py -- --preamble default
    hf jobs uv run --flavor t4-small --secrets HF_TOKEN \
        jobs/extract_job.py -- --preamble minimal

The two preamble runs are the ablation: Tiny Aya's chat template injects a large
"You are in contextual safety mode" system block, which plausibly shifts how readily
the model hedges — the behaviour under study. Running both says whether it does.

Writes back to the dataset repo under runs/<tag>/:
    generations.jsonl   answer, logprob baselines, verbalised confidence
    hidden.npy          float16 [n_items, n_layers+1, hidden] last-prompt-token states
    meta.json           layer count, timing, device, settings
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import HfApi, hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "CohereLabs/tiny-aya-global"
DATASET = "demiladepy/fuzzy-abstention-items"

VERBALISE = (
    "How likely is your previous answer to be correct? "
    "Reply with only a number from 0 to 100."
)
MINIMAL_SYSTEM = "You are a helpful assistant."


def build_prompt(tok, question: str, preamble: str, device):
    """Render one user turn, with either the model's default system block or a minimal one."""
    msgs = [{"role": "user", "content": question}]
    if preamble == "minimal":
        msgs = [{"role": "system", "content": MINIMAL_SYSTEM}] + msgs
    return tok.apply_chat_template(
        msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to(device)


def answer_logprobs(model, prompt_ids, answer_ids) -> tuple[float, float]:
    """Mean and min token logprob of the answer, teacher-forced in one pass.

    Re-scoring rather than reading generate()'s scores keeps this identical under
    greedy or sampled decoding and avoids depending on generate() internals.
    """
    full = torch.cat([prompt_ids, answer_ids], dim=1)
    with torch.no_grad():
        logits = model(full).logits
    start = prompt_ids.shape[1] - 1
    span = logits[0, start : full.shape[1] - 1].float().log_softmax(-1)
    chosen = span.gather(-1, answer_ids[0].unsqueeze(-1)).squeeze(-1)
    if chosen.numel() == 0:
        return float("nan"), float("nan")
    return float(chosen.mean()), float(chosen.min())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="8 items, for a cheap first run")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--preamble", choices=["default", "minimal"], default="default")
    # 32 truncated real answers mid-token in the smoke run ("...was 112," and
    # "Tradit"). A truncated answer cannot be graded for correctness, and a hedge
    # cut off halfway reads as a non-hedge — both would corrupt the labels the whole
    # experiment rests on. 64 costs a few extra cents.
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--dataset", default=DATASET)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        print("HF_TOKEN missing — pass --secrets HF_TOKEN", flush=True)
        return 1

    tag = args.tag or ("smoke" if args.smoke else args.preamble)
    limit = 8 if args.smoke else args.limit

    path = hf_hub_download(args.dataset, "items.jsonl", repo_type="dataset", token=token)
    items = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    if limit:
        # Stratify the smoke sample so a template bug in any one label shows up now
        # rather than in the paid run.
        by_label: dict[str, list] = {}
        for it in items:
            by_label.setdefault(it["label"], []).append(it)
        picked, i = [], 0
        while len(picked) < limit and any(by_label.values()):
            for lab in sorted(by_label):
                if by_label[lab] and len(picked) < limit:
                    picked.append(by_label[lab].pop(0))
            i += 1
            if i > limit:
                break
        items = picked
    print(f"{len(items)} items, preamble={args.preamble}, tag={tag}", flush=True)

    cuda = torch.cuda.is_available()
    dtype = torch.bfloat16 if cuda else torch.float32
    print(f"cuda={cuda} dtype={dtype}", flush=True)
    if cuda:
        print(f"gpu={torch.cuda.get_device_name(0)}", flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL, token=token)
    model = AutoModelForCausalLM.from_pretrained(
        # `dtype`, not `torch_dtype` — the latter is deprecated in transformers 5 and
        # logs a warning on every run.
        MODEL, token=token, dtype=dtype, device_map="auto" if cuda else None
    )
    model.eval()
    device = next(model.parameters()).device

    n_layers = model.config.num_hidden_layers
    hidden_size = model.config.hidden_size
    print(f"layers={n_layers} hidden={hidden_size}", flush=True)

    hidden = np.zeros((len(items), n_layers + 1, hidden_size), dtype=np.float16)
    rows: list[dict] = []
    t0 = time.time()

    for i, item in enumerate(items):
        q = item["question"]
        prompt = build_prompt(tok, q, args.preamble, device)
        prompt_ids = prompt["input_ids"]

        # Activations are taken from the prompt alone, BEFORE any answer exists, so
        # the probe is predicting rather than reacting to text it already produced.
        with torch.no_grad():
            pre = model(**prompt, output_hidden_states=True)
        hidden[i] = np.stack(
            [h[0, -1].float().cpu().numpy().astype(np.float16) for h in pre.hidden_states]
        )

        with torch.no_grad():
            gen = model.generate(
                **prompt, max_new_tokens=args.max_new_tokens, do_sample=False,
                pad_token_id=tok.eos_token_id,
            )
        answer_ids = gen[:, prompt_ids.shape[1] :]
        answer = tok.decode(answer_ids[0], skip_special_tokens=True).strip()
        mean_lp, min_lp = answer_logprobs(model, prompt_ids, answer_ids)

        v_msgs = [
            {"role": "user", "content": q},
            {"role": "assistant", "content": answer or "(no answer)"},
            {"role": "user", "content": VERBALISE},
        ]
        if args.preamble == "minimal":
            v_msgs = [{"role": "system", "content": MINIMAL_SYSTEM}] + v_msgs
        v_prompt = tok.apply_chat_template(
            v_msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True
        ).to(device)
        with torch.no_grad():
            v_out = model.generate(
                **v_prompt, max_new_tokens=8, do_sample=False, pad_token_id=tok.eos_token_id
            )
        v_text = tok.decode(
            v_out[0, v_prompt["input_ids"].shape[1] :], skip_special_tokens=True
        )
        digits = "".join(c for c in v_text if c.isdigit())[:3]
        verbalised = min(int(digits), 100) / 100.0 if digits else float("nan")

        rows.append({
            "id": item["id"], "label": item["label"], "lang": item.get("lang", "en"),
            "pair_id": item.get("pair_id"), "question": q, "gold": item.get("gold"),
            "answer": answer, "logprob_mean": mean_lp, "logprob_min": min_lp,
            "verbalised": verbalised, "verbalised_raw": v_text.strip()[:40],
            "n_answer_tokens": int(answer_ids.shape[1]),
            "n_prompt_tokens": int(prompt_ids.shape[1]),
        })

        if (i + 1) % 10 == 0 or i + 1 == len(items):
            rate = (time.time() - t0) / (i + 1)
            eta = rate * (len(items) - i - 1)
            print(f"  {i+1}/{len(items)}  {rate:.2f}s/item  eta {eta/60:.1f}min", flush=True)

    out = Path("out")
    out.mkdir(exist_ok=True)
    with open(out / "generations.jsonl", "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    np.save(out / "hidden.npy", hidden)
    meta = {
        "model": MODEL, "n_items": len(items), "n_layers": n_layers,
        "hidden_size": hidden_size, "dtype": str(dtype), "cuda": cuda,
        "preamble": args.preamble, "tag": tag, "smoke": args.smoke,
        "max_new_tokens": args.max_new_tokens,
        "mean_prompt_tokens": float(np.mean([r["n_prompt_tokens"] for r in rows])),
        "seconds": round(time.time() - t0, 1),
    }
    with open(out / "meta.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)

    api = HfApi(token=token)
    api.upload_folder(
        folder_path=str(out), path_in_repo=f"runs/{tag}",
        repo_id=args.dataset, repo_type="dataset",
    )
    print(f"\nuploaded to {args.dataset}/runs/{tag}", flush=True)
    print(json.dumps(meta, indent=2), flush=True)

    # A quick sanity read so the logs alone tell you whether it worked.
    print("\nfirst 3 answers:", flush=True)
    for r in rows[:3]:
        print(f"  [{r['label']}] {r['question'][:46]!r}", flush=True)
        print(f"     -> {r['answer'][:70]!r}  lp={r['logprob_mean']:.3f} "
              f"verb={r['verbalised']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
