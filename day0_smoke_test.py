"""Day-0 smoke test for the Lexsi Track 2 submission.

Verifies the things that cannot be fixed late: gated model access, the chat-template
preamble size, and whether SafeTune / AuditKIT import. Every check is independent and
reports PASS / FAIL / SKIP, so one failure does not hide the rest.

Usage:
    export HF_TOKEN=hf_...        # or: set HF_TOKEN=... on Windows
    python day0_smoke_test.py
    python day0_smoke_test.py --model CohereLabs/tiny-aya-global --no-generate

On Colab, run `pip uninstall -y torchao` first (SafeTune's documented workaround).
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback

DEFAULT_MODEL = "CohereLabs/tiny-aya-global"
results: list[tuple[str, str, str]] = []


def record(name: str, status: str, detail: str = "") -> None:
    results.append((name, status, detail))
    mark = {"PASS": "[PASS]", "FAIL": "[FAIL]", "SKIP": "[SKIP]", "WARN": "[WARN]"}[status]
    print(f"{mark} {name}" + (f" — {detail}" if detail else ""), flush=True)


def check_python() -> None:
    v = sys.version_info
    detail = f"{v.major}.{v.minor}.{v.micro}"
    # SafeTune needs >=3.12, AuditKIT >=3.10. Warn rather than fail: the dataset and
    # eval stages run fine on 3.10/3.11, only SafeTune is strict.
    if (v.major, v.minor) >= (3, 12):
        record("Python version", "PASS", detail)
    elif (v.major, v.minor) >= (3, 10):
        record("Python version", "WARN", f"{detail} — AuditKIT ok, SafeTune wants >=3.12")
    else:
        record("Python version", "FAIL", f"{detail} — too old for both")


def check_torch() -> "object | None":
    try:
        import torch
    except ImportError:
        record("torch", "FAIL", "not installed — pip install torch")
        return None
    except Exception as exc:  # noqa: BLE001 - a broken install raises OSError, not ImportError
        record("torch", "FAIL", f"installed but will not load: {str(exc).split(chr(10))[0][:140]}")
        return None
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        record("torch + GPU", "PASS", f"{torch.__version__}, {name}, {total:.1f} GiB")
    else:
        record("torch + GPU", "WARN", f"{torch.__version__}, CPU only — generation will crawl")
    return torch


def check_token() -> str | None:
    """Find a usable token from the environment *or* the HF CLI's stored login.

    `hf auth login` writes a token to the HF cache, which huggingface_hub picks up
    automatically. Checking only env vars reports a false FAIL on a machine that is
    in fact authenticated — and worse, it reports it next to a passing gated-access
    check, which makes the suite look self-contradictory.
    """
    for var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACEHUB_API_TOKEN"):
        tok = os.environ.get(var)
        if tok:
            record("HF token", "PASS", f"found in ${var} (…{tok[-4:]})")
            return tok
    try:
        from huggingface_hub import get_token

        tok = get_token()
    except Exception:  # noqa: BLE001
        tok = None
    if tok:
        record("HF token", "PASS", f"stored HF CLI login (…{tok[-4:]})")
        return tok
    record(
        "HF token",
        "FAIL",
        "no token — run `hf auth login`, or set HF_TOKEN; Tiny Aya is gated and will 401",
    )
    return None


def check_gated_access(model_id: str, token: str | None) -> bool:
    """The critical check: licence accepted AND token valid for this repo.

    Deliberately downloads a real file rather than calling model_info(). A gated
    Cohere repo is *publicly listable* — model_info() succeeds with no token at all —
    but the files 401 until the licence is accepted. Checking metadata would report a
    false PASS on the one thing that cannot be fixed on day 4.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        record("gated access", "SKIP", "huggingface_hub not installed")
        return False
    try:
        hf_hub_download(model_id, filename="config.json", token=token)
    except Exception as exc:  # noqa: BLE001 - want the message, whatever it is
        msg = str(exc).split("\n")[0][:140]
        record(
            "gated access",
            "FAIL",
            f"cannot fetch config.json: {msg}  ->  accept the licence at "
            f"https://huggingface.co/{model_id} with the same account as your token",
        )
        return False
    record("gated access", "PASS", f"{model_id} files downloadable (config.json fetched)")
    return True


def check_preamble(model_id: str, token: str | None) -> None:
    """Measure Tiny Aya's chat-template system preamble.

    SafeTune's README warns this is ~366 tokens and that a smaller explicit max_len
    makes the data loaders train on zero supervised tokens. Measure it rather than
    trusting the number, and record it for the write-up.
    """
    try:
        from transformers import AutoTokenizer
    except ImportError:
        record("chat-template preamble", "SKIP", "transformers not installed")
        return
    except Exception as exc:  # noqa: BLE001 - transformers imports torch, which may raise OSError
        record("chat-template preamble", "FAIL", f"transformers will not import: "
               f"{str(exc).split(chr(10))[0][:120]}")
        return
    try:
        tok = AutoTokenizer.from_pretrained(model_id, token=token)
        # Render to a string, then tokenize that. Do NOT call apply_chat_template with
        # tokenize=True and len() the result: in transformers v5 it returns a dict, so
        # len() gives the number of KEYS (2), not the token count. That reads as a
        # reassuringly tiny preamble and is completely wrong.
        rendered = tok.apply_chat_template(
            [{"role": "user", "content": ""}], add_generation_prompt=True, tokenize=False
        )
        n = len(tok(rendered, add_special_tokens=False)["input_ids"])
        has_tmpl = bool(getattr(tok, "chat_template", None))
        # An implausibly SMALL count is the dangerous case, not a large one: it means
        # the template contributed almost nothing, so the instruction-tuned model is
        # being fed something it was never trained on. Don't pass that silently.
        if not has_tmpl:
            record("chat-template preamble", "FAIL",
                   "tokenizer defines no chat_template — prompts would be unformatted")
        elif n < 8:
            record("chat-template preamble", "WARN",
                   f"only {n} tokens for an empty user turn; template may not be applying — "
                   "inspect apply_chat_template(..., tokenize=False) before trusting results")
        elif n >= 512:
            record("chat-template preamble", "WARN",
                   f"{n} tokens of preamble — keep max_len >= {n * 2}")
        else:
            record("chat-template preamble", "PASS",
                   f"{n} tokens for an empty user turn — keep max_len >= 512")
    except Exception as exc:  # noqa: BLE001
        record("chat-template preamble", "FAIL", str(exc).split("\n")[0][:160])


def check_generate(model_id: str, token: str | None, torch_mod) -> None:
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        record("load + generate", "SKIP", "transformers not installed")
        return
    except Exception as exc:  # noqa: BLE001 - transformers imports torch, which may raise OSError
        record("load + generate", "FAIL", f"transformers will not import: "
               f"{str(exc).split(chr(10))[0][:120]}")
        return
    try:
        tok = AutoTokenizer.from_pretrained(model_id, token=token)
        dtype = "bfloat16" if (torch_mod and torch_mod.cuda.is_available()) else "float32"
        model = AutoModelForCausalLM.from_pretrained(
            model_id, token=token, torch_dtype=dtype, device_map="auto"
        )
        inputs = tok.apply_chat_template(
            [{"role": "user", "content": "Name the capital of Kenya in one word."}],
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(model.device)
        out = model.generate(**inputs, max_new_tokens=12, do_sample=False)
        text = tok.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        record("load + generate", "PASS", f"{dtype}, replied {text.strip()[:60]!r}")
    except Exception as exc:  # noqa: BLE001
        record("load + generate", "FAIL", str(exc).split("\n")[0][:160])
        traceback.print_exc(limit=2)


def check_libs() -> None:
    for mod, hint in (
        ("auditkit", "pip install 'auditkit[transformers]'"),
        ("safetune", "pip install safetune  (needs Python >= 3.12)"),
    ):
        try:
            m = __import__(mod)
            record(mod, "PASS", getattr(m, "__version__", "version unknown"))
        except ImportError:
            record(mod, "FAIL", hint)
        except Exception as exc:  # noqa: BLE001 - import-time errors, e.g. torchao on Colab
            record(mod, "FAIL", f"imported but raised: {str(exc).split(chr(10))[0][:120]}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--no-generate", action="store_true", help="skip the weight download")
    args = ap.parse_args()

    print(f"Day-0 smoke test — target model: {args.model}\n")

    check_python()
    torch_mod = check_torch()
    token = check_token()
    reachable = check_gated_access(args.model, token)

    if reachable:
        check_preamble(args.model, token)
        if args.no_generate:
            record("load + generate", "SKIP", "--no-generate")
        else:
            check_generate(args.model, token, torch_mod)
    else:
        record("chat-template preamble", "SKIP", "model not reachable")
        record("load + generate", "SKIP", "model not reachable")

    check_libs()

    failures = [r for r in results if r[1] == "FAIL"]
    print("\n" + "-" * 60)
    print(f"{len(results) - len(failures)}/{len(results)} checks passed or warned.")
    if failures:
        print("\nBlocking:")
        for name, _, detail in failures:
            print(f"  - {name}: {detail}")
        print("\nFix the gated-access and token failures first; nothing else matters until then.")
    else:
        print("Day 0 is clear. Create the HF repos and start the dataset.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
