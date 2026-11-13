#!/usr/bin/env bash
# Launch an extraction run on HF Jobs. Wraps the env setup that Windows needs.
#
#   ./run_job.sh smoke             # 8 stratified items, ~3 cents — always first
#   ./run_job.sh default           # full run, model's own system preamble
#   ./run_job.sh minimal           # full run, minimal preamble (the ablation)
#
# Then:  ./run_job.sh logs <job_id>   |   ./run_job.sh ps
set -euo pipefail

# Prefer `hf` on PATH; fall back to the Windows launcher's Scripts dir.
HF="$(command -v hf || true)"
if [ -z "$HF" ]; then
  HF="$(py -3.12 -c 'import sysconfig,os;print(os.path.join(sysconfig.get_path("scripts"),"hf.exe"))' 2>/dev/null || true)"
fi
if [ -z "$HF" ] || [ ! -x "$HF" ]; then
  echo "hf CLI not found. Install with: pip install 'huggingface_hub[cli]'" >&2
  exit 1
fi
PY="$(command -v python3 || command -v python || echo 'py -3.12')"

# The hf CLI streams job logs to the console. On Windows that console is cp1252,
# and the first non-ASCII byte in a log line kills the CLI with
#   Error: Invalid value. 'charmap' codec can't encode characters...
# The JOB keeps running remotely — only the local stream dies — but it looks like a
# crash and tempts you into relaunching and paying twice. Force UTF-8.
export PYTHONIOENCODING=utf-8
export PYTHONUTF8=1
export HF_HUB_DISABLE_EXPERIMENTAL_WARNING=1

# hf jobs needs the token as a secret; read it from the stored CLI login.
HF_TOKEN="$($PY -c 'from huggingface_hub import get_token; print(get_token() or "")' 2>/dev/null | tr -d '\r')"
export HF_TOKEN
if [ -z "$HF_TOKEN" ]; then
  echo "No HF token. Run: hf auth login" >&2
  exit 1
fi
# The token must carry job.write, or every launch 403s. repo.write alone is not enough.

FLAVOR="${FLAVOR:-t4-small}"
mode="${1:-smoke}"

case "$mode" in
  smoke)
    exec "$HF" jobs uv run --flavor "$FLAVOR" --secrets HF_TOKEN \
      --name abstention-smoke jobs/extract_job.py -- --smoke
    ;;
  default|minimal)
    exec "$HF" jobs uv run --flavor "$FLAVOR" --secrets HF_TOKEN \
      --name "abstention-$mode" jobs/extract_job.py -- --preamble "$mode" --tag "$mode"
    ;;
  ps)    exec "$HF" jobs ps -a ;;
  logs)  exec "$HF" jobs logs "${2:?usage: ./run_job.sh logs <job_id>}" ;;
  wait)  exec "$HF" jobs wait "${2:?usage: ./run_job.sh wait <job_id>}" ;;
  *)     echo "usage: $0 {smoke|default|minimal|ps|logs <id>|wait <id>}" >&2; exit 2 ;;
esac
