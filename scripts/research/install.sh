#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export PIP_CACHE_DIR=/workspace/.cache/pip
export HF_HOME=/workspace/.cache/huggingface
export HF_HUB_DOWNLOAD_TIMEOUT=60
export HF_HUB_ETAG_TIMEOUT=30
python -c 'import sys; assert sys.version_info[:2] == (3,12), "Verified lock requires Python 3.12"'
python -m venv .venv
.venv/bin/python -m pip install --timeout 60 --retries 2 'torch==2.8.0+cpu' --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install --timeout 60 --retries 2 -r config/research/requirements-cpu.lock
.venv/bin/python -m pip install --no-deps --no-build-isolation -e .
.venv/bin/python -m pip check
.venv/bin/python -c 'import torch; from timesfm3 import TimesFM3Evaluator; assert torch.version.cuda is None; print("CPU research imports passed")'
if [[ "${DOWNLOAD_TIMESFM3:-0}" == 1 ]]; then
 .venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download
from tw_scenario.research.model import MODEL, REVISION
print(snapshot_download(MODEL, revision=REVISION, allow_patterns=['config.json','model.safetensors','LICENSE','README.md'],max_workers=1))
PY
fi
