# ChatGPT cloud execution — NO_GO at sample materialization

Execution date: 2026-10-17 Asia/Taipei. Sole model-code baseline:
`b4912f465436cb7811c797caaf24ed08d645d52a`, tree
`429309ce618c2092244b219defd762d0bc1ff1a4`.
No source, tests, model configuration, lockfile, production switch, Sheet or task
was changed in the delivered evidence commit. This is a blocked execution report,
not a completed training experiment or a failed accuracy gate.

## Executed

- Read AGENTS, README, PROJECT_STATUS, CURRENT_GOAL, CODEX_HANDOFF and draft PR #2;
  fetched the branch and fast-forwarded from 33143e8 to b4912f4.
- Installed the frozen dev/research/timesfm3 environment, plus `socksio==1.0.0`
  required by the cloud SOCKS proxy (environment-only addition).
- Verified pinned TimesFM source provenance and downloaded the pinned weights.
  CPU load passed in 5.78 seconds, peak RSS 2,844,912 KiB. No GPU is available.
- Collected FinMind data through the existing CLI, reconciled against official
  TWSE/TAIFEX records and passed the existing retrospective DQ gate. New batch
  identity differs from the local batch because fetch timestamps/provenance differ.
- Ran JOINT candidates 0, 1, 2 and 4, and TAIEX/TX independent candidate 0.
  Every attempt exited 2 before optimizer construction or parameter updates.
- Audited all three folds' requested training, early-stop and validation sample
  dates at both frozen context lengths for each market. The audit contains only
  dates/status/errors, not OHLC, features or licensed raw rows.
- Exact baseline checks: 76 pytest passed; Ruff passed; git diff --check passed.

## Blocker and decision

`run_output_head_walk_forward` begins its training dates at `train_days[context:]`.
However, `research_history` requires a full contiguous context of the expanded
features, which themselves require 60 prior sessions and a published night bar.
The certified no-night dates remain real exceptions; they are not synthetic bars.
The first requested sample encounters a context gap: `13 < 256`, or `269 < 512`.
`build_head_samples` / `build_joint_head_samples` raises on the first such day;
there is no auditable per-day training exclusion mechanism in the pinned tree.

All indexed candidates share this deterministic sample materialization blocker.
Indices 3 and 5–7 were not redundantly model-loaded. They are **NOT_RUN**, not
accuracy FAILs. No selection, strongest-baseline comparison, validation metric,
promotion gate or bootstrap claim exists. No model checkpoint exists because no
training step ran. The stopping reason is a shared implementation blocker, not
three completed no-gain validation rounds or exhausted compute budget.

| JOINT index | Context | Learning rate | Weight decay | Execution | Close improvement / gate |
|---|---:|---:|---:|---|---|
| 0 | 256 | 1e-4 | 0 | preparation failed: 13 < 256 | not evaluated |
| 1 | 256 | 1e-4 | 1e-4 | preparation failed: 13 < 256 | not evaluated |
| 2 | 256 | 3e-4 | 0 | preparation failed: 13 < 256 | not evaluated |
| 3 | 256 | 3e-4 | 1e-4 | not run: shared blocker | not evaluated |
| 4 | 512 | 1e-4 | 0 | preparation failed: 269 < 512 | not evaluated |
| 5 | 512 | 1e-4 | 1e-4 | not run: shared blocker | not evaluated |
| 6 | 512 | 3e-4 | 0 | not run: shared blocker | not evaluated |
| 7 | 512 | 3e-4 | 1e-4 | not run: shared blocker | not evaluated |

Independent TAIEX and TX candidate 0 each failed with `13 < 256`.

**NO_GO / BLOCKED_SAMPLE_MATERIALIZATION.** An updated authorized source revision
must explicitly preserve and report unavailable training days while retaining
uncompressed contiguous context and every expected evaluation date. Then rerun
the same grid/folds, including JOINT and independent controls. This report makes
no decision to impute features, move split dates, shorten context, or change the
fixed 08:30 information set. No source fix is included under the user's exact
b4912f4 execution constraint.

The old 2025-10-09–2026-10-08 final stays permanently **FAIL**. All historical
inspection here is redesign development evidence only. New acceptance requires
a future holdout. `production_switch=false`; no schedules or Sheets were touched.

## Reproduce

In a fresh clone, check out b4912f4. Use the repository's original lockfile:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv sync --frozen --extra dev --extra research --extra timesfm3
UV_CACHE_DIR=/tmp/uv-cache uv pip install --python .venv/bin/python socksio==1.0.0
.venv/bin/pytest -q
.venv/bin/ruff check .
git diff --check
.venv/bin/tw-scenario quant timesfm-preflight --config config/quant/timesfm_output_head.json
```

Download only the pinned model/config/license into the ignored model cache:

```python
from huggingface_hub import snapshot_download
snapshot_download(
    'google/timesfm-3.0-pytorch',
    revision='43046b85ec22d584a13f8098c2ed39c889e129c2',
    cache_dir='data/raw/timesfm3',
    allow_patterns=['config.json', 'model.safetensors', 'LICENSE'],
)
```

Rebuild data using the existing entrypoints (FinMind attribution; academic,
non-commercial use; raw data must not be redistributed). Read the dataset path
from each command's output; a new download has a new provenance identity:

```bash
.venv/bin/tw-scenario quant collect --start 2018-01-01 --end 2026-10-08 --source finmind --data-root data
.venv/bin/tw-scenario quant reconcile data/processed/UNCERTIFIED_ID --output reports/generated/reconciliation.json
.venv/bin/tw-scenario quant quality data/processed/CERTIFIED_ID --output reports/generated/quality.json
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 .venv/bin/tw-scenario quant timesfm-head-walk-forward data/processed/CERTIFIED_ID --instrument JOINT --candidate-index 0
```

The exact executed commands, exit codes, elapsed times and RSS are in
`attempts.json`. `sample-audit.json` records every requested precheck date and
each fold boundary. It is **sample feasibility**, not daily model predictions.
Reproduce that audit by calling the existing `expanding_folds` (3, 80, 1000),
then for each market/context/fold call `build_head_samples` separately for each
date in `train[context:][:-40:5]`, `train[context:][-40:]`, and validation.
Record ValueError as FAILED and successful materialization as OK without joining
across gaps. JOINT requires both market histories to materialize and align.

`provenance.json` records package/data/model/file hashes and cgroup limits.
The repo runtime probe reads host MemAvailable; the container's actual 8 GiB
cgroup memory ceiling is recorded separately. Cgroup peak includes downloads,
page cache and concurrent precheck work, so it is not training peak memory.
No full-training resource estimate can be inferred from this failed preparation.
