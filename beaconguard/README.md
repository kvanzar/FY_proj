# BeaconGuard — ML-Based Detection of Encrypted C2 Beaconing Traffic

Phase 1 implementation per `prd.md`. See `exec.md` for a plain-language
architecture walkthrough (what the system does and how data flows through
it) and `run.md` for full step-by-step setup/run/test instructions —
this file is a short version of `run.md`.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data acquisition

`config/default.yaml` is already pointed at `engraqeel/iot23preprocesseddata`
(verified to retain `ts`/`uid`/`id.orig_h`/`id.resp_h` — see `run.md` §4.2
for the exact schema, label distribution, and an important caveat about
how few positive sessions this specific variant contains):

```bash
pip install kagglehub
python3 -c "
import kagglehub
path = kagglehub.dataset_download('engraqeel/iot23preprocesseddata')
print('Path to dataset files:', path)
"
```

If it downloads to a different path/version than the one already in
`config/default.yaml` → `data.raw_paths`, update that path to match. See
`run.md` §4 for the full walkthrough, alternative datasets, and manual
download instructions.

`ingest.py` (FR-1.2) validates the required columns at load time and
aborts with a clear error if any are missing, rather than degrading
silently.

## Running

```bash
python run_pipeline.py --config config/default.yaml
```

This single command (FR-8.9) runs the full pipeline — ingestion through
evaluation — and writes:

- `outputs/models/` — trained, calibrated models + metadata (feature list,
  CV score, training/inference timing)
- `outputs/logs/decision_log.csv` — every simulated ALLOW/ALERT/BLOCK
  decision with branch scores and a top-3 SHAP explanation
- `outputs/logs/evaluation_report.json` — the full metric suite, ablation,
  jitter degradation curve, and fusion-method comparison

## Tests

```bash
pytest
```

Tests run against synthetic flow data (`tests/conftest.py`) and do not
require the real dataset — they check feature-function correctness
(FR-2 tier features, FR-2.1 sessionization, FR-5 decision thresholds).

## Project layout

See `exec.md` for the full module-by-module explanation. Directory
structure follows `prd.md` §7.
