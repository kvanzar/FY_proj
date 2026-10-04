# Run & Test Guide

Step-by-step commands to set up, test, and run the system. For what the
code actually does, see `exec.md`. All commands below assume your
terminal is at the repo root (`FY_proj/`) unless noted.

---

## 1. Prerequisites

- Python 3.10 or newer (the code was developed and tested against 3.12).
- No GPU required — everything targets CPU (NFR-4: < 8 GB peak memory).

Check your Python version:

```bash
python3 --version
```

If it's older than 3.10, install a newer one (e.g. via `brew install python@3.12`
on macOS) and use that instead of `python3` below.

## 2. Set up the environment

From the repo root:

```bash
cd c2-beaconing-detection
python3 -m venv .venv
source .venv/bin/activate          # on Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

This installs everything the project needs: pandas/numpy/scikit-learn for
the data pipeline, XGBoost, PyTorch (for the anomaly-detection
autoencoder), SHAP (for explanations), imbalanced-learn (for SMOTE), and
pytest.

> If you're using a pre-existing environment (conda, etc.) instead of a
> fresh venv, just make sure `pip install -r requirements.txt` succeeds in
> it — the test suite and pipeline don't care which environment manager
> you used, only that the packages are importable.

Every command from here on assumes you're inside `c2-beaconing-detection/` with this
environment active.

## 3. Run the tests (no dataset required)

```bash
pytest
```

You should see all tests pass, e.g.:

```
................                                                         [100%]
16 passed in 0.05s
```

**What this actually checks:** the test suite builds small, synthetic,
fake network data on the fly (`tests/conftest.py`) — a hand-crafted
"perfectly regular beacon" session and a "normal, irregular" session — and
checks that each measurement function computes what its name claims (e.g.
that the beacon session really does get a low irregularity score, that a
session with fewer than 4 connections gets correctly discarded, that a
risk score correctly maps to ALLOW/ALERT/BLOCK). It does **not** need the
real IoT-23 dataset and does **not** train any real ML model — it's
checking the building blocks, not end-to-end accuracy.

To run just one file, or one test:

```bash
pytest tests/test_features_timing.py
pytest tests/test_decision.py -v
```

To see print/log output while debugging a test:

```bash
pytest -s tests/test_sessionize.py
```

## 4. Get the real dataset (needed to run the full pipeline)

The tests above don't need this — skip to §5 if you only want to verify
the code works. This step is only needed to actually train and evaluate
the system on real traffic.

### Option A — kagglehub (what this project is currently configured for)

```bash
pip install kagglehub
python3 -c "
import kagglehub
path = kagglehub.dataset_download('engraqeel/iot23preprocesseddata')
print('Path to dataset files:', path)
"
```

This downloads to a local cache — typically
`~/.cache/kagglehub/datasets/engraqeel/iot23preprocesseddata/versions/<N>/iot23_combined_new.csv`
— and prints the exact path. **`config/default.yaml` already points at
this path** (version 1), so if your download lands there too, you can
skip straight to §5. If `kagglehub` prints a different version number or
a different path, update `data.raw_paths` in `config/default.yaml` to
match — see §4.3 below for the exact key to edit.

No Kaggle API key was needed to download this particular dataset in
testing; if kagglehub prompts you to authenticate, follow its prompt (it
will point you to `kaggle.com/settings` for an API token).

### Option B — download manually from Kaggle

Any of these work if you'd rather not use kagglehub:
- `kaggle.com/datasets/engraqeel/iot23preprocesseddata` (what this project is configured for — see below)
- `kaggle.com/datasets/surajsooraj26/iot-23` (full dataset — see §4.2 for why you might want this)
- `kaggle.com/datasets/abdullahhashmi/iot23` (fallback mirror)

Put the file(s) somewhere under `c2-beaconing-detection/data/` (this folder is
git-ignored, so it's safe to drop large files there), then point
`data.raw_paths` at it as in §4.3.

### 4.1 Verifying the columns you need are present

**Before running the pipeline, check the file has the columns
`ts`, `uid`, `id.orig_h`, and `id.resp_h`.** Some preprocessed Kaggle
versions strip these out to save space — but this project's entire
timing-analysis idea depends on them (you can't measure "how regular
are these connections" without a timestamp and who's talking to whom).
If your download is missing them, use the full-dataset variant instead,
or download `conn.log.labeled` directly from
[Stratosphere Laboratory](https://www.stratosphereips.org/datasets-iot23).

```bash
head -1 path/to/your/file.csv
# look for ts, uid, id.orig_h, id.resp_h in the column names
```

`engraqeel/iot23preprocesseddata` (the one this project is configured
for) **does** retain all four — already verified, see §4.2.

### 4.2 What's actually in `engraqeel/iot23preprocesseddata` (verified)

This variant is a single 903 MB CSV, `iot23_combined_new.csv`, with
6,046,623 rows and this exact header:

```
,ts,uid,id.orig_h,id.orig_p,id.resp_h,id.resp_p,proto,service,duration,
orig_bytes,resp_bytes,conn_state,local_orig,local_resp,missed_bytes,
history,orig_pkts,orig_ip_bytes,resp_pkts,resp_ip_bytes,label
```

Three things differ from raw Zeek `conn.log.labeled` that the code now
handles automatically (`src/ingest.py::_read_single_file`):

- **A stray leading unnamed column** (the row index from however this CSV
  was exported) — dropped automatically on load.
- **One `label` column, not two.** Native Zeek files have a separate
  `label` (Malicious/Benign) and `detailed-label` (e.g.
  `C&C-HeartBeat`) column. This file collapses them into one `label`
  column that actually holds the fine-grained value. The loader detects
  this (no `detailed-label` column present) and renames `label` →
  `detailed-label` so the rest of the pipeline works unmodified.
- **An extra malicious-but-not-C2 category, `Okiru-Attack`** (3 rows),
  not in `prd.md`'s original exclusion list. Already added to
  `data.excluded_labels` in `config/default.yaml`.

The full-file label breakdown (measured directly, one pass over the
real file):

| Label | Count |
|---|---:|
| PartOfAHorizontalPortScan | 3,389,036 |
| Okiru | 1,313,012 |
| Benign | 688,812 |
| DDoS | 638,506 |
| **C&C** | **15,286** |
| **C&C-HeartBeat** | **1,332** |
| Attack | 538 |
| **C&C-FileDownload** | **46** |
| **C&C-Torii** | **30** |
| FileDownload | 13 |
| **C&C-HeartBeat-FileDownload** | **8** |
| Okiru-Attack | 3 |
| **C&C-Mirai** | **1** |

Bolded rows are the positive class per `prd.md` §4.4 — 16,703 positive
flow-rows total, all excluded/dropped rows are the non-C2 attack types
(§5, §6 of `exec.md`).

**Important, measured finding — read this before running the full
pipeline:** those 16,703 positive flow-rows collapse into only **26
distinct beaconing conversations** (unique infected-host → C2-destination
pairs — 12 infected source IPs talking to 19 C2 destination IPs), and
only **21 of those survive** the `min_session_flows: 4` filter. The
skewed part isn't fragmentation — most of the volume already sits inside
a handful of very large sessions (one session alone has 4,110 flows) —
it's simply that this specific preprocessed subset contains beaconing
traffic from very few distinct infection instances. This matches
`prd.md`'s own repeated warning (§4.5, §4.6, risk R-5): *"verify these
figures against your own download before quoting them"* — the counts in
the PRD came from a different published subset than this one.

**What this means practically:**
- The pipeline runs correctly end to end on this data — this was
  verified directly, see §4.4 below.
- But 21-26 positive sessions is a very small sample to split three ways
  (random/temporal/held-out-family) and get statistically stable numbers
  from. Expect the evaluation report's metrics to swing noticeably
  between runs with different seeds, and expect the held-out-family
  split (FR-8.7) to hold out only 1-2 families at a time.
- If you want a capstone-quality evaluation (not just a working pipeline
  demo), consider also trying `kaggle.com/datasets/surajsooraj26/iot-23`
  (the full, un-subsampled dataset) for a richer positive class, and
  reporting whichever variant you used explicitly in your write-up
  (`prd.md` §4.1's own critical-implementation-note already asks for
  this). Treat a run against `engraqeel/iot23preprocesseddata` as a
  correctness/pipeline-validation run first.

### 4.3 Pointing the pipeline at your download

Open `c2-beaconing-detection/config/default.yaml` and check/edit `data.raw_paths`:

```yaml
data:
  raw_paths:
    - "~/.cache/kagglehub/datasets/engraqeel/iot23preprocesseddata/versions/1/iot23_combined_new.csv"
```

(`~` is expanded automatically.) You can list more than one file here if
you're combining multiple capture files.

### 4.4 Confirmed working against the real file

Loading and feature-building (`src/pipeline.py::load_and_prepare` +
`build_feature_matrix`) were run directly against the real, complete
6,046,623-row file on a 16 GB machine:

```
load_and_prepare:      705,515 rows kept (target built) in 12.0s
build_feature_matrix:  435 sessions × 39 features in 2.5s
raw CSV load alone:    ~9.4s, ~3.5 GB in memory
```

Both comfortably clear NFR-3 (< 30 min full pipeline) and NFR-4 (< 8 GB)
for the ingest/feature stages — model training/tuning (§5) is the slower
part, but on a positive class this small it will also finish quickly;
grid search is the bottleneck on a *large* training set, not this one.

## 5. Run the full pipeline

Once `data.raw_paths` points at a real file:

```bash
python run_pipeline.py
```

This runs everything end to end: loads and validates the data, builds
features, trains both detection branches, fuses their scores, calibrates
decision thresholds, generates explanations, runs the jitter-robustness
sweep, and evaluates the result three different ways (random / temporal /
held-out-family splits — see `exec.md` §3, step 9).

It logs progress stage by stage to the terminal, e.g.:

```
=== Stage 1/8: ingest ===
=== Stage 2/8: sessionize + feature matrix ===
=== Stage 3/8: categorical encoding + splits ===
...
```

**Expect this to take a while** on the full dataset — model tuning does a
cross-validated grid search per model, and the pipeline trains fresh
models three times (once per split). If you just want to sanity-check
that everything runs without errors before committing to a long run, set
a small working sample size first:

```yaml
data:
  working_sample_size: 20000   # null = use everything; set a number for a fast dev run
```

### Where the output goes

After a run finishes, look in `c2-beaconing-detection/outputs/`:

| File | What it is |
|---|---|
| `outputs/models/random_forest.joblib` | the trained, calibrated supervised model |
| `outputs/models/random_forest_metadata.json` | its hyperparameters, training time, and feature list |
| `outputs/logs/decision_log.csv` | every session scored, with its action (ALLOW/ALERT/BLOCK) and top-3 explanation |
| `outputs/logs/thresholds.json` | the calculated ALLOW/ALERT/BLOCK cutoff scores |
| `outputs/logs/evaluation_report.json` | the full results: metrics for all three splits, the per-feature-family ablation, the jitter degradation curve, SHAP importances |
| `outputs/logs/pipeline.log` | a plain-text log of everything the run did |

Open `evaluation_report.json` first — `metrics.pr_auc` under each split
key is the primary number to look at (see `prd.md` §8.2 for what counts
as a good result and why accuracy alone is misleading here).

### Using a different config file

```bash
python run_pipeline.py --config config/my_experiment.yaml
```

Useful if you want to try different settings (e.g. a different fusion
weighting, or a different alerts-per-day budget) without overwriting your
main config.

## 6. Explore the results interactively (optional)

```bash
cd c2-beaconing-detection
jupyter notebook notebooks/
```

- `01_data_exploration.ipynb` — load the raw data and see the label
  breakdown before any modelling.
- `02_feature_analysis.ipynb` — inspect what the computed measurements
  actually look like, split by malicious vs. benign.
- `03_results.ipynb` — load `evaluation_report.json` from a completed
  `run_pipeline.py` run and inspect it as tables/plots.

(`jupyter` isn't in `requirements.txt` since it's optional tooling, not a
pipeline dependency — install it separately if you want the notebooks:
`pip install jupyter`.)

## 7. Common issues

| Symptom | Fix |
|---|---|
| `SchemaValidationError: Missing required column(s)` | Your dataset variant is missing `ts`/`uid`/`id.orig_h`/`id.resp_h`. Re-download the full variant — see §4.1. |
| `ValueError: No raw data paths configured` | You haven't filled in `data.raw_paths` in `config/default.yaml` yet, or cleared it out. |
| `FileNotFoundError` on a configured path | Check the path — `~` is expanded, but a relative path is resolved against wherever you run `python run_pipeline.py` from (normally `c2-beaconing-detection/`). Re-run the kagglehub snippet in §4 to print the exact current cache path if you're not sure where it downloaded to. |
| `ImportError: xgboost is not installed` / `PyTorch is required` / SHAP import error | `pip install -r requirements.txt` wasn't run in the environment you're using, or a package failed to install — re-run that command and check for errors. |
| Pipeline runs for a very long time | Set `data.working_sample_size` to a smaller number (see §5) while iterating, then remove it for a final full run. |
| Very few positive sessions / noisy metrics | Expected on `engraqeel/iot23preprocesseddata` — this is a measured, real property of that dataset variant (only ~21-26 positive sessions total), not a bug. See §4.2 for the full explanation and the alternative dataset suggestion. |
| Very few or zero sessions after sessionization overall | Check the log line `Sessionization discard rate: ...%`. A high rate is expected (§4.2 measured ~95% on the full file — most host pairs only exchange 1-2 connections total). If it's discarding sessions you specifically expected to keep, try lowering `sessionize.min_session_flows` from 4 to 3 in the config (documented as a fallback in `prd.md` R-2) — though note this only helps sessions sitting exactly at size 3; check the size distribution first if you need to know whether it'll help. |

## 8. Quick reference — command summary

```bash
# one-time setup
cd c2-beaconing-detection
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# verify the code works (fast, no dataset)
pytest

# get the dataset (config/default.yaml is already pointed at this path)
pip install kagglehub
python3 -c "import kagglehub; print(kagglehub.dataset_download('engraqeel/iot23preprocesseddata'))"

# run the full pipeline
python run_pipeline.py

# inspect results
cat outputs/logs/evaluation_report.json | python3 -m json.tool | less
```
