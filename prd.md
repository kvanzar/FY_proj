# Product Requirements Document
## Machine Learning-Based Detection and Blocking of Encrypted Command-and-Control Beaconing Traffic

| Field | Value |
|---|---|
| Document version | 1.0 |
| Status | Approved for implementation |
| Project type | Final year B.Tech Computer Science capstone |
| Scope of this PRD | Phase 1 — 75% of total system scope |
| Target attack | Command-and-Control beaconing (MITRE ATT&CK TA0011, T1071.001, T1573) |
| Primary dataset | Aposemat IoT-23 (Kaggle mirror) |
| Companion documents | `Literature_Survey_C2_Beaconing_Detection_v2.xlsx`, `C2_Beaconing_Literature_Review_Notes.md` |

---

## 1. Executive summary

This system detects malware command-and-control beaconing inside encrypted network traffic without decrypting anything. It works by observing behavioural metadata — how regularly a host contacts a destination, how the byte volume is distributed between upload and download, how connections terminate — and classifying that behaviour with machine learning.

**Phase 1 (this document, 75% of scope)** delivers a complete, working, offline detection engine validated on a public labelled dataset. It ingests Zeek connection logs, derives beaconing features, trains a dual-branch model (supervised classifier plus benign-trained anomaly detector), fuses their scores into a calibrated risk value, explains each verdict, and simulates the three-tier enforcement decision. It is fully evaluable and demonstrable.

**Phase 2 (deferred, 25% of scope)** adds live packet capture, TLS handshake metadata including JA3 fingerprinting, DNS and reputation context, genuine inline enforcement on a real network interface, and the closed-loop retraining pipeline.

The split is deliberate and defensible: Phase 1 delivers everything that can be rigorously validated against ground-truth labels on a public dataset, while Phase 2 covers everything that requires live infrastructure, a private testbed, or data the public dataset does not contain.

---

## 2. Scope definition

### 2.1 What "75%" means concretely

The 75% figure is measured against the eight-stage reference architecture. Each stage is scored by the proportion of its functional requirements delivered in Phase 1.

| Architecture stage | Phase 1 delivery | Weight | Complete |
|---|---|---|---|
| 0. Traffic ingestion | Offline Zeek `conn.log` ingestion from dataset | 8% | 70% |
| 1. Flow assembly | Flow record parsing, session grouping by host-pair | 10% | 100% |
| 2. Feature extraction — Tier 1 (timing) | Full: periodicity, jitter, regularity statistics | 14% | 100% |
| 2. Feature extraction — Tier 2 (volume) | Full: byte ratios, packet counts, size distribution | 12% | 100% |
| 2. Feature extraction — Tier 3 (TLS metadata) | Deferred — schema and extractor stub only | 10% | 15% |
| 2. Feature extraction — Tier 4 (context) | Partial: destination fan-in derived from dataset | 8% | 40% |
| 3. Supervised branch | Full: RF and XGBoost, imbalance handling, tuning | 14% | 100% |
| 4. Anomaly branch | Full: autoencoder trained on benign only | 10% | 100% |
| 5. Score fusion and calibration | Full: threshold calibration to alert budget | 6% | 100% |
| 6. Explanation layer | Full: SHAP global and per-verdict attribution | 5% | 100% |
| 7. Enforcement engine | Offline simulation with decision logging | 2% | 60% |
| 8. Feedback loop | Deferred — interface defined only | 1% | 10% |
| **Weighted total** | | **100%** | **≈ 76%** |

### 2.2 In scope for Phase 1

- Ingestion and parsing of Zeek `conn.log` flow records from the IoT-23 dataset
- Grouping of flows into host-pair communication sessions for temporal analysis
- Derivation of beaconing periodicity features from connection timestamps
- Derivation of volume asymmetry and connection-state features
- Supervised binary and multi-class classification with explicit class-imbalance handling
- Unsupervised anomaly detection trained exclusively on benign traffic
- Score fusion producing a single calibrated risk value per flow
- Threshold calibration against an operator-defined alerts-per-day budget
- SHAP-based global feature importance and per-verdict local explanation
- Offline simulation of the allow / alert / block decision with full logging
- Controlled synthetic jitter injection to measure detection degradation
- Temporal train/test splitting and held-out-family generalisation testing
- Full evaluation report with PR-AUC as the primary metric

### 2.3 Out of scope for Phase 1 (deferred to Phase 2)

- Live packet capture from a network interface
- TLS handshake parsing and JA3 / JA3S fingerprint extraction
- Certificate property extraction (validity period, self-signed detection, issuer)
- DNS query correlation and domain-age lookup
- Third-party destination reputation enrichment
- Genuine inline traffic interception or connection termination
- Analyst-facing dashboard or web UI
- Automated retraining triggered by drift detection
- Adversarial hardening through iterative retraining
- Multi-tenant or distributed deployment

### 2.4 Explicit non-goals

The following are permanently out of scope and should be stated as such in the report:

- **Traffic decryption.** The system never decrypts. Any approach requiring TLS interception is rejected on privacy and architectural grounds.
- **Detection of non-C2 attacks.** DDoS, port scanning, phishing and web application attacks are outside the threat model. They require disjoint feature families (see literature review §2.4).
- **Malware family attribution.** The system determines whether traffic is C2, not which specific malware produced it. Family classification is an evaluation aid only, not a product feature.
- **Endpoint agent or host-based detection.** This is a network-side system.

---

## 3. Problem statement

Command-and-control beaconing is the pivot point of the modern intrusion lifecycle: the phase at which an attacker converts an initial foothold into persistent, remotely directed access. Every subsequent stage — lateral movement, privilege escalation, data staging and exfiltration — depends on it. It is therefore the highest-leverage single point of interdiction available to a network defender.

This channel is now effectively opaque. The overwhelming majority of malicious traffic is TLS-encrypted, rendering deep packet inspection and signature-based firewall rules inoperative against C2 payloads. Attackers compound this by rotating infrastructure, forging or self-signing certificates, mimicking mainstream browser TLS fingerprints, and above all by configuring sleep and data jitter to destroy the timing regularity on which classical periodicity detection depends.

Existing defences fail along four axes. Signature and blocklist approaches are defeated by infrastructure rotation and domain generation algorithms. Rule-based periodicity detectors suppress the enormous volume of legitimate beaconing only through hand-designed, environment-specific filter cascades that must be retuned per network. Supervised machine learning detectors report excellent laboratory figures yet fall sharply against real-world traces, and by construction cannot recognise C2 frameworks absent from their training data. Finally, virtually all published work terminates at a classification verdict, offering no calibrated enforcement decision, no measured latency, and no explanation an analyst can act upon.

**Phase 1 addresses the detection core of this problem:** to distinguish C2 beaconing flows from benign traffic using only timing and volume metadata derivable without decryption; to remain measurably effective as beacon timing is deliberately randomised; to surface C2 families absent from training data through a complementary anomaly branch; and to produce a calibrated, explainable verdict whose false-positive cost is expressed in operational rather than purely statistical terms.

---

## 4. Dataset specification

### 4.1 Primary dataset — Aposemat IoT-23

| Attribute | Value |
|---|---|
| Name | Aposemat IoT-23 |
| Origin | Stratosphere Laboratory, CTU University, Prague; funded by Avast Software |
| Published | January 2020 (captures from 2018–2019) |
| Scale | 23 captures — 20 malicious, 3 benign; over 760 million packets, ~325 million labelled flows |
| Format | Zeek `conn.log.labeled` (flow records, not raw packets) |
| Citation | Garcia, S., Parmisano, A., & Erquiaga, M.J. (2020). *IoT-23: A labeled dataset with malicious and benign IoT network traffic.* Zenodo. DOI: 10.5281/zenodo.4743746 |

**Kaggle access points:**

| Variant | URL | Use |
|---|---|---|
| Preprocessed subset (~6M rows) | `kaggle.com/datasets/engraqeel/iot23preprocesseddata` | Primary — development and training |
| Full dataset | `kaggle.com/datasets/surajsooraj26/iot-23` | Secondary — validation at scale |
| Alternate mirror | `kaggle.com/datasets/abdullahhashmi/iot23` | Fallback |

> **CRITICAL IMPLEMENTATION NOTE.** Several circulating preprocessed variants of IoT-23 have dropped the `ts`, `uid`, `id.orig_h` and `id.resp_h` columns as "dataset-specific." **Those variants are unusable for this project.** Beacon periodicity is computed as the time delta between consecutive connections from the same source to the same destination — which requires the timestamp *and* both host identifiers. Before beginning work, verify your download contains `ts`, `id.orig_h` and `id.resp_h`. If it does not, use the full-dataset variant or download the original `conn.log.labeled` files directly from Stratosphere.

### 4.2 Why this dataset justifies the model

This dataset is not merely convenient — it is the strongest publicly available ground truth for this specific attack, for four reasons.

**It labels C2 beaconing explicitly.** IoT-23 does not force you to infer which flows are beacons. The dataset authors define a `C&C-HeartBeat` detailed label meaning, in their own description, that the packets on the connection are used by the C2 server to keep track of the infected host — identified by filtering connections with response bytes under 1 byte, periodic similar connections, and a known-suspicious destination port or IP. **The dataset's own labelling criterion is beaconing behaviour.** This is exact ground truth for the phenomenon the model targets, which no generic IDS dataset provides.

**The labels come from real malware on real hardware.** The malicious captures are real malware samples (Mirai, Torii, Okiru, Gagfyt, Kenjiro, Hakai, IRCBot, Muhstik, Hide-and-Seek and others) executed on a Raspberry Pi. The benign captures come from genuine consumer IoT devices — a Philips Hue smart lamp, an Amazon Echo, and a Somfy smart door lock. Neither side is simulated.

**Benign traffic contains real legitimate beaconing.** This is the property that makes the dataset valuable for Gap 6. Smart home devices poll cloud services on a schedule constantly. So the benign class is not "aperiodic human browsing" — it is genuinely periodic machine-to-machine traffic. A model that separates `C&C-HeartBeat` from IoT-23 benign traffic has learned to distinguish *malicious periodicity from legitimate periodicity*, not merely *periodic from aperiodic*. That is the harder and more meaningful problem.

**It supports held-out-family generalisation testing.** Because captures are separated by malware family, the model can be trained on some families and tested on others entirely unseen — the evaluation protocol used by EarlyCrow and the standard the literature review identifies as separating credible work from optimistic work.

### 4.3 Schema

Zeek `conn.log` fields present in the full variant:

| Field | Type | Description | Phase 1 use |
|---|---|---|---|
| `ts` | float | Timestamp of first packet | **Essential** — beacon interval derivation |
| `uid` | string | Zeek unique connection ID | Row identity; dropped from features |
| `id.orig_h` | string | Originator IP | **Essential** — session grouping; dropped from features |
| `id.orig_p` | int | Originator port | Feature (ephemeral-port behaviour) |
| `id.resp_h` | string | Responder IP | **Essential** — session grouping; dropped from features |
| `id.resp_p` | int | Responder port | Feature (destination port) |
| `proto` | string | Transport protocol (tcp/udp/icmp) | Feature (categorical) |
| `service` | string | Application protocol if identified | Feature (categorical, high-null) |
| `duration` | float | Connection duration | Feature |
| `orig_bytes` | int | Payload bytes sent by originator | Feature |
| `resp_bytes` | int | Payload bytes sent by responder | Feature |
| `conn_state` | string | Connection state (S0, SF, REJ, RSTO, …) | Feature (categorical) |
| `history` | string | State history as letter sequence | Feature (engineered) |
| `orig_pkts` | int | Packets sent by originator | Feature |
| `orig_ip_bytes` | int | IP-level bytes from originator | Feature |
| `resp_pkts` | int | Packets sent by responder | Feature |
| `resp_ip_bytes` | int | IP-level bytes from responder | Feature |
| `label` | enum | Benign / Malicious | Binary target |
| `detailed-label` | enum | Specific behaviour category | **Primary target source** |

Known `detailed-label` values include: `Benign`, `C&C`, `C&C-HeartBeat`, `C&C-FileDownload`, `C&C-HeartBeat-FileDownload`, `C&C-Torii`, `C&C-Mirai`, `DDoS`, `Okiru`, `Attack`, `FileDownload`, `PartOfAHorizontalPortScan`.

### 4.4 Target variable construction

Phase 1 defines a **binary** target, deliberately narrower than the dataset's own malicious/benign split:

```
target = 1  if detailed-label ∈ {C&C, C&C-HeartBeat, C&C-FileDownload,
                                  C&C-HeartBeat-FileDownload, C&C-Torii, C&C-Mirai}
target = 0  if detailed-label == Benign
excluded    if detailed-label ∈ {DDoS, Okiru, Attack, PartOfAHorizontalPortScan,
                                  FileDownload}
```

**Rationale for exclusion.** DDoS, port scan and generic attack flows are malicious but are *not C2 beaconing*. Including them as positives would train a general "malicious traffic" detector and directly contradict the attack-specific scoping that defines this project. Including them as negatives would teach the model that malicious traffic is benign. Excluding them is the only coherent option, and this decision must be stated explicitly in the report — an examiner will ask.

A secondary multi-class configuration (`Benign` / `C&C-HeartBeat` / `other-C&C`) is retained as an analysis aid to check whether pure heartbeat beacons are detected more reliably than C2 flows carrying file transfers.

### 4.5 Class imbalance — the central data challenge

C2 flows are a vanishing fraction of IoT-23. In one published sampled subset, `C&C-HeartBeat` accounts for roughly 3,900 flows against approximately 826,000 `PartOfAHorizontalPortScan` and 198,000 `Benign` records. In the full dataset, the rarest C2 categories are extraordinarily sparse — `C&C-Mirai` and `C&C-HeartBeat-FileDownload` number in the single or low double digits of flows.

> **Verify these figures against your own download before quoting them.** Counts vary substantially between the full dataset and the circulating preprocessed subsets, and the numbers above come from a published sampling study rather than from a canonical distribution table.

Three consequences flow directly into the requirements:

1. **Accuracy is a meaningless metric here.** A classifier predicting "benign" universally would score above 99%. The evaluation protocol therefore mandates PR-AUC as primary.
2. **Stratification is mandatory** at every split, or rare C2 classes will vanish entirely from a fold.
3. **The rarest categories cannot support reliable evaluation.** Categories with fewer than 50 flows after sampling are folded into a combined `other-C&C` group rather than evaluated individually.

### 4.6 Known dataset limitations

These must appear in the report's limitations section. Stating them proactively is a strength, not a weakness.

| Limitation | Consequence | Mitigation |
|---|---|---|
| No TLS metadata — `conn.log` only, no `ssl.log` | JA3 and certificate features (Tier 3) cannot be validated in Phase 1 | Deferred to Phase 2; Phase 1 explicitly reports Tier 1+2-only performance as the encryption-resistant floor |
| IoT device traffic, not enterprise desktop traffic | Beacon intervals and benign baselines may differ from an enterprise network | Report as a generalisation caveat; Phase 2 adds enterprise captures |
| Captures from 2018–2019 | Predates TLS 1.3 ubiquity and current C2 frameworks | Acknowledged; Tier 1+2 features are protocol-agnostic and least affected by this |
| Labels are analyst-assigned, not cryptographically verified | Some label noise is likely present | Follows Anderson & McGrew (2017); note as a bounded source of error |
| Severe class imbalance | Naive models will appear excellent while detecting nothing | Addressed by FR-3.4 and the evaluation protocol |

### 4.7 Supplementary datasets

| Dataset | Kaggle / source | Role |
|---|---|---|
| CTU-13 | `stratosphereips.org/datasets-ctu13`; multiple Kaggle mirrors | Cross-dataset validation — does an IoT-23-trained model transfer? |
| CIC-IDS2017 | `unb.ca/cic/datasets/ids-2017.html` | Optional — provides a labelled botnet day with CICFlowMeter features |
| Synthetic jitter set | Generated locally (FR-6.2) | Adversarial degradation curve |

---

## 5. Functional requirements

Requirements use MoSCoW priority: **M** must-have, **S** should-have, **C** could-have.

### FR-1 — Data ingestion and validation

| ID | Requirement | Priority |
|---|---|---|
| FR-1.1 | Load IoT-23 `conn.log.labeled` or CSV exports into a dataframe, handling Zeek's `-` null convention | M |
| FR-1.2 | Validate that `ts`, `id.orig_h` and `id.resp_h` are present; abort with a clear error if absent | M |
| FR-1.3 | Coerce numeric columns, treating Zeek `-` as null rather than string | M |
| FR-1.4 | Emit a data profile report: row count, null fraction per column, label distribution | M |
| FR-1.5 | Support loading multiple capture files with a capture-ID column preserved for held-out-family splitting | M |
| FR-1.6 | Support stratified sampling to a configurable working size for development iteration | S |

### FR-2 — Feature engineering

#### FR-2.1 Session grouping (prerequisite for all timing features)

| ID | Requirement | Priority |
|---|---|---|
| FR-2.1.1 | Group flows into sessions keyed by `(id.orig_h, id.resp_h, id.resp_p)` | M |
| FR-2.1.2 | Sort each session's flows by `ts` ascending | M |
| FR-2.1.3 | Compute inter-arrival deltas between consecutive flows within a session | M |
| FR-2.1.4 | Discard sessions with fewer than 4 flows as insufficient for periodicity assessment; log the discard rate | M |

#### FR-2.2 Tier 1 — Timing features

| ID | Feature | Definition | Priority |
|---|---|---|---|
| FR-2.2.1 | `iat_mean` | Mean inter-arrival time between session flows | M |
| FR-2.2.2 | `iat_std` | Standard deviation of inter-arrival times | M |
| FR-2.2.3 | `iat_cv` | Coefficient of variation (`std / mean`) — the core beacon regularity signal; low CV means machine-regular | M |
| FR-2.2.4 | `iat_median`, `iat_min`, `iat_max` | Distribution shape, robust to outliers | M |
| FR-2.2.5 | `iat_mad` | Median absolute deviation — jitter-robust regularity measure | M |
| FR-2.2.6 | `periodicity_score` | Normalised autocorrelation peak of the interval series | M |
| FR-2.2.7 | `fft_dominant_power` | Power at the dominant frequency of the interval series | S |
| FR-2.2.8 | `session_flow_count` | Number of flows in the session | M |
| FR-2.2.9 | `session_span_hours` | Wall-clock duration from first to last flow | M |
| FR-2.2.10 | `duty_cycle` | Sum of flow durations divided by session span | S |
| FR-2.2.11 | `hour_of_day`, `is_weekend` | Temporal context; malware beacons ignore human schedules | S |

#### FR-2.3 Tier 2 — Volume features

| ID | Feature | Definition | Priority |
|---|---|---|---|
| FR-2.3.1 | `orig_resp_byte_ratio` | `orig_ip_bytes / (resp_ip_bytes + 1)` — captures the upload-heavy signature of exfiltration and the near-null response of a heartbeat | M |
| FR-2.3.2 | `total_bytes`, `orig_bytes`, `resp_bytes` | Raw volume, log-transformed | M |
| FR-2.3.3 | `orig_pkts`, `resp_pkts`, `pkt_ratio` | Packet counts and their ratio | M |
| FR-2.3.4 | `mean_pkt_size_orig`, `mean_pkt_size_resp` | Bytes divided by packet count | M |
| FR-2.3.5 | `bytes_per_second` | Throughput; beacons are extremely low | M |
| FR-2.3.6 | `resp_bytes_is_zero` | Boolean — directly mirrors the dataset's own heartbeat labelling criterion | M |
| FR-2.3.7 | `session_byte_variance` | Variance of flow sizes within a session; beacons are uniform | M |
| FR-2.3.8 | `duration` | Per-flow connection duration | M |

#### FR-2.4 Connection-state features

| ID | Feature | Definition | Priority |
|---|---|---|---|
| FR-2.4.1 | `conn_state` | One-hot encoded Zeek connection state | M |
| FR-2.4.2 | `history_len`, `history_has_S`, `history_has_D` | Engineered from the Zeek history string | S |
| FR-2.4.3 | `proto`, `service` | One-hot encoded, with an explicit unknown category | M |
| FR-2.4.4 | `resp_port`, `is_ephemeral_orig_port`, `is_common_service_port` | Port-derived features | M |

#### FR-2.5 Tier 4 (partial) — Context features derivable from the dataset

| ID | Feature | Definition | Priority |
|---|---|---|---|
| FR-2.5.1 | `dest_fanin` | Count of distinct source hosts contacting this destination | S |
| FR-2.5.2 | `src_fanout` | Count of distinct destinations contacted by this source | S |
| FR-2.5.3 | `dest_flow_count` | Total flows to this destination across the capture | S |

#### FR-2.6 Feature hygiene constraints

| ID | Requirement | Priority |
|---|---|---|
| FR-2.6.1 | **No raw IP addresses may enter the feature set.** Identity-agnostic design per PETNet; a model keying on specific IPs is defeated by infrastructure rotation and inflates test performance through leakage | M |
| FR-2.6.2 | No `uid` or capture filename in features | M |
| FR-2.6.3 | Every feature must be tagged with its visibility tier so per-tier ablation is possible | M |
| FR-2.6.4 | Log-transform heavy-tailed volume features | M |
| FR-2.6.5 | Fit scalers on training data only; never on the full dataset before splitting | M |

### FR-3 — Supervised detection branch

| ID | Requirement | Priority |
|---|---|---|
| FR-3.1 | Implement Random Forest as the baseline classifier | M |
| FR-3.2 | Implement XGBoost as the comparison classifier | M |
| FR-3.3 | Implement Logistic Regression as an interpretable floor baseline | S |
| FR-3.4 | Handle class imbalance by at least two of: class weighting, focal loss, SMOTE oversampling — and compare their effect | M |
| FR-3.5 | Hyperparameter tuning via stratified cross-validation optimising PR-AUC, never accuracy | M |
| FR-3.6 | Output a calibrated probability, not a hard label (Platt scaling or isotonic regression) | M |
| FR-3.7 | Persist trained models with version metadata and the exact feature list used | M |
| FR-3.8 | Record training wall-clock time and per-flow inference latency | M |

### FR-4 — Anomaly detection branch

| ID | Requirement | Priority |
|---|---|---|
| FR-4.1 | Implement an autoencoder trained **exclusively on benign flows** — no malicious data may touch this branch during training | M |
| FR-4.2 | Score flows by reconstruction error | M |
| FR-4.3 | Set the anomaly threshold from the benign reconstruction-error distribution (e.g. 95th percentile), not from test-set performance | M |
| FR-4.4 | Benchmark against Isolation Forest and One-Class SVM baselines | S |
| FR-4.5 | Evaluate specifically on C2 families entirely held out of all training — this is the branch's whole justification | M |
| FR-4.6 | Normalise reconstruction error to `[0,1]` for fusion | M |

### FR-5 — Fusion, calibration and decision

| ID | Requirement | Priority |
|---|---|---|
| FR-5.1 | Combine supervised probability and normalised anomaly score into one risk value; weighting configurable and its choice justified empirically | M |
| FR-5.2 | Support at minimum weighted-average and max-score fusion, comparing both | M |
| FR-5.3 | Derive thresholds `τ_low` and `τ_high` from an operator-specified alerts-per-day budget, projected onto a stated flow volume | M |
| FR-5.4 | Map risk to `ALLOW` / `ALERT` / `BLOCK` | M |
| FR-5.5 | Log every decision with flow ID, risk score, branch scores, action and top contributing features | M |
| FR-5.6 | Provide a configuration file for all thresholds — no hardcoded magic numbers | M |

### FR-6 — Adversarial evaluation harness

| ID | Requirement | Priority |
|---|---|---|
| FR-6.1 | Implement synthetic jitter injection: perturb C2 flow timestamps by ±J% of the base interval | M |
| FR-6.2 | Sweep J from 0% to 50% in 10% steps; measure recall at fixed precision at each level | M |
| FR-6.3 | Produce the **jitter degradation curve** — the project's headline robustness result | M |
| FR-6.4 | Implement packet-padding simulation: inflate byte counts by a random factor and measure impact | S |
| FR-6.5 | Evaluate against the DReLAB adversarial botnet benchmark if time permits | C |

### FR-7 — Explainability

| ID | Requirement | Priority |
|---|---|---|
| FR-7.1 | Compute SHAP global feature importance for the supervised branch | M |
| FR-7.2 | Produce a per-verdict local explanation listing the top three contributing features with signed direction | M |
| FR-7.3 | Render explanations in human-readable form, e.g. `"interval CV 0.03 (highly regular) · response bytes 0 · destination fan-in 1"` | M |
| FR-7.4 | Compare SHAP rankings against Random Forest built-in feature importance and note disagreements | S |

### FR-8 — Evaluation and reporting

| ID | Requirement | Priority |
|---|---|---|
| FR-8.1 | Report PR-AUC as the **primary** metric | M |
| FR-8.2 | Report precision, recall and F1 at the operating threshold | M |
| FR-8.3 | Report ROC-AUC and accuracy as **secondary** metrics, with explicit commentary on why they mislead under this imbalance | M |
| FR-8.4 | Report projected false positives per day at a stated flow volume | M |
| FR-8.5 | Produce a confusion matrix at the operating threshold | M |
| FR-8.6 | Perform **temporal** splitting (train on earlier captures, test on later) in addition to random splitting, and report both | M |
| FR-8.7 | Perform **held-out-family** evaluation: train on a subset of malware families, test on entirely unseen ones | M |
| FR-8.8 | Perform a **per-tier ablation**: Tier 1 only, Tier 1+2, Tier 1+2+4, quantifying each tier's contribution | M |
| FR-8.9 | Generate all evaluation artefacts reproducibly from a single command | M |

---

## 6. Non-functional requirements

| ID | Requirement | Target |
|---|---|---|
| NFR-1 | Inference latency per flow | < 10 ms on commodity hardware (CPU only) |
| NFR-2 | Batch scoring throughput | ≥ 10,000 flows/second |
| NFR-3 | Training time, full pipeline | < 30 minutes on a standard laptop |
| NFR-4 | Peak memory | < 8 GB, so the project runs without special hardware |
| NFR-5 | Reproducibility | Fixed random seeds; identical results across runs |
| NFR-6 | Environment | Pinned dependency versions in `requirements.txt` |
| NFR-7 | Code quality | Modular, typed function signatures, docstrings on public functions |
| NFR-8 | Configuration | All parameters in a single YAML/JSON config; no hardcoded constants |
| NFR-9 | Logging | Structured logging at every pipeline stage |
| NFR-10 | Documentation | README with setup, data acquisition, and run instructions |

---

## 7. System modules

```
c2-beaconing-detection/
├── config/
│   └── default.yaml              # all thresholds and hyperparameters
├── beacon_detection/
│   ├── ingest.py                 # FR-1: loading, validation, profiling
│   ├── sessionize.py             # FR-2.1: host-pair grouping, interval derivation
│   ├── features/
│   │   ├── timing.py             # FR-2.2: Tier 1
│   │   ├── volume.py             # FR-2.3: Tier 2
│   │   ├── connstate.py          # FR-2.4
│   │   ├── context.py            # FR-2.5: Tier 4 partial
│   │   └── tls_stub.py           # Phase 2 placeholder, interface defined
│   ├── models/
│   │   ├── supervised.py         # FR-3
│   │   ├── anomaly.py            # FR-4
│   │   └── fusion.py             # FR-5.1, FR-5.2
│   ├── decision.py               # FR-5.3–5.6: enforcement simulation
│   ├── explain.py                # FR-7: SHAP
│   ├── adversarial.py            # FR-6: jitter sweep
│   └── evaluate.py               # FR-8: metrics and reports
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_analysis.ipynb
│   └── 03_results.ipynb
├── tests/
├── outputs/                      # models, figures, decision logs
└── requirements.txt
```

### 7.1 Module contracts

| Module | Input | Output |
|---|---|---|
| `ingest` | Raw `conn.log` / CSV paths | Validated dataframe + profile report |
| `sessionize` | Flow dataframe | Flows annotated with session ID and inter-arrival delta |
| `features/*` | Sessionised flows | Feature matrix with tier tags |
| `models/supervised` | Feature matrix + labels | Calibrated probability per flow |
| `models/anomaly` | Feature matrix (benign-only for fit) | Normalised anomaly score per flow |
| `models/fusion` | Both scores | Single risk value |
| `decision` | Risk value + config thresholds | Action + decision log entry |
| `explain` | Model + feature vector | Top-3 signed attributions |
| `adversarial` | Trained model + test set | Degradation curve data |
| `evaluate` | Predictions + ground truth | Metrics, plots, report |

---

## 8. Evaluation protocol

### 8.1 Splitting strategy

Three splits are produced, and **all three must be reported**:

1. **Random stratified split** (70/15/15) — the optimistic baseline that most student projects report alone.
2. **Temporal split** — train on chronologically earlier captures, test on later. Follows Anderson & McGrew (2017) on non-stationarity.
3. **Held-out-family split** — train on a subset of malware families, test on families entirely absent from training. Follows EarlyCrow.

The gap between split 1 and splits 2 and 3 is itself a finding, and should be presented as one. A large gap indicates the model is memorising rather than generalising.

### 8.2 Acceptance criteria

| Criterion | Threshold | Rationale |
|---|---|---|
| AC-1 | PR-AUC ≥ 0.70 on random split | Meaningfully above the positive-class base rate |
| AC-2 | PR-AUC ≥ 0.50 on held-out-family split | Demonstrates genuine generalisation, not memorisation |
| AC-3 | Recall ≥ 0.60 at precision ≥ 0.90 | Operationally usable false-alarm rate |
| AC-4 | Anomaly branch recall on held-out families ≥ 0.30 | Justifies the branch's existence |
| AC-5 | Jitter degradation curve produced for J = 0–50% | Core robustness deliverable |
| AC-6 | Per-tier ablation completed | Quantifies encryption-resistant floor |
| AC-7 | Inference latency < 10 ms/flow | Validates inline feasibility claim |
| AC-8 | Every verdict carries a top-3 explanation | Explainability requirement |

> **On acceptance thresholds.** These are set deliberately below the 99% figures common in the literature, because that literature is evaluated on random splits of lab data. Ramos & Wang detected roughly half of real-world Cobalt Strike traffic at a 1.4% false positive rate, and that is a credible published result. A project reporting 99.8% PR-AUC on a held-out-family split should be treated as evidence of leakage, not success.

### 8.3 Deliverable artefacts

- PR curve and ROC curve for all three splits
- Confusion matrix at the operating threshold
- Jitter degradation curve (recall vs. jitter percentage)
- Per-tier ablation bar chart
- SHAP global importance plot
- Sample decision log with worked explanations
- Model comparison table (RF / XGBoost / LR / anomaly / fused)

---

## 9. Implementation plan

| Phase | Weeks | Deliverable | Requirements |
|---|---|---|---|
| P1 — Data foundation | 1–2 | Dataset acquired and validated; profile report; exploration notebook | FR-1 |
| P2 — Sessionisation | 3 | Host-pair grouping, interval derivation, discard-rate analysis | FR-2.1 |
| P3 — Feature engineering | 4–5 | Full Tier 1/2/4 feature matrix with tier tags | FR-2.2–2.6 |
| P4 — Supervised branch | 6–7 | Trained, tuned, calibrated RF and XGBoost | FR-3 |
| P5 — Anomaly branch | 8 | Autoencoder + baselines, held-out-family evaluation | FR-4 |
| P6 — Fusion and decision | 9 | Score fusion, threshold calibration, decision logging | FR-5 |
| P7 — Explainability | 10 | SHAP integration, human-readable verdicts | FR-7 |
| P8 — Adversarial harness | 11 | Jitter sweep, degradation curve | FR-6 |
| P9 — Evaluation | 12 | All three splits, ablation, full metric suite | FR-8 |
| P10 — Documentation | 13 | Report, README, reproducibility check, presentation | NFR-10 |

### 9.1 Critical path and risk

The critical path runs P2 → P3 → P4. **Sessionisation is the highest-risk early step**: if the dataset variant lacks `ts` or host identifiers, no timing feature can be built and the project's central hypothesis becomes untestable. Validate this in week 1, not week 3.

---

## 10. Technology stack

| Layer | Choice | Justification |
|---|---|---|
| Language | Python 3.10+ | Ecosystem maturity |
| Data | pandas, numpy | Standard; dataset fits in memory when sampled |
| Classical ML | scikit-learn | RF, LR, IsolationForest, OCSVM, calibration, metrics |
| Gradient boosting | XGBoost | Strong tabular performance; native imbalance handling |
| Deep learning | PyTorch (autoencoder only) | Small model; TensorFlow acceptable alternative |
| Imbalance | imbalanced-learn | SMOTE and variants |
| Explainability | shap | Standard SHAP implementation |
| Signal processing | scipy | Autocorrelation, FFT for periodicity |
| Visualisation | matplotlib, seaborn | Report figures |
| Config | PyYAML | NFR-8 |
| Testing | pytest | Feature-function correctness |

Deliberately excluded from Phase 1: any deep sequence model (LSTM, Transformer). RAID 2024 found Random Forest outperforms CNN and GRU when features are sparse, and the 2026 CTU-13 study found RF trains over 90% faster than CNN baselines with competitive results. Classical ML is the evidence-backed choice, not a compromise.

---

## 11. Risks and mitigations

| ID | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| R-1 | Downloaded dataset variant lacks `ts` / host columns | Medium | **Critical** | Validate in week 1; fall back to full dataset or direct Stratosphere download |
| R-2 | Too few sessions survive the ≥4-flow filter | Medium | High | Lower threshold to 3; report discard rate; consider longer aggregation windows |
| R-3 | Model memorises destination ports rather than behaviour | Medium | High | Ablate port features; verify SHAP does not rank them first |
| R-4 | Data leakage inflates results | Medium | High | Split by session before featurising, never after; fit scalers on train only |
| R-5 | Rarest C2 classes too sparse to evaluate | High | Medium | Fold classes with <50 flows into `other-C&C`; state explicitly |
| R-6 | Autoencoder produces unusable false-positive rate | Medium | Medium | Tune threshold on benign distribution; fall back to Isolation Forest |
| R-7 | Full dataset exceeds available memory | Medium | Low | Stratified sampling; chunked reading |
| R-8 | Results look "too good" | Medium | Medium | Treat as a leakage signal; audit against the Arp et al. pitfall checklist |

---

## 12. Traceability — gaps to requirements

| Gap (from literature review) | Addressed by | Phase |
|---|---|---|
| GAP-1 — Lab-to-real-world collapse | FR-8.6, FR-8.7 (temporal and held-out-family splits) | 1 |
| GAP-2 — Jitter evasion under-modelled | FR-6.1–6.3 (jitter degradation curve) | 1 (partial) |
| GAP-3 — Supervised blind to unknown C2 | FR-4.1, FR-4.5 (benign-trained anomaly branch) | 1 |
| GAP-4 — TLS 1.3 / ECH / DoH erosion | FR-2.6.3, FR-8.8 (tier tagging and ablation) | 1 (partial) |
| GAP-5 — Imbalance masked by bad metrics | FR-3.4, FR-8.1, FR-8.3, FR-8.4 | 1 |
| GAP-6 — Benign-beacon false positives | Dataset choice §4.2 (IoT benign traffic is genuinely periodic) | 1 |
| GAP-7 — Detection decoupled from enforcement | FR-5.3–5.6 (offline), full inline deferred | 1 (partial) |
| GAP-8 — Alerts lack explanation | FR-7.1–7.3 (SHAP) | 1 |

Six of eight gaps are fully addressed in Phase 1; two are partially addressed with the remainder scheduled for Phase 2.

---

## 13. Future scope — the remaining 25%

Phase 2 is scoped here in the same detail as Phase 1, so the report demonstrates that the deferred work is *planned* rather than *unconsidered*. Weights refer to the remaining 25% of total project scope.

### 13.1 FS-1 — TLS metadata extraction and JA3 fingerprinting (8% of total scope)

**Why deferred.** The Kaggle IoT-23 distribution provides Zeek `conn.log` only. TLS handshake data lives in `ssl.log`, which the preprocessed variants omit. Phase 1 cannot validate Tier 3 features against ground truth, and building unvalidatable features would be poor practice.

**Phase 2 work.**
- Reprocess the original IoT-23 PCAPs with Zeek configured to emit `ssl.log` and `x509.log`
- Extract JA3 and JA3S client/server fingerprints
- Extract cipher suite offers, TLS version, extension sets
- Extract certificate features: validity period, self-signed flag, issuer, subject-CN entropy
- Join TLS records to connection records on Zeek `uid`
- **Constraint:** all features must remain identity-agnostic per PETNet — encode certificate *properties*, never certificate *identities*, or the model will be defeated by infrastructure rotation
- Re-run the per-tier ablation with Tier 3 present and quantify its marginal contribution over the Tier 1+2 floor

**Success criterion.** Measurable PR-AUC improvement from adding Tier 3, plus a quantified statement of exactly how much is lost when TLS 1.3 hides those fields.

### 13.2 FS-2 — Live packet capture and real-time inference (6%)

**Why deferred.** Requires a controlled network testbed, a traffic mirror or tap, and infrastructure not available for offline dataset work.

**Phase 2 work.**
- Deploy Zeek in live-capture mode on a span port or virtual testbed
- Implement a streaming sessionisation buffer with a sliding time window — the hardest engineering problem here, since periodicity needs history but memory is bounded
- Handle partial sessions: a beacon that has fired only twice is not yet assessable
- Measure genuine end-to-end latency from packet arrival to verdict
- Validate NFR-1 and NFR-2 under live load rather than batch conditions

**Key open question.** What is the minimum observation window before a periodicity verdict is trustworthy? This is a genuine research question and would make a strong Phase 2 contribution in its own right.

### 13.3 FS-3 — DNS, domain age and reputation context (4%)

**Why deferred.** Requires external API access (WHOIS, passive DNS, reputation feeds) and DNS query logs correlated to connections — none present in the Phase 1 data.

**Phase 2 work.**
- Correlate Zeek `dns.log` queries to subsequent connections by timing and resolved address
- Enrich with domain registration age via WHOIS
- Integrate a destination reputation feed
- Add DGA likelihood scoring on queried domain names using character-level lexical features
- **Document the DoH limitation explicitly:** when the victim uses DNS-over-HTTPS, this entire tier goes dark. Report accuracy with and without it.

### 13.4 FS-4 — Genuine inline enforcement (3%)

**Why deferred.** Phase 1 simulates the enforcement decision offline. Actually dropping traffic requires firewall integration and carries real operational risk.

**Phase 2 work.**
- Integrate with `iptables` / `nftables` or a pfSense testbed
- Implement connection termination on `BLOCK` verdicts
- Build an analyst override path that reverses a block and records the correction
- Implement fail-open behaviour: if the model errors or times out, traffic passes rather than being dropped
- Add a rate limiter so a model malfunction cannot mass-block the network
- Measure the true latency cost of inline placement

### 13.5 FS-5 — Feedback loop and drift-triggered retraining (2%)

**Why deferred.** Requires deployment history, which does not exist before Phase 2.

**Phase 2 work.**
- Log analyst overrides and confirmed detections as new labelled examples
- Implement drift detection on the feature distribution (population stability index or KL divergence against the training distribution)
- Trigger retraining when drift exceeds a threshold
- Version models and support rollback
- Address the noisy-label problem: analyst corrections are themselves imperfect (Anderson & McGrew, 2017)

### 13.6 FS-6 — Adversarial hardening (1%)

**Why deferred.** Phase 1 *measures* adversarial degradation; Phase 2 *reduces* it.

**Phase 2 work.**
- Iterative adversarial retraining following Novo & Morla (2020)
- Distinguish generated adversarial samples (feature space) from crafted ones (physically realisable packets) — the paper's central finding is that real evasion is much harder than feature-space results suggest
- Evaluate against the DReLAB benchmark
- Report the arms-race equilibrium rather than claiming robustness

### 13.7 FS-7 — Analyst interface (1%)

**Phase 2 work.** A web dashboard showing live risk scores, alert queue with SHAP explanations rendered readably, one-click override, and host-level risk aggregation (following Apruzzese et al., 2017, on host-level rather than flow-level output).

### 13.8 Phase 2 summary

| ID | Item | Weight | Blocker |
|---|---|---|---|
| FS-1 | TLS metadata and JA3 | 8% | Requires PCAP reprocessing |
| FS-2 | Live capture and streaming inference | 6% | Requires network testbed |
| FS-3 | DNS, domain age, reputation | 4% | Requires external APIs |
| FS-4 | Inline enforcement | 3% | Requires firewall integration |
| FS-5 | Feedback and retraining | 2% | Requires deployment history |
| FS-6 | Adversarial hardening | 1% | Depends on FS-2 |
| FS-7 | Analyst dashboard | 1% | None — deferred by priority |
| | **Total** | **25%** | |

---

## 14. Open questions

| ID | Question | Owner | Needed by |
|---|---|---|---|
| OQ-1 | Which Kaggle variant retains `ts` and host columns? | Team | Week 1 |
| OQ-2 | What minimum session length yields stable periodicity estimates? | Team | Week 3 |
| OQ-3 | Should fusion weights be learned or fixed? | Team | Week 9 |
| OQ-4 | What alerts-per-day budget should calibration assume? | Team + supervisor | Week 9 |
| OQ-5 | Is CTU-13 cross-validation feasible within the timeline? | Team | Week 11 |

---

## Appendix A — Glossary

| Term | Definition |
|---|---|
| Beacon | A periodic outbound connection from a compromised host to a C2 server |
| C2 | Command and Control — the infrastructure through which an attacker directs compromised hosts |
| Jitter | Deliberate randomisation of beacon intervals to defeat periodicity detection |
| JA3 | A hash fingerprint of a TLS client's handshake parameters, identifying the client software |
| Flow | A unidirectional or bidirectional sequence of packets sharing a 5-tuple |
| Zeek | A network analysis framework producing structured connection logs from packet captures |
| `conn.log` | Zeek's connection-level log; the primary Phase 1 data source |
| PR-AUC | Area under the precision-recall curve; the correct primary metric under class imbalance |
| SHAP | SHapley Additive exPlanations — a method attributing a model's output to input features |
| Visibility tier | A grouping of features by how much encryption degrades their availability |
| ECH | Encrypted Client Hello — a TLS extension hiding the destination server name |
| DoH | DNS over HTTPS — encrypts DNS lookups, removing them from network visibility |

## Appendix B — Key references

1. Abu Talib, M. et al. (2022). *APT beaconing detection: A systematic review.* Computers & Security, 122, 102875.
2. Anderson, B. & McGrew, D. (2016). *Identifying encrypted malware traffic with contextual flow data.* ACM AISec.
3. Anderson, B. & McGrew, D. (2017). *Machine learning for encrypted malware traffic classification.* ACM SIGKDD.
4. Arp, D. et al. (2022). *Dos and don'ts of machine learning in computer security.* USENIX Security.
5. Alageel, A. & Maffeis, S. (2022). *EarlyCrow: Detecting APT malware C2 over HTTP(S).* Springer ISC.
6. Fu, C. et al. (2023). *Detecting unknown encrypted malicious traffic via flow interaction graph analysis.* NDSS.
7. Garcia, S., Parmisano, A. & Erquiaga, M.J. (2020). *IoT-23: A labeled dataset with malicious and benign IoT network traffic.* Zenodo.
8. Hu, X. et al. (2016). *BAYWATCH: Robust beaconing detection.* IEEE/IFIP DSN.
9. Känzig, N. et al. (2019). *Machine learning-based detection of C&C channels.* IEEE CyCon.
10. Novo, C. & Morla, R. (2020). *Flow-based detection and proxy-based evasion of encrypted malware C2 traffic.* ACM AISec.
11. Ramos, R. & Wang, X. (2023). *Detecting stealthy Cobalt Strike C&C activities from encrypted network traffic.* Springer MLN.
12. Sommer, R. & Paxson, V. (2010). *Outside the closed world: On using machine learning for network intrusion detection.* IEEE S&P.

---

*End of document.*