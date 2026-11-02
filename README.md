# Label efficient calibration for cross dataset IoT intrusion detection

Public repository: https://github.com/haryoko-beep/jjkk-label-efficient-calibration

Research code and aggregate evidence supporting **Limits of Label Efficient Calibration in Cross Dataset IoT Intrusion Detection** by Haryoko, Windha Mega Pradnya Dhuhita, Agus Purwanto, Nuraini, Ikmah, and Lukman.

The experiments investigate when a frozen intrusion detector can be adapted using a limited target-label budget. They compare acquisition policies, inverse-inclusion weighting, budget-accounted thresholds, and retraining. The original 100-label calibrators produce near-unit false-positive rates. Mixed retraining reaches the joint criterion on the evaluated common-feature CIC subset but does not do so consistently on the expanded IoT-23 cohort. The code preserves these negative findings.

## What is included

- `scripts/`: original acquisition, preprocessing, fitting, and analysis scripts, plus release verification utilities.
- `evidence/results/`: unchanged historical CSV/JSON outputs, kept separate from rerun outputs.
- `evidence/logs/`: derived scalar event summaries; the provenance file records hashes of the original logs. These are not full query-level logs.
- Four fixed experiment protocols and the feature-semantics audit.
- `data/official_manifest.json`: provider URLs and original download checksums; no raw data.
- `docs/`: reproduction instructions, publication guidance, and proposed data-availability wording.

No manuscript, author registration form, cookies, raw packet capture, feature matrix, fitted model, or prediction array is distributed. Original scripts are copied byte for byte; the release adds documentation and inspection utilities without changing the experiment protocol. Relative paths are resolved from the repository root.

## Inspect the published evidence first

Use Python 3.12 (the recorded full-study run used 3.12.10). In a terminal opened at the repository root:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe scripts/verify_release.py
.venv\Scripts\python.exe scripts/reproduce_main_results.py
```

On macOS/Linux, replace `.venv\Scripts\python.exe` with `.venv/bin/python`. The checksum utility uses only the standard library. The evidence summary requires pandas and does not fit models or download data. The requirements are the recorded project versions; a fresh online dependency installation has not been validated as part of this packaging step.

## Reproduce the experiments

See [REPRODUCING.md](docs/REPRODUCING.md). Full reproduction requires obtaining official datasets, scanning tens of millions of CSV rows, and running the complete models. The repository does not bundle caches or claim that the full pipeline can run without provider access. No new training was performed when preparing this release.

## Interpretation

Historical target-test reuse is exploratory, not prospective deployment validation. Seeds measure acquisition variability within the evaluated captures. Packet-window extraction and label eligibility define a distinct evaluation task. Exact floating-point thresholds can depend on parallel RF reductions; see the fixed-membership controls and supplementary methods.

## Citation and licensing

Author metadata is in `CITATION.cff`. Add the public repository URL and version DOI after publication. The MIT license covers the supplied code and documentation. Dataset access and reuse remain governed by their providers; aggregate evidence is supplied as a research record and does not transfer rights to underlying datasets.
