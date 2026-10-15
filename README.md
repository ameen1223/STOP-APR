# STOP-APR

**Preservation-Constrained Sequential Stopping for Agentic Pull-Request Review**

This repository provides the code, experiment definitions, held-out numerical outputs,
figures, and verification information for the STOP-APR study. The repository contains
implementation and reproducibility materials only; **it does not contain the manuscript**.

## Experimental scope

- 9,217 decision-ready historical pull requests (PRs), from 1,294 repositories
- 53,776 observable interaction states
- Empirical historical-merge preservation targets: 0.90, 0.95, and 0.98
- Three repository-disjoint splits (seeds 2026, 2027, 2028)
- Strict temporal transfer, five leave-one-agent-out tests, component comparisons,
  calibration-margin and cohort-exposure analyses
- 2,000 repository-cluster bootstrap samples for paired comparisons

For the repository-disjoint evaluation, 8 of 9 STOP-APR operating points met their
held-out preservation target, and 7 of 9 had larger avoided-event rates than their
validation-selected comparator. Temporal transfer met 0 of 3 targets. These are
**retrospective evaluations of recorded downstream review/comment events**, not
measured savings in developer time or causal outcomes under deployment.

## Repository contents

| Location | Content |
| --- | --- |
| `scripts/` | Frozen data-building and evaluation implementations (02, 06, 07, 08), PowerShell drivers, and plot scripts |
| `outputs/` | Held-out comparisons, ablations, transfer diagnostics, calibration and cohort sensitivity, integrity summaries |
| `figures/` | Six publication figures, including the workflow schematic |
| `requirements.txt` | Python dependencies |
| `DATA.md` | Original data locations, four expected input filenames, dataset SHA-256 verification |
| `SHA256_MANIFEST.tsv` | Hashes for every included non-manifest file |

## Prerequisites and input data

The experimental drivers target **Windows PowerShell and Python 3.11**, with a
project root at `D:\IJIES_STOP_APR`. Scripts intentionally preserve the tested paths
and evaluation logic. For other operating systems, adapt the launch paths without
changing the experimental decision rules.

Obtain the original public AIDev corpus from its data providers. See [DATA.md](DATA.md)
for links and the exact expected filenames. **Do not upload source Parquet tables to
this repository.** The source snapshot must match the recorded SHA-256 digests for
exact replication; newer AIDev versions can differ.

Directory layout on Windows:

```text
D:\IJIES_STOP_APR\
├── scripts\
├── data\      # four input Parquet files (locally only)
├── logs\      # generated outputs (locally only)
├── .venv\     # Python 3.11 environment (locally only)
└── requirements.txt
```

Set up the environment in PowerShell:

```powershell
py -3.11 -m venv D:\IJIES_STOP_APR\.venv
D:\IJIES_STOP_APR\.venv\Scripts\python.exe -m pip install -r D:\IJIES_STOP_APR\requirements.txt
```

Place the dataset files in `D:\IJIES_STOP_APR\data`, then run:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "D:\IJIES_STOP_APR\scripts\07_STOP_APR_LeakageFreeAblation.ps1"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "D:\IJIES_STOP_APR\scripts\08_STOP_APR_FinalEvidence.ps1"
```

The scripts write fresh results to the local `logs/` directory. Compare these with
archived `outputs/` records; `outputs/08_final_evidence_summary.json` reports the
12/12 primary protocol reproduction test and source/data SHA-256 checks.

## Key archived outputs

- `outputs/07_corrected_comparison.csv`: primary held-out repository, temporal and agent-family policy comparison.
- `outputs/07_component_ablation.csv`: outcome-only and burden-only comparisons.
- `outputs/07_all_policy_results.csv`: policy-level metrics and empirical target attainment.
- `outputs/08_frozen_reproduction.csv`: direct primary-protocol agreement.
- `outputs/08_calibration_sensitivity.csv`: preservation-margin trade-off.
- `outputs/08_temporal_covariate_drift.csv`: observed temporal covariate and score change.
- `outputs/08_single_event_cohort_sensitivity.csv`: one-interaction exposure analysis.

Run `python scripts/plot_stopapr_figures.py` from the repository root to regenerate
Figures 2–5 from archived CSV outputs. Run
`python scripts/plot_calibration_figure.py` to regenerate Figure 6.
Figure 1 is an explanatory workflow schematic included as a publication asset.

## Replication and interpretation

STOP decisions use current observed PR-state fields and model predictions, never
true future remaining-event counts or terminal merged labels. Recorded later events
and historical merged outcomes are used to score the first selected STOP retrospectively.
The validation preservation condition does **not** constitute a formal guarantee of
preservation on new data. All reported targets refer to historical-merge preservation.

## Source and citation

This work uses the public **AIDev: Studying AI Coding Agents on GitHub** corpus by
H. Li, H. Zhang and A. E. Hassan. See [DATA.md](DATA.md). The repository is the
manuscript-facing code and evidence package tagged `v1.0.0` upon publication of the
GitHub Release. Cite the research article using its final bibliographic details when
available; no article DOI is asserted here.
