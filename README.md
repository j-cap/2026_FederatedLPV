# Federated LPV Identification and Control

This repository studies whether measurable operating-point variation and persistent
client heterogeneity should be represented separately in federated control. The
initial benchmark uses lateral vehicle dynamics: longitudinal speed is the LPV
scheduling variable, while mass, inertia, and tire parameters define persistent
vehicle families.

The first milestone is deliberately an **oracle feasibility study**. Before adding
federated learning, limited data, or learned clustering, it tests whether a small
number of family-specific LPV models and scheduled controllers is useful at all.

## Repository layout

```text
code/       Python package, experiment entry points, configurations, and tests
results/    Reproducible outputs; generated data are not committed by default
report/     Living LaTeX development report and bibliography
```

## Scientific comparison

The core two-factor comparison is:

| Model | Speed scheduling | Structural specialization |
|---|---|---|
| Global LTI | No | No |
| Oracle clustered LTI | No | Yes |
| Global LPV | Yes | No |
| Oracle clustered LPV | Yes | Yes |
| Gridded LTI oracle | Discrete | Yes |

See [`report/main.tex`](report/main.tex) for the motivation, benchmark definition,
experiment plan, decision gates, and the progressively updated findings.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e "./code[dev]"
python -m unittest discover -s code/tests
python code/experiments/phase0_validate_plant.py
```

Generated experiment artifacts are written below `results/`.

The latest Phase-4 studies are `experiment_4e_identification_diagnosis.py` and
`experiment_4f_structured_identification.py` in `code/experiments/`. Run them with
`PYTHONPATH=code/src python code/experiments/<script>.py` from the repository root.
4E diagnoses the earlier failed redesign. 4F tests structured parameter-ratio
identification. The report preserves both the negative results and the initial
recovery, including its single-fleet and changed-baseline limitations.

Experiment 4G validates the frozen structured method on ten independent
train/test fleet pairs and three held-out scenarios, including direct LTI fits.
Run `PYTHONPATH=code/src OPENBLAS_NUM_THREADS=1 python code/experiments/experiment_4g_independent_validation.py`.
The frozen protocol is in `code/config/experiment_4g.json`. Per-seed client,
parameter, and stability CSVs preserve all runs. Aggregate comparisons use seed
pairs as independent units. Use `--summarize-only` to rebuild summaries and the
figure from those CSVs.
