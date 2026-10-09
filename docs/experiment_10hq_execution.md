# 10H-Q execution and restoration

This run continues the reconstructed parent at `6afb002`, using the unchanged
frozen Q protocol and learned N libraries. Original historical M–P artifacts
are not claimed to have been recovered. Confirmation seeds remain sealed.

Run from the repository root with NumPy, SciPy and pandas installed. The numerical
environment is recorded in `results/tables/experiment_10hq_environment.json`.
Keep BLAS threads at one for every command below.

```bash
export PYTHONPATH=code/src:code/experiments
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
python code/experiments/experiment_10hq_preflight.py
python code/experiments/experiment_10hq_restore_trajectories.py --experiment 10hp --workers 4
python code/experiments/experiment_10hq_parent_overlap.py --workers 4
python code/experiments/experiment_10hq_feedback_bottleneck.py --workers 4
python code/experiments/experiment_10hp_result_audit.py --mode 10hq --workers 4
```

The overlap precheck regenerates all 1,600 P cases using Q's oracle-enabled design
setup and requires exact equality of every stored array. Q's runner resumes
completed seed/panel jobs automatically after checking their source and table
hashes. It does not require refitting M/N or changing P's frozen manifests.

Push Q's completed execution tables and manifests before running post-execution
audits. If an interrupted job has no completion manifest, rerun that job through
the same entry point. Completed jobs remain locked.

Ignored trajectories can subsequently be restored without changing the frozen
metrics or manifests:

```bash
python code/experiments/experiment_10hq_restore_trajectories.py --experiment both --workers 4
python code/experiments/experiment_10hq_restore_trajectories.py --verify-regeneration --experiment both --workers 4
```

Restoration checks exact committed NPZ hashes and stops if the numerical
environment produces different bytes. Saved-step integrity, independent
full-path sensitivity, and exact parent overlap are separate checks. Their
original tolerances remain unchanged.
