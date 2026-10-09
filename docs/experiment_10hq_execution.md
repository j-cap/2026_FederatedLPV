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
python code/experiments/experiment_10hq_restore_trajectories.py --experiment both --workers 4
python code/experiments/experiment_10hq_parent_overlap.py --workers 4
python code/experiments/experiment_10hq_feedback_bottleneck.py --workers 4
python code/experiments/experiment_10hq_restore_trajectories.py --verify-regeneration --experiment both --workers 4
python code/experiments/experiment_10hq_result_audit.py --workers 4
python code/experiments/experiment_10hq_reporting.py
python code/experiments/experiment_10hq_final_verification.py
python code/experiments/experiment_10hq_plot.py
```

These commands reproduce checks from a checkout containing the committed Q
execution tables. Restoration is necessary before auditing completed jobs,
because the runner reuses their locked tables without regenerating ignored NPZs.
For the original fresh execution, P alone was restored before running Q.

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

The Q audit entry point includes the saved-step/full-path audit, followed by an
independent common-input KF replay and steering projection calculation. Compile the living report
with `make -C report`; its checked PDF is also committed.

Current status: all 4,800 Q runs are finite and all 1,600 P overlaps are exact.
Saved-step and projection integrity pass. Strict alternate-order full-path
replay passes for 4,684 records, with 116 sensitive records retained. Paired
contrasts use two donor deployments per recipient and five fleets as the paired
units. Oracle interventions remain diagnostics, not deployable selected methods.
