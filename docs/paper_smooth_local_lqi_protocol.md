# Smooth Local LQI comparison

## Question and frozen protocol

Does the development-selected recipient-only curvature prior close the downstream
tracking gap to federated grouped maps? This supplement uses exactly the ten
10F confirmation fleets (411–420), their original calibration RNG stream,
36 reset transitions at each of two/three local speeds, L=7 QR basis, and
alpha=1e6 with numerical ridge 1e-16. The smoothness strength was selected on
321–330, not these confirmation fleets. C11 weights, gain/prefilter interpolation,
12-second moderate/hard demands, available noise-free feedback, constraints,
and anti-windup remain frozen. Smooth Local is evaluated over the full speed
interval; it does not receive Restricted Local's nearest-observed-speed rule.
No new controller or smoothness selection is performed.

The historical 10F source/results are read-only. The supplement regenerates
all five parent methods and verifies every per-client metric, group count, and
feasibility flag against archived confirmation records (absolute tolerance
1e-8). Only a verified fleet is checkpointed. This regeneration is needed because
10F saved metrics but not its targets/models. Identical target regeneration is
saved alongside Smooth Local maps, controller schedules, and ExactLPV reference
maps. Files and dependencies are protected by normalized-LF SHA256 locks.

Synthesis and nonfinite-rollout failures remain explicit records. A fleet RMSE
is missing if any client fails; comparisons with incomplete fleets receive no
gain estimate. Finite constraint violations remain in the RMSE analysis and
are counted separately. Frozen-point spectral radius is a diagnostic, not a
scheduled or nonlinear stability certificate. No fallback or retuning is used.

## Run from the repository root in Windows CMD / Anaconda Prompt

```bat
git pull --ff-only
conda activate federated_lpv
set OPENBLAS_NUM_THREADS=1
set OMP_NUM_THREADS=1
python code\experiments\paper_smooth_local_lqi.py preflight
python code\experiments\paper_smooth_local_lqi.py smoke
python code\experiments\paper_smooth_local_lqi.py run --workers 5
```

Smoke reproduces the full fleet identification but evaluates one client on both
maneuvers with all parent controllers and Smooth Local. It checks the simulator
against 10F and writes/restores NPZ trajectories in temporary storage, verifying
array hashes after reopening. It creates no production result files. The full
run is resumable; completed JSONs are checked before reuse, and stale checkpoints
are rejected. Never move aside a stale checkpoint without first understanding
its provenance mismatch. Reduce workers if memory/CPU usage is excessive.

```bat
python code\experiments\paper_smooth_local_lqi.py status
python code\experiments\paper_smooth_local_lqi.py summarize
python code\experiments\paper_smooth_local_lqi.py restore-data --seed 411
```

Restoration uses saved schedules and deterministic plants, without fitting or
selection. It checks all restored array hashes against checkpoints and reopens
each NPZ to check persisted values. Output is ignored under
`results/data/paper_smooth_local_lqi/`. This restoration path is exercised in smoke.

## Outputs and analysis

Tracked `results/tables/paper_smooth_local_lqi_*` JSON/CSV files contain:
- ten complete fleet checkpoints with targets, maps, schedules, rows, and hashes;
- per-client metrics and per-fleet means, failure/infeasibility counts, and rho;
- paired Federated-versus-Smooth/Global/Local/ExactLPV gains, absolute differences,
  fleet-bootstrap intervals (20,000 draws), and wins;
- a manifest locking checkpoints and generated summaries.

The experimental unit is the fleet. Both maneuvers are reported separately;
intervals are descriptive and pointwise. This is an additional comparison on the
existing confirmation cohort, not independent confirmation. Existing manuscript
results are not replaced until these outputs have been audited and interpreted.

After execution, commit and push the execution checkpoint before further audits:

```bat
git add code/config/paper_smooth_local_lqi.json code/experiments/paper_smooth_local_lqi.py code/tests/test_paper_smooth_local_lqi.py docs/paper_smooth_local_lqi_protocol.md results/tables/paper_smooth_local_lqi_*.json results/tables/paper_smooth_local_lqi_*.csv
git commit -m "Run frozen Smooth Local LQI comparison"
git push origin main
```
