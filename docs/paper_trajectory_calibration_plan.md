# Continuous calibration trajectories: protocol and execution plan

Status: implementation and local handoff; no scientific outcomes yet.

## Question and scope

Does the benefit of selective fleet sharing survive when local matrix estimates
come from continuous, bounded steering maneuvers instead of independently reset
state perturbations? This is a simulated test-track calibration study at fixed
speed. It is not recorded road data, ordinary-driving identification, an observer
validation, or a road-safety certification. Longitudinal speed is prescribed and
the existing two-state nonlinear tire model is unchanged.

The three questions are deliberately separate:

1. Does the maneuver produce continuous, physically plausible motion within the
   declared limits, at the intended cornering operating point?
2. Can its observations identify useful local A/B matrices, including on a
   different validation steering signal?
3. With the same incomplete local speed coverage, do learned groups still help
   relative to Global and Smooth Local?

Do not replace the manuscript evidence until these questions have been evaluated.

## Maneuver, sensing, and local fitting

Each client/speed experiment starts once from straight motion, with beta=r=0.
A common yaw-rate PI controller gradually establishes the paper's cornering
condition r_ref = v * 0.005 /m. It uses measured yaw rate and nominal wheelbase,
not the client's physical parameters or analytical equilibrium. Steering is
amplitude- and rate-limited. After 20 s, the mean steering over the last 2 s is
held fixed for another 4 s. The final 2 s of that quiet hold provide an empirical
state/input center. The controller is off during the hold and identification
window, avoiding a feedback identification confound in that window.

A 20 s, tapered sum of sinusoids perturbs this constant steering. Frequencies
are 0.2, 0.5, 0.9, and 1.4 Hz. Phases are seeded and paired across amplitude
candidates. Only steering is excited; sideslip and yaw rate always evolve from
the previous simulated state. The sample time remains 0.01 s. Candidate peak
steering deviations at 20 m/s are 0.15, 0.30, and 0.60 degrees, scaled by
(20/v)^2 and capped at 0.8 degrees. This is a prescribed design, not an optimized
or standardized vehicle test.

Each training run therefore lasts 44 s, of which 20 s supplies the fitting
records. A separate 44 s validation run is used for the research diagnostic.
Settling was extended from an initial 12 s engineering prototype to 20 s after
smoke checks found remaining yaw-rate offset in handling-family vehicles.
No pilot or confirmation fleet was used to choose this setup correction.

A separate run uses a tapered 0.15--1.2 Hz linear chirp for validation, with its
own measurement noise and empirically established center. It never supplies
training rows. Both runs start independently; neither resets states within a run.
Use different input data for model estimation and validation, consistent with
[system-identification guidance](https://www.mathworks.com/help/ident/select-data-for-estimation.html).
Sum-of-sinusoids inputs are an established identification input type
([documentation](https://www.mathworks.com/help/ident/ref/idinput.html)); all
frequencies, amplitudes, limits, and acceptance thresholds here are our choices.

The practical sensor condition adds one noise sequence to the state trace and
one to steering. Therefore a measured state used as the next-state target of
one row is exactly the current-state regressor of the next row. Center estimates
use only the quiet measurements. No true center, true A/B matrix, family label,
or physical parameter enters local OLS or learned grouping. Sideslip availability
still assumes instrumented calibration; no observer is introduced. OLS with noisy
regressors is intentionally tested as implemented, not claimed to be unbiased.

## Stages and frozen choices

### A. Technical checks

Preflight verifies source hashes, dependencies, distinct seed sets, and frozen
baseline settings. Unit checks cover continuous propagation, sensor consistency,
non-use of true centers in fitting, bounded steering, affine-linear recovery,
checkpoint corruption detection, and out-of-sample rollout alignment. Smoke
exercises all branches on a separate seed with a small fleet and writes only
temporary outputs. It is not an amplitude-selection experiment.

### B. Pilot: local calibration only

Three fresh fleets (1301--1303), all 30 clients, their assigned two/three speeds,
and all three amplitudes. No group fitting is performed during selection.
Record physical-limit checks, settling error, scaled regressor singular values,
local reference-matrix errors, and validation prediction errors. True simulator
quantities enter evaluation only, after the fits.

Select the smallest amplitude satisfying the predeclared pilot screen:

- Every calibration AND validation trajectory respects |beta| <= 5 degrees,
  |a_y| <= 6 m/s^2, |steering| <= 12 degrees and rate <= 30 degrees/s, with no
  nonfinite states or limiter activation during the identification window.
- Quiet yaw-rate centers are within max(0.001 rad/s, 1% of target) and quiet
  drift is below 0.02 in fixed normalized state units per second in every cell.
- At least 95% of clean local regressions have rank 3 and scaled condition <= 1000.
- At least 95% of clean local fits have relative matrix error <= 5% and
  independent one-second free-run normalized RMSE <= 0.25.
- At least 95% of sensor-condition fits have relative matrix error <= 10% and
  independent one-second free-run normalized RMSE <= 0.50.

Both accuracy screens also require the free-run prediction to beat holding the
initial state constant over the same validation window. Normalization uses fixed
state scales (0.25 degrees sideslip and 2 degrees/s yaw rate), not a data-dependent
variance. Each validation window has a common exact initial state for all models;
this diagnostic isolates model error and does not test state-estimation error.

These are engineering screening thresholds, not statistical guarantees. Report
all candidates and failure fractions. If none qualifies, stop; do not silently
pick the best or loosen thresholds. Diagnose excitation, noise, centering and
local nonlinearity before defining a new version. Pilot measurements are not
confirmation results. Commit the completed pilot before selection analysis,
then commit the selection and its input hashes before confirmation.

### C. Confirmation: frozen sharing comparison

Ten fresh fleets (1311--1320). Preserve the existing family distribution,
label-free speed allocation, seven-function basis, clustering settings, and
previously selected Smooth Local penalty. Fit the following conditions on the
same vehicles and speed blocks:

| Condition | Records per speed | Purpose |
| --- | ---: | --- |
| Reset36 | 36 | Re-run original reset acquisition as a paired reference |
| Trajectory36Bridge | 36 evenly spaced transitions | Match fitted row count, exact regressors and target-only noise |
| TrajectoryAllBridge | 2000 | Effect of using all consecutive transitions under the old noise convention |
| TrajectoryAllClean | 2000 | Diagnose information and local approximation without measurement noise |
| TrajectoryAllSensor | 2000 | Main continuous-data condition, coherent noisy state/input traces |

Both bridge cases use the same empirical clean center and the same target-noise
realization. Their rows are nested. Reset36 uses the original equilibrium and
independent state perturbations. Comparisons with Reset36 therefore change both
acquisition and centering; they are not a pure isolation of temporal correlation.
Thirty-six selected trajectory rows still require collecting the whole maneuver:
matched row count is NOT matched driving time or independent information.

The main comparison is Federated versus Global and Smooth Local under
TrajectoryAllSensor. Restricted Local is also reported. All fitted maps see the
same targets within a condition. Report full/observed/unobserved speed errors,
paired fleet-level relative gains and pointwise 95% bootstrap intervals, selected
K and group ranks. Ten fleets, not transitions or clients, are the independent
units. Keep weak/failed calibration cells in diagnostic counts. A rank-deficient
or nonfinite local fit blocks that condition's fleet fit rather than dropping
clients. Do not replace a failed group fit with another method without labeling it.

Local diagnostics include entrywise absolute error (especially B), state-scaled
one-step RMSE and 1 s free-run RMSE on the chirp, compared with persistence.
Reference matrices are evaluated at the original prescribed cornering curve;
centering mismatch is separately reported. Low one-step error alone is insufficient
at 100 Hz. Full-envelope matrix error is not a controller-performance measure.
No new LQI experiment is included in this first stage.

The analysis distinguishes calibration failure from a missing sharing benefit.
It recommends review for manuscript integration only if the sensor condition
passes the same calibration screen on confirmation fleets and the full-envelope
paired-gain confidence intervals are positive against both Global and Smooth
Local. This is a predeclared decision aid, not automatic acceptance of a claim.
Unobserved-speed results, per-entry errors, client variation, and all other
conditions remain available for interpretation.

## Efficiency and reproducibility

Generate each physical training/validation trajectory once per client/speed and
derive all four trajectory conditions from it. Parallelize across fleet/profile
jobs using Windows-safe spawned workers, with one BLAS thread per worker.
Atomic JSON checkpoints allow resumption and reject changed sources, environment,
protocol, or pilot selection. Worker progress is printed after each fleet job.

Commit configurations, source locks, selection, compact input targets, fitted maps,
per-cell diagnostics, array hashes, CSV summaries, and PDF figures. Full trajectory
arrays are saved as ignored NPZ files under results/data. The restore command
uses saved vehicle parameters and seeded acquisition inputs and verifies every
array hash; it never overwrites frozen checkpoints. If floating-point differences
prevent exact reproduction, restoration fails rather than claiming equivalence.
Keep the original NPZ files locally until analysis is complete.

## Windows commands

Run from the repository root in the existing environment:

```bat
git pull --ff-only
conda activate federated_lpv
python code/experiments/paper_trajectory_calibration.py preflight
python code/experiments/paper_trajectory_calibration.py smoke
python code/experiments/paper_trajectory_calibration.py pilot --workers 5
git add results/tables/paper_trajectory_calibration_*.json
git commit -m "Checkpoint continuous calibration pilot"
git push
python code/experiments/paper_trajectory_calibration.py select
git add results/tables/paper_trajectory_calibration_* results/figures/paper_trajectory_calibration_*.pdf
git commit -m "Analyze pilot and freeze calibration trajectory"
git push
```

Only if selection reports a passing profile:

```bat
python code/experiments/paper_trajectory_calibration.py run --workers 5
git add results/tables/paper_trajectory_calibration_*.json
git commit -m "Checkpoint continuous calibration confirmation"
git push
python code/experiments/paper_trajectory_calibration.py analyze
git add results/tables/paper_trajectory_calibration_* results/figures/paper_trajectory_calibration_*.pdf
git commit -m "Analyze continuous calibration confirmation"
git push
```

Resume with the same pilot/run command. Inspect completion with `status`.
Restore missing ignored trajectories with `restore --stage pilot --workers 5`
or `restore --stage confirmation --workers 5`.

## Implementation verification before handoff

- Ten unit checks passed (`python code/tests/test_trajectory_calibration.py`).
- The separate six-client smoke test exercised all five acquisition conditions,
  all four methods, source validation, checkpoint writing, and Windows-style
  spawned-process execution. The final smoke worker completed in about 9 s in
  the preparation environment; this is not a full-campaign runtime estimate.
- Temporary smoke fixtures exercised pilot selection, confirmation summaries,
  diagnostic PDF generation, checkpoint resume, and deletion/restoration of an
  NPZ file. Restored arrays matched their hashes exactly, and the frozen JSON
  checkpoint was unchanged. Diagnostic plots were visually checked.
- The original 12 s settling prototype missed the intended yaw center in some
  smoke vehicles. The 20 s setup passed the targeted nominal/heavy/handling
  entry checks at 10, 20, and 30 m/s. Full-fleet feasibility remains a pilot test.
- Smoke diagnostics exposed sensitivity of ordinary least squares to coherent
  regressor noise. This was not hidden by selecting on one-step prediction error
  or weakening the thresholds. If all amplitudes fail, inspect local matrix
  errors and noise-free/bridge/sensor differences before considering a different
  estimator. No pilot or confirmation result has been generated during handoff.

The historical experiment files, manuscript, and frozen numerical results are
unchanged by this addition.
