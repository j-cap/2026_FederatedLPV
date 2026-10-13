# Review point 5: focused coverage and calibration robustness experiment

Status: protocol and executable code prepared; the ten-fleet production run
and manuscript integration are pending. Preparation smoke tests use seed
1200 only. Production seeds 1201--1210 have not been fitted in preparation.

Entry point: `code/experiments/paper_robustness_validation.py`.
Frozen configuration: `code/config/paper_robustness_validation.json`.

## Questions and controlled conditions

1. Does the selected partition retain scheduling information when a compatible
   subgroup loses coverage, although the fleet still covers the whole grid?
2. What happens when the fleet itself cannot supply all seven scheduling
   directions, and how does that bound a Global fallback?
3. How sensitive are the fixed estimators to the calibration assumptions:
   excitation, next-state noise, exact current regressors, and correct centers?

Each of ten fresh 30-client fleets is evaluated under seven predeclared
conditions, giving 70 resumable fleet/condition checkpoints. Conditions
within a fleet are paired; the ten fleets are the statistical units.

| Condition | Change from the reference | Intended information check |
| --- | --- | --- |
| `reference` | Original label-free balanced block allocation and calibration constants. | Fleet rank seven. |
| `group_coverage` | All heavy-family clients observe the first two or three speeds of [10,12,14], matching each client's original speed count. Other clients retain their original blocks. | Known-family heavy group rank three with ten clients; fleet union retains all eleven speeds and rank seven. Learned-group ranks are measured, not forced. |
| `fleet_coverage` | Map the four original blocks to [10,12,14], [16,18,20], [10,12,14], [16,18]. | Fleet union has six distinct speeds and rank six, so every selected seven-function group is deficient. |
| `weak_excitation` | Halve the state and steering perturbation standard deviations; retain next-state noise. | Calibration signal-to-noise sensitivity at unchanged scheduling support. |
| `target_noise` | Double next-state observation-noise standard deviations. | Output-noise sensitivity with identical current regressors/physical transitions. |
| `regressor_noise` | Add current beta/yaw noise of 0.01 degrees / 0.03 degrees/s and steering measurement noise of 0.01 degrees. | Errors-in-variables sensitivity; physical transitions and next-state observations stay paired with the reference. |
| `center_bias` | Bias assumed beta/yaw/steering centers by magnitudes 0.05 degrees / 0.2 degrees/s / 0.1 degrees, with independent reproducible +/- signs per component and client, constant across its speeds. | Centering/preprocessing sensitivity with identical physical/measured records. |

Every condition keeps 36 one-step records per observed speed and each
client's original two- or three-speed count. Thus the recipient budget remains
72 or 108 transitions, including in both coverage interventions. There is
no simultaneous change of basis order, plant, fleet scatter, calibration
curvature, grouping settings, minimum size, or ridge strengths.

Only the stress generator uses known families to restrict heavy-client
coverage. The learned fitting/selection interface receives speed and matrix
targets, not families or physical parameters. OracleFamily is a separate
known-family diagnostic constructed after label-free model selection. It
shows the deficiency of a compatible partition even if the learned method
merges groups and thereby avoids deficient selected maps.

The fleet-wide restriction explicitly tests a case where no partition can
recover the missing direction. This condition remains in all outputs even
if the grouped and Global predictions fail badly. No favorable seed, group
count, sensitivity level, or result is selected after execution.

## Pairing, models, and endpoints

The fleet seed determines physical parameters and baseline block allocation.
Independent reproducible RNG streams for each client and speed-slot index
generate state/input perturbations, next-state noise, current measurement
noise, and client center biases. Condition names do not seed these draws.
Coverage changes therefore preserve the latent draws and recipient budgets;
calibration changes reuse the appropriate underlying random variates.

All physical records are generated at the true benchmark turning equilibrium
on curvature 0.005 1/m. For biased centering, only the assumed preprocessing
center changes: the measured arrays remain those of the reference condition.
The same assumed state center is subtracted from current and next-state
observations, and its steering center is subtracted from measured steering.
Local matrix regression retains the existing zero-intercept formulation.

The six methods are:

- Restricted Local: the existing L3 recipient-only ridge estimate.
- Smooth Local: the existing L7 curvature prior, with alpha=1e6 and numerical
  ridge 1e-16 from the already frozen development-only selection. No retuning.
- Global: the existing L7 fleet-statistic fit.
- Federated: the original BIC-selected label-free grouped fit (K=1--5,
  twelve restarts for K>1, 30 iterations, minimum size five, ridge 1e-8).
- RankFallback: retain full-rank selected group maps; otherwise give affected
  recipients the existing Global map. The rule checks rank only and is applied
  after original grouping. Global deficiency is explicitly flagged. This is
  a diagnostic safeguard, not a performance or stability guarantee.
- OracleFamily: a known-family L7 pooling diagnostic on the same records.

The fixed-coordinate relative six-entry matrix error is evaluated on the
same eleven speeds 10,12,...,30 m/s. Full-grid error is the main fixed-domain
endpoint. Recipient-observed/unobserved regions are secondary; their eligible
sets follow each condition's actual support. Across coverage conditions,
paired unobserved-error changes compare this endpoint, not identical held-out
speed pairs. Fleet-unobserved error is additionally reported where such pairs
exist; absent regions are omitted rather than reported as zero error.

Group tables record stacked scheduling rank, full-rank status, unique speeds,
singular values, design condition when full rank, and nonzero-spectrum
condition otherwise. Local transition-regressor ranks, singular values,
condition numbers, and residual RMS are saved for calibration diagnosis.
The observed speed design is unchanged by measurement noise; regressor
quality and scheduling information are distinct checks.

Method gains average paired fleet-relative improvements, with 20,000
fleet-bootstrap draws. Predeclared comparisons are Federated vs Global,
Federated vs Smooth Local, and RankFallback vs Federated, for full-grid and
recipient-unobserved endpoints in each condition. All intervals are pointwise
descriptive robustness intervals; no multiplicity-adjusted joint claim is
implied. Paired sensitivity tables also record error changes relative to the
reference in percentage points, their mean and fleet-bootstrap intervals,
and the number of deteriorating fleets. No controller or representation is tuned.

## Execution, provenance, and saved artifacts

The preflight checks historical fitting/calibration sources and the frozen
Smooth Local selection, software pins, seed separation, recipient budgets,
and planned ranks. Text hashes normalize CRLF to LF for Windows; substantive
changes are rejected. The new runner and protocol are included in every
checkpoint's source lock. Processes use the spawn start method and one
numeric-library thread per worker.

Each atomic JSON checkpoint contains measured matrix targets, speed support,
physical reconstruction parameters, true/assumed calibration centers,
record-array hashes, regression diagnostics, all model coefficients,
assignments, candidate BIC rows, group information, reference arrays, and
client-speed errors. A canonical payload checksum detects accidental edits
before resuming. Negative results and deficient groups are retained.

Checkpoints and summaries use the prefix `paper_robustness_validation_` under
`results/tables/`; historical results are not overwritten. The complete
manifest requires all 70 checkpoints. Repeating `--run` validates and resumes
the saved jobs. `--status` is read-only. A changed source or corrupted result
stops execution rather than silently replacing the checkpoint.

The optional `--audit` recomputes reference Jacobians once per fleet, checks
all saved method predictions, scheduling ranks, the rank-only fallback, and
stationarity of the saved local/Global/grouped/oracle coefficients without
refitting them. It writes a separate audit record after execution.

Independent transitions are regenerable and kept out of git. `--restore-data`
recreates the measured arrays using the frozen centers and parameters, checks
every canonical float64 record hash, and writes ignored NPZ files under
`results/data/paper_robustness_validation/`. It performs no matrix fitting,
grouping, or modification of checkpoints. Hashes cover record arrays rather
than ZIP timestamps. Exact reconstruction must pass the committed hashes;
it is not silently treated as approximate restoration.

## Windows Command Prompt / Anaconda Prompt

Run from the repository root in the existing environment:

```bat
git pull --ff-only
conda activate federated_lpv
set "OPENBLAS_NUM_THREADS=1"
set "OMP_NUM_THREADS=1"
set "MKL_NUM_THREADS=1"
python code\experiments\paper_robustness_validation.py --preflight
python code\experiments\paper_robustness_validation.py --smoke --workers 2
python code\experiments\paper_robustness_validation.py --run --workers 5
```

If needed, the environment can be created with
`conda env create -f environment.yml`. The runner also adds the source paths
it needs, so it does not depend on a separately set PYTHONPATH.
The environment now explicitly pins threadpoolctl=3.6.0 as well. If preflight
reports differing package pins in an existing environment, update it with
`conda env update -n federated_lpv -f environment.yml`, then deactivate and
reactivate it before running.

Progress or resume:

```bat
python code\experiments\paper_robustness_validation.py --status
python code\experiments\paper_robustness_validation.py --run --workers 5
```

After the run reports `complete_execution`, push its execution checkpoint
before auditing or editing the manuscript, as required by AGENTS.md:

```bat
git add results/tables/paper_robustness_validation_*.json results/tables/paper_robustness_validation_*.csv
git commit -m "Add focused coverage and calibration robustness results"
git push origin main
```

Subsequent saved-model audit:

```bat
python code\experiments\paper_robustness_validation.py --audit
```

Exact restoration of ignored records, without changing frozen outputs:

```bat
python code\experiments\paper_robustness_validation.py --restore-data --seed 1201 --case reference
```

Omit both filters to restore all 70 completed jobs. Regenerated records are
not required for the main result summaries or saved-model audit.

## Preparation checks and evidence boundary

The unit tests cover reference-calibrator parity, rank-deficiency despite
minimum group size, equal coverage budgets, common-random-number pairing,
centering-only versus physical-record changes, current measurement noise,
the deterministic fallback rule, CRLF-safe provenance, checkpoint corruption,
partial-summary rejection, and exact no-refit record restoration.

The full separate-seed smoke exercises all seven conditions, spawned
workers, atomic checkpoint loading, per-method saved-model audits, ignored
record restoration, and the summary/independent-reference audit paths in a
temporary directory. Its one-fleet summaries are software checks, not new
manuscript evidence. Native Windows execution remains the user's local run.

This focused study extends the controlled available-state benchmark. Noise
and center-bias interventions do not establish ordinary-driving calibration,
an observer, real-fleet performance, frozen-library transfer, or scheduled
stability. Existing manuscript results remain frozen. Point 5 remains open
until the local production results are pushed, audited, and interpreted.

Preparation verification: all 13 targeted tests pass; the full seven-condition
spawn-based smoke and independent-reference audit pass, including restoration
of every measured record cell. The three existing manuscript-evidence checks
also pass. Production status remains 0/70 jobs. Only new protocol/runner/tests,
this guide, the README handoff, and the explicit environment dependency pin
are changed; the manuscript PDF and historical scientific evidence are intact.
