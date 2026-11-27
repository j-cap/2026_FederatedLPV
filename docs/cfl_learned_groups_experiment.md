# Learned compatible priors for heterogeneous vehicles

This experiment tests whether training-data-based groups retain the useful
oracle-family advantage and whether complementary-speed peers help more than
equally sized same-speed data. The existing three fleets are development data.
Ten fresh fleets are reserved for confirmation after development review.

## Frozen setup

Compact, Sport and SUV parameters and all variations come from the completed
CFL-FF heterogeneity diagnostic. The same matched four-state tanh/relaxation
plant, known per-vehicle geometry, sample time, integration, steering inputs,
noise, bounds, starts, nominal center and penalty weights are inherited.
There are 18 physical vehicles per fleet, six per family. Each vehicle provides
one 12-second steering/yaw record. Steering and speed are known; latent states
and physical generating parameters do not enter fitting.

The pinned configuration is `code/config/paper_cfl_learned_groups.json`.
It locks the original protocol, numerical fitting code, and 129 source
checkpoints by hashes. It also freezes the ten confirmation fleets (seeds
2401–2410). Source results and the manuscript remain untouched.

## Two conditions, matched target records

| Condition | Local training records | Evaluated vehicles |
|---|---|---|
| `complement` | Two vehicles per family at each of 13, 21 and 29 m/s | All 18 |
| `same13` | The same 18 physical vehicles, each measured only at 13 m/s | The six vehicles originally measured at 13 m/s |

Both conditions use 18 records and 216 seconds of measured data per fleet.
For the six paired low-speed targets, the exact local arrays, noise, LocalOE
fit and nominal-prior fit are reused. Only the other vehicles' training speeds
change. The same two steering probes at 21 and 29 m/s define their paired
coverage endpoint. The other twelve `same13` clients supply peer information;
they are not counted as paired targets because their local training data change.

Existing local fits and fixed-group fits in the complementary condition are
explicitly attributed copies of the completed diagnostic. Thus development
requires only 36 new local jobs (three local starts plus two nominal-prior
starts each), 21 new fixed-group jobs and 18 learned-group restart jobs.
There are 1008 development test records across both conditions and all seven
test speeds; 216 complementary-condition records define the primary endpoint.

## Methods and information boundary

| Method | Sharing rule |
|---|---|
| LocalOE | No prior |
| NominalPrior | Fixed independent population-design center |
| SharedPrior | One learned center across all 18 clients |
| MixedGroups | The frozen three stratified mixed partitions |
| OracleFamily | The three known physical families, diagnostic only |
| LearnedGroups | Three centers and assignments learned from local updates |
| Exact | Generating parameters, evaluation reference only |

Every vehicle keeps its own six physical parameters. The local objective is
normalized yaw simulation-error SSE plus

\[
\left\|W\bigl(\log\theta_i-\eta_{c_i}\bigr)\right\|^2,
\qquad W=\operatorname{diag}(10,10,10,0,4,4).
\]

LearnedGroups fixes K=3. Weighted K-means initializes centers from selected
local OE log parameters. The server receives fitted log parameters and scalar
training losses. Each client solves its local problem using its current center;
the server updates nearest-center assignments and centers. The friction
coordinate has zero weight and does not determine assignments. Family labels,
generating parameters and test outputs do not enter initialization, updates,
convergence checks or restart selection.

Three deterministic restarts use seeds 1701–1703. Select the lowest final
penalized training objective among converged restarts, breaking ties by restart
key. No prediction error or family agreement enters selection. All restart
outcomes, including budget exhaustion, remain in the optimization table. If no
restart converges, execution stops with its checkpoints preserved. A nominal
K=3 solution may have fewer active groups; this remains visible rather than
being repaired using family labels. Communication accounting is available for
every restart, including unselected attempts.

The original 80-round and 50-evaluation-per-client-update limits remain fixed.
All condition selections finish before either condition's test outputs are
opened. Family and training-speed ARI are computed only during analysis.

## Endpoints and progression

The primary endpoint is clean yaw prediction RMSE at the two locally unobserved
anchor speeds and two new input probes, averaged first within vehicle and then
within fleet. Local/same-speed performance, interpolation/extrapolation,
families, training-speed strata, vehicle wins and error quantiles are also saved.
Operating failures are retained and prevent a positive screen.

For compatible sharing, development requires at least 5% pooled improvement
over each of SharedPrior and NominalPrior, with improvement in at least two
of three fleets for each comparison. Pooled LocalOE and MixedGroups errors
must also improve. All selected local, fixed-group and learned fits must
converge, and training/test operating checks must pass.

Coverage support is a separate decision: LearnedGroups with complementary
peers must beat LearnedGroups with same-speed peers on the six unchanged
targets, pooled and in at least two fleets. Failure here does not erase a
compatible-sharing result, but it does block an operating-envelope-completion
claim from this test.

Confirmation requires improvement in at least eight of ten fleets for the
two primary controls, plus positive 95% fleet-paired bootstrap intervals on
absolute improvements (10,000 resamples; fixed seed 552910). Coverage support
also requires improvement in eight fleets and a positive interval. The 5%
margin and 8/10 count are practical criteria, not significance tests. Bootstrap
replication units are fleets; the three-fleet development intervals are
descriptive. Report tail errors separately; there is no individual guarantee.

An oracle-gain-retention metric compares the learned reduction against the
shared prior with the corresponding oracle-family reduction. It is reported
only when that oracle reduction is positive; OracleFamily is one privileged
partition rather than an upper bound on possible performance.

## Run development locally

From the repository root in the existing Conda environment:

```bat
git pull --ff-only origin main
conda activate federated_lpv
python code\experiments\paper_cfl_learned_groups.py preflight
python code\experiments\paper_cfl_learned_groups.py smoke
python code\experiments\paper_cfl_learned_groups.py run --workers 5
```

Repeat the final command after an interruption. Jobs run across clients and
group restarts with one BLAS thread each. OS locks prevent competing writers;
each optimizer start, each client update within a round, and each evaluation
case is checkpointed. Resuming validates code, protocol, runtime and parent
hashes. No completed fit is silently rerun or replaced.

Commit the execution checkpoint before report generation:

```bat
git add results/tables/paper_cfl_learned_groups_development_*.json
git diff --cached --stat
git commit -m "Run learned vehicle groups and matched speed control"
git push origin main
```

Then generate and commit the analysis:

```bat
python code\experiments\paper_cfl_learned_groups.py analyze
start report\cfl_learned_groups_development.html
git add results/tables/paper_cfl_learned_groups_development_*.json
git add results/tables/paper_cfl_learned_groups_development_*.csv
git add report/cfl_learned_groups_development.html
git diff --cached --stat
git commit -m "Analyze learned vehicle groups and matched speed control"
git push origin main
```

If execution stops early, commit the JSON checkpoints before reporting the
failure. CSV/report paths do not exist until analysis finishes.

## Confirmation after development review

The default run executes development only. Once the development result has
been reviewed and the compatible-sharing screen passes, freeze the exact
protocol, code, runtime and development outputs:

```bat
python code\experiments\paper_cfl_learned_groups.py freeze
git add results/tables/paper_cfl_learned_groups_confirmation_freeze.json
git commit -m "Freeze learned vehicle groups confirmation protocol"
git push origin main
python code\experiments\paper_cfl_learned_groups.py run --phase confirmation --workers 5
git add results/tables/paper_cfl_learned_groups_confirmation_*.json
git diff --cached --stat
git commit -m "Run ten-fleet learned vehicle groups confirmation"
git push origin main
python code\experiments\paper_cfl_learned_groups.py analyze --phase confirmation
git add results/tables/paper_cfl_learned_groups_confirmation_*.json
git add results/tables/paper_cfl_learned_groups_confirmation_*.csv
git add report/cfl_learned_groups_confirmation.html
git diff --cached --stat
git commit -m "Analyze ten-fleet learned vehicle groups confirmation"
git push origin main
```

Missing or stale locks reject confirmation. No development score automatically
starts confirmation, changes settings, or opens its test results.

## Ignored data and exact restoration

The original raw arrays are reused if still present. Keep the existing
`results/data/paper_cfl_heterogeneity` folder on the original machine: it avoids
recreating source measurements and baseline predictions. Missing arrays are
regenerated and must match the recorded hashes; cross-platform regeneration
can differ at floating-point precision even when package versions match. In
that case use the original numerical platform or retain its original arrays.
The recorded source trajectories are never replaced with approximate copies.

To rebuild missing experiment arrays from saved checkpoints, with no refitting
and no changes to JSON manifests or tables:

```bat
python code\experiments\paper_cfl_learned_groups.py restore
python code\experiments\paper_cfl_learned_groups.py restore --phase confirmation
```

Restoration uses only available checkpoints, so either command also supports
an interrupted run. Use the phase you have executed.

## Interpretation limits

This first test learns assignments with a known nominal K=3. It does not select
the number of clusters, guarantee family recovery, test new nonlinear Pacejka
mismatch, or establish closed-loop benefits. All physical assumptions remain
matched to the preceding oracle experiment. Absolute errors and weak friction
identification must accompany percentage improvements.
