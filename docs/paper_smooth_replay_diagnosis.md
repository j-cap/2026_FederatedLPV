# Smooth Local replay diagnosis

## Scope and status

This diagnosis uses original seeds 411–420 and saved Smooth Local schedules.
No identification, DARE synthesis, retuning, or original-result replacement is
performed. Preparation: 100bac8. Replay execution: f6024ea. Audit: 49c5bee.
The manuscript and small PDF corrections are intentionally unchanged.

## Upstream discrepancy

Across all 600 client/scenario records, regenerated time arrays match the
original hashes, but regenerated speed and yaw-reference arrays do not.
These inputs are computed before any controller or plant integration. Therefore
DARE re-synthesis is not necessary to produce a replay discrepancy, and an
upstream scenario-input mismatch is established. It is not yet established that
this mismatch is the only difference between runtimes.

The original checkpoints contain hashes rather than numeric scenario arrays,
plant parameters, or execution-runtime metadata. A hash mismatch alone cannot
quantify the original input difference or identify a particular math library.
Plant parameters can be regenerated from seeds, but cannot be compared directly
to original saved numeric parameters because those were not recorded. Scalar
cosine and CPU-feature probes did not recover the archived speed hashes here;
this does not isolate the original backend cause.

New replay checkpoints save time, speed, curvature, numeric plant parameters,
package/platform/threadpool metadata, all metrics and component-wise hashes.
Original schedules are reused without modification. Their original provenance
and all original execution files remain protected by source hashes.

## All-fleet outcomes

| Scenario | Original RMSE [deg/s] | Replay RMSE [deg/s] | Original violations | Replay violations | Changed flags |
|---|---:|---:|---:|---:|---:|
| Moderate | 8.356546 | 8.366134 | 88/300 | 88/300 | 0/300 |
| Hard | 5.716094 | 5.715549 | 65/300 | 65/300 | 0/300 |

All 600 rollouts finish and all 600 feasibility flags agree. Aggregate metrics
are close across the two executions; this is a numerical replay check on the
same cohort, not new statistical confirmation. Seven moderate and three hard
client RMSEs differ by more than 1e-6 deg/s. The maximum individual differences
are 2.885241 and 0.167488 deg/s respectively. Original table values remain the
execution endpoint. No confidence interval based on these two numerical replays
is appropriate.

## Controlled sensitivity probe

With the same client heavy_01, seed 411, saved schedule, and moderate scenario,
identical-input repeated simulation matches every array hash. Advancing each
speed value upward or downward by one representable floating-point value changes
speed by at most 3.552714e-15 m/s. Curvature is fixed, so the yaw reference changes
as induced by speed. RMSE changes from 8.323126 deg/s to 5.347988 or 5.170831;
the feasibility flag stays true. This supplies direct evidence of strong numerical
sensitivity for this response. It does not recover the missing original arrays,
prove the precise historical cause, or imply every client is equally sensitive.
The probe is repeatable with:

```bat
python code\experiments\paper_smooth_replay_sensitivity.py
```

## Exact restoration and provenance

The new replay checkpoint for fleet 411 restores all 60 trajectories from saved
numeric inputs and schedules, and reopens every NPZ to verify exact replay hashes:

```bat
python code\experiments\paper_smooth_replay_diagnosis.py restore-data --seed 411
```

This command passed on the diagnosis runtime. It restores the new checkpoint,
not the old execution, and is not a promise of bitwise portability to a different
runtime. NPZ files are ignored under results/data/paper_smooth_replay_diagnosis.
The old strict restoration check remains intact and still rejects mismatches.

Audit summaries can be regenerated without rollouts or fitting:

```bat
python paper\analyze_smooth_replay_diagnosis.py
```

## Remaining original-runtime check

Run the lightweight exporter in the environment/machine used for the original
Smooth Local execution. It performs no fitting or rollouts. It saves regenerated
inputs and parameters, package/platform metadata, and comparisons against the
original input hashes. It refuses to overwrite an earlier export.

```bat
git pull --ff-only
conda activate federated_lpv
python code\experiments\paper_smooth_replay_reference.py
git add results/tables/paper_smooth_replay_execution_reference.json
git commit -m "Export Smooth Local execution-runtime replay inputs"
git push origin main
```

If the original hashes match, these arrays provide a recoverable reference input
for controlled cross-runtime comparison. If they do not, the current machine's
environment no longer reproduces the original inputs; the export records this
rather than claiming recovery. This check is required before attributing the
historical mismatch to a specific backend or quantifying its original size.

## Manuscript implication

The present feasibility finding is supported by agreement across all ten fleets,
not only fleet 411. A future revision can state that all 600 classifications
persist under replay and that aggregate errors are close, while some individual
responses amplify minute input differences. Retain the limitation that original
scenario generation was not bitwise recovered. The small PDF corrections remain
for the next step.
