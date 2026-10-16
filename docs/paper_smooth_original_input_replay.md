# Resolution: original-input Smooth Local replay

## Controlled result

The Windows export in execution-reference commit 0d0192d recovers the original
scenario input hashes for every fleet. Controlled replay code is checkpointed at
5cb9058, all ten replay outputs at 3264493, and their audit at 8ad22d0.
The exported numerical parameters equal the seed-generated parameters for all
300 vehicles. Saved controller schedules are used directly; no identification,
Riccati synthesis, retuning, or model selection is performed.

Using the exported time/speed/curvature arrays and numeric plant parameters on
the audit runtime reproduces all seven archived array hashes for every one of
600 trajectories: time, speed, reference, state, integral, steering, and raw
steering. Every saved Smooth Local tracking metric and feasibility flag is also
exactly equal to its original execution value. Moderate and hard fleet-mean
RMSEs are unchanged at 8.3565455162 and 5.7160943471 deg/s; sideslip violations
remain 88/300 and 65/300. The historical outputs and manuscript Table V are not
replaced by the replay.

## What this identifies

Regenerating the prescribed profiles in the two environments gives identical
time arrays but speed differences of at most 3.5527136788e-15 m/s; derived
curvatures also differ at floating-point precision. With exported original
profiles, the discrepancies disappear completely in this comparison. Therefore
profile regeneration accounts for the previously observed replay discrepancy;
re-synthesized gains or different numeric plant parameters are not required to
explain it. This does not identify a particular libm/SIMD implementation or
promise bitwise portability on every platform. The controlled ULP probe provides
separate evidence that minute speed/reference changes can amplify in sensitive
responses; it is not an additional statistical validation cohort.

The prior report's unresolved-input limitation is now superseded by this result.
Keep its regenerated-profile replay as a separate numerical sensitivity check.
Do not describe the new result as proving general robustness or scheduled
stability. Smooth Local's fixed-protocol feasibility failure remains the finding.

## Reproduction and verification

From the repository root in the pinned environment:

```bat
python code\experiments\paper_smooth_original_input_replay.py preflight
python paper\analyze_smooth_original_input_replay.py
python paper\check_original_replay_integration.py
python code\experiments\paper_smooth_original_input_replay.py restore-data --seed 411
```

Preflight validates original input hashes. Analysis recomputes fleet summaries
without rollouts. The integration checker independently verifies checkpoint and
source hashes, all 600 trajectory matches, every original metric, and the speed
difference used by the manuscript. Restoration loads the exported numerical
inputs and saved schedules, checks arrays against the new checkpoints (which
match original execution hashes), writes ignored NPZ files, and reopens them to
check persisted hashes. Fleet 411 restoration passed for all 60 trajectories.
A different runtime may still fail a strict hash check; do not weaken it.

To reproduce all replay checkpoints from scratch, use `run --workers 5` with
`paper_smooth_original_input_replay.py` in an explicitly separate checkout/output
archive. Existing valid checkpoints resume; do not silently overwrite them.

## Manuscript integration

Section VI replaces the former one-fleet unresolved-replay paragraph with the
all-fleet exact-input replay result, measured speed difference, and numerical
input archival. Generated macros and a separate evidence lock keep these claims
traceable. The primary identification results, controller comparison, and Table V
remain unchanged. Small PDF corrections from the latest review remain separate.
