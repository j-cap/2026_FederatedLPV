# CFL-FF heterogeneity: result review

The frozen oracle-family diagnostic passes its predeclared progression screen.
Family-compatible learned priors improve clean yaw prediction at a vehicle's
locally unobserved anchor speeds. This supports testing learned grouping.

## Evidence integrity

All 129 result checkpoints pass their payload checksums, source/protocol hashes,
parent hashes and completion checks. All 108 selected local/nominal-prior fits
and 21 learned-prior jobs converge. All 54 training and 756 test records pass
their operating checks. Exact-model test prediction error is zero.
Checkpoint metrics reproduce the saved case, parameter, vehicle and
fleet tables and the original progression decision. Original result files are
unchanged. This review performs no optimization or simulation.

## Primary result

Each of 54 vehicles contributes four clean-yaw tests: two new steering probes
at each of its two locally unobserved anchor speeds. The 216 test records are
repeated evaluations within three fleets, not 216 independent replications.
Errors below are mean vehicle RMSE, equally averaged across fleets, in deg/s.

| Method | Mean yaw RMSE [deg/s] | Vehicles improved by oracle / 54 |
|---|---:|---:|
| LocalOE | 0.008931 | 40 |
| NominalPrior | 0.007742 | 41 |
| SharedPrior | 0.007777 | 42 |
| OracleFamily | 0.004850 | — |
| MixedGroups | 0.007988 | 41 |

OracleFamily reduces mean error by **37.6% versus SharedPrior**, **37.4% versus
NominalPrior**, **39.3% versus MixedGroups**, and **45.7% versus LocalOE**.
The absolute reduction versus SharedPrior is **0.002928 deg/s**.

| Fleet | Shared prior [deg/s] | Oracle family [deg/s] | Reduction |
|---|---:|---:|---:|
| 2301 | 0.007592 | 0.005236 | 31.0% |
| 2302 | 0.008662 | 0.005302 | 38.8% |
| 2303 | 0.007078 | 0.004010 | 43.3% |

The oracle beats each of the shared, nominal and mixed-group controls in all
three fleets. The screen required at least 5% pooled reduction and improvement
in at least two fleets for each comparison, with convergence and valid
operating conditions. All requirements pass.

## Mechanism and limits of the comparison

SharedPrior and NominalPrior are almost equal on average: 0.007777 versus
0.007742 deg/s. A single learned center therefore adds no average prediction
advantage over the strong fixed population prior in this diagnostic. The
shared center is 0.45% worse on this pooled mean. MixedGroups is 2.71% worse
than SharedPrior despite having the same number and size of groups as
OracleFamily. These controls support attributing the oracle gain to the
chosen family-compatible partition, beyond generic partitioning or prior use.
Only one predeclared mixed partition is tested per fleet.

| Family | Shared prior [deg/s] | Oracle family [deg/s] | Reduction |
|---|---:|---:|---:|
| Compact | 0.005378 | 0.004501 | 16.3% |
| Sport | 0.012533 | 0.005832 | 53.5% |
| SUV | 0.005421 | 0.004216 | 22.2% |

Sport contributes 76.3% of the pooled absolute reduction versus SharedPrior,
but Compact and SUV also improve when pooled across fleets. Eight of nine
family/fleet strata improve; Compact in fleet 2303 becomes 2.95% worse.
OracleFamily beats SharedPrior for 42/54 vehicles, so it is not an individual
guarantee. The 90th percentile of vehicle error falls from 0.018468 to 0.009861
deg/s, supporting a reduction beyond only the mean.

All three local training-speed strata improve versus SharedPrior: 40.3% at
13 m/s, 27.0% at 21 m/s and 39.1% at 29 m/s. Most absolute improvement comes
from low-speed-trained vehicles. Secondary error reductions are 9.7% at the
locally observed speed, 35.4% for fleet interpolation and 30.9% for fleet
extrapolation. These secondary endpoints do not replace the frozen primary
endpoint or select a setting.

## What can be claimed

In this controlled heterogeneous population, one fleet-wide regularization
center is inferior to the three chosen family-specific centers for unseen-speed
yaw prediction. Keeping individual vehicle parameters while sharing compatible
priors provides a useful way to combine operating-envelope completion with
persistent vehicle differences.

The experiment does not establish that three groups are necessary or optimal,
that family labels can be recovered from short local records, or that clustering
improves closed-loop control. Labels are privileged information. The three
fleets are exploratory. The physical CFL-FF anchors and variations are mapped
to the existing matched tanh/relaxation plant; full nonlinear Pacejka mismatch
is not tested. Geometry is known and relaxation lengths are common to all
vehicles. Absolute yaw errors are small, so control relevance remains open.

Friction is weakly excited and unregularized; it reaches a bound in
25/54 oracle fits. Successful yaw prediction therefore
does not establish reliable identification of every physical parameter.

## Recommended next experiment

Replace oracle labels with groups learned from local training information,
keeping these fleets, records, bounds, penalty weights and test endpoints
fixed. Keep LocalOE, NominalPrior, SharedPrior and OracleFamily as references.
Report performance and per-vehicle losses alongside retrospective family
agreement, rather than selecting grouping settings using the outer test.
First test whether a practical learned partition retains the oracle advantage;
unknown group count and control validation can follow if it does.

No new experiment is started by this review.

## Reproduce this review

```bat
python code\experiments\analyze_cfl_heterogeneity_review.py
```

After regenerating review outputs locally, commit any resulting changes:

```bat
git add docs/cfl_heterogeneity_result_review.md
git add results/tables/paper_cfl_heterogeneity_review.json
git diff --cached --stat
git commit -m "Audit CFL family-prior heterogeneity results"
git push origin main
```
