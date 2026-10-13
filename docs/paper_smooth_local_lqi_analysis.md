# Smooth Local LQI comparison: analysis

Execution checkpoint: b8caaa2. Saved-map and summary audit: 2b82e07.

## Finding

The development-selected function-curvature prior is a strong matrix-error
baseline but does not reliably support the frozen LQI synthesis protocol.
All 300 clients synthesize schedules and all 600 rollouts complete numerically.
Nevertheless 114 clients have an ExactLPV-reference frozen-point spectral radius
at least one; the maximum is 50.8140. Recorded sideslip violations affect 88/300
moderate and 65/300 hard runs. Grouped controllers have no recorded violations.

| Method | Moderate yaw RMSE [deg/s] | Hard yaw RMSE [deg/s] |
|---|---:|---:|
| Restricted Local | 0.053961 | 0.169807 |
| Smooth Local | 8.356546 | 5.716094 |
| Global | 0.042903 | 0.156442 |
| Federated groups | 0.041080 | 0.154568 |
| ExactLPV | 0.040020 | 0.150743 |

These are client means within fleet, then means of ten fleet units. Smooth Local
medians across client instances are 0.058902 and 0.204715 deg/s: its large means
are driven by severe tails. Recorded mean saturation fractions are 14.44% and
10.23%. Some simulated sideslip peaks exceed 260 degrees, far outside the six-
degree feasibility bound and any plausible validity region for this lateral
model. Those values mark failed runs, not physically meaningful vehicle behavior.
The modest grouped-versus-Global gain remains 4.255%/1.207%; this supplement
must not turn it into a claim of enormous practical benefit from federation.

The executed paired gains over Smooth Local are 99.496% (95% interval
[99.444,99.548]) and 97.162% ([96.795,97.542]), with 10/10 fleet wins each.
These descriptive figures are dominated by failed closed-loop responses. The
main manuscript message should be feasibility and control suitability, with
these ratios relegated to the audit or omitted from the main argument.

## Verification and numerical limits

`python paper/audit_smooth_local_lqi.py` checks immutable input/output hashes,
all 3,600 row keys, parent metrics, saved-map stationarity, controller schedules,
frozen-point radii, and fleet means. It recreates all three summaries in temporary
storage and compares them to executed CSVs. It never fits group maps or changes
historical results. Maximum parent yaw RMSE discrepancy is 7.73e-10 deg/s.
Maximum normalized Smooth Local normal-equation residual is 6.28e-17.
Recomputed controller gains differ from saved gains by at most 2.26e-7 in the
normalized componentwise check; the saved schedules are used for radius auditing.
These checks establish consistency of the saved fitting and summary outputs,
not exact replay of every nonlinear trajectory.

A 60-rollout replay of fleet 411 using saved schedules agrees in all feasibility
flags but matches none of the original byte-level trajectory hashes. Most metric
differences are negligible, while at least one failing client has a material
RMSE difference (maximum 2.88524 deg/s). Thus exact trajectory replay across this
runtime and the execution runtime is not established. Small numerical differences
and sensitive failed responses are consistent with this observation, but the
specific cause is not isolated. The strict `restore-data --seed 411` command
correctly rejects mismatched hashes; it must not be reported as passing here.
Retain original execution outputs and do not weaken their hashes or substitute
replays. Restoration was tested on the preparation runtime's smoke outputs;
that success does not establish portability of the user's later execution.

Only one of 300 Smooth Local maps changes the yaw-input sign on a grid node.
Therefore a simple B2 sign check would not explain or repair all failures.
The unweighted coordinate-dependent matrix metric can mask dynamically important
errors; these results show that matrix accuracy alone is insufficient for frozen
controller synthesis. They do not prove that every smooth local prior, or a
controller tuned specifically for such a prior, fails.

## Recommended manuscript integration

1. Add Smooth Local to Table V, with explicit infeasibility counts in the caption
   or adjacent text. Keep the existing grouped/Global comparison as the main
   measure of the benefit of compatibility selection.
2. State that no prior penalty, controller weight, or fallback was retuned. This
   is an additional baseline on the existing ten confirmation fleets, not a new
   independent cohort.
3. Explain that the matrix-error prior lacks a guarantee of controller suitability;
   in this frozen test its schedules have frozen-point instability and infeasible
   nonlinear responses for a subset of clients.
4. Disclose sensitive failed-run replay in the reproducibility note. Avoid precise
   extreme-tail performance claims as the central evidence.
5. Replace the conclusion's statement that Smooth Local control remains untested.
   Limit the conclusion to the tested prior and frozen synthesis protocol.

No manuscript changes are included in this analysis checkpoint.
