# Interpretation of learned primary groups

Frozen descriptive analysis; no fitting, grouping changes, or simulations.
Execution checkpoint: 105d026. All counts refer to primary seeds 371–380.

## What the partitions contain

Eight fleets select two groups; 373 and 377 select one group and exactly retain
Global. In every two-group fleet all ten heavy clients share one group and all
ten handling clients share the other. Nominal clients are split between both.
The heavy-containing group receives 2–8 nominal clients; its handling-containing
counterpart receives the remaining 8–2. Group labels are arbitrary and are not
matched across fleets. This statement uses family composition, not group IDs.

| Fleet | Nominal with heavy | Nominal with handling |
|---|---:|---:|
| 371 | 6 | 4 |
| 372 | 6 | 4 |
| 374 | 2 | 8 |
| 375 | 6 | 4 |
| 376 | 4 | 6 |
| 378 | 8 | 2 |
| 379 | 2 | 8 |
| 380 | 4 | 6 |

All 18 selected groups span every local block and all eleven grid speeds, with
rank seven and design condition numbers 3.26–3.87. Two-group sizes are 12–18.
Each selected map therefore has complete scheduling coverage, but this audit
does not prove BIC enforces that condition or guarantees compatible dynamics.
Family labels enter only this retrospective diagnostic; they remain excluded
from assignment and fitting. Complete heavy/handling separation in the eight
selected two-group fleets does not constitute three-family recovery.

## Recipient tradeoffs

At recipient-unobserved speeds, client instances have the following outcomes
against Global, using the manuscript's 1e-12 dimensionless-error tie tolerance:

| Family | Better | Tied | Worse |
|---|---:|---:|---:|
| Nominal | 9 | 20 | 71 |
| Heavy | 80 | 20 | 0 |
| Handling | 80 | 20 | 0 |

The 20 ties per family come from the two K=1 fleets. All 71 deteriorating client
instances are nominal. Conditional on the eight K=2 fleets, mean error changes
(Grouped minus Global) are +2.0402 percentage points for nominal, -2.7451 for
heavy, and -2.5240 for handling. Each fleet gives equal weight to its ten clients
per family; nominal family means worsen in every K=2 fleet. These conditional
summaries are descriptive, not independent confirmation or causal effects.
The 300 clients remain nested in ten fleet units.

Observed standardized assignment residuals improve on average for all three
families, including nominal. They increase for 32 individual clients across the
primary cohort; neither assignment to a learned map nor lower fleet residual
ensures individual improvement over Global. This statistic uses saved noisy
matrix targets and the original pooled-entry scaling, not exact evaluation
matrices. The sum of reconstructed selected scores matches the archived selected
SSE in each fleet within 1e-8. It is an in-sample diagnostic and must not be
presented as out-of-sample evidence.

## Scientific interpretation and limits

Here 'compatibility' means that a shared regression map reduces aggregate
residual/error under the fitting and selection procedure. It is not an
identified physical class, a stability certificate, or a guarantee of recipient
benefit. The observed partitions separate the heavy and handling families while
sharing nominal clients with either, retaining complementary coverage. The
held-out-speed tradeoff shows that such a compromise can benefit the former
families while disadvantaging many nominal recipients.

The analysis does not establish that nominal clients are dynamically intermediate,
that speed blocks cause their assignments, or that coverage is the reason BIC
selects two rather than three groups. Those explanations require counterfactual
fits or targeted experiments. No new family-recovery guarantee should be added.
Coassignment CSVs provide label-invariant within/between-family descriptive rates,
and group CSVs give block counts; do not treat pair counts as independent samples.

## Reproducibility and manuscript placement

`python paper/learned_group_analysis.py` regenerates four CSVs and a source/output
manifest from frozen checkpoints. Membership, rank and conditioning are checked
against saved audits; original error records supply recipient comparisons.
Additional checks verified manifest hashes, all 300 memberships, and selected
SSEs. No original numerical evidence was overwritten.

Add a concise paragraph to the scheduling-information results subsection:
report heavy/handling separation, the nominal split, and full group rank; follow
with the family breakdown of client deterioration. Clarify that these are
retrospective diagnostics using labels withheld from learning. Keep the detailed
cross-tabs here if the eight-page manuscript has no room for another table.
The manuscript is unchanged in this analysis checkpoint.
