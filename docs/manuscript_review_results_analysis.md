# Matched manuscript-review validation: analysis

Date: 12 October 2026.
Result commit: `6b05653a2e0b5aeeadf5e2e66a96f11085ba1a02`.

The ten-fleet supplement resolves both targeted gaps within the controlled
available-state benchmark. Selected groups have sufficient scheduling
information, and the richer regularized Local comparator does not explain away
the improvement over restricted Local. This is matched evidence on the original
ten fleets, not a new independent confirmation campaign.

## Independent verification

Run from the repository root:

```text
python paper/audit_review_validation.py
```

The audit checks source and checkpoint hashes, reconstructs group design
diagnostics from memberships, evaluates saved model coefficients against the
existing reference Jacobians, compares parent records, and independently
rebuilds summaries and paired bootstrap intervals. No model fits or calibration
and tracking rollouts run. Windows provenance paths and LF/CRLF text conventions
are normalized for verification; original artifacts remain unchanged.

| Check | Outcome |
|---|---|
| Complete original fleets | 10/10, seeds 371--380 |
| Source hashes / checkpoints | 23 sources / 10 checkpoints verified |
| Client-speed-method errors | 13,200 complete finite records |
| Errors checked from saved Global/grouped/regularized coefficients | 9,900; maximum absolute difference 4.13e-12 |
| Parent error reproduction | Maximum client-speed difference 3.10e-10; fleet-mean difference 5.05e-12 |
| Aggregation and paired bootstrap intervals | Reproduced |

## Identification results

Relative matrix errors in percent, using the frozen physical coordinates and
averaging eligible client-speed records within each fleet before averaging
fleets:

| Method | Full envelope | Observed speeds | Unobserved speeds |
|---|---:|---:|---:|
| Restricted Local, L=3 | 49.562208 | 1.929577 | 65.568315 |
| LocalRegularized, L=7 with local L=3 prior | 49.562229 | 1.930125 | 65.568158 |
| Global, L=7 | 4.249846 | 3.839952 | 4.387583 |
| Federated groups, L=7 | 3.248925 | 2.422684 | 3.526569 |

Grouped fitting improves full-envelope error over LocalRegularized by mean
paired **93.412965%**, 95% fleet-bootstrap interval **[92.592761, 94.211729]%**,
with wins in all ten fleets. The unobserved-speed improvement is
**94.586550%**, interval **[93.915819, 95.252799]%**, also in all ten fleets.
At observed speeds, grouped fitting is approximately 25.08% worse than
LocalRegularized on the mean paired criterion.

LocalRegularized changes full-envelope grand-mean error by only +0.0000206
percentage points relative to Local. At unobserved speeds it reduces the grand
mean by about 0.0001566 percentage points. The paired unobserved gain is
0.00024697%, with a narrow positive bootstrap interval, but a negligible effect
size. These small differences are not exact numerical equality or substantive
improvement.

This evaluates one defined local alternative: a richer basis, local L=3 prior
and locally selected regularization. It does not establish failure of every
possible local estimator or a stronger independently justified physical prior.
Within the frozen LPV representation, regularization does not supply the missing
scheduling information contributed by other clients.

## Selected-group information

All **18 selected groups** have rank **7**, contain observations at all eleven
grid speeds, and have stacked design condition numbers between
**3.259876 and 3.873110**. Group sizes range from 12 to 30 clients.
Eight fleets select two groups and two select one.

All 300 local seven-function designs are rank deficient: 230 have rank three and
70 have rank two. The learned groups therefore actually attain the complementary
information condition, rather than merely being solvable after regularization.

These are condition numbers of the stacked scheduling design in the fixed QR
basis, not the Gram or full dynamical-identification information matrix.
Coefficient uniqueness is conditional on a fixed compatible partition and a
correct shared representation. Full rank does not remove within-group bias or
prove physical-family recovery. The minimum-size rule still does not guarantee
coverage in other fleets.

## Client tradeoff and scope

At unobserved speeds, grouped fitting improves on Global for 169/300 client
instances, worsens it for 71/300 and ties for 60/300 at absolute tolerance
1e-12. The ties arise from the two one-group fleets. Maximum client-average
deterioration is 4.40645 percentage points of matrix error. Clients are nested
within fleets; these counts are descriptive, not independent statistical units.

Local remains best at its observed speeds. Average improvement therefore does
not guarantee improvement for every vehicle or every operating region.
Controlled simulator-derived centering, available states, exact current
regressors, noisy next-state targets and one cornering curve remain assumptions.
No real-fleet, ordinary-driving or frozen-library transfer claim is added.
The modest supporting LQI evidence remains unchanged.

## Manuscript implications

The next text edit should add LocalRegularized to the main results table,
replace both pending-evidence statements with the audited findings, and state
the observed group rank/coverage/conditioning separately from any general
algorithm guarantee. Retain the observed-speed Local advantage and recipient
tails. Preserve the original Global-relative headline and primary interval;
this supplement is a matched check, not extra independent confirmation.

After those edits, prioritize novelty positioning against local LPV
identification and the IFAC contribution, clarity and presentation. These
results do not require another simulation campaign or reopening KF learning.

The independent audit is saved in
`results/tables/paper_review_validation_independent_audit.json`.
Local-rank/regularization diagnostics and client tails are saved in
`paper_review_validation_local_rank_audit.csv` and
`paper_review_validation_tail_audit.csv`. Original checkpoints and summaries
remain unchanged.
