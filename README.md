# Federated LPV Identification and Control

This repository studies whether measurable operating-point variation and persistent
client heterogeneity should be represented separately in federated control. The
initial benchmark uses lateral vehicle dynamics: longitudinal speed is the LPV
scheduling variable, while mass, inertia, and tire parameters define persistent
vehicle families.

The first milestone is deliberately an **oracle feasibility study**. Before adding
federated learning, limited data, or learned clustering, it tests whether a small
number of family-specific LPV models and scheduled controllers is useful at all.

## Repository layout

```text
code/       Python package, experiment entry points, configurations, and tests
results/    Reproducible outputs; generated data are not committed by default
report/     Living LaTeX development report and bibliography
```

## Scientific comparison

The core two-factor comparison is:

| Model | Speed scheduling | Structural specialization |
|---|---|---|
| Global LTI | No | No |
| Oracle clustered LTI | No | Yes |
| Global LPV | Yes | No |
| Oracle clustered LPV | Yes | Yes |
| Gridded LTI oracle | Discrete | Yes |

See [`report/main.tex`](report/main.tex) for the motivation, benchmark definition,
experiment plan, decision gates, and the progressively updated findings.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e "./code[dev]"
python -m unittest discover -s code/tests
python code/experiments/phase0_validate_plant.py
```

Generated experiment artifacts are written below `results/`.

The latest Phase-4 studies are `experiment_4e_identification_diagnosis.py` and
`experiment_4f_structured_identification.py` in `code/experiments/`. Run them with
`PYTHONPATH=code/src python code/experiments/<script>.py` from the repository root.
4E diagnoses the earlier failed redesign. 4F tests structured parameter-ratio
identification. The report preserves both the negative results and the initial
recovery, including its single-fleet and changed-baseline limitations.

Experiment 4G validates the frozen structured method on ten independent
train/test fleet pairs and three held-out scenarios, including direct LTI fits.
Run `PYTHONPATH=code/src OPENBLAS_NUM_THREADS=1 python code/experiments/experiment_4g_independent_validation.py`.
The frozen protocol is in `code/config/experiment_4g.json`. Per-seed client,
parameter, and stability CSVs preserve all runs. Aggregate comparisons use seed
pairs as independent units. Use `--summarize-only` to rebuild summaries and the
figure from those CSVs.

Experiment 5A sweeps training-selected nearest-speed controller grids against M4.
Run `PYTHONPATH=code/src OPENBLAS_NUM_THREADS=1 python code/experiments/experiment_5a_lti_complexity.py`.
It reuses the committed 4G parameter fits and fleet protocol. The 5A JSON config
records candidate counts and the matching tolerance. Per-seed CSVs retain
selection scores, selected anchors, all test metrics, and stability audits.
`--summarize-only` rebuilds the aggregate tables and figure.

Experiment 6A tests four family-separation levels and three speed envelopes.
Run `PYTHONPATH=code/src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python code/experiments/experiment_6a_regime_map.py`.
The protocol in `code/config/experiment_6a.json` uses fresh structured fits,
ten independent train/test fleet pairs, and unchanged within-family scatter.
Per-seed files preserve client metrics, fit diagnostics, common-data prediction
errors, and full-envelope frozen stability audits. `--summarize-only` rebuilds
the regime map and summaries. This remains a centralized oracle comparison.

Experiment 6B checks the four corner regimes with the 6A models and gains frozen.
Run `PYTHONPATH=code/src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python code/experiments/experiment_6b_acceleration_validation.py`.
It compares historical integration, within-step speed variation without the
coordinate correction, and lateral-momentum integration with consistent varying
speed. Each uses the original and doubled maneuver duration. The 6B JSON freezes
the protocol. Compressed client-level metrics, paired comparisons, and provenance
hashes accompany the report. `--summarize-only` rebuilds its evidence.

Experiment 7A compares batch output-error Local, Global, and oracle-Family fits
across two recording budgets, two speed-coverage conditions, and clean/noisy
outputs on the corrected plant.
Run `PYTHONPATH=code/src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python code/experiments/experiment_7a_local_identification.py --workers 4`.
The frozen protocol is `code/config/experiment_7a.json`; `--summarize-only`
rebuilds the evidence. All 2,720 fits converge and 21,600 evaluations are
amplitude-feasible. The primary collaboration gate fails: Family tracking
error is 11.62% higher than Local, although 36.33% lower than Global.
Local identification is already accurate with six seconds of data. Family
pooling offers fewer deployed models at a measurable performance cost.
There is no federation, streaming RLS, or learned clustering in this experiment.

Experiment 7B tests family-informed personalization on new fleet seeds 41--50
in the unchanged six-second noisy restricted-coverage setting.
Run `PYTHONPATH=code/src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python code/experiments/experiment_7b_personalization.py --workers 4`.
Each prior excludes the recipient client. Leave-one-episode-out validation
selects regularization from a fixed grid, followed by a full local refit.
The protocol is `code/config/experiment_7b.json`; `--summarize-only` rebuilds
summaries. All 6,284 fits converge and 2,700 evaluations are amplitude-feasible.
Personalization improves tracking over fixed Family pooling by 12.79%, but
is 0.0573% worse than Local. The 5% improvement gate fails. This is near-parity
with local accuracy, not a demonstrated federated accuracy benefit.
Fold-level diagnostics, selected strengths, donor lists, and data hashes are
retained. The LaTeX report explains the revised sequence and both negative gates.

Experiment 7C tests complementary coverage on fresh fleet seeds 51--60.
Run `PYTHONPATH=code/src:code/experiments OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python code/experiments/experiment_7c_complementary_coverage.py --workers 4`.
Each client records three one-second episodes at one assigned speed, while each
oracle family jointly covers 10--30 m/s. LocalExtra receives an equal transition
budget from the recipient itself over the complete speed set. All 640 fits
converge and all 3,600 evaluations are feasible. Family beats Global by 35.22%,
but is 13.61% worse than Local and fails the collaboration gate on every paired
seed. Complementary speed coverage therefore does not create a federated accuracy
benefit for the strongly structured three-ratio estimator.

Experiment 8A is the formal paper-readiness and contribution audit. It freezes
the supported claim set, excluded claims, minimum remaining robustness work,
and manuscript blueprint in `report/sections/15_experiment8a_paper_audit.tex`.
The decision is conditional readiness as an LPV/control paper, not readiness as
a federated-algorithm paper. Machine-readable audit outputs are
`results/tables/experiment_8a_claim_audit.csv` and
`results/tables/experiment_8a_conclusions.json`.

Experiment 8B implements distributed structured output-error identification on
fresh fleet seeds 61--70. Run
`PYTHONPATH=code/src:code/experiments OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python code/experiments/experiment_8b_federated_equivalence.py --workers 4`.
Clients retain trajectories and upload one loss plus three gradient entries per
server evaluation. Both Global and oracle-Family scopes are tested, together
with a pairwise-mask aggregate-only emulator. The equivalence gate passes:
maximum centralized--federated relative parameter error is 7.65e-8 and maximum
tracking difference is 6.12e-10 deg/s; all 5,400 evaluations are feasible.
The secure path is a numerical utility emulator, not production cryptography or
differential privacy. Experiment 8C can now study an explicit privacy budget.

Experiment 8C evaluates one-shot replacement-adjacent client-level differential
privacy on fresh fleets 71--80. Run
`PYTHONPATH=code/src:code/experiments OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python code/experiments/experiment_8c_private_frontier.py --workers 4`.
Bounded local log-parameter contributions are securely aggregatable and released
with analytically calibrated Gaussian noise at delta=1e-5. Ten privacy draws per
fleet are tested for epsilon 0.5, 1, 2, 4, and 8. No finite epsilon meets the
predeclared 10% tracking-degradation gate. Epsilon 4 and 8 remain feasible in all
runs; epsilon <=2 includes frozen-instability cases. The result motivates a
control-aware privacy mechanism rather than weakening the privacy definition.

Experiment 8D tests privacy-utility recovery on fresh training seeds 81--85 and
separate nonlinear test fleets. Run
`PYTHONPATH=code/src:code/experiments OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python code/experiments/experiment_8d_privacy_recovery.py --workers 5`.
It retains replacement-adjacent client-level DP and delta=1e-5 while comparing
the 8C broad box with tighter public benchmark-design bounds and compatible
family cohorts of 10, 30, and 100 clients. The fixed 10% gate passes: Tight-100
has 6.05% degradation at epsilon 2, full amplitude feasibility, and worst frozen
radius 0.9692. Tight-30 passes only at epsilon 8 (6.22%); Tight-10 does not pass.
This is a cohort-scale privacy recovery result, not evidence for private learned
clustering, repeated-round accounting, dropout tolerance, or production secure
aggregation.

Experiment 8E tests public control-curvature-shaped client-level DP on fresh
training seeds 86--95. Run
`PYTHONPATH=code/src:code/experiments OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python code/experiments/experiment_8e_control_aware_privacy.py --workers 5`.
The transformed-space Gaussian mechanism retains the same replacement adjacency
and delta=1e-5 as 8C--8D. It improves tracking in every finite cell and every
fleet, but the largest mean gain is 11.39%, below the frozen 15% requirement.
The 30-client, epsilon-4 privacy cost falls from 16.36% to 11.74% but still
misses the 10% utility threshold. The joint gate therefore fails and is reported
as a consistent but insufficient effect, not a successful proposed method.

Experiment 8F maps deployment robustness on fresh training seeds 96--105. Run
`PYTHONPATH=code/src:code/experiments OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python code/experiments/experiment_8f_deployment_robustness.py --workers 5`.
Across balanced, random, and low-speed-biased partial participation, the minimum
viable control-aware cohorts are 100, 50, and 20 clients at epsilon 2, 4, and 8;
the isotropic requirements are 100, 50, and 30. At the established 100-client,
epsilon-4 operating point, both mechanisms remain below the 10% tracking-cost
gate under 10% bound narrowing, a 5% bound shift, 5--10% bounded outlier
contributions, and nonlinear tire friction reduced from 0.9 to 0.7. The
30-client boundary and 10-client nonviable points correctly remain outside the
declared region. The deployment gate passes without reusing Experiment 8E's
arbitrary 15% method-improvement threshold.

Experiment 8G performs a blind confirmation on fresh seeds 106--115. Run
`PYTHONPATH=code/src:code/experiments OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python code/experiments/experiment_8g_blind_confirmation.py --workers 5`.
The frozen primary prediction at 20 clients and epsilon 8 fails narrowly:
control-aware DP costs 10.43% relative to its matching non-private family model,
just outside the 10% gate, while isotropic DP costs 14.40%. Control-aware DP is
better in all ten fleets (paired Wilcoxon p=0.000977). The secondary 50-client,
epsilon-4 point confirms, with costs of 3.48% and 5.08%, respectively. Local
models remain best in absolute tracking. Thus the robust paper claim uses the
confirmed 50-client point; the 20-client result is reported as a boundary, not
retrospectively relabelled as viable.

Experiment 9A removes known family labels from non-private model sharing on
fresh seeds 116--125. Run
`PYTHONPATH=code/src:code/experiments OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python code/experiments/experiment_9a_learned_groups.py --workers 5`.
Label-free clustering of normalized log-parameter estimates selects three groups
in every fleet, assigns all unseen test vehicles consistently with the simulated
families (test ARI 1.0), and matches the oracle-family tracking result (0.007554
versus 0.007556 rad/s) with three rather than 30 Local controllers. The proposed
reuse of the privacy control weights as a clustering metric fails its hard gate:
it merges two groups on seed 124 and is 2.66% worse than oracle on average.
Experiment 9A therefore supports learned compatibility discovery, but not the
claim that the existing control-aware DP geometry is also a valid clustering
geometry. Clustering is non-private in this gate and assumes a trusted server.

Experiment 9B stress-tests ordinary label-free parameter clustering on fresh
seeds 126--135. Run
`PYTHONPATH=code/src:code/experiments OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python code/experiments/experiment_9b_group_robustness.py --workers 5`.
The frozen gate passes for baseline and strongly unbalanced discrete fleets:
tracking remains within 0.08% and 1.49% of Oracle family, with unseen ARI 0.980
and 0.954. It fails for 0.25-second records and fourfold output noise, where
point-estimate clustering becomes unreliable (ARI 0.082 and 0.390). On a
continuously heterogeneous population, Learned-K selects two regions, matches
the true-ratio partition within 0.42%, and improves Global by 17.25%, narrowly
below the predeclared 20% gate. All 10,500 fits converge and all 21,600 nonlinear
evaluations are feasible and frozen-stable. This bounds the 9A claim and
motivates uncertainty-aware clustering before private group discovery.

Experiment 9C implements that uncertainty-aware repair on the locked 9B
short-data and high-noise seeds. Run
`PYTHONPATH=code/src:code/experiments OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python code/experiments/experiment_9c_uncertainty_groups.py --phase development --workers 5`.
Gauss--Newton log-parameter covariances feed a heteroscedastic Gaussian mixture;
BIC selects an unknown number of groups and posterior memberships blend shared
scheduled controllers. Under high noise this reduces point-clustering tracking
error by 17.27%, but remains 5.31% above Oracle family and narrowly misses the
frozen 5% gate. With 0.25-second records BIC selects one group in every fleet,
and the oracle gap remains 55.45%. The development gate therefore fails and the
untouched confirmation seeds 136--145 are deliberately not opened. All 4,200
fits converge and all 10,800 nonlinear evaluations are feasible and stable.
The result motivates acquiring more informative or repeated local records, not
forcing extra clusters from structurally uncertain estimates.

Experiment 9D replaces that fixed-data assumption with uncertainty-triggered
adaptive calibration. Run
`PYTHONPATH=code/src:code/experiments OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python code/experiments/experiment_9d_adaptive_calibration.py --phase development --workers 5`.
Clients start with three 0.25-second records and request a one-second maneuver at
a complementary speed while normalized covariance trace exceeds 0.05. The
development gate passes, after which the frozen policy also passes on untouched
seeds 136--145. Confirmation Adaptive-soft tracking is 0.007758 rad/s: 2.86%
above Oracle-adaptive and 33.50% better than the failed Short-point method. It
uses 1.75 seconds per client instead of the fixed-full 2.75 seconds and retains
2.8 shared models rather than 30 Local models. All 4,200 development and
confirmation fits converge, and all 10,800 nonlinear evaluations are feasible
and frozen-stable. Local remains best in absolute tracking; the positive result
is reduced calibration and model proliferation with near-oracle shared control.

Experiment 9E audits why Local remains best rather than weakening that baseline.
Run `PYTHONPATH=code/src:code/experiments OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python code/experiments/experiment_9e_local_advantage_audit.py --workers 5`.
Across ten fresh fleets, Exact-individual is 3.89% better than fitted Local,
while swapping Local fits only within the correct family is 22.88% worse and an
error-free Exact-family mean is 7.85% worse. The result attributes the gap to
genuine within-family specialization; neither clustering nor federated
optimization can remove it. Raw Local parameter error is effectively
uncorrelated with tracking, while scheduled-gain and held-out prediction errors
are more informative. Tire-fade and load-shift tests preserve the ranking. The
next model should therefore combine a shared family backbone with a small,
control-relevant local residual and evaluate tracking, calibration burden, and
deployed degrees of freedom separately.

Experiment 9F tests that personalized representation before introducing another
federated optimizer. Run
`PYTHONPATH=code/src:code/experiments OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python code/experiments/experiment_9f_personalized_manifold.py --workers 5`.
On ten fresh fleets, a one-scalar control-aware client head closes 69.5% of the
Exact-family-to-Exact-individual tracking gap and is 0.62% better than the 9D
Local fit on average; ordinary parameter PCA at the same rank closes only 6.6%
and remains 6.77% worse than Local. Two local coordinates close 97.7% of the gap
for the control-aware basis and 99.6% for parameter PCA. This is an oracle
representation result: true unseen-client parameters are used to project the
local coefficients. The next required gate is finite-data personalized
estimation of those coefficients with federated learning of the shared centers
and directions.

Experiment 9G performs that finite-data test. Run
`PYTHONPATH=code/src:code/experiments OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python code/experiments/experiment_9g_finite_personalized.py --workers 5`.
Training clients transmit structured estimates and covariances; the server
selects three unknown groups (mean ARI 0.951) and learns control-aware backbones.
Unseen clients select a group on a held-out complementary record and fit only
one or two local manifold coefficients. Rank two reaches near-parity at 0.61%
worse tracking than full three-parameter Local and matches its gain error, but
fails the strict no-worse gate. Rank one is 7.16% worse and fails its 2% gate.
Oracle-family variants give nearly identical results, excluding clustering as
the bottleneck. The remaining question is whether rank two can reach a fixed
control-quality target with less local calibration; otherwise a reduction from
three to two local coordinates is too modest to anchor the paper contribution.

Experiment 9H evaluates the full matched-prefix calibration frontier. Run
`PYTHONPATH=code/src:code/experiments OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python code/experiments/experiment_9h_calibration_frontier.py --workers 5`.
The strict time-to-target gate fails: personalized rank two at 1.25 seconds does
not match the 1.75-second Local reference, and both first cross its 2% tolerance
at 1.75 seconds. Sharing nevertheless gives a clear cold-start advantage at
equal budgets. Rank one improves Local tracking by 31.79% at 0.50 seconds and
20.92% at 0.75 seconds; rank two improves it by 0.94% at 1.25 seconds, and both
personalized variants reach Local parity at 1.75 seconds. The appropriate model
order changes with information: shared backbone first, then rank one, then rank
two. This motivates uncertainty-gated personalization rather than a fixed local
head and provides a defensible operational benefit for fleet information sharing.

Experiment 9I tests whether fleet knowledge can also select the next calibration
maneuver. Run
`PYTHONPATH=code/src:code/experiments OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python code/experiments/experiment_9i_active_identification.py --workers 5`.
Fleet control-active improves Fleet-fixed by 1.94% and Local-fixed by 1.78%, but
it is only 0.18% better than Local control-active and wins that paired comparison
in five of ten fleets. More importantly, fleet parameter-active, fleet control-
active, and Local control-active all select the same high-speed, high-frequency
maneuver for every client. Active excitation helps, but the tested library has a
globally dominant candidate and does not demonstrate personalized or control-
aware fleet selection. The report therefore retains 9I as a negative mechanism
boundary rather than pivoting the paper toward active learning.

Experiment 9J implements the missing distributed personalized-backbone stage.
Run `PYTHONPATH=code/src:code/experiments OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python code/experiments/experiment_9j_distributed_backbone.py --workers 5`.
Clients retain structured estimates and personalization coordinates locally and
send additive uncertainty-weighted mixture/manifold statistics. Full
participation reproduces the centralized-summary tracking result to numerical
precision. With 50% participation, degradation is at most 0.53%; with 20%, the
method is 2.34% better at 0.75 seconds and 0.04% worse at 1.25 seconds. Fed20
beats Local by 21.09% and 0.47% at the respective budgets. The cold-start result
therefore survives an explicit federated implementation and partial
participation, with mean numerical payloads falling from 1.160/0.670 MB
(upload/download) at full participation to 0.258/0.138 MB at 20%. The additive
interface is secure-aggregation-compatible, but cryptography and its overhead
are not implemented. Seeds 206--215 remain reserved for frozen confirmation.

Experiment 9K executes that frozen confirmation without tuning. Run
`PYTHONPATH=code/src:code/experiments OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python code/experiments/experiment_9k_blind_confirmation.py --workers 5`.
A machine-checked guard verifies that seeds 206--215 and every inherited 9J
setting remain unchanged. Fed20 confirms the cold-start result, improving Local
tracking by 17.27% at 0.75 seconds and winning nine of ten fleet comparisons.
At 1.25 seconds it is 0.06% worse than Local, confirming the predicted parity
after sufficient local calibration. Fed50 and Fed20 pass their 1% and 2%
partial-participation gates, and all 10800 nonlinear evaluations are feasible.
The deliberately strict Fed100 gate fails: it is 0.281% worse than Central at
0.75 seconds rather than within 0.1%. Its deterministic internal rerun remains
identical below `1e-12`; the failure means the distinct Central and federated
mixture implementations should be described as sub-percent control-equivalent,
not numerically identical.

Experiment 9L stresses the personalized federated backbone under nonrandom
availability. Run `PYTHONPATH=code/src:code/experiments OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python code/experiments/experiment_9l_nonrandom_participation.py --workers 5`.
At the 0.75-second cold-start budget, a fixed 20% subset, 20x family
underrepresentation, high-speed underrepresentation, and matched 50% message
dropout all remain within the predeclared 5% tolerance relative to uniform 20%
participation and improve aggregate tracking over Local by 12.83--17.17%.
Dropout reduces the mean message count from 1296 to 745 while retaining 97.2%
distinct-client coverage. The fleet-wise audit exposes an important boundary:
Persistent20 selects one group on seed 221 and becomes 34.43% worse than Local;
it wins only nine of ten fleets, whereas the rotating-bias and dropout regimes
win all ten. Repeated rounds therefore do not repair missing population support.
The deployment implication is to monitor cumulative group and operating-region
coverage, solicit underrepresented clients, or fall back to Local identification.

Experiment 9M reconstructs the main baselines on the same 9K confirmation
fleets and adds a 50000-resample paired fleet bootstrap. Run
`PYTHONPATH=code/src:code/experiments OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python code/experiments/experiment_9m_unified_ablation.py --workers 5`.
At 0.75 seconds, GlobalK1, LearnedCenter, and Fed20Rank1 improve Local by 8.43%,
14.47%, and 16.90%. The Fed20Rank1 95% fleet-bootstrap interval is
[11.79%, 21.10%]. LearnedCenter wins all ten fleets, showing that heterogeneous
group structure supplies most of the initial cold-start benefit; the additional
rank-one improvement over the center is 2.20% with an interval spanning zero.
At 1.25 seconds, Fed20Rank1 improves LearnedCenter by 4.21% with a [2.40%,
6.36%] interval and wins all ten fleets, showing that the local coordinate
becomes useful as client information accumulates. Rank two overfits at 0.75
seconds and provides no reliable advantage over rank one at 1.25 seconds. The
analysis is explicitly retrospective because the 9K fleet outcomes were known;
it quantifies the confirmed mechanism but is not another blind confirmation.

The paper-level method and evidence audit is in
`paper/METHOD_EVIDENCE_AUDIT.md`. It concludes that the existing privacy-centered
paper skeleton is obsolete and that the defensible IEEE IV core is unknown-group
federated LPV backbone learning for cold-start transfer. The audit maps every
intended claim to evidence, separates the final method from the earlier privacy
campaign, identifies the persistent-coverage boundary, and lists mandatory
pre-drafting corrections. In particular, the 9J--9K communication values count
only the selected group order and must be recomputed to include federated
selection over all candidate orders before they appear in the manuscript.

Experiment 10A starts the complementary-operating-coverage redesign with a
development-only basis-order and identifiability audit. Run
`PYTHONPATH=code/src:code/experiments OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python code/experiments/experiment_10a_basis_coverage_audit.py`.
It linearizes the existing nonlinear tanh-tire plant along steady-cornering
operating points and selects the smallest nested speed basis by whole-region
blocked validation. The straight-line negative control retains the physical
three-term reciprocal basis. The strongest nonlinear cornering condition
requires four additional orthogonalized polynomial terms, so the audit freezes
`L=7`. Every tested restricted client speed block is rank deficient for this
model, while the combined fleet envelope is full rank. This establishes a
legitimate representation and complementary-information premise for 10B, but
does not itself claim federated estimation or control improvement.

Experiment 10B tests that premise using noisy finite perturbation transitions
from ten new fleets. Run
`PYTHONPATH=code/src:code/experiments OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python code/experiments/experiment_10b_oracle_complementary_gate.py --workers 5`.
Oracle-family sharing reduces locally unseen-speed matrix error by about 97%
relative to Local and improves nonlinear closed-loop recovery by about 1.7%,
with positive fleet-bootstrap intervals and 100% feasibility. FamilyPool also
reduces error by 57.70% relative to Global, but their control difference is not
resolved. The personalized family model is 5.13% worse in prediction than the
plain family backbone, showing that a local head is premature at this cold-start
budget. Experiment 10B therefore passes the complementary-sharing gate while
supporting compatible backbone sharing rather than a personalization claim.

Experiment 10C sweeps progressive recipient coverage while freezing the
oracle-family backbone learned at cold start. Run
`PYTHONPATH=code/src:code/experiments OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python code/experiments/experiment_10c_personalization_transition.py --workers 5`.
Personalization is harmful at the initial 2--3 speeds, unresolved at five,
and first improves full-envelope model accuracy at eight covered speeds by
7.74% with a positive fleet-bootstrap interval. At all eleven speeds it improves
the backbone by 13.66%, but a fully local model is then another 13.79% more
accurate. Personalization never improves nonlinear recovery over FamilyPool.
The evidence supports a coverage-gated progression from shared backbone to
personalized model to fully local model for identification quality, not a claim
of personalized-control superiority.

Experiment 10D removes oracle family labels from the sharing rule. Run
`PYTHONPATH=code/src:code/experiments OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python code/experiments/experiment_10d_learned_compatible_groups.py --workers 5`.
It fits mixtures of the frozen order-seven LPV regression directly to clients'
partial speed--matrix observations, enforces at least five clients per group,
and selects one to five groups by BIC. The selected label-free model reduces
full-envelope matrix error by 94.06% and nonlinear recovery error by 1.40%
relative to restricted Local, with positive fleet-bootstrap intervals. It also
improves matrix error by 34.27% relative to Global, while remaining 62.25%
worse than OracleFamily. BIC chooses two compatibility groups in nine fleets
and one in one; it never recovers the simulator's three nominal families.
Thus the result supports learned task-compatible sharing, not latent physical
family discovery. The learned model retains about 97.7% of OracleFamily's
matrix-error benefit and essentially all of its recovery benefit over Local.

Experiment 10E implements the latent-group estimator as a federated
sufficient-statistic protocol on fresh fleets. Run
`PYTHONPATH=code/src:code/experiments OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python code/experiments/experiment_10e_federated_latent_groups.py --workers 5`.
Clients retain their partial speed--matrix observations and transmit only
order-seven Gram/right-hand-side statistics, six-number residual summaries,
and candidate-group assignment scores. The complete search over `K=1..5` and
all deterministic restarts is included in the communication count. Federated
and centralized latent grouping select identical partitions, with maximum LPV
coefficient discrepancy `3.20e-14`. Under globally randomized label-free
coverage, FederatedLearned reduces matrix error by 93.41% and recovery error by
1.22% relative to restricted Local, while remaining feasible on every fleet.
It improves matrix error by 23.74% over Global, but their recovery differs by
only 0.031%; a stronger control-relevance test is therefore still required.

Experiment 10F performs that control-relevance test with strict
development/confirmation separation. Run
`PYTHONPATH=code/src:code/experiments OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python code/experiments/experiment_10f_control_relevance.py all --workers 5`.
Twelve LQI weight combinations are audited using ExactLPV only; the frozen
choice is `Q=diag(25,50,1000)`, `R=0.15`. Final confirmation uses fresh seeds
411--420, label-free coverage, speed-varying lateral-acceleration references,
steering amplitude/rate constraints, anti-windup, and dense-grid stability
checks. FederatedLearned improves mean yaw tracking over Global by 4.26% in the
moderate maneuver and 1.21% in the hard maneuver, with positive paired
bootstrap intervals. Hard-maneuver worst-client tracking improves by 3.31%.
All federated controllers are feasible and small-signal stable. The result
establishes control relevance, although the absolute gains over Global remain
small at 0.0018--0.0019 deg/s and must not be presented as a large practical
improvement.

Experiment 10G tests whether plausible compact, SUV, and sport classes plus
class-dependent steering-actuator dynamics create a stronger oracle control
advantage before extending the federated estimator. Run
`PYTHONPATH=code/src:code/experiments python code/experiments/experiment_10g_vehicle_class_gate.py all`.
The nonlinear tanh-tire plant is augmented by a first-order actuator, and
Global, oracle-class, and client-exact scheduled LQI controllers are compared
on ten untouched fleets after client-exact-only controller selection. The
prerequisite gate fails: OracleClass does not improve on Global in either
maneuver, and client-exact nominal LQI is less robust near nonlinear tire and
steering constraints. The negative result prevents an unjustified federated
extension and motivates an explicitly robust or constraint-aware control
question rather than further post-hoc LQI tuning.

Experiment 10H keeps the gain-scheduled LQI setting but replaces the easy
two-state identification problem by a five-state LPV model with front/rear
tire-relaxation and steering-actuator dynamics. Run
`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=code/src:code/experiments python code/experiments/experiment_10h_higher_order_lpv_gate.py all`.
Every client sees only one speed block and one steering-frequency band. Its
30-column LPV regression has rank 12--18, whereas every class aggregate has
full rank 30. On fresh fleets, Global pooling improves over an optimistic
noise-free LocalRestricted controller by 9.42%/2.55% in broadband/transient
mean tracking and 14.71%/13.65% in worst-client tracking. All paired intervals
are positive and every controller is stable and feasible. The strict 5% gain
gate fails for transient mean tracking, so 10H supports a strong fleet-tail and
calibration-coverage claim, not a uniformly large nominal-mean claim. Oracle
class specialization remains unnecessary in this setting.

Experiment 10H-A audits whether the latent sideslip and normalized tire-force
states used by 10H can be estimated from production-like signals. Run
`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=code/src:code/experiments python code/experiments/experiment_10ha_tire_force_observer.py all`.
A scheduled Kalman observer uses yaw rate, lateral acceleration, applied
steering, and speed. All exact systems are observable. On ten fresh fleets,
the shared GlobalKF obtains 0.048 deg sideslip RMSE and 7.96%/5.32% front/rear
normalized-force NRMSE; an ExactKF reaches 3.72%/3.12% force NRMSE. State
estimation therefore passes its gate. Directly feeding GlobalKF states into
the frozen 10H LQI does not: mean tracking degrades by 13--21% and worst-client
tracking by 37--45%, concentrated in SUV observer mismatch. Experiment 10H-A
supports tire-force observability but blocks an unqualified full-state-to-
output-feedback transition.

Experiment 10H-B isolates why the GlobalKF control interface fails. Run
`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=code/src:code/experiments python code/experiments/experiment_10hb_observer_mechanism.py`.
Using frozen 10H/10H-A fleets, controller, observer tuning, and common random
numbers, it compares ExactKF, oracle-class, and fleet-global observers both
with and without sensor bias. Within-class mismatch adds only 0.7--2.4% to
tracking error, while cross-class mismatch adds 11--34%; sensor bias adds
5.5--11.3%. OracleClassKF recovers 90--95% of the GlobalKF-to-ExactKF gap at
all mean/worst and broadband/transient endpoints. The audit therefore supports
class-compatible shared observers as a concrete CFL mechanism, while leaving
learned (non-oracle) grouping as the next required gate.

Experiment 10H-C removes the known-class assumption at the observer-grouping
level. Run
`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=code/src:code/experiments python code/experiments/experiment_10hc_learned_observer_groups.py --phase all --workers 5`.
Clients share only partial noisy LPV matrix-update summaries from their local
speed blocks. A multi-start hard mixture selects an unknown group count from
one to five by BIC; physical labels are used only for OracleClassKF and ARI.
On ten untouched fleets, BIC selects three groups in nine fleets and four in
one, with mean ARI 0.986. LearnedClusterKF improves over GlobalKF by
9.31--10.34% in mean tracking and 21.74--22.92% in worst-client tracking,
retaining 85--94% of the Global-to-Exact observer gap and matching the oracle
class observer closely. This establishes a label-free CFL observer-grouping
mechanism, but the update summaries are controlled noisy matrix surrogates;
end-to-end local output-error observer learning remains the next gate.

Experiment 10H-D performs that measured-output interface gate. Run
`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=code/src:code/experiments python code/experiments/experiment_10hd_measured_output_observer_groups.py --phase development --workers 5`.
Clients receive only yaw rate, lateral acceleration, applied steering, speed,
and steering command. A common nominal KF generates local pseudo-states, from
which regularized partial LPV updates enter the unchanged unknown-K grouping.
The development gate fails: modal K is two, mean ARI is 0.365, local matrices
have 52.2% relative error, and LearnedClusterKF improves over GlobalKF by only
0.7% in mean and 2.4--2.8% in worst-client tracking. The tiny pseudo-state
prediction residual is therefore not evidence of physical model recovery.
Confirmation seeds 551--560 remain sealed. The next method must jointly
estimate states and physically anchored parameters, rather than regress on a
single mismatched observer's pseudo-states.
