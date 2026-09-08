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
