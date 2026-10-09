# Consolidation: full-envelope LPV identification from incomplete local coverage

Decision date: 9 October 2026. Evidence baseline: commit `aef68883d4caa4c97d06cfa244b13ba1671cb9a5`.
This is a scope and evidence consolidation. No new fitting or simulations are part of this decision.

## Primary contribution

> Clients with incomplete local operating coverage can combine compatible information through a federated protocol to identify reusable LPV models across the speed envelope. Scheduling represents operating variation; learned groups represent persistent differences between clients.

The primary outcome is full-envelope model identification. Scheduled LQI provides supporting evidence of model usability. The contribution does not require a large tracking advantage over a competent global controller.

The active evidence chain is **10A -> 10B -> 10D -> 10E**, with **10F** as control validation and **10C** as a boundary on personalization. Earlier experiments explain why the low-dimensional physical-ratio estimator did not require sharing. The 10G/10H and observer branches remain documented development evidence. They are paused and are not prerequisites for this paper.

## Fixed scope and assumptions

| Item | Consolidated scope |
|---|---|
| State | Two-state lateral model, `x = [beta, yaw rate]`. States are available for identification and feedback. No KF or inferred tire-force state. |
| Scheduling | Measured longitudinal speed, 10-30 m/s in the benchmark. |
| Representation | Existing orthogonalized seven-function span `[1, 1/v, 1/v^2, z, z^2, z^3, z^4]`, `z=(v-20)/10`. Keep the order and coordinates frozen. |
| Local information | Each vehicle supplies finite perturbation transitions at two or three speeds in one contiguous block. Compatible clients may jointly cover the envelope. |
| Heterogeneity | Unknown persistent compatibility groups; a single group remains an eligible outcome. No family labels in primary coverage assignment or fitting. |
| Federation | Additive regression statistics and local assignment scores; centralized grouped fitting is the numerical reference. |
| Primary evaluation | Full-envelope relative matrix error, with locally unobserved-speed error distinguished where existing tables support it. |
| Supporting evaluation | Nonlinear yaw tracking, worst-client errors, actuator limits and frozen-speed stability diagnostics under the existing LQI protocol. |
| Boundary | Controlled simulation proof of concept. No production-sensor, formal privacy, scheduled-stability, or unseen-recipient transfer claim. |

**Available states are an assumption at both stages.** Substituting true states only at feedback time in the current output-fitted N models is not the proposed simplification; Q already tested that intervention. The active formulation identifies matrices from state/input observations.

**Full envelope means the speed interval along the benchmark operating curve.** It does not mean an arbitrary nonlinear state/input envelope. The seven-function order was chosen using exact Jacobians on development fleets at prescribed cornering conditions; those exact matrices are representation-selection and evaluation targets, not finite-data fitting targets in 10B-10F.

The current finite-data construction has additional privileged setup information. `estimate_matrix` in `experiment_10b_oracle_complementary_gate.py` computes an equilibrium using simulator parameters, generates independent state/input perturbations about it, and centers the regression on that equilibrium. Current state/input regressors are exact, while next-state targets receive noise. The server and grouping algorithm receive estimated matrix entries and speed, not mass, inertia, stiffness, true matrices, or family labels. Nevertheless, simulator-derived centering is part of the identification pipeline and must be disclosed. Available states alone do not establish a learning interface that accepts only ordinary driving records.

## Method that the manuscript describes

1. Estimate local discrete state/input matrices from the available perturbation transitions at each observed speed.
2. Represent each group's stacked matrix entries by `M_g(v) = phi(v)^T Theta_g` in the frozen basis.
3. Fit the group maps through client statistics `G_i = Phi_i^T Phi_i`, `H_i = Phi_i^T Y_i` and group sums. Ridge regularization provides a numerical solution; genuine coverage still requires rank in the unregularized aggregate Gram matrix.
4. Broadcast candidate maps. Clients return variance-normalized residual scores on their own observed speeds. Alternate model fitting and hard group assignment, with the existing minimum-size rule and BIC selection over `K=1,...,5`.
5. Evaluate predictions throughout the speed envelope. Use the resulting maps for scheduled LQI as a separate downstream check.

This is hard-assignment alternating regression, not the later heteroscedastic EM, low-rank personalization or KF/LPV iteration. The emulator keeps trajectories local but transmits informative statistics and initialization summaries. It provides neither differential privacy nor implemented cryptographic secure aggregation. All candidate orders and restarts enter communication accounting.

## Existing evidence and permitted claims

| Evidence | Result from committed tables | Interpretation and limitation |
|---|---|---|
| 10A representation | At curvature 0.005 1/m, order 3 has 2.6607% mean blocked-speed matrix error; order 7 has 0.0007%. Straight-line selection retains order 3. | Richer speed dependence is justified in the audited nonlinear cornering condition, not for every vehicle task. |
| 10A coverage | Each local speed block has rank 2 or 3 for seven basis coefficients; the full speed union has rank 7 and condition 3.406. | The union supplies missing scheduling directions. This alone is not a statistical benefit or a guarantee that learned groups have sufficient coverage. |
| 10B finite-data oracle | On locally unobserved speeds, FamilyPool matrix error is 1.816% versus Local 66.121% and Global 4.296%; paired gains are 97.24% and 57.70%. | Compatible information can overcome restricted coverage. Family labels and centralized aggregation are oracle prerequisites, not the proposed algorithm. |
| 10C personalization boundary | FamilyPersonalized worsens prediction at restricted coverage; with eight/all eleven speeds it improves matrix error by 7.74%/13.66%, but gives no positive recovery gain over FamilyPool. Full-coverage Local is more accurate still. | Personalization is coverage-dependent and is not needed for the active restricted-coverage method. |
| 10D compatibility | Alternating regression learns groups from observed speed/matrix data without fitting to family labels. | Preparatory centralized method; primary label-free coverage evidence comes from 10E. |
| 10E primary label-free identification | Full-envelope matrix error: Local 49.562%, Global 4.250%, federated grouped 3.249%. Mean paired fleet gains: 93.41% vs Local and 23.74% vs Global; the latter 95% interval is [15.39%,31.04%]. | Main quantitative claim. These are whole-envelope errors, not an unseen-speed-only statistic. The same clients contribute fitting data and are evaluated at other speeds. |
| 10E federation | Same partitions and group counts as centralized fitting in all 20 fleet/protocol jobs; maximum grouped coefficient discrepancy 3.20e-14 and BIC discrepancy 1.87e-9. | Numerical equivalence of the implemented protocol, not a new convergence theorem. |
| 10E complexity | Label-free selection retains two groups in 8/10 fleets and one in 2/10. Mean full-search traffic is 2.243 MB per fleet, assuming 64-bit scalars. | One group is legitimate; exhaustive multistart search is not demonstrated communication-efficient. |
| 10F supporting control | Mean tracking gains vs Local: 23.86% moderate and 8.96% hard; vs Global: 4.26% and 1.21%. Absolute Global-relative gains: 0.001824 and 0.001874 deg/s. | Positive confirmation evidence of control relevance; practical headroom over Global is modest. |
| 10F feasibility | All federated evaluations satisfy the tested constraints; maximum frozen spectral radius 0.9759. | Empirical feasibility and frozen-speed diagnostics, not a scheduled or nonlinear stability certificate. |

The reported gains average paired fleet-relative changes; they are not ratios of the grand means. Ten fleets are the independent paired units for 10E label-free results and for 10F confirmation. The 10E controlled protocol uses family-stratified coverage and is an equivalence reference, not the primary application claim. The 10F controller weights were selected using ExactLPV development results only; discarded pilot seeds are disclosed in the historical protocol.

Evidence is not evidence for arbitrary new vehicles: fresh fleets replicate the complete fitting/evaluation procedure, whereas client membership is inferred from each evaluated client's own restricted records. Frozen-library transfer to a wholly held-out recipient is outside this consolidated claim. Global in these comparisons is fitted from all finite-data fleet observations; it is distinct from Q's public nominal controller.

## Baselines and fairness

| Baseline | Why retain it |
|---|---|
| Restricted Local | Establishes the cost of missing operating coverage. State the lower-order local model and nearest-observed-speed control rule explicitly. |
| LocalRegularized | 10B tests whether a higher-order local fit with a local prior can create the missing information; it does not improve on Local there. |
| Global LPV | Tests sharing without compatibility specialization under the same fleet observations. |
| Centralized learned groups | Separates the grouping/model benefit from distributed implementation. |
| Federated learned groups | Active proposed method. |
| Oracle family | Favorable compatibility reference, with missing-block failures retained. |
| LocalFull / ExactLPV | Distinguish full recipient coverage from sharing and finite-data error from approximation error. These have additional information and are not equal-budget deployable baselines. |

Sharing provides more total information than an isolated client. The claim is access to a full-envelope model without full-envelope collection by every vehicle, not superiority at equal total fleet data. Missing local coverage does not justify choosing an unnecessarily rich basis; the straight-line negative control and blocked-speed order selection remain part of the argument.

## One unresolved evidence boundary

The existing evidence is sufficient for a controlled available-state identification study. It does **not** yet validate an estimator that learns from ordinary sequential `(x_k, u_k, v_k, x_{k+1})` records using no simulator-provided equilibrium or exact matrix.

Before any new experiment, decide whether the manuscript deliberately retains controlled equilibrium-centered perturbation data. If so, consolidate the current draft with that limitation and do not add a generic-record claim. If state-record-only identification is required, the single next gate is a narrowly specified available-state regression interface: fit from supplied state/input/speed arrays and empirically available centering only, retain the existing basis and primary matrix-error endpoint, and audit that hidden physical parameters, true Jacobians and family labels never enter fitting or selection. Freeze that protocol before execution. No simultaneous KF, basis-order, plant-order, personalization or controller-cost changes belong in this gate.

No calibration-time saving percentage is currently supported by 10A-10F. Missing-block participation, errors in state regressors, real fleet data, and generalization to wholly unseen vehicles remain limitations or later work. They are not bundled into the next gate.

## Manuscript and repository disposition

- `paper/main.tex` is the active available-state identification draft. Its title, abstract, problem, method, evidence and limitations follow this scope.
- `paper/archive/9m_cold_start/` preserves the preceding personalized cold-start draft and its audits byte-for-byte. Its 17.27% headline belongs to a different method and is not reused here.
- The front of the development report states the current decision. Historical experiment chapters retain their original protocols and conclusions, including older planned next steps.
- 10H-Q and its restoration/audit artifacts remain intact. The current manuscript makes no joint KF/controller-improvement claim.
- The next manuscript task is evidence review and positioning against the existing IFAC contribution. LPV scheduling, compatible coverage and the distributed regression construction must carry the distinction; clustering and additive statistics alone are not novel claims.

## Source map

The compact verified evidence manifest is `results/tables/consolidation_identification_evidence.json`. Rebuild it without fitting or simulations using:

```bash
python paper/check_consolidation.py
make -C paper
make -C report
```

Primary source tables: `experiment_10a_selection.csv`, `experiment_10a_coverage_summary.csv`, `experiment_10b_summary.csv`, `experiment_10b_comparisons.csv`, `experiment_10e_summary.csv`, `experiment_10e_comparisons.csv`, `experiment_10e_diagnostics.csv`, `experiment_10f_summary.csv`, `experiment_10f_comparisons.csv` and their existing conclusions/provenance JSONs, all under `results/tables/`. These commands leave historical results and sources unchanged.
