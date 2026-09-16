# Paper-level method and evidence audit

Date: 2026-09-16

Target: IEEE Intelligent Vehicles Symposium 2027

Repository state audited through Experiment 9M

## Executive verdict

The project contains a scientifically defensible conference-paper core, but the
current `paper/main.tex` no longer describes it. That skeleton is centered on
known-family private aggregation, whereas the strongest final evidence concerns
unknown-group federated backbone learning, cold-start transfer, low-dimensional
local adaptation, and participation coverage. It should be replaced rather than
incrementally edited.

The defensible central claim is:

> A mature fleet can federatively learn latent, control-relevant LPV group
> backbones that regularize identification for a newly deployed,
> information-limited vehicle. The benefit is largest during cold start,
> survives unknown group membership and random partial participation, and
> vanishes as sufficient local excitation becomes available.

The blind-confirmed quantitative headline is the 9K Fed20 result: 17.27% lower
tracking RMSE than Local at 0.75 s, with nine of ten fleet-wise wins, followed by
practical parity at 1.25 s. The retrospective 9M fleet bootstrap gives a 16.90%
mean Fed20Rank1 improvement and a 95% interval of [11.79%, 21.10%]. These values
differ because 9K reports a ratio of aggregate means and 9M reports the mean of
paired fleet-wise relative improvements; both are valid when labeled precisely.

The scientific core is ready for manuscript design after the mandatory method
and accounting corrections below. No further broad exploratory campaign is
needed.

## 1. Audited method chain

### 1.1 Plant, identified model, and downstream controller

- The data-generating and evaluation plant is the nonlinear heterogeneous
  bicycle benchmark with nonlinear tire forces.
- Each client identifies a structured LPV approximation with longitudinal
  speed as the scheduling variable and three positive physical ratios,
  \(p_i=[C_{f,i}/m_i, C_{r,i}/m_i, m_i/I_{z,i}]^\top\).
- Identification uses batch output error on closed-loop local records. It is not
  streaming RLS and is not end-to-end neural federated learning.
- The learned ratios are converted into scheduled state-feedback/feedforward
  gains. Performance is evaluated by nonlinear closed-loop simulation.
- Feasibility and frozen spectral-radius checks are empirical diagnostics, not a
  nonlinear or time-varying LPV stability certificate.

Verdict: **internally consistent and well supported**, provided the paper clearly
separates the structured LPV model used for identification/controller synthesis
from the nonlinear plant used for evaluation.

### 1.2 Local uncertainty-aware summary

Client \(i\) computes a log-parameter estimate \(z_i=\log p_i\) and a
Gauss--Newton covariance \(C_i\), normalized by the declared public parameter
box. The covariance is derived from the local output-error Jacobian and known
output-noise scaling.

Implementation:

- `code/src/federated_lpv/output_error.py`
- `code/src/federated_lpv/uncertainty_clustering.py`

Verdict: **supported**, but the covariance is a local asymptotic/Gauss--Newton
approximation. It should not be called a calibrated posterior covariance.

### 1.3 Latent compatibility mixture

The fleet model is a heteroscedastic Gaussian mixture,

\[
 z_i\sim\sum_{g=1}^{K}\pi_g\,\mathcal N(\mu_g,C_i+S_g),
\]

where \(C_i\) represents local estimation uncertainty and \(S_g\) represents
within-group physical variation. Candidate orders \(K=1,\ldots,6\) are selected
by BIC subject to a minimum effective group size.

The federated EM-style update uses locally computed responsibilities and
additive group statistics. For group \(g\), the required quantities are
responsibility mass, precision, information vector, and uncertainty-corrected
scatter. The server updates \(\pi_g,\mu_g,S_g\) with fixed damping.

Implementation:

- Central reference: `code/src/federated_lpv/uncertainty_clustering.py`
- Federated path: `code/src/federated_lpv/federated_manifold.py`

Verdict: **methodologically valid as federated summary learning**, not FedAvg.
The paper should call it federated heteroscedastic mixture estimation or
federated empirical-Bayes representation learning.

### 1.4 Control-aware group backbone

For each learned group, the method builds a low-dimensional affine manifold in
log-parameter space,

\[
 z_i\approx \mu_g+U_g a_i.
\]

A positive metric \(W_g\) is formed from finite-difference sensitivities of the
scheduled controller gains to the log parameters. A weighted PCA/SVD then
constructs \(U_g\). The client-specific coordinates \(a_i\) are fitted locally
from the new client's short record with a small quadratic prior.

Implementation:

- Gain metric and weighted basis: `experiment_9f_personalized_manifold.py`
- Finite-data coefficient fit: `experiment_9g_finite_personalized.py`
- Distributed group moments/backbones: `experiment_9j_distributed_backbone.py`

Verdict: **supported with a terminology qualification**. The basis is
controller-gain-aware, not directly optimized for nonlinear tracking loss. The
paper should avoid the stronger phrase “closed-loop optimal manifold.” The final
backbone covariance uses hard group assignments and ordinary moments; it is not
debiased by the local estimation covariances after clustering.

### 1.5 Deployment of a new client

A mature fleet of 180 clients contributes to the shared representation. Each
mature client has 4.75 s of complementary calibration data. A separate new
client receives the learned backbones and uses only 0.75 or 1.25 s of local data
to select a group and estimate zero, one, or two continuous coordinates.

Verdict: **fair for the intended transfer setting**, but the comparison must be
framed as amortized fleet knowledge versus cold-start local calibration. It is
not a same-total-data comparison between Local and federation.

## 2. Privacy and trust audit

The final 9J--9M personalized method does not implement differential privacy or
cryptographic secure aggregation. It keeps raw trajectories and new-client
coordinates local by design and expresses the fleet updates as additive
statistics compatible with secure aggregation.

Experiments 8C--8G are scientifically useful but belong to an earlier,
known-family shared-mean formulation. Their privacy mechanism is not integrated
with unknown-group mixture learning or personalized backbones. The current paper
skeleton incorrectly makes that privacy mechanism the principal contribution.

Allowed wording:

- raw trajectories remain local;
- client updates are additive and secure-aggregation-compatible;
- personalization coordinates remain local;
- cryptography, resistance to inference from released models, and formal DP for
  the final method are outside the present paper.

Disallowed wording:

- private federated personalized LPV identification;
- formally privacy-preserving clustering;
- secure aggregation was implemented;
- the 8E--8G DP guarantee applies to the 9J--9M algorithm.

Recommendation: **remove privacy from the title, abstract, contributions, and
main results**. Retain one limitations paragraph and consider Experiments 8C--8G
as the basis of a separate privacy paper or supplementary research report.

## 3. Evidence-to-claim matrix

| Intended claim | Best evidence | Verdict | Required wording |
|---|---|---|---|
| Speed scheduling and structural heterogeneity are distinct | 6A--6B | Keep as motivation | Regime map within the tested nonlinear benchmark; not a universal architecture theorem |
| Structured LPV identification produces useful controllers on nonlinear plants | 4G, 6B | Keep briefly | Data-driven structured model approaches the physics reference on tested fleets and trajectories |
| The number of compatibility groups need not be known | 9A--9C, 9J--9M | Keep | BIC-selected latent groups in synthetic fleets generated from three populations |
| Fleet sharing improves cold-start control | 9H, blind 9K, 9M | Primary claim | 9K is confirmatory; 9M intervals are retrospective mechanism analysis |
| Continuous personalization causes the entire cold-start gain | 9M | Reject | Learned group centers provide most of the 0.75 s gain; the rank-one increment is uncertain |
| A local head becomes useful as information grows | 9M | Keep | At 1.25 s, Fed20Rank1 improves LearnedCenter by 4.21%, CI [2.40%, 6.36%], 10/10 wins |
| Rank two is required | 9H versus 9M | Reject | Rank ordering does not replicate; a very low-dimensional head suffices |
| Federated learning matches Central exactly | 9J development, 9K | Reject | Full participation is control-equivalent within 0.3%; the frozen 0.1% gate failed |
| Random 20% participation preserves the cold-start result | 9J--9K | Keep | Random rotating cohorts with nearly complete cumulative coverage |
| Arbitrary partial participation is safe | 9L | Reject | A fixed narrow subset can collapse group discovery and be 34.43% worse than Local |
| Rotating bias and 50% dropout are tolerated | 9L | Keep as secondary | Development evidence only; all tested fleets improved over Local at 0.75 s |
| Exact family recovery is necessary for good control | 9J--9L | Reject | Control can remain good at imperfect ARI; groups are control-useful latent components |
| Formal privacy for the final method | 8C--8G versus 9J--9M | Reject | Data locality and secure-aggregation compatibility only |
| Closed-loop stability is guaranteed | All experiments | Reject | Empirical nonlinear feasibility plus frozen linear spectral-radius audit only |

## 4. Statistical audit

### Supported

- Independent fleet seeds are used as the statistical units in the decisive
  studies.
- 9K is a genuinely frozen ten-fleet confirmation of the 9J development result.
- 9M resamples ten fleet-level paired effects rather than clients or scenarios,
  avoiding pseudoreplication.
- Failed gates are retained: 9H time-to-target, 9K 0.1% Fed100 equivalence, and
  the 9L persistent-subset fleet failure.

### Qualifications

- Ten fleets provide useful paired evidence but limited tail resolution.
- 9M uses already observed 9K fleets and is explicitly retrospective. Its
  bootstrap intervals quantify consistency; they are not a new preregistered
  confirmation.
- Multiple 9M comparisons are mechanistic/descriptive and do not carry a
  simultaneous family-wise error guarantee.
- Relative improvements must state whether they are a ratio of aggregate means
  or a mean of paired fleet-wise percentages.
- The 1% non-inferiority margin was introduced after earlier development results
  and should be described as a practical retrospective margin, not a prospectively
  registered clinical-style threshold.

## 5. Mandatory corrections before manuscript drafting

### M1. Replace the obsolete paper skeleton

`paper/main.tex` currently assumes known classes and a privacy-centered
contribution. Its title, abstract, contributions, method, and evidence map
conflict with the final method. Do not preserve this story during condensation.

### M2. Correct communication accounting

The 9J/9K communication tables count only the finally selected value of \(K\).
However, the implemented unknown-order selection fits all candidates
\(K=1,\ldots,6\). Therefore, the reported 1.160/0.670 MB (Fed100) and
0.258/0.138 MB (Fed20) are conditional selected-model payloads, not full
discovery-stage payloads.

The manuscript must report two quantities:

1. representation-discovery communication including all candidate orders;
2. selected-model communication when \(K\) is fixed or reused.

The scalar count must also be derived explicitly. Per group, the current local
statistics contain one mass, two symmetric 3-by-3 matrices, and one length-three
vector: 16 independent scalars, or 17 if a separate likelihood scalar is
included. The code currently hard-codes 17 without documenting that distinction.

This correction changes communication numbers, not tracking results.

### M3. Make every distributed operation explicit

The mathematical algorithm should show additive implementations for:

- initialization from first and second moments;
- local responsibility and likelihood evaluation;
- BIC comparison across candidate orders;
- final hard/soft group moment aggregation;
- broadcast of mixture state and learned backbones.

The simulation currently holds all summaries in one process and emulates these
local/server operations. That is acceptable, but the paper must not imply a
networked or cryptographic deployment.

### M4. Reconcile the personalization-order claim

9J--9K freeze rank one at 0.75 s and rank two at 1.25 s. The matched 9M ablation
shows rank one has lower mean tracking at both budgets, with no reliable rank-two
advantage at 1.25 s. The manuscript should not present rank two as necessary.

Safest wording: the blind-confirmed progressive policy remains the primary
reported protocol, while the retrospective ablation shows that a single local
coordinate is sufficient in these fleets and that exact rank selection is not
stable across studies.

### M5. Attribute the mechanism correctly

At 0.75 s, LearnedCenter versus Local is the cleanest all-fleet effect. The
increment from LearnedCenter to Fed20Rank1 has a confidence interval spanning
zero. The paper must attribute the initial benefit primarily to learned
heterogeneous fleet structure, with continuous personalization becoming clearly
useful at 1.25 s.

### M6. State the trust and information model

The paper must say what is public, local, released, and retained:

| Quantity | Status |
|---|---|
| Raw state/input/speed records | Local |
| Local structured estimate and covariance | Local input to message computation |
| Mixture parameters/backbones | Broadcast fleet model |
| Additive sufficient statistics | Released per cohort, ideally through secure aggregation |
| New-client group choice and coordinates | Local |
| Client identity/participation history | Server-visible in the simulated coverage audit |
| True family label | Evaluation only; never used by the learned method |

### M7. Separate empirical safety from guarantees

All final runs are feasible and the frozen linear audit is useful. Neither is a
proof of nonlinear closed-loop stability under time-varying scheduling. Use
“empirically feasible” and “frozen-point stability diagnostic.”

### M8. Clarify novelty relative to the accepted IFAC paper

The earlier IFAC work already contributes uncertainty-aware clustered federated
identification for controller design. The IV paper must not claim clustering
alone as new. Its distinct contribution is the combination of:

- explicit LPV scheduling versus persistent heterogeneity;
- learned low-dimensional group backbones;
- local cold-start adaptation for unseen clients;
- iterative additive-statistic federation with unknown group count;
- nonlinear closed-loop evidence and participation-coverage boundary.

This distinction should appear in the introduction and related work.

## 6. Recommended manuscript evidence set

### Main paper

1. **Structural motivation:** one compact 6A regime-map panel or table showing
   why both scheduling and heterogeneity matter.
2. **Method figure:** mature clients, local structured estimates/covariances,
   federated latent-group/backbone learning, and new-client local coordinate.
3. **Primary result:** 9K blind cold-start comparison with Local, Central,
   Fed100/Fed50/Fed20 and the failed 0.1% equivalence gate disclosed.
4. **Mechanism ablation:** selected 9M methods and the fleet-level confidence
   interval for Fed20Rank1 versus Local.
5. **Participation boundary:** concise 9L coverage-versus-performance panel,
   including the persistent-subset failure.
6. **Protocol table:** mature/new-client data, fleet counts, budgets, nonlinear
   scenarios, participation, metrics, and statistical unit.

### Supplement/report only

- Full oracle development chronology (0--6).
- Pooling and early personalization failures (7A--7C).
- Privacy campaign (8C--8G), unless developed as a separate paper.
- Detailed learned-group robustness development (9A--9E).
- Oracle manifold construction and intermediate gates (9F--9G).
- Active-selection negative result (9I).
- Full per-seed and per-scenario tables.

### Evidence to omit from the central narrative

- The 15-versus-30 controller-bank compactness result unless space remains.
- Claims that Fed20 is intrinsically better than Central; observed improvements
  are treated as stochastic finite-fleet effects.
- Exact physical interpretation of every learned component.
- Any claim that differential privacy has been solved for the personalized
  algorithm.

## 7. What is still needed

### Required before submission

1. Correct and rerun the communication audit, including model-order discovery.
2. Write a clean algorithm with client/server operations and trust assumptions.
3. Replace the paper skeleton with the final non-private personalized-FL story.
4. Consolidate all final comparison values from 9K and 9M into one source table
   to prevent metric-definition drift.
5. Add a concise comparison with the authors' earlier IFAC method.
6. Perform a final consistency audit of units: radians versus degrees, seconds,
   tracking definition, and aggregation convention.

### Valuable but not required

- A server-observable coverage safeguard motivated by the 9L failure.
- Networked secure-aggregation implementation.
- Formal DP for the mixture/backbone messages.
- Real-vehicle or public-dataset validation.
- Higher-dimensional LPV parameterization demonstrating scalability beyond
  three ratios.

## 8. Stop/go decision

**Go for paper consolidation**, subject to M1--M8 and the corrected communication
audit. The main closed-loop claim has both development and blind-confirmation
support. The project should not add another broad experiment before drafting.

The paper becomes scientifically weaker if it tries to combine the privacy,
active-learning, regime-map, clustering, personalization, and participation
stories as equal contributions. It becomes stronger if it presents one causal
chain:

1. scheduling handles operating variation;
2. latent groups handle persistent heterogeneity;
3. fleet backbones regularize cold-start identification;
4. a small local head adapts the prior as information grows;
5. additive federation preserves the effect under rotating partial
   participation;
6. insufficient cumulative coverage is the identified failure boundary.
