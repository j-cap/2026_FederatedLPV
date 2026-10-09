# Novelty and overlap boundary

## Relationship to the accepted IFAC paper

The accepted IFAC study supplies the closest methodological foundation: clients estimate compact physical parameters with uncertainty, the fleet discovers compatible groups, and group information supports controller design. The IV paper must cite that work explicitly and must not claim clustering, uncertainty-aware grouping, or compact physical summaries as new in isolation.

| Dimension | Accepted IFAC work | Proposed IEEE IV paper |
|---|---|---|
| Dynamics | fixed-form/local vehicle models | speed-scheduled LPV representation evaluated on a nonlinear varying-speed plant |
| Estimation | local recursive estimates and covariance-aware clustering | batch output-error estimates with heteroscedastic latent-mixture federation |
| Group count | clustering-focused compatibility structure | unknown (K\in\{1,\ldots,6\}) selected by federated BIC statistics |
| Shared representation | group centers/controllers | control-gain-weighted low-rank group backbones |
| Personalization | group membership/controller assignment | unseen-client local coordinate fitted from 0.75--1.25 s of data |
| Main question | identify compatible groups for control | whether prior fleet knowledge improves cold-start LPV controller calibration |
| Federation stress test | limited | rotating partial participation, biased participation, and a persistent-coverage failure boundary |
| Primary evidence | held-out clients | frozen blind fleets, nonlinear closed-loop tracking, paired fleet statistics, communication audit |

## Defensible novelty claim

The contribution is the combination of (i) explicit LPV scheduling, (ii) federated unknown-order latent backbone learning, (iii) a control-gain-aware low-rank representation with a private local head for an unseen client, and (iv) a demonstrated coverage boundary under partial participation. The scientifically strongest result is transfer during information-limited deployment, not improved fitting of previously observed clients and not clustering accuracy by itself.

## Claims to avoid

- first federated clustering method for vehicle dynamics;
- formal privacy or implemented secure aggregation;
- a closed-loop-optimal manifold;
- universal superiority to local identification;
- a nonlinear stability guarantee;
- exact recovery of physical vehicle families.
