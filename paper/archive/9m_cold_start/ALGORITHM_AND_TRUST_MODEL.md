# Federated algorithm and trust model

## Objects and roles

Previously observed client (i) retains its trajectory \(D_i\) and computes a three-dimensional log-parameter estimate \(z_i=\log p_i\), with \(p_i=[C_f/m,C_r/m,m/I_z]^T\), and a Gauss--Newton covariance \(C_i\). The coordinator learns latent group distributions and group-specific low-rank backbones. A previously unseen client downloads those backbones, selects a component, and estimates its private coordinate from a short local record.

## Discovery protocol

For each candidate order (K=1,\ldots,6):

1. Clients contribute additive counts, first moments, and symmetric second moments for deterministic initialization (10 independent scalars per contribution).
2. For each of 30 fixed EM rounds, the server broadcasts \((\pi_g,\mu_g,S_g)\) (10 scalars per group). Each participating client computes responsibilities locally and contributes, for every group, responsibility mass, precision, information vector, and symmetric scatter (16 scalars per group).
3. Clients contribute the (K) responsibility masses and one log-likelihood scalar used for the BIC score. The server rejects components below the minimum effective size and selects the admissible order with minimum BIC.
4. With the selected assignments, clients contribute group counts, first moments, and symmetric second moments (10 scalars per group). The server forms the group centers and the controller-gain-weighted rank-one basis.

The reported algorithm is an EM-style federated sufficient-statistic method, not FedAvg. Fixed damping, iteration count, candidate orders, and minimum component size are frozen by the experiment configuration.

## New-client personalization

For each learned group (g), the new client solves its local output-error problem restricted to
\[
z=\mu_g+U_g a,
\]
including the stated quadratic coordinate prior. It selects the group with the best local criterion and keeps (a) local. The resulting positive ratios are mapped to scheduled controller gains and evaluated on the nonlinear plant. Rank zero is a learned group center; rank one adds one client-specific continuous coordinate.

## Trust and visibility

| Object | Client | Aggregator | Released/broadcast |
|---|---|---|---|
| Raw trajectory (D_i) | retained locally | never received | no |
| (z_i,C_i) | computed locally | not required individually by the additive protocol | no |
| Per-round sufficient statistics | computed locally | cohort sums | no individual contribution |
| Participation identity | known in the simulation/accounting | visible | no anonymity claim |
| Mixture state and group backbones | downloaded | computed | yes |
| New-client group and coordinate | local | not required | no |
| True synthetic family | evaluation only | evaluation only | no training use |

The repository emulates client and server operations in one process. Its updates are compatible with an additive secure-aggregation layer, but no cryptographic protocol or differential privacy is implemented for Experiments 9J--9M. Consequently the supported claim is data locality and secure-aggregation compatibility, not formal privacy.

## Communication convention

Payload counts use independent float64 scalars and include every candidate order during discovery. They exclude transport headers, authentication, cryptographic expansion, retransmissions, and the one-off download by a new client. Two costs are reported: complete unknown-(K) discovery and a fixed-(K) update when the selected order is reused.
