# Experiment 10H-Q: isolate the feedback bottleneck

## Purpose and frozen parent

Identify whether the limited 10H-P tracking benefit comes from state reconstruction,
the learned controller model, or limited headroom under the fixed maneuvers and
LQI costs. This is an opened-development diagnostic, not confirmation or a new
federated-transfer claim. Freeze this protocol before any 10H-Q rollouts.

The required parent is the completed 10H-P checkout at `93f24cc`, with its original
execution inputs, selected 10H-N model libraries, recipient assignments and
provenance locks. Do not refit, reselect memberships, tune bounds/covariances,
replace the parent with 10H-L models, or infer model coefficients from reported
metrics. If the parent inputs are unavailable, record a blocked preparation
status and do not report experiment results.

Inherit both donor-library deployments, development seeds 601–605, ten held-out
recipients per seed, both maneuvers, noise/bias draws, quiet-bias correction,
initialization, sample timing, scheduling grids/interpolation, steering clipping,
measured-yaw integrator, antiwindup, LQI costs and metric windows from 10H-P.
The two fitting panels remain separate. Primary diagnostic panel: Q_fit=0.01;
comparison panel: Q_fit=1. These panels differ in learned plants and noise
settings; their difference is not a causal estimate of Q's effect. Seeds 591–600
and 611–620 remain sealed.

## Controller/state-source factorial

Run all nine combinations of controller design {nominal, learned, simulator
oracle} and supplied plant state {nominal KF, learned KF, true state}. Add three
learned-controller/learned-KF cases with truth corrections {beta only, both axle
forces together, beta plus both axle forces}. Thus there are twelve cases and
4,800 planned rollouts: 5 seeds × 10 recipients × 2 donor deployments × 2 panels
× 2 maneuvers × 12 cases. Do not select a better donor deployment per recipient.

The five plant-state coordinates are [beta, yaw rate, F_yf/m, F_yr/m, applied
steering]. Preserve these coordinates in every design. An oracle controller uses
the simulator's actual discrete matrices sampled at the same controller-design
grid, with the same costs and interpolation as the other controllers. It does
not optimize tracking weights or use family labels. It is a diagnostic reference,
not a guaranteed optimum for tracking RMSE or an LPV stability certificate.

True-state cases replace only the five states passed to the controller. The
integrator continues to use the current noisy measured yaw error. Preserve all
other reference, limit and antiwindup terms. A partial correction replaces the
specified entries in the controller input only: never overwrite the KF's
internal state. The observer predicts using the actual clipped command and
corrects using the resulting measured outputs. All outputs remain causal under
the 10H-P step convention. Log raw KF state and supplied controller state
separately. In full-state cases keep the chosen observer out of the command path.

Use identical exogenous noise and references for paired rollouts. Different
controllers legitimately generate different inputs, states and integrator
histories; do not interpret their own-trajectory state RMSE as a common-data
observer comparison. Retain failed designs/nonfinite runs in the failure counts.

## Common-input observer replay and command-error decomposition

Replay nominal and learned KFs on each fixed nominal and learned-Both parent
record. Include a true-model KF with the same Q/R and bias handling as an oracle
diagnostic. Each replay receives identical command, speed and measurement arrays,
and never receives latent states. Latent states enter only the scoring boundary.
Do not apply alternate observer commands to the recorded plant. Replay with the
same current-estimate initialization and predict/correct convention as 10H-P.

For each fixed controller gain K_x(v), evaluate the pre-clipping error

    delta_u = u(x_hat, eta, reference) - u(x, eta, reference)
            = -K_x(v) (x_hat - x).

The instantaneous identity holds at the same integrator state and reference;
it does not decompose total tracking loss across different closed-loop paths.
Report raw state RMSE, individual command contributions, the signed joint force
contribution, total command-error RMS, and the contribution second-moment matrix.
Preserve cross terms: a sum of component RMS values is not total command RMS.
Use steering radians internally and convert command summaries to degrees only
when reporting. Never combine mixed-unit state errors into one raw norm.

## Paired endpoints and interpretation

For each recipient average its two separate deployment metrics first. Compute
fleet means and paired percentage gains within panel/maneuver; five fleets are
the paired units. Also report recipient improved/degraded counts, median/p90/max,
command effort/rate, saturation, violation counts and numerical failures. Keep
both all-sample tracking and inherited post-warmup state/command diagnostics.
Retain frozen-speed feedback checks and the saved-step/full-path replay distinction
from 10H-P; do not conceal sensitivity by loosening a tolerance.

Prespecified comparisons:

1. Learned versus nominal controller with true states: learned control-design
   benefit after removing plant-state estimation error.
2. Oracle versus nominal controller with true states: attainable reference
   headroom under the fixed cost/scheduling/maneuver setup.
3. Each controller with learned KF versus the same controller with true states:
   feedback performance lost to state reconstruction and its interaction.
4. Each partial correction versus learned-Both, and versus full-state learned
   control: whether beta/force corrections recover tracking benefit. Corrections
   can interact; do not force an additive explanation or rank using one mean.
5. Common-input nominal/learned/oracle KF comparisons: whether steering-relevant
   reconstruction improves and whether force errors cancel or amplify.

A large learned-controller gain with true states that disappears with learned
KF supports an observer/interaction bottleneck. Poor learned full-state control
when oracle full-state control improves supports a control-model/design bottleneck.
Little oracle headroom supports reconsidering this fixed task/cost setup. Mixed
outcomes must remain mixed; no single diagnosis is required. Report all panel,
maneuver and tail outcomes. No post hoc deployable winner or blind-confirmation
gate is created. All oracle interventions are simulation diagnostics; truth and
labels remain excluded from fitting, assignment and deployable methods.

## Recovery and execution status

At preparation time the current workspace lacks the earlier checkout and GitHub
main is `91bbdd6` (10H-L). The parent `93f24cc` is unavailable on GitHub and the
10H-M–P source/models cannot be recovered here. Prepare reusable calculations and
tests on a separate branch, but defer the rollout adapter and all fleet results
until the original committed checkout is restored. After recovery, verify its
provenance, commit the adapter before execution, reproduce the four overlapping
10H-P cases, then execute the complete locked comparison. Historical files stay
unchanged. Commit locally; pushing is a separate action.
