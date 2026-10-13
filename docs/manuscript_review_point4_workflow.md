# Review point 4: vehicle calibration and controller implementation

This clarification pass audits the existing implementation and revises the
active manuscript. It introduces no new fleet, fitting, controller tuning,
or scientific simulation run. Frozen sources, configurations, calibration
targets, model fits, selection records, and results retain their original bytes.

## Calibration workflow

For each client's assigned two- or three-speed block:

1. At each speed v, fix curvature kappa0=0.005 1/m and yaw rate r*=v*kappa0.
2. Solve for beta* and steering u* such that both continuous plant derivatives
   vanish. This setup uses the simulated client's physical parameters.
3. Draw 36 independent zero-mean Gaussian state and steering perturbations.
4. Reset the state to x*+Delta x and hold u*+Delta u and v fixed during one
   0.01 s RK4 step. There is no sequential carry-over between records.
5. Add two-state Gaussian observation noise to the next state, subtract x*,
   and regress the centered next-state targets on [Delta beta, Delta r, Delta u].
   The three-column design must have rank three. Current regressors are exact.
6. Pass only the estimated six matrix entries and their observed speeds to
   local/group LPV fitting and client assignment.

Source: `estimate_matrix` and `rk4_step` in
`code/experiments/experiment_10b_oracle_complementary_gate.py`, and
`equilibrium` in `experiment_10a_basis_coverage_audit.py`.

The recipient budget is 72 or 108 independently initialized transitions.
The Gaussian perturbation standard deviations are about 0.25 degrees for
beta, 2 degrees/s for yaw rate, and 0.30 degrees for steering. Next-state
noise standard deviations are 0.01 degrees and 0.03 degrees/s.

Exact matrix references are separately computed by centered finite differences
of the continuous nonlinear dynamics (step 1e-6), then an augmented matrix
exponential at 0.01 s. They use the same speed/equilibrium curve. They enter
development representation/penalty tuning and evaluation, not finite-data
group fitting. RK4 records have finite perturbations and are not analytically
generated Jacobian targets.

## Controller synthesis and runtime workflow

`experiment_10f_control_relevance.py` builds fresh calibration/model fits for
each confirmation fleet. These maps stay fixed during each maneuver. It uses
the shared `design_lqi_gain` and `augmented_tracking_matrices` helpers in
`code/src/federated_lpv/control.py`.

For yaw output c_r=[0,1], the augmentation is

$$
A_a=\begin{bmatrix}\widehat A&0\\T_sc_r&1\end{bmatrix},
\qquad B_a=\begin{bmatrix}\widehat B\\0\end{bmatrix}.
$$

The integral sign is yaw minus reference. Discrete Riccati synthesis uses
Q=diag(25,50,1000), R=0.15, with a controllability check before the solve.
For gain K=[K_x,K_xi], solve

$$
(I-\widehat A)h-\widehat Bw=0,\qquad c_rh=1,
\qquad F=w+K_xh.
$$

This defines the homogeneous linear-model prefilter at unit yaw reference;
it does not identify a nonlinear turning equilibrium or an affine offset map.
Local substitutes its nearest observed-speed matrix at unobserved synthesis
nodes. Every method then linearly interpolates its gain and prefilter arrays
on the same eleven-speed grid. The contiguous local blocks make Local's
outside-block design equal to the corresponding boundary-speed design.

At sample k, with reference r_ref=a_y_req/v,

$$
u_k^o=-K(v_k)[\beta_k,r_k,\xi_k]^\top+F(v_k)r_k^{ref}.
$$

The implementation proceeds in this order:

1. Interpolate the gain and prefilter at the current prescribed speed.
2. Calculate the unconstrained command from the **absolute**, noise-free
   available beta and yaw rate, current integral, and yaw reference.
3. Clip the command to u_previous +/- Ts*120 degrees/s, then +/-12 degrees.
4. Advance the nonlinear plant one RK4 step with applied steering and speed
   held fixed over that step.
5. Form the clipped integral candidate xi+Ts*(r-r_ref), bounded by +/-0.6 rad,
   using the current sample's yaw error.
6. Accept the candidate if the command is effectively unsaturated or its
   integral contribution counteracts the limiter. Otherwise retain xi.

The exact acceptance condition in `simulate_tracking` is

$$
u_k\simeq u_k^o\quad\text{or}\quad
\operatorname{sign}(u_k^o-u_k)\ne
\operatorname{sign}[-K_{\xi}(v_k)(r_k-r_k^{ref})].
$$

The approximate equality uses NumPy's `isclose` defaults in the frozen code.
Initial physical states, integral, and previous steering are zero. Physical
plant parameters enter nonlinear integration, but they are not an argument
to gain/prefilter synthesis or the online control law. Calibration centering
offsets are neither subtracted from runtime feedback nor added to steering.

## Evidence boundaries

| Stage | Benchmark information | Information supplied to the evaluated method |
| --- | --- | --- |
| Record generation | Physical parameters determine each equilibrium and RK4 transition. | Exact current deviations, perturbed steering, speed, noisy centered next-state targets. |
| Finite-data LPV fitting/assignment | Simulator knows fleet families and exact Jacobians. | Observed speeds, estimated matrix targets, and specified local/distributed statistics. |
| Development tuning | Exact Jacobians choose the basis/local penalty; ExactLPV tracking chooses controller weights. | A frozen representation, penalty protocol, and common Q/R for confirmation. |
| Controller synthesis | Each method's own fitted matrices; ExactLPV is a privileged exact-Jacobian reference. | A/B predictions, the fixed speed grid, sample time, Q/R, and Local's observed support. |
| Runtime tracking | True parameters generate the plant response. | Absolute beta/r, current speed, reference, scheduled gain/prefilter, integral, previous command. |
| Stability diagnostic | ExactLPV reference Jacobian map on the calibration curve. | Interpolated gains and 81 frozen-speed augmented eigenvalue checks. |

Calibration is a controlled, simulator-centered experiment. Available states
alone do not imply a demonstrated ordinary-driving estimator; measured
operating-point centers remain an untested replacement for privileged centering.

The control law applies matrices calibrated at curvature 0.005 1/m to variable
turn requests and absolute states. It therefore demonstrates empirical
nonlinear tracking usability, rather than reconstructing or compensating a
scheduled family of equilibrium offsets. No scheduled/nonlinear stability
guarantee follows from the frozen-speed unsaturated linear diagnostic.

The moderate/hard 12 s profiles, limits, reference definition, and all frozen
numerical results are unchanged. Tracking RMSE is computed per client over
all 1201 state samples, including the terminal sample; fleet means average
30 clients, followed by ten fleet means. Smooth Local's matched matrix-error
test does not extend to this separate LQI comparison.

Weights were frozen from ExactLPV-only development fleets 381--385 using
feasibility, mean saturation, and frozen-speed spectral-radius gates. The
discarded low-intensity pilot used fleets 391--400. Confirmation fleets
411--420 do not select weights or maneuver intensity.

## Verification

```bat
python paper/check_consolidation.py
python paper/check_review_integration.py
python paper/check_local_baseline_integration.py
```

The existing control, nonlinear-calibration, and 10F tests check the shared
augmentation/prefilter helpers, feasible turning equilibria, and maneuver
speed profiles. Reproduction of the original confirmation experiment is not
needed for this text clarification. No new simulation command is required.

All 10 existing calibration/control/10F tests pass. All three evidence checks
pass, and the six directly traced source/configuration/selection files are
byte-identical to the preceding `7e8379d` checkpoint. The revised IEEEtran PDF
remains seven pages; all pages were visually inspected. There are no overfull
boxes, unresolved-reference warnings, or column-balance warnings. Minor
flush-bottom underfull spacing messages remain. No scientific source, result,
or tuning record was modified to obtain this layout.
