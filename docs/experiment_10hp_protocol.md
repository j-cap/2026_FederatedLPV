# 10H-P reconstruction: core output-feedback baseline

Rebuild the four core nominal / learned KF / learned LQI / learned Both methods
from audited reconstructed N, not the unavailable historical P. 1,600 rollouts:
5 development seeds × 2 Q panels × 2 donor libraries × 10 recipients × 2 maneuvers
× 4 methods. No new fit, reassignment, covariance calibration or cost tuning.
Original O and local/global extensions are deferred; this core is sufficient
for the frozen Q bottleneck diagnostic. Push the protocol before execution,
and the audited results before Q. Confirmation seeds remain sealed.

Use public HF nominal coefficients, selected N plants/prefix assignments and
the historical H03 frozen LQI costs. Generate true benchmark matrices with the
intact H simulator on its 2 m/s grid. Design KF/LQI schedules on a 1 m/s grid,
ZOH each learned continuous model at grid points, then linearly interpolate.
No physical process noise is added. Use the original 12-second H broadband and
transient references. All methods, panels and donor deployments share the same
exogenous per-recipient/maneuver draws: seed=3900000+1000*fleet_seed+10*position
+scenario_index, sensor bias plus independent 100-sample quiet calibration and
measurement noise from HA. This RNG convention is a new reconstruction choice.

KF gain uses DARE prior P without an extra covariance prediction. At step k,
compute command from the current corrected state, current integrator and reference;
clip at ±15 degrees. Integrate current bias-corrected measured yaw error, freezing
an increment that would worsen command saturation. Propagate true plant and KF
with the actual clipped command and correct with y[k+1]. Start physical state
and integrator at zero, correct initial observer mean using y[0]. Report all-sample
yaw tracking and state errors after 2 seconds, command effort/rate, saturation,
6-degree beta and 15-degree applied-steer diagnostic violations and failures.

Freeze matrices/gains/assignments; save regenerable trajectories outside git with
hashes in committed tables. Audit each saved equation independently and distinguish
strict saved-step (1e-10) from independent full-path replay (1e-8). Keep sensitivity
failures visible rather than relaxing tolerances. Sample frozen unsaturated feedback
spectral radius at 201 speeds as a diagnostic, not an LPV stability certificate.
Average two deployment metrics per recipient, then fleet means; five fleets are
paired units. Retain individual tails. Q_fit panels also differ in learned plants,
so a panel difference does not isolate covariance's effect.
