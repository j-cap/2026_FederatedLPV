# Units, metrics, and aggregation audit

## Canonical paper metrics

| Quantity | Definition | Unit | Aggregation/statistical unit |
|---|---|---|---|
| Tracking RMSE (9J--9M) | RMS yaw-rate tracking error over time | rad/s | client/scenario mean within fleet; fleet seed is independent unit |
| Tracking RMSE (4G, 6A--6B) | same physical error after `rad2deg` | deg/s | used only for structural motivation; convert before any numeric cross-phase comparison |
| Parameter error | Euclidean norm of log-ratio error | dimensionless | mean within fleet, then across fleets |
| Gain error | relative Frobenius norm of scheduled gain tables | dimensionless | mean within fleet, then across fleets |
| Feasibility rate | finite simulation and steering/acceleration limits satisfied | fraction | observations or fleets as explicitly labeled |
| Frozen radius | maximum spectral radius over frozen scheduled linear models | dimensionless | diagnostic only; not a nonlinear stability proof |
| ARI | adjusted Rand index against synthetic generating populations | dimensionless | evaluation only; true family labels are not used for learning |
| Communication | independent float64 payload scalars multiplied by 8 | bytes | upload and download reported separately |

The paper uses rad/s for its primary 9K/9M results. Any retained 4G/6A/6B value must be converted to rad/s or confined to a separately labeled structural figure in deg/s.

## Percentage conventions

The blind 9K comparison reports a ratio of aggregate means,
\[
100(\bar e_{\mathrm{method}}/\bar e_{\mathrm{reference}}-1).
\]
Thus Fed20 is 17.27% below Local at 0.75 s. The retrospective 9M statistic first computes the paired percentage within each fleet and then averages those ten percentages; it reports a 16.90% improvement with a fleet-bootstrap 95% interval of [11.79%, 21.10%]. These are different estimands and must retain distinct labels.

## Statistical conventions

- Fleet seeds, not clients or maneuvers, are the independent units.
- Confidence intervals resample paired fleet effects.
- Scenario/client observations are averaged within each fleet before inference.
- 9K is the frozen confirmation; 9M is a retrospective mechanism ablation on the same fleets.
- The 1% late-budget non-inferiority margin is retrospective and practical, not preregistered.
- All error bars must state whether they are across-fleet standard deviations or bootstrap confidence intervals.
