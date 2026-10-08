# 10H-M reconstruction: freeze actual donor libraries

The original M–P sources and learned coefficients are unavailable. This is a
fresh reconstruction authorized on 2026-10-08, not recovery of those executions.
Keep the locked I–L inputs unchanged. Commit and push this protocol before the
run and the frozen results before starting N. Confirmation seeds stay sealed.

Use the ten intact 10H-L development jobs (601–605, Q_fit 0.01 and 1). Verify
their source/output hashes and reconstructed measured-array hashes. For each
of the two inner folds, retain its actual ten-donor K1/K2 fitted models. Choose
a numerically eligible K2 restart and fallback margin using that fold's ten
validation clients and the existing mean log forecast-error ratio at h=5,20,50.
Ties within 1e-12 prefer larger margin, then smaller restart. Never refit on all
donors, average matrices, or use outer scores to choose a library. If no eligible
K2 exists, deploy the eligible K1 only. Freeze all three plants and the margin.

For each outer recipient use only its first-record 100-sample measured prefix
to choose K1 or one of the two components by the inherited h=5,20,50 score and
margin. Save these choices, coefficients, donor/validation/recipient positions,
and hashes. Score the same outer causal forecasts as L. Average the two donor
deployments per recipient before fleet aggregation. This is development evidence;
it does not establish a transfer gate or recover the original M result.
