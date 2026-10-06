# Results

Every experiment writes to a dedicated subdirectory and records its configuration,
summary metrics, tables, and figures. Large generated artifacts are ignored by Git;
publication-ready selected figures and tables may be force-added once validated.

Expected structure:

```text
data/       generated datasets and serialized models
figures/    diagnostic and publication candidate figures
tables/     machine-readable and LaTeX summary tables
```

Experiment 10H-I uses `tables/experiment_10hi_*` and
`figures/experiment_10hi_physically_coupled_fit.pdf`. Part A is a paired
eight-coordinate structure comparison at the inherited covariance. Part B
selects covariance using only client-blocked inner training validation.
The execution manifest records the source revisions used for archived jobs.
The prediction comparison and innovation calibration outputs are post-hoc
diagnostics that do not change fits, selection, or predeclared gates.
