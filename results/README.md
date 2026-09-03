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

