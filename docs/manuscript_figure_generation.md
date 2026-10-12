# Manuscript figure preparation

The visual revision is integrated in `paper/main.tex` as Figures 1--4. The
generator reads committed, frozen results only. It performs no model fitting,
plant simulations, controller retuning, or changes to manuscript numbers.

## Overview draft with PaperBanana

Use the complete brief in [paperbanana_overview_prompt.txt](paperbanana_overview_prompt.txt).
It specifies the scientific content, two-panel layout, typography, exchanged
quantities, and boundaries of the claim. Treat the generated image as an author
review draft: inspect mathematical labels and arrow directions before using it.
Keep the final overview editable when possible and save the approved paper
figure as `results/figures/paper_method_overview.pdf`. A vector-style generated
image is not necessarily a vector file.

The method diagram must show additive statistics **and** the other transmitted
summaries/scores. It must not imply known vehicle-family labels, noisy-state
reconstruction, cryptographic privacy, transfer to unseen recipients, or
guaranteed rank recovery for arbitrary partitions.

## Generate the evidence figures

From the repository root in **Windows Command Prompt**:

```bat
git pull --ff-only
conda activate federated_lpv
python paper/prepare_identification_figures.py --preflight
python paper/prepare_identification_figures.py --generate
```

If the environment has not yet been created, first run
`conda env create -f environment.yml` from the repository root. Plotting requires
only NumPy, pandas, and Matplotlib from that existing environment. No LaTeX
installation is needed. Do not use Unix-style environment-variable prefixes in
Windows Command Prompt.

`--preflight` is read-only. It verifies 22 input hashes and cross-checks raw
records against frozen fleet summaries, regional summaries, client tradeoffs,
basis selection, local ranks, and selected-group diagnostics. CSV/JSON hashes
normalize CRLF to LF so Windows checkout line endings are accepted. A changed
input is rejected; do not update the lock simply to bypass a failure.

`--generate` produces these combined figures in `results/figures/`:

| File | Content | Manuscript placement |
| --- | --- | --- |
| `paper_basis_information.pdf` | Basis-order validation and observed-speed/rank audit | Figure 2, referenced in Sections V-A and V-C |
| `paper_identification_coverage_v2.pdf` | Four speed-error panels with distinct line styles and explicit observed grid marks | Figure 3, Section V-B; replaces the older speed-error figure |
| `paper_fleet_client_tradeoff.pdf` | Paired full-envelope fleet errors and client unobserved-error changes | Figure 4, Section V-B |

Each combined figure has a PNG preview. Each panel also has a separate PDF and
PNG, with suffixes `_a`, `_b` (and `_c`, `_d` for coverage). There are **11 PDFs
and 11 PNG previews** in total. Separate panels permit LaTeX composition without
embedding panel labels in each panel image. PDF fonts are embedded; the output
uses consistent serif typography and physical sizes suitable for IEEE columns.

The generation manifest is `results/tables/paper_visuals_manifest.json`. It
records input/output hashes, package versions, generator hash, numerical checks,
and statistical/display conventions. PNG previews are ignored by git; PDFs and
the manifest are intended to be committed. Existing tables, figures, and
`paper/main.tex` are left unchanged.

For a scratch preview, use `--generate --output-dir YOUR_DIRECTORY`. In that
case the manifest is saved in that directory as well.

## Interpretation and captions

**Basis and information figure:** (a) Mean blocked-speed relative matrix error
over ten development fleets; error bars are one standard error across fleets.
The straight-line negative control and the selected cornering condition are
shown. Rings mark the frozen selected orders. The dotted line is the 0.01%
practical selection floor, not a noise estimate. (b) Filled cells mark observed
grid speeds. The four local blocks have ranks 3, 3, 3, and 2 for seven basis
functions; the development fleet union has rank seven. The last row summarizes
all 18 audited learned groups from the ten primary fleets, each with eleven
speeds and rank seven. Their stacked-design condition numbers range from 3.26
to 3.87. Development union coverage and empirically verified learned-group
coverage are distinct checks. Full rank does not remove within-group bias or
guarantee recovery of physical families.

**Fleet and client figure:** (a) Paired Global and Grouped full-envelope matrix
errors for all ten independent primary fleets, in seed order. A connector joins
the methods within each fleet; `K=1` marks the two effectively tied fleets.
The mean paired relative gain is 23.74%, with the original frozen fleet-bootstrap
95% interval [15.39, 31.04]%. (b) Empirical cumulative distribution of changes
in client mean unobserved-speed error, expressed as **percentage points**:
`100 * (Grouped error - Global error)`. Negative values are improvements,
positive values are deterioration. Counts are 169 better, 60 tied, and 71 worse,
using the existing numerical tie tolerance of `1e-12` in dimensionless error.
These 300 client instances are nested within ten fleets. This panel is
descriptive and does not treat clients as independent statistical replicates.

**Speed coverage figure:** Curves average client errors within block and fleet,
then average the ten fleets. Bands are pointwise 95% fleet-bootstrap intervals,
using 20,000 draws and the same random seeds as the existing figure. Light
shading indicates a local block; short green marks identify the actual observed
grid speeds. The log scale preserves Local extrapolation failure. These are
alternative views of the primary records, not additional confirmation fleets.
The separate fleet/client figure resolves the smaller Global/Grouped difference.

## Commit the generated results

Review the three combined PNG previews, then use these commands from the
repository root (Windows Command Prompt):

```bat
git status --short
git add "results/figures/paper_basis_information*.pdf"
git add "results/figures/paper_fleet_client_tradeoff*.pdf"
git add "results/figures/paper_identification_coverage_v2*.pdf"
git add results/tables/paper_visuals_manifest.json
git diff --cached --stat
git commit -m "Generate manuscript identification figures from frozen evidence"
git push
```

The author's overview PDF has been reviewed and integrated as Figure 1,
using `results/figures/paper_method_overview.pdf`. The caption explains the
seven-function basis, conditional rank recovery, illustrative groups, and
controlled calibration setting. The three committed combined diagnostic PDFs
are included unchanged as Figures 2--4, with final captions and results
references. Their hashes match the generation manifest, and the frozen-input
preflight passes. The result tables remain alongside the figures to retain
the baseline and observed/unobserved endpoint comparisons. A control
time-history figure is outside this integration; it would need trajectory
capture from the fixed evaluation protocol.
