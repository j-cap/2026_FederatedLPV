# IEEE IV review revision: novelty and closest prior work

Review point 1 is addressed in the introduction and bibliography of
`paper/main.tex`. This revision changes positioning and exposition; it adds no
fitting, simulations, or numerical results. The other five major review points
remain open.

## Supported distinction

The contribution is a distributed LPV construction and controlled empirical
study of **locally deficient scheduling information combined with vehicle
heterogeneity**. Each client observes a fixed subset of speeds. Additional
transitions at those same speeds cannot supply missing scheduling-basis
directions. Compatible pooling can supply them, conditional on the group
design, while heterogeneous clients still introduce approximation bias.

| Prior work | Established contribution | Distinction used in this manuscript |
| --- | --- | --- |
| IFCA, Ghosh et al. (2020) | Alternating client assignment and cluster-model optimization, with conditional convergence analysis | Specialization to speed-wise LPV matrix targets with fixed, locally deficient scheduling support and approximate shared maps |
| Clustered SysID, Toso et al. (2023) | Alternating assignment and collaborative identification of LTI system types; analyzed models are identical within each true cluster | Scheduling-dependent coefficient maps, complementary recipient coverage, and remaining within-group vehicle variation |
| FedSysID, Wang et al. (2023) | Collaborative LTI identification with a quantified heterogeneity tradeoff | Separating missing scheduling directions from compatibility bias in the LPV benchmark |
| IC-SYSID/ClusterCraft, Kececi et al. (2025) | Incremental cluster formation and stability promotion, including a vehicle-fleet study | The present study evaluates scheduling coverage and identification at recipient-unobserved grid speeds |

These are distinctions in the studied setting and evidence. They do not assert
that IFCA cannot handle a finite rank-deficient local sample, that its framework
cannot be adapted to LPV regression, or that the current method outperforms those
algorithms. No direct algorithm comparison with IFCA or Clustered SysID was run.
The inherited alternating structure, additive regression, and fixed-partition
rank condition are established tools. Their general convergence or cluster
recovery guarantees are not transferred to this benchmark.

The contribution paragraph now identifies three concrete pieces: development
basis selection; distributed fitting without full local coefficient maps or
family labels, followed by a group-rank audit; and paired identification evidence
that separates complementary coverage from compatibility selection. Numerical
centralized--federated agreement remains implementation verification, and LQI
remains a supporting usability check.

## Sources consulted

- IFCA: [official NeurIPS paper](https://papers.nips.cc/paper_files/paper/2020/file/e32cc80bf07915058ce90722ee17bb71-Paper.pdf), including the linear-model assumptions and conditional convergence analysis.
- Clustered SysID: [author manuscript](https://arxiv.org/pdf/2304.01395), especially Sections 2--3; [published IEEE record](https://ieeexplore.ieee.org/document/10383950).
- FedSysID: [official L4DC/PMLR paper and bibliographic record](https://proceedings.mlr.press/v211/wang23d.html).
- IC-SYSID/ClusterCraft: [author preprint](https://arxiv.org/abs/2505.16857). The manuscript cites this verified 2025 version.

## Frozen-input recovery

The revision starts from main commit `7df744f`, the merge of the Overleaf updates.
That merge removed historical result files, including two inputs required by
`paper/check_consolidation.py`: `experiment_10f_summary.csv` and
`experiment_10f_seed_summary.csv`. Only these two required files were restored
from the previously verified commit `3895999`. Their SHA-256 hashes match the
existing consolidation lock exactly. No frozen values or provenance locks were
changed.

Validation commands from the repository root:

```bash
python paper/check_consolidation.py
python paper/check_review_integration.py
python paper/prepare_identification_figures.py --preflight
make -C paper
```

The final build remains six pages. All pages were visually inspected, all nine
bibliographic references resolve, and the evidence and figure-input checks pass.
The three diagnostic plots are included at 0.95 text width to accommodate the
additional references; their committed PDF assets and numerical content are
unchanged. The final build has no overfull boxes or unresolved references.
