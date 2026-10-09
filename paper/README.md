# Active manuscript: full-envelope LPV identification

The current contribution is **full-envelope identification from incomplete local operating coverage**, assuming available two-state lateral observations. The scope decision and evidence boundaries are in [the consolidation](../docs/identification_consolidation.md).

`main.tex` uses the 10A-10E identification chain and 10F as supporting LQI validation. It explicitly describes controlled equilibrium-centered perturbation data, whole-envelope matrix error, client participation in fitting, modest control headroom and the absence of a general stability or privacy guarantee. It is a working draft, not a submission-ready deployment claim.

Build and check from the repository root:

```bash
python paper/check_consolidation.py
python paper/analyze_identification_records.py
make -C paper
```

The PDF is `2026_FederatedLPV_identification_draft.pdf`. The unmodified CTAN
IEEEtran 1.8b class is included and conference formatting is required.

The [review revision and run instructions](../docs/manuscript_review_revision.md)
describe the resolved text issues, the existing-record region analysis, and the
prepared selected-group/regularized-Local supplement. The supplement has passed
preflight and a one-fleet smoke check; its full results are still pending.

The preceding personalized cold-start manuscript and its audits are preserved in [archive/9m_cold_start](archive/9m_cold_start). They describe a different claim and method. Their quantitative headlines and assumptions do not apply to this draft. The chronological development report remains the full experimental record.
