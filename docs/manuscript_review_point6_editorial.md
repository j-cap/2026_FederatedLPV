# Review point 6: editorial and formatting pass

The supplied eight-page compiled draft was checked against the active manuscript
source. This pass refines presentation of the existing identification and control
evidence. It adds no experiment, fitting, selection, controller tuning, or claim
of broader validation.

## Scientific clarity and consistency

- Define the physical states as sideslip angle and yaw rate, and define the
  observed-speed count n_i explicitly. It counts local matrix targets, not the
  36 transitions used to estimate each target.
- Describe g_i as the learned group assignment rather than a group that is
  "unknown during fitting."
- Distinguish the full-envelope identification objective from its fixed-grid
  error metric, and define the fleet index s.
- Use Restricted Local consistently where the three-function comparator is
  intended. Retain Prior-centered Local and Smooth Local as distinct names.
- Move the assignment-tie rule next to assignment scoring, avoiding ambiguity
  with BIC model selection. Replace "candidate orders" by "candidate group
  counts" in the communication accounting.
- Define LQI, LTI, OLS, RK4, and BIC where introduced. Preserve the mathematical
  objectives, numerical ridge values, centering workflow, and controller signs.
- Reduce repeated statements about independent confirmation and local-prior
  equivalence while retaining the distinction between reused primary records,
  fresh robustness fleets, and separate control confirmation fleets.
- Replace slash-delimited maneuver parameters and tracking gains by explicit
  corresponding lists. Preserve all reported values and uncertainty intervals.

## Formatting

The Smooth Local objective is set on two lines with an explicit minimization
scope. Removing its same-page wrapper avoids excessive display spacing on the
float-heavy page. Captions are tightened without removing pairing, uncertainty,
rank, or scope definitions. Figure files are unchanged. Standard IEEE conference
fonts, margins, and the original class are retained. The PDF now carries title,
author, and keyword metadata.

The compiled manuscript is visually checked for equation fit, table readability,
float order, section transitions, and last-page column balance. The author block
still uses the original "Jakob Weber et al." and working-draft designation.
Final authors and affiliations cannot be inferred from this editorial request
and must be supplied before submission.

## Reference metadata

The Toth book now includes its verified series volume, publisher location, and
DOI. The FedAvg entry lists all five authors and the proceedings-series volume.
FedSysID uses the same PMLR series convention. Existing journal, conference,
preprint, and accepted-paper entries retain their verified publication type.
No DOI, page range, or publication status is invented for the accepted IFAC
reference.

Primary metadata sources, checked on 13 October 2026:

- [Springer book record](https://link.springer.com/book/10.1007/978-3-642-13812-6).
- [FedAvg proceedings record](https://proceedings.mlr.press/v54/mcmahan17a.html)
  and [author manuscript](https://proceedings.mlr.press/v54/mcmahan17a/mcmahan17a.pdf).
- [FedSysID proceedings record](https://proceedings.mlr.press/v211/wang23d.html).

## Verification

The four evidence checks cover the original consolidation, matched review
supplement, distinct Smooth Local baseline, and focused robustness study:

```bat
python paper\check_consolidation.py
python paper\check_review_integration.py
python paper\check_local_baseline_integration.py
python paper\check_robustness_integration.py
```

All pass after the editorial changes. The numerical includes and frozen evidence
remain intact. The original uploaded PDF is retained as the review input; the
edited manuscript and compiled PDF are saved in the repository.

Final layout verification: eight pages, with Figures 2--4 placed on pages 4--6
next to the identification discussion. All pages have been rendered and
inspected. Tables appear in order, equations fit their columns, the final
columns are balanced, and the build reports no overfull boxes or reference warnings. A remaining
underfull vertical-box diagnostic on the float-heavy page does not produce
clipping or overlap in the rendered output. Fonts and conference margins are unchanged.
