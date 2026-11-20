# Corrected figure and analysis map

> **Scope of this inherited document:** the workflow/numbered artifact map below records the earlier October 23 audit. For the current manuscript, use `make verify`, the root `README.md` and `CURRENT_FIGURE_MAP.json`. The current two-endpoint clinical protocol is preserved in the unchanged reviewer archive. The inherited claim-audit target also includes secondary residual analyses; it is not a one-command reproduction of every current or historical manuscript result.

This map supersedes the legacy numbered scripts. Those entry points are blocked because they target
the obsolete 10,653-condition analysis.

## Public aggregate figures

| Artifact | Content | Public source |
|---|---|---|
| Figure 1 | Data-free PATCH model and governance schematic | `figures/release/render_aggregate_figures.py` |
| Figure 2 | Architecture-relevant discrimination and probability metrics | `sensitivity_and_support.json`, `shared_k_sensitivity.json` |
| Figure 3 | Corrected five-profile size, outcome, HIV, and resistance summaries | `shared_profile_reference_composition.json` |
| Figure 4 | Profile-by-regimen coefficient map with support and uncertainty | `shared_profile_full_pipeline_stability.json` |
| Figure 5 | Strict SINAN prognostic transport | `brazil_source_rebuild_audit.json` |
| Supplementary K | K=3 through K=8 held-out sensitivity | `shared_k_sensitivity.json` |
| Supplementary simulation | Twenty-seed coefficient recovery | `expanded_simulation_benchmark.json` |

All six are rendered by:

```bash
make aggregate-figures
```

## DUA-local analysis producers

| Analysis | Producer |
|---|---|
| Episode, time zero, endpoint, and culture sensitivity | `scripts/run_episode_endpoint_audit.py` |
| Context-only, regimen-only, additive, shared, and residual architecture checks | `scripts/run_architecture_composition_ablations.py` |
| Exposure policy, endpoint, culture, support, and fold-local sensitivities | `scripts/run_aggregate_sensitivities.py` |
| Shared K=3 through K=8 validation | `scripts/run_shared_k_sensitivity.py` |
| Full-pipeline continuous-profile and coefficient uncertainty | `scripts/run_shared_profile_full_pipeline_stability.py` |
| Corrected hard-bin composition and modality attribution | `scripts/run_shared_profile_reference_composition.py` |
| Patient-disjoint temporal and geographic transport | `scripts/run_internal_transport.py` |
| Locked out-of-fold validation for private predictions | `scripts/run_locked_residual_analysis.py` |
| Exploratory internal decision-curve sensitivity | `scripts/run_internal_decision_curve.py` |
| Twenty-seed synthetic recovery | `scripts/run_expanded_simulation.py` |
| SINAN source reconstruction and prognostic transport | `scripts/run_brazil_source_rebuild_audit.py` |

Run the first seven TB Portals aggregate producers with `make claim-audits-dua`. Run the locked
validation and internal decision curve together with `make locked-validation-dca-dua`. This second
target requires separate aggregate and private output roots.

## Interpretation rules

- Five is a clinically useful descriptive resolution. It is not a uniquely identified number of
  biological classes.
- Soft profiles and the continuous coefficient surface are primary. Hard assignments are summaries.
- The heatmap remains in the main paper, with suppressed cells and uncertainty marks.
- Brazil validates prognostic transport only.
- The unrecalibrated external decision curve is retired.
- A tuned general-purpose gradient boosting model is an internal audit comparator, not a required
  PATCH architecture component.
- Synthetic recovery supports the low-rank representation. It does not validate causal treatment
  effects.
