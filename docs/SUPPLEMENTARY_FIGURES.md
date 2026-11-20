# PATCH figure map

The manuscript line uses the observed-record culture-positive cohort and the endpoint-explicit
`died_or_failure` matrix. The current builder does not impose a strict day-zero culture window.
A figure is not ready for release until its script loads
`data/processed/unfavorable_model_matrices.npz`, checks the endpoint name, and writes only aggregate
outputs.

## Main figures

| Figure | Role | Primary source |
|---|---|---|
| 1 | PATCH method schematic | `scripts/17_method_schematic.py` |
| 2 | Five treatment-response profiles and modality attribution | `scripts/22_multimodal_subtypes.py` |
| 3 | Model ladder, temporal check, and learned geometry | `scripts/15_model_ladder_unfavorable.py`, `scripts/23_multimodal_umap.py`, `scripts/28_temporal_leakfree.py` |
| 4 | Drug-specific coefficient decomposition | `scripts/25_multimodal_drug_effects.py` |
| 5 | Context-conditioned coefficient atlas | `scripts/14_heterogeneity_deepening.py` |
| 6 | Cross-country transport and Brazil transfer | `scripts/19_holdout_country_validation.py`, `scripts/26_cross_population.py`, `scripts/32_external_validation.py` |

## Supplementary figures

| Figure | Role | Primary source |
|---|---|---|
| S1 | Cross-seed profile stability | `figures/supp/figS1_reproducibility.py` |
| S2 | Archetype-count selection | `figures/supp/figS2_kselection.py` |
| S3 | Entropy-weight sensitivity | `scripts/27_entropy_sensitivity.py` |
| S4 | Rifampicin receipt alignment | `scripts/16_subtype_validation.py` |
| S5 | Full coefficient atlas | `scripts/10_heterogeneity_atlas.py` |
| S6 | Context dependence by drug | `scripts/29_modality_attribution_ci.py` |
| S7 | Calibration and precision-recall | `scripts/30_calibration_auprc.py` |
| S8 | Nested model comparison | `scripts/34_fair_comparison.py` |
| S9 | Temporal and country transport | `scripts/24_temporal_validation.py`, `scripts/19_holdout_country_validation.py` |
| S10 | Internal decision-curve analysis | `scripts/37_decision_curve.py` |
| S11 | Correlation-preserving enrichment null | `scripts/26_enrichment_permutation_null.py` |
| S12 | Matched profile-stability null | `scripts/27_profile_stability_null.py` |
| S13 | Semi-synthetic effect-recovery benchmark | `scripts/28_effect_recovery_benchmark.py` |
| S14 | Internal and external decision-curve comparison | `scripts/39_external_decision_curve.py` |

## Endpoint and release rules

The culture-conversion builder `scripts/05b_build_conversion_matrices.py` is retained for a
separate analysis. It must not be used for figures in this manuscript. The manuscript scripts load
`unfavorable_model_matrices.npz` and should assert `outcome_name == "died_or_failure"` before fitting.

The public source-only release omits generated figures and tables by default. After a gated-data
run, figures can be added to a release only after checking captions, small cells, embedded metadata,
and the public scanner. Synthetic smoke outputs are not manuscript evidence.
