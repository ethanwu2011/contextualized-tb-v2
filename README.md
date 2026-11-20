# PATCH

Patient-specific Archetypal Treatment-effect Contextualization of Heterogeneity in Tuberculosis.

PATCH is a linear archetype contextualized logistic model. It uses the recorded regimen line
`X` and a multimodal patient context `C` to estimate a context-conditioned coefficient vector for
each recorded drug.

```text
Z(C) = softmax(M · C + q)
logodds(Y | X, C) = X · (Z(C) · A_X) + Z(C) · a_0
```

Here `A_X` contains the drug coefficients for `K = 5` shared archetype profiles and `a_0` contains
their intercepts. `Z(C)` gives each episode its mixture over those profiles. The model is supervised
by treatment and outcome. Resistance and drug-susceptibility variables are included in context. The
coefficient map is descriptive in this observational registry and is not a causal treatment effect
estimate or a prescribing rule.

## Data-use boundary

The NIH TB Portals data are governed by a Data Use Agreement and are not included here. This
repository contains no real patient rows, identifiers, patient-level labels, patient-level
predictions, embeddings, or trained patient-level artifacts. The `data/` directory is ignored by
Git. The public smoke path below creates only synthetic tables. Synthetic results are execution
checks and have no biological meaning.

## Historical analyses in the manuscript

The historical analyses use 10,653 TB episodes with at least one interpretable positive
culture in the observed record, 335 context features, and eight data modalities. The current
matrix builder does not impose a strict day-zero culture window. A confirmatory release must lock
that time-zero rule and rerun the analysis before the cohort is described as baseline-only.

- Five recurring treatment-response profiles are resolved by the shared archetype model.
- Resistance class separates several favorable-response profiles, while failure is distributed
  across region, comorbidity, bacillary burden, and resistance rather than one axis alone.
- These analyses use the regimen-line `died_or_failure` outcome. It includes recorded `died` or
  `failure` outcomes and excludes loss to follow-up. See `docs/PATCH_OUTCOME_DEFINITION.md`. The
  primary validation uses the final recorded outcome (next section).
- The tuned PATCH model is close to the tuned context logistic baseline on internal discrimination.
  The value of the model is the interpretable coefficient and subtype structure, not a claim of
  superior AUROC.
- Temporal and cross-country analyses test transport. The Brazil analysis is a prognostic transfer
  check, not a causal validation of a treatment switch.
- Drug-specific and modality results are descriptive, positivity-audited, and hypothesis-generating.
  They should not be read as prescribing recommendations.

## Current validation pipeline

`reproducibility/current_pipeline/` holds the later, self-contained pipeline behind the primary
assessed-outcome validation (9,647 conditions from 9,481 patients, patient-grouped folds, train-only
preprocessing), the prespecified culture-conversion secondary endpoint and the assessed-outcome
profile refits. Its locked protocol (`validation/protocol_before_run.json`, written before any
clinical fit) records SHA-256 hashes of the exact source and driver, and its README maps each
manuscript result to code. It has its own locked environment and checks.

```bash
cd reproducibility/current_pipeline
uv sync --frozen --extra dev --python 3.11
make verify
```

## Public synthetic smoke run

Install the pinned environment, then run the complete safe path.

```bash
make setup
make public-smoke
```

The command generates fake TB Portals tables, builds the endpoint-explicit matrix, and fits a small
PATCH model. It writes only aggregate smoke metadata under `results/metrics/`. The runner forces its
raw-data path to a synthetic directory under `CTB_DATA_DIR` and never reads a user-supplied
clinical-data directory. Set the three output directory variables to scratch paths if you want to
keep the clone unchanged.

To check the release firewall, use a temporary staging directory.

```bash
make public-release-check
```

The underlying commands are also available directly.

```bash
bash scripts/sync_to_public.sh /tmp/patch-public
bash scripts/scan_public.sh /tmp/patch-public
```

The staged tree contains source, tests, CI, and release documentation. It omits `data/`, generated
results, notebooks, logs, and local review material. The scanner fails on data-shaped files,
symlinks, local paths, common secret patterns, notebook outputs, and excluded directories.

## Run with approved TB Portals data

Researchers with an approved Data Use Agreement can place the August 2023 CSV bundle outside this
repository. The acquisition helper prints the expected filenames and checks that they are present.
It does not download data.

```bash
uv run --locked python scripts/00_download_data.py
export CTB_RAW_DIR=/path/to/TB_Portals_Published_Clinical_Data_August_2023
uv run --locked python scripts/01_build_dataset.py
uv run --locked python scripts/05c_build_unfavorable_matrices.py
```

The endpoint-explicit matrix is the input for the manuscript analyses. The culture-conversion
builder `scripts/05b_build_conversion_matrices.py` is needed only for Supplementary Figs. S1 and S2,
which were fitted on culture conversion; it must not be used for the primary endpoint.

For a release build with provenance enforcement, set `CTB_REQUIRE_CLEAN_RELEASE=1` before running
the endpoint builder. The generated manifest records the commit, dirty-tree state, critical source
hashes, preprocessing contract, and hashes of the local endpoint artifacts.

The core analysis scripts are numbered. The figure table below maps every figure to the script
that draws it and the script that writes its input. The canonical temporal leakage check is
`scripts/28_temporal_leakfree.py`. Every imputation, standardization step, and category vocabulary
used in a holdout analysis must be fitted on the training split only.

## Figures

Each figure in the paper and the code that draws it. Run a figure module with
`uv run --locked python -m <module>` from the repository root, for example
`uv run --locked python -m figures.fig02a_subtypes`. The summary tables are written to
`results/tables/` by the analysis scripts in the last column, which need the approved TB Portals data
(and, for Fig. 5 and S14, the SINAN registry). The only table shipped here is the reviewed
`figures/source_data/fig4_literature_validation.csv`. Figures marked *patient level* refit or
re-read the processed patient matrices, so only an approved data holder can redraw them.

| Figure | Drawn by | Reads | Input written by |
|---|---|---|---|
| 1 | By hand (BioRender and PowerPoint), not code | | |
| 2 | `figures.fig02a_subtypes` | Summary tables | `scripts/22_multimodal_subtypes.py`, `scripts/29_modality_attribution_ci.py` |
| 3 | `figures.fig03_structure` | Patient level | `scripts/22_multimodal_subtypes.py`, `scripts/05c_build_unfavorable_matrices.py` |
| 4a | `figures.fig04_atlas_plotly` | Summary table | `scripts/10_heterogeneity_atlas.py` |
| 4b | `figures.pptx_export` (`fig4_subtype_drug`) | Summary table | `scripts/25_multimodal_drug_effects.py` |
| 4c, 4d | `figures.pptx_export` (`fig4_validation_heatmap`, `fig4_validation_lollipop`) | Summary table and shipped review table | `scripts/25_multimodal_drug_effects.py`, `figures/source_data/fig4_literature_validation.csv` |
| 5 | `figures.fig05_external_brazil` | Summary tables | `scripts/32_external_validation.py` |
| 6 | `figures.fig06_clinical_utility` | Summary tables | `scripts/37_decision_curve.py` |
| S1 | `figures.supp.figS1_reproducibility` | Patient level, refits | `scripts/05b_build_conversion_matrices.py` (culture conversion, as published) |
| S2 | `figures.supp.figS2_kselection` | Patient level, refits | `scripts/05b_build_conversion_matrices.py` (culture conversion, as published) |
| S3 | `figures.supp.figS3_entropy_sensitivity` | Summary tables | `scripts/27_entropy_sensitivity.py` |
| S4 | `figures.supp.figS4_resistance_recovery` | Patient level | `scripts/05c_build_unfavorable_matrices.py`, `scripts/22_multimodal_subtypes.py` |
| S5 | `figures.supp.figS5_atlas_full` | Summary table | `scripts/10_heterogeneity_atlas.py` |
| S6 | `figures.supp.figS6_drug_ranking` | Summary tables | `scripts/14_heterogeneity_deepening.py` |
| S7 | `figures.supp.figS7_calibration_pr` | Summary table | `scripts/30_calibration_auprc.py` |
| S8 | `figures.supp.figS8_frontier` | Summary tables | `scripts/15_model_ladder_unfavorable.py`, `scripts/33_hparam_sweep.py` |
| S9 | `figures.supp.figS9_internal_generalization` | Summary tables | `scripts/24_temporal_validation.py`, `scripts/26_cross_population.py` |
| S10 | `figures.fig06_clinical_utility` (panel a) | Summary tables | `scripts/37_decision_curve.py` |
| S11 | `scripts/26_enrichment_permutation_null.py` | Patient level, refits | the same script |
| S12 | `scripts/27_profile_stability_null.py` | Patient level, refits | the same script |
| S13 | `scripts/28_effect_recovery_benchmark.py` | Patient level, semi-synthetic | the same script |
| S14 | `scripts/39_external_decision_curve.py` | Patient level | the same script |

Fig. 4 is assembled from its four panels by hand. Supplementary Figs. S1 and S2 take
`--endpoint died_or_failure` to refit on the primary endpoint instead; that run writes separately
named files. All figures share one palette, `figures/palette.py`: the five subtype colours are used
only for subtypes, PATCH is one ink colour in every model comparison, and comparators are grey.

## Repository layout

```text
src/ctb                         installable package and model code
scripts/                        data checks, endpoint builder, analyses, and smoke runner
figures/                        figure-generation code
tests/                          synthetic schema and model tests
docs/PUBLIC_REPRODUCIBILITY.md public release and DUA instructions
docs/PATCH_OUTCOME_DEFINITION.md manuscript endpoint contract
docs/EVIDENCE_LEDGER.md         row-free source review and interpretation rules
PUBLIC_RELEASE_MANIFEST.json   row-free release metadata
reproducibility/current_pipeline locked patient-grouped validation pipeline (own environment)
```

## Development checks

```bash
make test
make lint
make type
```

The same checks and the synthetic smoke path run in GitHub Actions. No workflow step downloads or
uploads the gated clinical data.

## Citation and license

Please cite the PATCH manuscript when using this code. The code is released under the MIT license.
The TB Portals data remain governed by the NIH Data Use Agreement.
