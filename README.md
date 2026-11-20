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

The endpoint-explicit matrix is the input for the manuscript analyses. The older
`scripts/05b_build_conversion_matrices.py` script is retained for a separate culture-conversion
analysis and must not be used for the manuscript endpoint.

For a release build with provenance enforcement, set `CTB_REQUIRE_CLEAN_RELEASE=1` before running
the endpoint builder. The generated manifest records the commit, dirty-tree state, critical source
hashes, preprocessing contract, and hashes of the local endpoint artifacts.

The core analysis scripts are numbered and their inputs are documented in
`docs/SUPPLEMENTARY_FIGURES.md`. The canonical temporal leakage check is
`scripts/28_temporal_leakfree.py`. Every imputation, standardization step, and category vocabulary
used in a holdout analysis must be fitted on the training split only.

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
