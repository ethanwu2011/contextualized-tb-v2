# Public reproducibility

This repository contains the PATCH source code and a schema-faithful synthetic-data path. It does
not contain the NIH TB Portals export, patient identifiers, patient-level labels, patient-level
predictions, model embeddings, or trained patient-level artifacts.

## Run the public smoke path

After installing the pinned environment, run

```bash
make public-smoke
```

The command generates fake TB Portals tables, builds the ordinary ETL artifacts, builds the
endpoint-explicit `died_or_failure` matrix, and fits a small five-archetype PATCH model. The result
is an aggregate JSON summary under `results/metrics/public_synthetic_smoke.json`. Every value from
this run is synthetic and has no biological interpretation.

The synthetic runner always sets `CTB_RAW_DIR` to a synthetic directory under `CTB_DATA_DIR`.
With the default settings this is `data/raw/synthetic`. It never follows a user-supplied raw-data
path. The generator also refuses to overwrite a directory that looks like a full-size real export
unless a user explicitly passes its dangerous `--force` option. The public runner never passes that
option. To keep a clone completely unchanged, set `CTB_DATA_DIR`, `CTB_RESULTS_DIR`, and
`CTB_LOG_DIR` to scratch directories before running the smoke path.

For a local gated-data check without writing patient-level intermediates into the working copy,
set `CTB_DATA_DIR`, `CTB_RESULTS_DIR`, and `CTB_LOG_DIR` to scratch directories. Keep
`CTB_RAW_DIR` pointed at the approved data location. These overrides change only local output
locations and do not relax the public export rules.

The model stack is declared explicitly in `pyproject.toml`, including PyTorch, PyTorch Lightning,
and `contextualized-ml`. The committed `uv.lock` records those direct dependencies and their
transitive packages. Refresh it whenever `pyproject.toml` changes and inspect the lock diff before
tagging a public release. The smoke and CI commands install from the project metadata, so this is a
release bookkeeping check rather than a data-access requirement.

## Run the gated-data analysis

Researchers with an approved Data Use Agreement can place the August 2023 TB Portals CSV bundle
outside the repository and set `CTB_RAW_DIR` to that directory. The acquisition helper only checks
that the expected files exist. It does not download data.

```bash
uv run --locked python scripts/00_download_data.py
export CTB_RAW_DIR=/path/to/TB_Portals_Published_Clinical_Data_August_2023
uv run --locked python scripts/01_build_dataset.py
uv run --locked python scripts/05c_build_unfavorable_matrices.py
```

The manuscript endpoint is `died_or_failure`. It is the indicator for any recorded `died` or
`failure` outcome among episodes with at least one interpretable positive culture in the observed
record. Loss to follow-up is excluded from this primary endpoint and retained only for sensitivity
analyses. The current builder does not impose a strict day-zero culture window. A confirmatory
release must lock that time-zero rule and rerun the full analysis before calling the cohort
baseline-only. The endpoint builder writes the local `unfavorable_context_meta.parquet` file used by
interpretation figures. It contains raw context fields and is excluded from the public export. The older
`05b_build_conversion_matrices.py` script is a separate culture-conversion analysis and must not be
used as the source for the PATCH manuscript figures.

The numbered analysis scripts can then be run in the order documented in the manuscript and in
`docs/SUPPLEMENTARY_FIGURES.md`. Scripts that use a temporal or country holdout must fit imputation,
standardization, and category vocabularies on the training split only. The canonical leakage-
controlled temporal check is `scripts/28_temporal_leakfree.py`.

## Rebuild the Brazil transfer input

The external SINAN analysis requires the public yearly `TUBEBR15.dbc` through `TUBEBR19.dbc` files.
Obtain those files from DATASUS, keep them outside the repository, and run the local converter. The
converter writes the derived `brazil_harmonized.parquet` file under the configured processed-data
directory. It never copies the source files into the repository and logs only aggregate summaries.

```bash
uv run --with datasus-dbc --with dbfread python scripts/31_build_brazil_cohort.py \
  --years 15 16 17 18 19 --src /path/to/sinan-files
uv run --locked python scripts/32_external_validation.py
uv run --locked python scripts/39_external_decision_curve.py
```

The seven harmonized transfer fields and the outcome definition are built by
`scripts/31_build_brazil_cohort.py`. The Brazil files and the derived parquet are excluded by the public sync
and scanner. The public synthetic smoke path does not claim to reproduce Brazil results.

## Release checks

Before publishing a staged tree, run

```bash
bash scripts/sync_to_public.sh /tmp/patch-public
bash scripts/scan_public.sh /tmp/patch-public
```

The sync script uses an allowlist and omits all generated `data/` content and all generated result
tables and figures by default. This keeps stale or locally generated row-shaped files out of a
public release. Static figures can be added only after they are regenerated from the endpoint-
explicit matrix and reviewed for disclosure risk. The scanner fails on binary data artifacts,
row-shaped CSV files, notebooks with outputs, excluded directories, local paths, and common secret
patterns.

## Provenance

`PUBLIC_RELEASE_MANIFEST.json` records the public input path, endpoint contract, commands, and the
fact that the manuscript is not redistributed with the code. A final gated-data release
should be built from a clean reviewed checkout. Set `CTB_REQUIRE_CLEAN_RELEASE=1` when running the
endpoint builder to enforce that check. The generated manifest records the exact commit, dirty-tree
state, critical source hashes, refreshed dependency lock hash, preprocessing schema, random seeds,
split definitions, and hashes for the local endpoint artifacts.
