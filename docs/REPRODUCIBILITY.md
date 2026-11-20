# Reproducibility

PATCH has two reproducibility paths. The public path runs on synthetic tables and is safe for any
reviewer. The gated-data path uses the August 2023 TB Portals release under an approved Data Use
Agreement. The two paths share code and schema, but synthetic values do not reproduce biological
results.

## Public path

```bash
make setup
make public-smoke
make public-release-check
```

The smoke runner generates synthetic tables, builds the standard ETL artifacts, builds the named
`died_or_failure` matrix, and fits a small PATCH model. It writes aggregate metadata only. The
generator uses a fixed seed and refuses to write into the configured real-data directory.

## Gated-data path

Request access and execute the TB Portals Data Use Agreement at
<https://tbportals.niaid.nih.gov/>. Place the August 2023 CSV bundle outside this repository or set
`CTB_RAW_DIR` to that directory.

```bash
uv run --locked python scripts/00_download_data.py
export CTB_RAW_DIR=/path/to/TB_Portals_Published_Clinical_Data_August_2023
uv run --locked python scripts/01_build_dataset.py
uv run --locked python scripts/05c_build_unfavorable_matrices.py
```

The endpoint builder currently retains episodes with at least one interpretable positive culture in
the observed record and defines the primary outcome as any recorded `died` or `failure` code. It
does not impose a strict day-zero culture window. Loss to follow-up is excluded from that endpoint.
The older
`05b_build_conversion_matrices.py` script is a separate culture-conversion analysis.

After the endpoint matrix has been reviewed, run the numbered analyses documented in
`docs/SUPPLEMENTARY_FIGURES.md`. The temporal leakage-controlled analysis is
`scripts/28_temporal_leakfree.py`. Holdout analyses must fit imputation, standardization, and
categorical vocabularies on the training split only.

## Determinism

The project seed is `SEED = 42` in `src/ctb/config.py`. Python, NumPy, PyTorch, model initialization,
cross-validation splits, and bootstrap routines are seeded. LightGBM is configured for one worker
where it is used. Figure bytes can vary across operating systems even when the aggregate results
agree.

## Quality and release checks

```bash
make test
make lint
make type
bash scripts/sync_to_public.sh /tmp/patch-public
bash scripts/scan_public.sh /tmp/patch-public
```

The sync script publishes only an allowlist of source code and documentation. The scan rejects raw
data, patient-shaped files, local paths, symlinks, notebook outputs, excluded directories, and
common secret patterns. For a final gated-data endpoint build, set `CTB_REQUIRE_CLEAN_RELEASE=1` so
the builder refuses a dirty checkout. Its manifest records the exact commit, critical source hashes,
dependency lock hash, preprocessing schema, split definitions, random seeds, and hashes for the
local endpoint artifacts.
