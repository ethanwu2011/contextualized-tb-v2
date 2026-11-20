# Public reproducibility

> **Scope of this inherited document:** the workflow/numbered artifact map below records the earlier October 23 audit. For the current manuscript, use `make verify`, the root `README.md` and `CURRENT_FIGURE_MAP.json`. The current two-endpoint clinical protocol is preserved in the unchanged reviewer archive. The inherited claim-audit target also includes secondary residual analyses; it is not a one-command reproduction of every current or historical manuscript result.

This repository is a source release with a synthetic execution path and a frozen
disclosure-safe aggregate result snapshot. It contains no restricted TB Portals data or patient-level
derived artifacts.

## Reviewer commands

```bash
make setup
make manuscript-smoke
make aggregate-figures
make public-release-check
```

The synthetic workflow verifies the endpoint artifact and preprocessing contracts. The figure
renderer reads only `manuscript/aggregate_results`. Neither path downloads or opens restricted
data.

## Claim-bearing rerun

An approved local researcher can rerun the corrected TB Portals analyses with:

```bash
export CTB_RAW_DIR=/secure/TB_Portals_August_2023
export CTB_DUA_OUTPUT_ROOT=/secure/patch-fresh-run
make claim-audits-dua
```

The primary endpoint is the final Patient Cases outcome, with Died or Failure as events and Cured or
Completed as observed non-events. Culture positivity is not required for the primary cohort.
The governed analysis contains 9,647 assessed conditions from 9,481 patients, 1,770 events, 279
retained context features, and 26 recorded regimen components.

## Brazil rebuild

```bash
uv sync --frozen --extra brazil
export CTB_RAW_DIR=/secure/TB_Portals_August_2023
export CTB_SINAN_SRC=/secure/SINAN_DBC_2015_2019
export CTB_DUA_OUTPUT_ROOT=/secure/patch-brazil-fresh-run
make brazil-audit-dua
```

The optional dependency group pins the DBC decompressor and DBF reader. The rebuild uses public
`TUBEBR15.dbc` through `TUBEBR19.dbc`, records their checksums, and writes only aggregate
output. It assigns no regimen to Brazilian records.

## Release boundary

The source stager refuses an existing destination and uses an explicit allowlist. The scanner rejects
symlinks, row-shaped files, model artifacts, clinical-data directories, local home paths, secrets,
notebooks, altered aggregate checksums, and any file outside the reviewed release manifest.

The exact commands and cohort contract are also documented in `README.md`,
`docs/REPRODUCIBILITY.md`, and `PUBLIC_RELEASE_MANIFEST.json`.
