# Reproducibility

> **Scope of this inherited document:** the workflow/numbered artifact map below records the earlier October 23 audit. For the current manuscript, use `make verify`, the root `README.md` and `CURRENT_FIGURE_MAP.json`. The current two-endpoint clinical protocol is preserved in the unchanged reviewer archive. The inherited claim-audit target also includes secondary residual analyses; it is not a one-command reproduction of every current or historical manuscript result.

PATCH has three separate reproducibility layers. They must not be conflated.

## Public execution path

The public path tests installation, endpoint-artifact contracts, figure rendering, and the release
firewall without opening clinical data.

```bash
make setup
make manuscript-smoke
make aggregate-figures
make public-release-check
```

The synthetic workflow generates fake TB Portals-shaped tables and verifies the governed schema.
The aggregate renderer reads only the frozen disclosure-safe JSON reports in
`manuscript/aggregate_results`. Neither output is new scientific evidence.

## TB Portals claim audit

The claim-bearing TB Portals rerun requires the August 2023 published clinical bundle and an approved
Data Use Agreement.

```bash
export CTB_RAW_DIR=/secure/TB_Portals_August_2023
export CTB_DUA_OUTPUT_ROOT=/secure/patch-fresh-run
make claim-audits-dua
```

The output root must be outside the repository. The analysis drivers require an explicit
`--dua-acknowledge` flag and write aggregate reports only. The suite reconstructs the earliest
validated nonempty dated regimen, the recorded final Patient Cases endpoint, fold-local
preprocessing, patient-grouped validation, architecture ablations, K sensitivity, soft-profile
stability, coefficient support, internal transport, and simulation.

The governed primary analysis includes 9,647 assessed conditions from 9,481 patients with 1,770
events, 279 retained context features, and 26 regimen components. Culture positivity is a
sensitivity, not an inclusion requirement. The strict baseline culture-positive sensitivity includes
7,913 conditions.

The internal decision-curve sensitivity depends on locked patient-grouped out-of-fold predictions.
Those predictions must remain in a second private root.

```bash
export CTB_PRIVATE_OUTPUT_ROOT=/secure/patch-private-oof
export CTB_DUA_OUTPUT_ROOT=/secure/patch-dca-aggregate
make locked-validation-dca-dua
```

The target writes only `internal_decision_curve.json` and the locked-validation summary to the
aggregate root. Identifiers, outcomes, fold assignments, and predictions remain under
`CTB_PRIVATE_OUTPUT_ROOT` and are not public artifacts.

## Public SINAN rebuild

Brazilian SINAN DBC files for 2015 through 2019 are public but are kept outside the repository.

```bash
export CTB_RAW_DIR=/secure/TB_Portals_August_2023
export CTB_SINAN_SRC=/secure/SINAN_DBC_2015_2019
export CTB_DUA_OUTPUT_ROOT=/secure/patch-brazil-fresh-run
make brazil-audit-dua
```

The rebuild reproduces the legacy 298,900-record cohort and defines a stricter 296,079-record cohort
that excludes postmortem notification entries. The strict cohort is the corrected external
validation population. SINAN is used only for prognostic transport because it lacks a comparable
observed regimen vector.

## Frozen public results

Each JSON in `manuscript/aggregate_results` is row-free and has a SHA-256 checksum in
`manuscript/aggregate_results/CHECKSUMS.sha256`. The reports are not edited to harmonize their
wording. Later analyses supersede earlier limitations where the release index says so.

`figures/release/render_aggregate_figures.py` converts those JSON reports to PDF and PNG. It does
not recompute a statistic.

## Determinism and uncertainty

The project seed is defined in `src/ctb/config.py`. Validation uses patient-grouped folds and
fold-local preprocessing. Full-pipeline resampling repeats preprocessing and model fitting. Hard
profile labels are secondary because the learned representation is continuous.

Floating-point results can differ slightly across supported platforms. Aggregate estimates,
cohort contracts, feature vocabularies, seed lists, source hashes, and uncertainty definitions must
agree before an output is accepted.

## Release firewall

```bash
make lint
make type
make test
make public-release-check
```

Legacy analysis and figure entry points are blocked. The public staging script uses an allowlist and
the scanner rejects restricted directories, patient-shaped files, local paths, secrets, symlinks,
notebook outputs, altered aggregate checksums, and any staged file not named and hashed in
`PUBLIC_FILE_MANIFEST.sha256`.
