# PATCH current validation pipeline

Source code for the patient-grouped analyses reported in the manuscript. The repository root holds
the numbered historical pipeline (10,653 condition records, 335 context features, regimen-line
outcome) behind the profile, atlas, literature-triage, decision-curve and Brazil analyses. This
folder holds the later pipeline behind:

- the primary assessed-outcome validation (9,647 conditions from 9,481 patients, final died or
  failure versus cured or completed, five patient-grouped five-fold repeats, train-only
  preprocessing),
- the prespecified secondary endpoint (observed sustained culture conversion, 8,979 conditions) and
  its fixed-horizon day-60 sensitivity,
- the assessed-outcome profile refits (six initializations) and modality attribution.

No TB Portals data, fitted models, predictions or result tables are included. The data are governed
by the NIH TB Portals Data Use Agreement. Aggregate results are in Supplementary Data 1 of the paper.

## Provenance

`validation/protocol_before_run.json` was written before any clinical fit (locked
2026-10-24 17:55 UTC). It records the endpoint hierarchy, model settings, seeds, the secondary
claim rule and SHA-256 hashes of the 33 files under `src/ctb` and of the driver
`validation/run_validation.py`. The files here match all 34 hashes, so this is the exact code that
produced the reported validation numbers. Do not edit `src/ctb` or the driver, or the run refuses
to start.

## Install and test (synthetic data only)

```bash
uv sync --frozen --extra dev --python 3.11
make verify        # lint, types, unit tests and the bounded synthetic manuscript workflow
```

The tested environment is Python 3.11 with the locked versions in `uv.lock` (NumPy 1.26.4,
PyTorch 2.4.0, scikit-learn 1.5.1, pandas 2.2.2).

## Rerun the locked validation (DUA holders only)

The driver checks its own hash, the source hashes and the hashes of the four August 2023 TB Portals
tables before fitting, and writes only aggregate JSON next to itself.

```bash
bash validation/prepare_run.sh /secure/patch-validation-run
uv run --frozen python /secure/patch-validation-run/run_validation.py --synthetic-only
uv run --frozen python /secure/patch-validation-run/run_validation.py \
    --raw-dir /secure/tb_portals_august_2023 --dua-acknowledge
```

## Where each manuscript result comes from

| Manuscript item | Code |
| --- | --- |
| Primary validation, AUROC 0.8042 versus 0.8078 (Results, Methods Validation) | `validation/run_validation.py` with `src/ctb/models/locked_validation.py`, `fold_preprocessing.py` |
| Secondary endpoint, culture conversion +0.009 (Results) | same driver, `src/ctb/data/conversion.py` |
| Assessed-outcome profile refit, five profiles, six seeds (Results) | `scripts/run_shared_profile_reference_composition.py`, `scripts/run_shared_profile_full_pipeline_stability.py` |
| Synthetic end-to-end workflow | `scripts/run_manuscript.py --mode synthetic` |
| Historical profiles, atlas, triage, Brazil, decision curves (Figs. 1 to 6, most Supplementary Figs.) | numbered scripts in the repository root |

## Notes

- `src/ctb/data/conversion.py` is hash-locked, so its docstring still points to
  `docs/PREREGISTRATION.md`, an earlier planning document that is not part of this release. The
  endpoints used here are fixed in `validation/protocol_before_run.json`.
- The renderers under `figures/release/` read aggregate reports from the reviewer archive
  (`manuscript/aggregate_results`), which is not part of this repository.
  `render_shared_primary_submission.py` also calls a separate Figure 4 layout script that is not
  included.
