# PATCH figure map

The table mapping every main and supplementary figure to the code that draws it, and to the analysis
script that writes its input, is in the Figures section of `README.md`.

## Endpoint and release rules

The manuscript line uses the observed-record culture-positive cohort and the endpoint-explicit
`died_or_failure` matrix (`scripts/05c_build_unfavorable_matrices.py`). Analysis scripts load
`unfavorable_model_matrices.npz` and assert `outcome_name == "died_or_failure"` before fitting.

Supplementary Figs. S1 and S2 are the exception. They were fitted on observed culture conversion, so
their scripts read `conversion_model_matrices.npz` from `scripts/05b_build_conversion_matrices.py`
by default (`figures/supp/_endpoint.py`). `--endpoint died_or_failure` refits them on the primary
endpoint and writes separately named files.

The public source-only release omits generated figures and tables. The one exception is the reviewed,
hash-pinned table in `figures/source_data/`. After a gated-data run, figures can be added to a
release only after checking captions, small cells, embedded metadata, and the public scanner.
Synthetic smoke outputs are not manuscript evidence.
