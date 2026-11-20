# PATCH outcome contract

The primary PATCH analysis uses one binary endpoint named `died_or_failure`.

An episode is positive when at least one recorded regimen line has `outcome_cd` equal to `died` or
`failure`. An episode is negative when it has neither code. `ltfu` is not part of the primary
endpoint. Episodes enter the modelling matrix only after the cohort rules have been applied and at
least one interpretable positive culture is present in the observed DST record. The current matrix
builder does not impose a strict day-zero or prespecified baseline window. Any confirmatory analysis
must lock that time-zero rule before rebuilding the cohort and downstream results.

The model is supervised by the recorded treatment matrix and this outcome. Context includes
resistance and drug-susceptibility variables. The resulting coefficients and subtype profiles are
context-conditioned associations in an observational registry. They are not causal effects, and
they do not identify a treatment switch without stronger design assumptions.

The endpoint is built by `scripts/05c_build_unfavorable_matrices.py`, which writes the local
patient-level matrix `data/processed/unfavorable_model_matrices.npz` and the disclosure-safe
aggregate manifest `results/metrics/unfavorable_model_matrices.json`. The conversion matrix created
by `scripts/05b_build_conversion_matrices.py` is a different endpoint and is not an acceptable input
for the manuscript's PATCH figures.
