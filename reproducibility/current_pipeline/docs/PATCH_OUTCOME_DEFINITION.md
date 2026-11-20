# PATCH endpoint and cohort contract

The primary PATCH endpoint is the recorded final Patient Cases outcome.

- `Died` or `Failure` is an event.
- `Cured` or `Completed` is an observed non-event.
- Loss to follow-up, still on treatment, palliative care, unknown, missing, and unclassified states
  are excluded from the primary binary analysis.

The primary governed cohort requires a defined resistance class, an assessable final outcome, a
valid patient link, and an earliest validated nonempty dated recorded regimen. Culture is not an
eligibility requirement. This produces 9,647 assessed conditions from 9,481 patients, with 1,770
events.

A strict culture sensitivity requires an interpretable positive culture on or before the selected
regimen index and contains 7,913 conditions. A separate WHO-style unfavorable sensitivity includes
loss to follow-up and contains 10,562 conditions with 2,685 events.

The endpoint and strict culture sensitivity are constructed by
`scripts/05b_build_conversion_matrices.py`. The primary local artifact is
`data/processed/died_failure_model_matrices.npz`. It carries an endpoint identifier, cohort
identifier, ordered condition and patient hashes, feature vocabularies, and source provenance. The
culture-conversion matrix produced in the same workflow is a separate sensitivity and cannot be
substituted for the primary endpoint.

The model is supervised by recorded regimen, registry context, and this endpoint. The learned
coefficients are regularized regimen-outcome associations conditional on recorded context. They are
not causal effects or counterfactual treatment outcomes.
