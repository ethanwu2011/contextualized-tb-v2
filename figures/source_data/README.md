# Reviewed source data for figures

Files here are the only tabular files allowed in the public tree. Each one is listed with its SHA-256
in `SHA256SUMS`, and `scripts/scan_public.sh` rejects any table that is not listed or whose hash has
changed. Add a file only after a disclosure review, then update `SHA256SUMS`.

## fig4_literature_validation.csv

The post hoc literature triage of the 23 nominally significant subtype by drug cells shown in
Figure 4c and 4d. `figures/pptx_export.py` reads it to draw those two panels. It was curated by hand
after the model fit and is described in `docs/EVIDENCE_LEDGER.md`.

| Column | Meaning |
|---|---|
| `subtype`, `subtype_short` | Subtype name and the code used in Figure 4 (DS, HB, RR, SR-HIV) |
| `drug` | Recorded drug variable |
| `direction` | Sign of the recorded association: more favorable (Δβ < 0) or less favorable (Δβ > 0) |
| `dbeta`, `p` | Drug-specific coefficient contrast and its bootstrap P value, as printed in Figure 4 |
| `category` | Review state: confirmed, confounded, discordant or novel (display labels in the ledger) |
| `concordant`, `confidence` | Reviewer's agreement call and confidence |
| `reference`, `source_note` | Source checked and a short note where the source disagrees or is absent |

Disclosure review (20 November 2026): the table holds model estimates and literature calls for 23
cells. It has no patient rows, identifiers, counts or denominators, so no cell can be recovered by
subtraction. Every value is already printed in Figure 4c and 4d.
