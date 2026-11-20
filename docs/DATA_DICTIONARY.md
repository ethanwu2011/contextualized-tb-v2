# Data Dictionary — TB Portals (August 2023 export)

> **DUA note:** This dictionary documents *schema and aggregate properties only*. No row-level
> patient data is reproduced here or anywhere in this repository.

Source: NIH TB Portals Published Clinical Data, August 2023 (access-gated; obtain via the DUA at
<https://tbportals.niaid.nih.gov/>). Point the pipeline at the CSVs with the `CTB_RAW_DIR` environment
variable (or symlink them to `data/raw/tb_portals_aug2023`, git-ignored). Files live outside the repo.

Grain: a **patient** (`patient_id`) may have multiple **conditions/episodes** (`condition_id`).
Cohort: **11,282 conditions** across **11,067 patients** (215 patients have >1 episode).
Category counts below 30 are merged with a neighbouring category.

---

## 1. `TB_Portals_Patient_Cases_August_2023.csv`  (11,282 × 222)

One row per condition (episode). Source of context features and resistance class.

| Field | Type | Use | Notes / missingness |
|---|---|---|---|
| `condition_id` | id | join key (episode) | unique, 0% missing |
| `patient_id` | id | join key (patient) | 11,067 unique |
| `registration_date` | date | episode start, re-initiation logic | |
| `age_of_onset` | num | context: demographics | |
| `gender` | cat | context | |
| `country` | cat | context: socioeconomic | |
| `education`, `employment` | cat | context: socioeconomic | |
| `bmi` | num | context | |
| `case_definition` | cat | context / cohort | New 7354, Relapse 2122, Failure 767, LTFU 471, Chronic 284, Other or unknown 284 |
| `type_of_resistance` | cat | **expected-duration mapping**, context | MDR non XDR 4679, Sensitive 3758, XDR 1248, Mono DR 826, Pre-XDR 494, Poly DR or negative 277 |
| `diagnosis_code` | cat | context | |
| `lung_localization`, `dissemination`, `lungcavity_size`, `cavities…`, `timika_score`, imaging fields | mixed | context: imaging/radiology | many CT/CXR-derived fields; high missingness expected |
| `culture`, `microscopy`, `bactec_*`, `le_*`, `hain_*`, `genexpert_*` | cat | context: baseline micro/DST | per-drug susceptibility flags |
| `status` | cat | record status | final 10988, amended 294 |

## 2. `TB_Portals_Regimens_August_2023.csv`  (about 16,000 × 13) — **outcome source**

One row per **regimen line** within an episode (not a sequential dated phase — see caveat).

| Field | Type | Use | Notes |
|---|---|---|---|
| `patient_id`, `condition_id` | id | join | |
| `period_start` | date | episode start | **0% missing**; *equals* `period_end` per row (see caveat) |
| `period_end` | date | — | 3.8% missing; per-row `period_end − period_start = 0` for ALL rows |
| `period_span` | num (days) | **treatment duration** (authoritative) | 4.3% missing; median 251 (max/condition), max 10,242 (outliers) |
| `outcome_cd` | cat | **unfavorable-outcome component**; final status | cured 9282, completed 2399, died 1365, ltfu 1243, failure 922, stillOnTreatment 422, unknown 184, palliativeCare 182 |
| `activities_period_start` | date | (intended) phase start | **constant within a condition** — all rows share one date |
| `activities_period_end` | date | (intended) phase end | per-row span = 0 |
| `activities_statusreason_cd` | cat | **escalation signal** | ended or terminated 7666, NaN 3375, continuation 2945, patientStoppedTreatment 845, **resistance 660**, **adverse 343**, **newDrugAvailable 162** |
| `regimen_drug` | set-string | **treatment matrix X**, escalation | e.g. `{E,H,R,Z}`; 1,935 distinct set-strings; `{}` empty in 201 |
| `dose`, `collected`, `reinfusioned` | mixed | not used Phase 1 | |

**Drug abbreviations** (WHO-style): H=isoniazid, R=rifampicin, E=ethambutol, Z=pyrazinamide,
S=streptomycin, Bdq=bedaquiline, Lzd=linezolid, Dld=delamanid, Pa=pretomanid, Cfz=clofazimine,
Cs=cycloserine, Lfx=levofloxacin, Mfx=moxifloxacin, Am=amikacin, Cm=capreomycin, Km=kanamycin,
Pas=PAS, Pto=prothionamide, Amx/Clv=amox-clav, Ipm/Cln=imipenem-cilastatin, ART=antiretroviral.

> **CRITICAL CAVEAT (timeline semantics).** `period_start == period_end` for every row, and within
> any multi-row condition every `activities_period_start` is identical (spread = 0 across all 3,486
> multi-row conditions). Therefore the regimen table does **not** encode a longitudinal, distinctly
> dated sequence of regimen changes. The multiple rows per episode are concurrent/alternative
> regimen lines stamped with the episode start date; `period_span` carries each line's duration.
> Consequence: the originally-specified "drug class added **after day 30**" escalation rule is **not
> literally derivable**.

## 3. `TB_Portals_Specimen_August_2023.csv`  (… × 6)
`specimen_id`, `specimen_collection_site`, `specimen_collection_date` — links to culture/biochem.
Candidate source for culture-conversion timing (Phase 2).

## 4. `TB_Portals_Biochemistry_August_2023.csv`  (… × 42)
Longitudinal labs keyed by `specimen_id`/`test_date`: Hemoglobin (g/l), ALT, AST, Albumin, WBC-
derived (ANC, ALC, NLR), Hepatitis B surface antigen, Hepatitis C virus, glucose/HbA1c, etc.
→ baseline lab context features (take earliest per condition).

## 5. `TB_Portals_DST_August_2023.csv`  (114 MB, … × 100+)
Per-specimen drug-susceptibility results (`bactec_*`, `le_*`, `lpaother_*`, `hain_*`,
`genexpert_*`) plus culture/microscopy results. → baseline resistance-pattern context.

---

## Derived analysis features

These are constructed by the pipeline (not raw fields). Source code is authoritative.

### Treatment matrix `X` — `src/ctb/data/treatments.py`
Binary indicators for each drug in the **initial regimen** (first regimen line per condition).
Columns `drug_<CODE>` for drugs used by ≥1% of the cohort; rarer drugs collapse into
`drug_other_second_line`. 26 columns at 1% threshold on the analysis cohort.

### Context matrix `C` — `src/ctb/data/context.py`
Un-imputed (NaNs preserved; imputation happens inside the CV pipeline, train-fold only).

| Feature | Source field | Transform | Missingness |
|---|---|---|---|
| `age_of_onset`, `bmi`, `number_of_children`, `number_of_daily_contacts` | same | numeric coercion | bmi ~29% |
| `timika_score` | `timika_score` | numeric | high |
| `gender`, `education`, `employment` | same | categorical | ~0% (sentinel "Not Reported") |
| `country` | `country` | categorical | 0% |
| `comorb_hiv/diabetes/hepatitis_c/hepatitis_b/anemia`, `comorb_missing` | `comorbidity` | token parse of free text | field missing 44% (→ `comorb_missing`) |
| `type_of_resistance`, `case_definition`, `lung_localization` | same | categorical | ~0% |
| `cavitation` | `lungcavity_size` | yes / no / unknown | derived |
| `baseline_culture`, `baseline_smear` | `culture`, `microscopy` | positive / negative / unknown | derived |
| `lab_alt`, `lab_ast`, `lab_creatinine`, `lab_glucose`, `lab_albumin`, `lab_hemoglobin` | Biochemistry | earliest non-null per condition | sparse (hemoglobin ~79% missing) |

**Imputation strategy** (applied in `models/baselines.py`, per training fold only): numeric →
`IterativeImputer(random_state=42)`; categorical → constant `"missing"` category then one-hot
(`min_frequency=0.01`, `handle_unknown="ignore"`). This prevents leakage across CV folds.
