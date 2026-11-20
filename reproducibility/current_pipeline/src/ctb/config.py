"""Project-wide configuration: paths, constants, and seeds.

All paths are derived from ``PROJECT_ROOT`` via :mod:`pathlib` so that nothing is hardcoded to a
particular machine. The location of the raw NIH TB Portals export can be overridden with the
``CTB_RAW_DIR`` environment variable. The default is ignored local scratch space and is not an
authorized DUA data location.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Final

# --- Reproducibility ---------------------------------------------------------------------------
SEED: Final[int] = 42
N_BOOTSTRAP: Final[int] = 1000
N_CV_FOLDS: Final[int] = 5

# --- Core paths --------------------------------------------------------------------------------
PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
# Synthetic/integration runs may redirect every generated artifact away from the tracked
# submission aggregates. Source code still resolves from PROJECT_ROOT.
OUTPUT_ROOT: Final[Path] = Path(
    os.environ.get("CTB_OUTPUT_ROOT", PROJECT_ROOT / ".artifacts" / "local")
).expanduser()

DATA_DIR: Final[Path] = OUTPUT_ROOT / "data"
# The ignored scratch default supports synthetic development only. DUA runs must set CTB_RAW_DIR
# to the authorized external folder containing the CSV files.
_DEFAULT_RAW: Final[Path] = DATA_DIR / "raw" / "tb_portals_aug2023"
RAW_DIR: Final[Path] = Path(os.environ.get("CTB_RAW_DIR", _DEFAULT_RAW)).expanduser()
INTERIM_DIR: Final[Path] = DATA_DIR / "interim"
PROCESSED_DIR: Final[Path] = DATA_DIR / "processed"

RESULTS_DIR: Final[Path] = OUTPUT_ROOT / "results"
TABLES_DIR: Final[Path] = RESULTS_DIR / "tables"
FIGURES_DIR: Final[Path] = RESULTS_DIR / "figures"
METRICS_DIR: Final[Path] = RESULTS_DIR / "metrics"

LOGS_DIR: Final[Path] = OUTPUT_ROOT / "logs"

# --- TB Portals source files (August 2023 export) ----------------------------------------------
# Filenames within RAW_DIR. Override RAW_DIR (env CTB_RAW_DIR) to point at the real export.
TB_PORTALS_FILES: Final[dict[str, str]] = {
    "patient_cases": "TB_Portals_Patient_Cases_August_2023.csv",
    "regimens": "TB_Portals_Regimens_August_2023.csv",
    "specimen": "TB_Portals_Specimen_August_2023.csv",
    "biochemistry": "TB_Portals_Biochemistry_August_2023.csv",
    "dst": "TB_Portals_DST_August_2023.csv",
}

# --- Clinical constants (WHO expected treatment durations, days) -------------------------------
EXPECTED_DURATION_DAYS: Final[dict[str, int]] = {
    "DS": 180,  # drug-susceptible: 6 months
    "MDR": 540,  # multidrug-resistant: 18 months
    "XDR": 600,  # extensively drug-resistant: 20 months
}

# Mapping from TB Portals ``type_of_resistance`` to expected treatment duration (days).
# Clinical-standard mapping (locked decision): Sensitive & Mono DR -> DS length; Poly DR &
# MDR non XDR -> MDR length; Pre-XDR & XDR -> XDR length. Classes with no defined length
# (Negative/unknown) map to None and are NOT assessed for the extension component.
RESISTANCE_EXPECTED_DAYS: Final[dict[str, int]] = {
    "Sensitive": 180,
    "Mono DR": 180,
    "Poly DR": 540,
    "MDR non XDR": 540,
    "Pre-XDR": 600,
    "XDR": 600,
}

EXTENSION_THRESHOLD: Final[float] = 1.20  # >120% of expected duration => "treatment extension"
REINITIATION_WINDOW_DAYS: Final[int] = 365  # new episode within 365 days of prior end

# Drugs whose addition most strongly signals genuine treatment intensification (WHO codes).
KEY_INTENSIFICATION_DRUGS: Final[frozenset[str]] = frozenset(
    {"Bdq", "Lzd", "Dld", "Pa", "Am", "Cm", "Km", "Cfz"}
)
# ``activities_statusreason_cd`` values that indicate a regimen change consistent with escalation.
ESCALATION_STATUSREASONS: Final[frozenset[str]] = frozenset(
    {"resistance", "adverse", "newDrugAvailable"}
)
# WHO "unfavorable" outcome codes (kept as a separate validation column, NOT in primary composite).
UNFAVORABLE_OUTCOMES: Final[frozenset[str]] = frozenset({"died", "failure", "ltfu"})
# PATCH manuscript endpoint.  LTFU remains in the WHO sensitivity outcome above but is not an event
# for the primary died/failure analysis.
DIED_FAILURE_OUTCOMES: Final[frozenset[str]] = frozenset({"died", "failure"})
# Explicit observed non-events for the exploratory died/failure endpoint. Missing, unknown,
# ongoing, palliative-care, and loss-to-follow-up states are not terminal non-events.
DIED_FAILURE_NONEVENT_OUTCOMES: Final[frozenset[str]] = frozenset({"cured", "completed"})


def ensure_output_dirs() -> None:
    """Create output directories (results, logs, interim, processed) if absent.

    Raw data directories are intentionally not created here, since raw data lives outside the
    repository and must never be written to by this pipeline.

    Returns
    -------
    None
    """
    for path in (INTERIM_DIR, PROCESSED_DIR, TABLES_DIR, FIGURES_DIR, METRICS_DIR, LOGS_DIR):
        path.mkdir(parents=True, exist_ok=True, mode=0o700)
