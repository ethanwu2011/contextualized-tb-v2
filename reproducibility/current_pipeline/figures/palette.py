"""Shared colour definitions for the PATCH figure set.

Every category — subtype, resistance class, modality, model — is defined here once and imported
everywhere, so a category is the same colour in every panel of every figure (including the Fig. 1
schematic). Base colours are the Okabe-Ito colourblind-safe set, extended deliberately; nothing is
left to matplotlib auto-assignment. Canonical public profile labels are declared in
``PUBLIC_RELEASE_MANIFEST.json`` and contract-tested against this module.

Semantic encoding of the five descriptive profiles follows the paper's observed outcome gradient.
The three lower-event profiles are cool (teal/blue), and the two higher-event profiles are warm
(orange/red), so the observed gradient is legible from colour alone.
"""

from __future__ import annotations

from matplotlib.colors import LinearSegmentedColormap

# --- Okabe-Ito base (colourblind-safe) ---------------------------------------------------------
_OKABE_ITO = {
    "blue": "#0072B2",
    "sky": "#56B4E9",
    "green": "#009E73",
    "orange": "#E69F00",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "yellow": "#F0E442",
    "black": "#000000",
    "grey": "#949494",
}

# --- 5 descriptive profiles -------------------------------------------------------------------
# Keyed by the historical encoder index retained for compatibility with the diagnostic figure code.
# Each entry carries the corrected display name, hex, and observed-event family. The explicit
# NARRATIVE_ORDER below maps these indices to the risk-rank order in the public manifest.
SUBTYPES: dict[int, dict[str, str]] = {
    0: {"name": "Drug-sensitive responders", "hex": "#009E73", "family": "lower_event"},
    1: {"name": "High-burden regional failure", "hex": "#D55E00", "family": "higher_event"},
    2: {
        "name": "Mixed-spectrum resistance responders",
        "hex": "#0072B2",
        "family": "lower_event",
    },
    3: {
        "name": "HIV-enriched severe-resistance failure",
        "hex": "#E69F00",
        "family": "higher_event",
    },
    4: {"name": "MDR-predominant responders", "hex": "#56B4E9", "family": "lower_event"},
}
# Corrected risk-rank order from PUBLIC_RELEASE_MANIFEST.json.
NARRATIVE_ORDER: list[int] = [0, 2, 4, 3, 1]


def subtype_color(i: int) -> str:
    """Hex colour for canonical subtype index ``i`` (0-4)."""
    return SUBTYPES[i]["hex"]


def subtype_name(i: int) -> str:
    """Display name for canonical profile index ``i``."""
    return SUBTYPES[i]["name"]


def subtype_family(i: int) -> str:
    """Observed event-rate family for canonical profile index ``i``."""
    return SUBTYPES[i]["family"]


# --- Resistance classes (ordinal severity -> single-hue purple sequential) ---------------------
RESISTANCE_COLORS: dict[str, str] = {
    "Sensitive": "#DADAEB",
    "Mono DR": "#BCBDDC",
    "Poly DR": "#9E9AC8",
    "MDR non XDR": "#756BB1",
    "Pre-XDR": "#54278F",
    "XDR": "#3F007D",
}

# --- Context modalities (Paul-Tol "muted" family — deliberately DISTINCT from the vivid Okabe-Ito
# subtype hues so inputs (modalities) never read as outputs (subtypes), incl. in the schematic) ---
MODALITY_COLORS: dict[str, str] = {
    "resistance": "#882255",  # wine
    "sociodemographic": "#332288",  # indigo
    "region": "#999933",  # olive
    "burden_clinical": "#AA4499",  # purple
    "comorbidity": "#CC6677",  # rose
    "imaging": "#DDCC77",  # sand
    "genomics": "#117733",  # forest green
    "labs": "#949494",  # grey
}

# --- Baseline / model comparison colours (kept consistent with src/ctb/viz/style.py) -----------
MODEL_COLORS: dict[str, str] = {
    "LR-no-context": "#949494",
    "LR-with-context": "#0072B2",
    "GBDT-no-context": "#E69F00",
    "GBDT-with-context": "#009E73",
    "PATCH": "#CC79A7",
}

# --- Binary clinical-status overlays (Fig 3) ---------------------------------------------------
# Coloured by the MODALITY each variable belongs to (HIV -> comorbidity, smear -> bacillary burden),
# never by a subtype hue, so a status overlay can never be misread as a subtype. 'off' is the shared
# neutral background class (one defined grey everywhere).
STATUS_COLORS: dict[str, str] = {
    "HIV+": "#CC6677",  # comorbidity modality
    "smear+": "#AA4499",  # burden_clinical modality
    "off": "#BFBFBF",  # neutral background (unhighlighted points)
}

# --- Countries / regions (one canonical qualitative scale, distinct from subtype hues) ----------
# Used wherever a country is encoded by identity (e.g. Fig 3 region overlay). A separate scale so a
# country is never coloured with a subtype hue. Fig 5c instead encodes powered vs under-powered with
# the neutral pair below, since there colour means "enough events", not country identity.
COUNTRY_COLORS: dict[str, str] = {
    "Ukraine": "#332288",
    "Georgia": "#DDCC77",
    "Nigeria": "#117733",
    "South Africa": "#88CCEE",
    "Belarus": "#882255",
    "Moldova": "#44AA99",
    "Azerbaijan": "#999933",
    "Kazakhstan": "#661100",
    "Romania": "#6699CC",
}
POWERED = "#4C566A"  # neutral slate: a country with >=30 events (Fig 5c)
UNDERPOWERED = "#BFBFBF"  # neutral grey: an under-powered country (Fig 5c)

# --- Colormaps ----------------------------------------------------------------------------------
# Diverging scale for signed descriptive coefficients and correlations.
DIVERGING_CMAP = LinearSegmentedColormap.from_list(
    "patch_diverging",
    ["#2166AC", "#67A9CF", "#F7F7F7", "#EF8A62", "#B2182B"],
)
DIVERGING_POLES = ("negative coefficient shift", "positive coefficient shift")

# Sequential single-hue (magnitudes: importance, agreement counts, rates).
SEQUENTIAL_CMAP = LinearSegmentedColormap.from_list(
    "patch_sequential",
    ["#F7FBFF", "#9ECAE1", "#4292C6", "#08519C", "#08306B"],
)
