"""THE single source of truth for colour across the PATCH figure set.

Every category — subtype, resistance class, modality, model — is defined here once and imported
everywhere, so a category is the same colour in every panel of every figure (including the Fig. 1
schematic). Base colours are the Okabe-Ito colourblind-safe set, extended deliberately; nothing is
left to matplotlib auto-assignment.

Semantic encoding of the 5 treatment-response subtypes follows the paper's story: the three
good-prognosis responder subtypes are cool (teal/blue), the two poor-outcome failure subtypes are
warm (orange/red), so the good→bad axis is legible from colour alone.
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

# --- 5 treatment-response subtypes -------------------------------------------------------------
# Keyed by the CANONICAL subtype index as it appears in data/processed/multimodal_subtypes.npz and
# results/tables/multimodal_subtype_characterization.csv (row label "S{i}"). Each entry carries the
# display name, hex, and prognostic family. Colour encodes prognosis: the three good-outcome
# responder subtypes are COOL (green/blue/sky), the two poor-outcome failure subtypes are WARM
# (orange/vermillion), so the good→bad axis is legible from colour alone. Defining this against the
# real index (not narrative order) removes the old per-figure CANON2PAL remap, which was duplicated
# across scripts and a latent mislabelling hazard.
# "short" is the figure label (the short names given with Table 1); "name" is the full Table 1 name.
SUBTYPES: dict[int, dict[str, str]] = {
    0: {"name": "Drug-sensitive responders", "short": "Drug-sensitive responders",
        "hex": "#009E73", "family": "responder"},  # 4.5% unfav
    1: {"name": "High-burden / regional failure", "short": "High-burden / regional",
        "hex": "#E69F00", "family": "failure"},  # 34.0%
    2: {"name": "Mixed-spectrum resistance responders", "short": "Resistant responders",
        "hex": "#0072B2", "family": "responder"},  # 4.6%
    3: {"name": "HIV-enriched severe-resistance failure", "short": "Severe-resistant / HIV",
        "hex": "#D55E00", "family": "failure"},  # 34.9%
    4: {"name": "MDR-predominant responders", "short": "MDR responders",
        "hex": "#56B4E9", "family": "responder"},    # 8.0%
}
# Canonical display order: the three responders, then the two failure subtypes (manuscript §2.3).
NARRATIVE_ORDER: list[int] = [0, 2, 4, 1, 3]


def subtype_color(i: int) -> str:
    """Hex colour for canonical subtype index ``i`` (0-4)."""
    return SUBTYPES[i]["hex"]


def subtype_name(i: int, *, full: bool = False) -> str:
    """Figure label for canonical subtype index ``i``; ``full=True`` gives the full Table 1 name."""
    return SUBTYPES[i]["name" if full else "short"]


def subtype_family(i: int) -> str:
    """Prognostic family ('responder' | 'failure') for canonical subtype index ``i``."""
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
    "resistance": "#882255",        # wine
    "sociodemographic": "#332288",  # indigo
    "region": "#999933",            # olive
    "burden_clinical": "#AA4499",   # purple
    "comorbidity": "#CC6677",       # rose
    "imaging": "#DDCC77",           # sand
    "genomics": "#117733",          # forest green
    "labs": "#949494",              # grey
}

# --- PATCH versus comparators: one colour for PATCH, greys for everything it is compared with ----
# PATCH is the same blue in every model comparison (no subtypes appear in those panels); comparator
# models are greys; reference lines (treat all / none, chance, perfect calibration, null
# expectations) are a darker grey drawn dashed or dotted. Resistance-class-only targeting is drawn in
# the resistance-modality wine so "resistance" is one colour across Fig. 2c, Fig. 6 and Supplementary
# Figs. S10 and S14.
PATCH_COLOR = "#0072B2"
COMPARATOR_GREY = "#8C8C8C"
COMPARATOR_LIGHT = "#BDBDBD"
REFERENCE_GREY = "#666666"
RESISTANCE_CLASS_COLOR = MODALITY_COLORS["resistance"]

MODEL_COLORS: dict[str, str] = {
    "LR-no-context": COMPARATOR_LIGHT,
    "LR-with-context": COMPARATOR_GREY,
    "GBDT-no-context": COMPARATOR_LIGHT,
    "GBDT-with-context": COMPARATOR_GREY,
    "PATCH": PATCH_COLOR,
}

# --- Binary clinical-status overlays (Fig 3) ---------------------------------------------------
# Coloured by the MODALITY each variable belongs to (HIV -> comorbidity, smear -> bacillary burden),
# never by a subtype hue, so a status overlay can never be misread as a subtype. 'off' is the shared
# neutral background class (one defined grey everywhere).
STATUS_COLORS: dict[str, str] = {
    "HIV+": "#CC6677",     # comorbidity modality
    "smear+": "#AA4499",   # burden_clinical modality
    "off": "#BFBFBF",      # neutral background (unhighlighted points)
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
POWERED = "#4C566A"        # neutral slate: a country with >=30 events (Fig 5c)
UNDERPOWERED = "#BFBFBF"   # neutral grey: an under-powered country (Fig 5c)

# --- Colormaps ----------------------------------------------------------------------------------
# Diverging (signed quantities: drug-specific Δβ, interactions, correlations). Poles are named in
# every colorbar: negative (blue) = more favorable recorded association; positive (red) = less favorable.
DIVERGING_CMAP = LinearSegmentedColormap.from_list(
    "patch_diverging",
    ["#2166AC", "#67A9CF", "#F7F7F7", "#EF8A62", "#B2182B"],
)
DIVERGING_POLES = ("more favorable recorded association (Δβ < 0)",
                   "less favorable recorded association (Δβ > 0)")

# Sequential single-hue (magnitudes: importance, agreement counts, rates).
SEQUENTIAL_CMAP = LinearSegmentedColormap.from_list(
    "patch_sequential",
    ["#F7FBFF", "#9ECAE1", "#4292C6", "#08519C", "#08306B"],
)
