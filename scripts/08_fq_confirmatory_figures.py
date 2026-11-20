"""Part B confirmatory figures: the fluoroquinolone (moxi vs levo) culture-conversion anchor.

Generates the rigorously-identified-slice figures:

* Kaplan-Meier culture-conversion curves by FQ choice (the endpoint, with censoring).
* Forest plot of the three prespecified FQ x context interactions (adjusted Cox HRs).
* An E-value panel quantifying robustness of the surviving interaction to unmeasured confounding.

Usage
-----
    uv run python scripts/08_fq_confirmatory_figures.py
"""

from __future__ import annotations

import json
import logging
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter

from ctb.config import (
    METRICS_DIR,
    PROCESSED_DIR,
    RAW_DIR,
    TABLES_DIR,
    TB_PORTALS_FILES,
    ensure_output_dirs,
)
from ctb.data.conversion import derive_conversion
from ctb.utils.logging import configure_logging
from ctb.utils.reproducibility import set_global_seed
from ctb.viz.style import PALETTE, save_figure, set_style

logger = logging.getLogger(__name__)


def _evalue(hr: float, lo: float, hi: float) -> tuple[float, float]:
    """Return E-values for an HR point estimate and its CI bound nearest the null.

    Parameters
    ----------
    hr, lo, hi : float
        Hazard ratio and 95% CI bounds.

    Returns
    -------
    tuple of float
        ``(evalue_point, evalue_ci)``.
    """

    def ev(rr: float) -> float:
        r = rr if rr >= 1 else 1 / rr
        return float(r + np.sqrt(r * (r - 1)))

    if lo <= 1 <= hi:
        return ev(hr), 1.0
    near = hi if hr < 1 else lo
    return ev(hr), ev(near)


def _build_fq_frame() -> pd.DataFrame:
    """Assemble the FQ-susceptible, exclusive-FQ, baseline-positive analysis frame."""
    analysis = pd.read_parquet(PROCESSED_DIR / "analysis_frame.parquet")
    dst = pd.read_csv(
        RAW_DIR / TB_PORTALS_FILES["dst"],
        low_memory=False,
        usecols=["condition_id", "specimen_collection_date", "cultureresults"],
    )
    conv = derive_conversion(dst)
    af = analysis.join(conv, how="left")
    fqs = af["type_of_resistance"].isin(["Sensitive", "Mono DR", "Poly DR", "MDR non XDR"])
    excl = (af["drug_Mfx"] == 1) ^ (af["drug_Lfx"] == 1)
    frame = af[fqs & excl & af["event"].notna() & (af["time_to_event"] > 0)].copy()
    frame["moxi"] = frame["drug_Mfx"].astype(int)
    return frame


def main() -> int:
    """Generate the FQ confirmatory figures.

    Returns
    -------
    int
        Exit code.
    """
    configure_logging()
    set_global_seed()
    ensure_output_dirs()
    set_style()

    frame = _build_fq_frame()
    logger.info(
        "FQ confirmatory frame: n=%d (moxi %d, levo %d)",
        len(frame),
        int(frame["moxi"].sum()),
        int((frame["moxi"] == 0).sum()),
    )

    # --- Figure: KM conversion curves by FQ choice ---
    fig, ax = plt.subplots(figsize=(4.5, 3.4))
    kmf = KaplanMeierFitter()
    for arm, lab, col in [(0, "Levofloxacin", PALETTE[0]), (1, "Moxifloxacin", PALETTE[3])]:
        g = frame[frame["moxi"] == arm]
        kmf.fit(g["time_to_event"], g["event"], label=f"{lab} (n={len(g)})")
        # plot conversion probability = 1 - survival(still positive)
        ax.plot(
            kmf.survival_function_.index,
            1 - kmf.survival_function_.iloc[:, 0].to_numpy(),
            color=col,
            label=f"{lab} (n={len(g)})",
        )
    ax.set(
        xlabel="Days from treatment start",
        ylabel="Cumulative culture conversion",
        title="Culture conversion by fluoroquinolone",
        xlim=(0, 365),
    )
    ax.legend(loc="lower right", fontsize=7)
    save_figure(fig, METRICS_DIR.parent / "figures" / "fq_km_conversion")

    # --- Adjusted Cox: 3 prespecified interactions ---
    frame["HIV"] = frame["comorb_hiv"].astype(int)
    frame["smear_pos"] = (frame["baseline_smear"] == "positive").astype(int)
    frame["MDR"] = (frame["type_of_resistance"] == "MDR non XDR").astype(int)
    frame["age"] = pd.to_numeric(frame["age_of_onset"], errors="coerce")
    frame["age"] = frame["age"].fillna(frame["age"].median())
    res = pd.get_dummies(frame["type_of_resistance"], prefix="r", drop_first=True).astype(int)

    interactions = [("HIV", "FQ x HIV"), ("MDR", "FQ x MDR"), ("smear_pos", "FQ x bacillary load")]
    forest_rows = []
    for mod, label in interactions:
        d = pd.DataFrame(
            {
                "time_to_event": frame["time_to_event"].to_numpy(),
                "event": frame["event"].to_numpy(),
                "moxi": frame["moxi"].to_numpy(),
                "age": frame["age"].to_numpy(),
                mod: frame[mod].to_numpy(),
                f"moxi_x_{mod}": (frame["moxi"] * frame[mod]).to_numpy(),
            }
        )
        for col in res.columns:
            d[col] = res[col].to_numpy()
        cph = CoxPHFitter(penalizer=0.1)
        cph.fit(d, "time_to_event", "event")
        s = cph.summary
        t = f"moxi_x_{mod}"
        forest_rows.append(
            {
                "interaction": label,
                "hr": float(np.exp(s.loc[t, "coef"])),
                "lo": float(np.exp(s.loc[t, "coef lower 95%"])),
                "hi": float(np.exp(s.loc[t, "coef upper 95%"])),
                "p": float(s.loc[t, "p"]),
            }
        )
    forest = pd.DataFrame(forest_rows)
    forest["p_bonferroni"] = (forest["p"] * 3).clip(upper=1.0)
    forest.to_csv(TABLES_DIR / "fq_interaction_forest.csv", index=False)

    # --- Figure: interaction forest ---
    fig, ax = plt.subplots(figsize=(5.0, 2.6))
    ypos = np.arange(len(forest))
    err = np.vstack([forest["hr"] - forest["lo"], forest["hi"] - forest["hr"]])
    ax.errorbar(forest["hr"], ypos, xerr=err, fmt="o", color=PALETTE[0], capsize=3, markersize=5)
    ax.axvline(1.0, color="#888", linestyle="--", linewidth=0.8)
    ax.set_yticks(ypos)
    ax.set_yticklabels(forest["interaction"])
    ax.invert_yaxis()
    ax.set(
        xlabel="Interaction hazard ratio (conversion)", title="FQ x context interactions (adjusted)"
    )
    for i in range(len(forest)):
        sig = "*" if forest["p"].iloc[i] < 0.0167 else ""  # Bonferroni 0.05/3
        ax.text(
            forest["hi"].iloc[i] + 0.02,
            i,
            f"p={forest['p'].iloc[i]:.3f}{sig}",
            va="center",
            fontsize=6,
        )
    save_figure(fig, METRICS_DIR.parent / "figures" / "fq_interaction_forest")

    # --- E-value for the surviving interaction (smallest Bonferroni p) ---
    best = forest.loc[forest["p"].idxmin()]
    ev_pt, ev_ci = _evalue(best["hr"], best["lo"], best["hi"])

    summary = {
        "n": int(len(frame)),
        "n_moxi": int(frame["moxi"].sum()),
        "n_levo": int((frame["moxi"] == 0).sum()),
        "interactions": forest.round(4).to_dict(orient="records"),
        "surviving_interaction": str(best["interaction"]),
        "evalue_point": round(ev_pt, 2),
        "evalue_ci": round(ev_ci, 2),
    }
    (METRICS_DIR / "fq_confirmatory_summary.json").write_text(json.dumps(summary, indent=2))
    logger.info(
        "FQ confirmatory: %s HR=%.2f p=%.3f | E-value point=%.2f CI=%.2f",
        best["interaction"],
        best["hr"],
        best["p"],
        ev_pt,
        ev_ci,
    )
    logger.info("FQ figures written.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
