"""Rebuild the entire PATCH figure set and a contact sheet. ``uv run python -m figures.render_all``."""

from __future__ import annotations

import importlib
import sys

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from ctb.config import FIGURES_DIR
from figures.style import mm, set_style

FIG_DIR = FIGURES_DIR
MODULES = [
    ("figures.fig01_schematic", "fig01_patch_schematic"),
    ("figures.fig02a_subtypes", "fig02_subtypes"),
    ("figures.fig03_structure", "fig03_structure"),
    ("figures.fig04_drugspecific", "fig04_drugspecific"),
    ("figures.fig05_external_brazil", "fig05_external_brazil"),
    ("figures.fig06_clinical_utility", "fig06_clinical_utility"),
]
SUPP = [
    ("figures.supp.figS1_reproducibility", "figS1_reproducibility"),
    ("figures.supp.figS2_kselection", "figS2_kselection"),
    ("figures.supp.figS3_entropy_sensitivity", "figS3_entropy_sensitivity"),
    ("figures.supp.figS4_resistance_recovery", "figS4_resistance_recovery"),
    ("figures.supp.figS6_drug_ranking", "figS6_drug_ranking"),
    ("figures.supp.figS5_atlas_full", "figS5_atlas_full"),
    ("figures.supp.figS7_calibration_pr", "figS7_calibration_pr"),
    ("figures.supp.figS8_frontier", "figS8_frontier"),
    ("figures.supp.figS9_internal_generalization", "figS9_internal_generalization"),
]


def main() -> int:
    set_style()
    stems = []
    for mod, stem in MODULES:
        m = importlib.import_module(mod)
        m.main()
        stems.append(stem)

    # Supplementary figures (some retrain models, so this is the slow part).
    for mod, _stem in SUPP:
        importlib.import_module(mod).main()

    # contact sheet (2 columns x 3 rows) to check cross-figure consistency at a glance
    fig, axes = plt.subplots(3, 2, figsize=(mm(180), mm(220)))
    for ax, stem in zip(axes.ravel(), stems, strict=False):
        img = mpimg.imread(FIG_DIR / f"{stem}.png")
        ax.imshow(img)
        ax.set_title(stem, fontsize=6)
        ax.axis("off")
    for ax in axes.ravel()[len(stems):]:
        ax.axis("off")
    fig.tight_layout()
    out = FIG_DIR / "_contact_sheet.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
