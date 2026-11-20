"""Export every figure's individual panels as standalone high-res files.

Each main/supplementary figure is normally saved as one composite PDF+PNG. For users who want to
re-format or re-arrange panels, this re-renders every figure and additionally writes each lettered
panel on its own — vector PDF + 600-dpi PNG — to ``results/figures/panels/<stem>_<letter>.{pdf,png}``.

Panels with an attached colorbar / inset / twin axis are cropped to include those neighbours (so a
heatmap panel keeps its colourbar). The Fig 1 schematic draws its four stages inside a single axes, so
it is cropped by region instead.

Run: ``uv run python -m figures.export_panels``
"""

from __future__ import annotations

import importlib
import sys

import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox

import figures.panels as panels_mod
from ctb.config import FIGURES_DIR
from figures.style import mm, set_style

OUT = FIGURES_DIR / "panels"

# (module, stem) for every figure that has multiple panels worth exporting individually.
FIGURES = [
    ("figures.fig01_schematic", "fig01_patch_schematic"),
    ("figures.fig02a_subtypes", "fig02_subtypes"),
    ("figures.fig03_structure", "fig03_structure"),
    ("figures.fig04_drugspecific", "fig04_drugspecific"),
    ("figures.fig05_external_brazil", "fig05_external_brazil"),
    ("figures.fig06_clinical_utility", "fig06_clinical_utility"),
    ("figures.supp.figS1_reproducibility", "figS1_reproducibility"),
    ("figures.supp.figS3_entropy_sensitivity", "figS3_entropy_sensitivity"),
    ("figures.supp.figS7_calibration_pr", "figS7_calibration_pr"),
    ("figures.supp.figS8_frontier", "figS8_frontier"),
    ("figures.supp.figS9_internal_generalization", "figS9_internal_generalization"),
]
# Single-panel figures: the composite already IS the panel, no per-panel split needed.
SINGLE = ["figS2_kselection", "figS4_resistance_recovery", "figS5_atlas_full", "figS6_drug_ranking"]

# Fig 1 draws 4 stages in one axes (xlim 0..4.3); crop by x-fraction bands [letter -> (x0, x1)].
FIG01_BANDS = {"a": (0.00, 0.255), "b": (0.258, 0.508), "c": (0.508, 0.755), "d": (0.755, 1.00)}


def _panel_letter(ax):
    """Return (letter, text_artist) for the bold single-letter panel label on ``ax``, else (None, None)."""
    for t in ax.texts:
        s = t.get_text().strip()
        if len(s) == 1 and s in "abcdefgh" and str(t.get_fontweight()) in ("bold", "700"):
            return s, t
    return None, None


def _save_crop(fig, bbox_disp, stem: str, letter: str) -> None:
    """Save the display-space bbox region of ``fig`` as a standalone PDF + 600-dpi PNG.

    The bbox is taken as-is (any padding/clamping is applied by the caller), so an adjacent panel's
    content is never re-included by a blanket pad.
    """
    bb_in = bbox_disp.transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(OUT / f"{stem}_{letter}.pdf", bbox_inches=bb_in)
    fig.savefig(OUT / f"{stem}_{letter}.png", dpi=600, bbox_inches=bb_in)


def _export_panels(fig, stem: str) -> list[str]:
    """Write each lettered panel of ``fig`` (with adjacent colorbar/inset axes) standalone."""
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    letters: list[str] = []

    if stem == "fig01_patch_schematic":
        W, H = fig.get_size_inches() * fig.dpi  # display units
        for letter, (xf0, xf1) in FIG01_BANDS.items():
            bbox = Bbox.from_extents(xf0 * W, 0.0, xf1 * W, H)
            _save_crop(fig, bbox, stem, letter)
            letters.append(letter)
        return letters

    lettered = [(ax, *_panel_letter(ax)) for ax in fig.axes]
    panel_axes = [(ax, lt, tx) for ax, lt, tx in lettered if lt]
    attach_axes = [ax for ax, lt, _ in lettered if not lt]
    W, H = fig.get_size_inches() * fig.dpi  # display px
    for pax, letter, _ in panel_axes:
        bbox = pax.get_tightbbox(rend)
        ppos = pax.get_position()
        # union any unlettered axes (colorbar / inset / twin) whose position is adjacent to this panel
        for q in attach_axes:
            qpos = q.get_position()
            near_x = qpos.x0 < ppos.x1 + 0.10 and qpos.x1 > ppos.x0 - 0.10
            near_y = qpos.y0 < ppos.y1 + 0.10 and qpos.y1 > ppos.y0 - 0.10
            if near_x and near_y:
                try:
                    bbox = Bbox.union([bbox, q.get_tightbbox(rend)])
                except (ValueError, RuntimeError):
                    pass
        # Clamp the crop (in px) so adjacent panels never bleed in: (1) to the MIDPOINT toward each
        # neighbour panel, and (2) explicitly clear of each neighbour's panel-LETTER glyph, which is
        # placed in the inter-panel gap and can sit on our side of the midpoint.
        lf, rt, bt, tp = 0.0, W, 0.0, H
        for q, qlt, qtx in panel_axes:
            if qlt == letter:
                continue
            qp = q.get_position()
            v = qp.y0 < ppos.y1 and qp.y1 > ppos.y0
            h = qp.x0 < ppos.x1 and qp.x1 > ppos.x0
            if v and qp.x0 >= ppos.x1 - 1e-6:
                rt = min(rt, 0.5 * (ppos.x1 + qp.x0) * W)
            if v and qp.x1 <= ppos.x0 + 1e-6:
                lf = max(lf, 0.5 * (ppos.x0 + qp.x1) * W)
            if h and qp.y0 >= ppos.y1 - 1e-6:
                tp = min(tp, 0.5 * (ppos.y1 + qp.y0) * H)
            if h and qp.y1 <= ppos.y0 + 1e-6:
                bt = max(bt, 0.5 * (ppos.y0 + qp.y1) * H)
            # neighbour letter glyph: exclude it if it falls within this panel's vertical band
            te = qtx.get_window_extent(rend)
            if te.y0 < bbox.y1 and te.y1 > bbox.y0:
                if 0.5 * (te.x0 + te.x1) > 0.5 * (bbox.x0 + bbox.x1):
                    rt = min(rt, te.x0 - 2.0)
                else:
                    lf = max(lf, te.x1 + 2.0)
        pad = 0.035 * fig.dpi  # px breathing room, never past the clamps
        bbox = Bbox.from_extents(max(bbox.x0 - pad, lf), max(bbox.y0 - pad, bt),
                                 min(bbox.x1 + pad, rt), min(bbox.y1 + pad, tp))
        _save_crop(fig, bbox, stem, letter)
        letters.append(letter)
    return letters


def main() -> int:
    """Re-render every figure and additionally export its panels."""
    set_style()
    OUT.mkdir(parents=True, exist_ok=True)
    orig_save = panels_mod.save_figure
    for modname, stem in FIGURES:
        mod = importlib.import_module(modname)
        captured: dict[str, object] = {}

        def _capture(fig, s, outdir, _stem=stem, _cap=captured):
            _cap["fig"] = fig
            outdir.mkdir(parents=True, exist_ok=True)
            fig.savefig(outdir / f"{s}.pdf")
            fig.savefig(outdir / f"{s}.png", dpi=300)
            return outdir / f"{s}.pdf", outdir / f"{s}.png"  # do NOT close yet

        mod.save_figure = _capture  # intercept the module-global save_figure
        try:
            mod.main()
            fig = captured.get("fig")
            if fig is not None:
                letters = _export_panels(fig, stem)
                print(f"{stem}: panels {letters}")
                plt.close(fig)
        finally:
            mod.save_figure = orig_save
    print(f"Single-panel figures (composite is the panel): {SINGLE}")
    print(f"Wrote per-panel files to {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
