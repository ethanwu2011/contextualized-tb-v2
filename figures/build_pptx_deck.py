"""Assemble the PATCH figure deck with python-pptx (PowerPoint-native, unlike the old pptxgenjs build).

One slide per figure; multi-panel figures are composed from the individually-rendered Plotly panels in
results/figures/plotly/pptx_assets so spacing is controlled here, not by Plotly subplots. python-pptx
produces clean OOXML that PowerPoint and Keynote render reliably (pptxgenjs+rezip did not).

Run: ``.venv/bin/python -m figures.build_pptx_deck``  ->  results/figures/plotly/PATCH_figures.pptx
"""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.opc.constants import RELATIONSHIP_TYPE as RT
from pptx.opc.package import Part
from pptx.opc.packuri import PackURI
from pptx.oxml import parse_xml
from pptx.util import Inches, Pt

from ctb.config import FIGURES_DIR

FULL = FIGURES_DIR / "plotly"
A = FULL / "pptx_assets"
OUT = FULL / "PATCH_figures.pptx"

INK = "1A1A1A"
MUTE = "6B7177"
EMU = 914400
SW, SH = 13.333, 7.5


def rgb(h):
    return RGBColor.from_string(h)


def fit(path, bx, by, bw, bh):
    """Return (x, y, w, h) in inches, image scaled to fit the box preserving aspect, centred."""
    w, h = Image.open(path).size
    s = min(bw / w, bh / h)
    fw, fh = w * s, h * s
    return bx + (bw - fw) / 2, by + (bh - fh) / 2, fw, fh


_SVG_EXT = ('<a:extLst xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">'
            '<a:ext uri="{96DAC541-7B7A-43D3-8B79-37D633B846F1}">'
            '<asvg:svgBlip xmlns:asvg="http://schemas.microsoft.com/office/drawing/2016/SVG/main" '
            'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" '
            'r:embed="%s"/></a:ext></a:extLst>')


def _embed_svg(pic_shape, slide, svg_path):
    """Attach an SVG blip (with the just-added PNG as raster fallback) so PowerPoint shows the vector
    and 'Convert to Shape' can decompose it into movable objects."""
    pkg = slide.part.package
    used = {p.partname for p in pkg.iter_parts()}
    i = 1
    while PackURI(f"/ppt/media/svgfig{i}.svg") in used:
        i += 1
    svg_part = Part(PackURI(f"/ppt/media/svgfig{i}.svg"), "image/svg+xml", pkg, Path(svg_path).read_bytes())
    rid = slide.part.relate_to(svg_part, RT.IMAGE)
    pic_shape._element.blipFill.blip.append(parse_xml(_SVG_EXT % rid))


def pic(slide, path, bx, by, bw, bh):
    path = str(path)
    x, y, w, h = fit(path, bx, by, bw, bh)
    p = slide.shapes.add_picture(path, Inches(x), Inches(y), Inches(w), Inches(h))
    svg = path[:-4] + ".svg"
    if path.lower().endswith(".png") and Path(svg).exists():
        try:
            _embed_svg(p, slide, svg)
        except Exception as e:                     # never let SVG-embed break the build; PNG still shows
            print(f"  [svg-embed skipped] {Path(path).name}: {e}")
    return p


def _tf(tb):
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    return tf


def text(slide, runs, x, y, w, h, size, *, bold=False, color=INK, align=PP_ALIGN.LEFT,
         anchor=MSO_ANCHOR.TOP, italic=False, name="Arial"):
    """runs = str OR list of (text, {opts}). Single paragraph, multiple coloured runs supported."""
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = _tf(tb)
    tf.vertical_anchor = anchor
    p = tf.paragraphs[0]
    p.alignment = align
    items = [(runs, {})] if isinstance(runs, str) else runs
    for t, o in items:
        r = p.add_run()
        r.text = t
        f = r.font
        f.size = Pt(o.get("size", size))
        f.bold = o.get("bold", bold)
        f.italic = o.get("italic", italic)
        f.name = name
        f.color.rgb = rgb(o.get("color", color))
    return tb


def multiline(slide, lines, x, y, w, h, size, color="333333", name="Arial"):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = _tf(tb)
    for i, ln in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        r = p.add_run()
        r.text = ln
        r.font.size = Pt(size)
        r.font.name = name
        r.font.color.rgb = rgb(color)


def header(slide, t):
    text(slide, t, 0.4, 0.18, 12.5, 0.6, 19, bold=True, anchor=MSO_ANCHOR.MIDDLE)


def label(slide, ltr, x, y):
    text(slide, ltr, x, y, 0.45, 0.4, 22, bold=True)


def pagenum(slide, n):
    text(slide, str(n), 12.7, 7.05, 0.4, 0.3, 10, color=MUTE, align=PP_ALIGN.RIGHT)


def box(slide, shp, x, y, w, h, fill, line=None, lw=1.0):
    s = slide.shapes.add_shape(shp, Inches(x), Inches(y), Inches(w), Inches(h))
    s.fill.solid()
    s.fill.fore_color.rgb = rgb(fill)
    if line:
        s.line.color.rgb = rgb(line)
        s.line.width = Pt(lw)
    else:
        s.line.fill.background()
    s.shadow.inherit = False
    return s


# ---- Figure 1: native, fully-editable compare-and-contrast schematic --------------------------
# Every element below is a discrete PowerPoint shape/textbox, so each item can be moved individually.
SUBC = ["009E73", "0072B2", "56B4E9", "E69F00", "D55E00"]           # subtype colours (narrative order)
ACCENT, WARN = "009E73", "D55E00"


def _dcol(v):
    st = [(-1, (33, 102, 172)), (-0.5, (103, 169, 207)), (0, (247, 247, 247)),
          (0.5, (239, 138, 98)), (1, (178, 24, 43))]
    v = max(-1.0, min(1.0, v))
    for (a, ca), (b, cb) in zip(st, st[1:]):
        if a <= v <= b:
            t = 0 if b == a else (v - a) / (b - a)
            return "".join(f"{round(ca[i] + t * (cb[i] - ca[i])):02X}" for i in range(3))
    return "F7F7F7"


def _ctext(s, txt, size, color, bold, align=PP_ALIGN.CENTER):
    tf = s.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = Inches(0.03)
    tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    p.alignment = align
    for i, ln in enumerate(txt.split("\n")):
        pp = p if i == 0 else tf.add_paragraph()
        pp.alignment = align
        r = pp.add_run()
        r.text = ln
        f = r.font
        f.size, f.bold, f.name = Pt(size if i == 0 else size - 1.5), bold, "Arial"
        f.color.rgb = rgb(color if i == 0 else MUTE)


def chip(slide, x, y, w, h, fill, txt="", size=10, tcolor=INK, line=None, lw=1.0, bold=True,
         shp=MSO_SHAPE.ROUNDED_RECTANGLE, align=PP_ALIGN.CENTER):
    s = box(slide, shp, x, y, w, h, fill, line, lw)
    if txt:
        _ctext(s, txt, size, tcolor, bold, align)
    return s


def dot(slide, cx, cy, d, fill, line=None, lw=1.0):
    return box(slide, MSO_SHAPE.OVAL, cx - d / 2, cy - d / 2, d, d, fill, line, lw)


def flow(slide, x, y, w=0.32, h=0.13, fill="9AA0A6"):
    return box(slide, MSO_SHAPE.RIGHT_ARROW, x, y, w, h, fill)


def bars(slide, x, y, cw, h, vals, gap=0.007):
    for i, v in enumerate(vals):
        box(slide, MSO_SHAPE.RECTANGLE, x + i * (cw + gap), y, cw, h, _dcol(v), "FFFFFF", 0.4)


def fig1_native(slide):
    import random
    rng = random.Random(7)
    text(slide, "Figure 1 — PATCH: from patient buckets to a treatment-response spectrum",
         0.4, 0.12, 12.6, 0.4, 20, bold=True)
    text(slide, "Conventional models sort patients into a few fixed buckets and fit one model per group; "
         "contextualized modeling places every patient on a continuous spectrum of treatment-response "
         "archetypes, giving each a context-conditioned coefficient model even from sparse records.",
         0.4, 0.54, 12.6, 0.4, 10.5, italic=True, color=MUTE)
    box(slide, MSO_SHAPE.RECTANGLE, 6.62, 1.02, 0.014, 3.38, "DDDFE1")                  # divider

    # ---------- LEFT: conventional / buckets ----------
    chip(slide, 0.45, 1.0, 5.9, 0.34, MUTE, "CONVENTIONAL  ·  POPULATION MODELS", 11, "FFFFFF")
    text(slide, "patients forced into a few discrete buckets — one shared model each",
         0.45, 1.4, 5.9, 0.22, 9, italic=True, color=MUTE)
    bx = [0.6, 2.5, 4.4]
    for bxc, title in zip(bx, ["Drug-sensitive", "MDR", "XDR"]):
        text(slide, title, bxc, 1.66, 1.75, 0.2, 8.5, bold=True, color="55595E", align=PP_ALIGN.CENTER)
        box(slide, MSO_SHAPE.ROUNDED_RECTANGLE, bxc, 1.88, 1.75, 1.2, "F5F6F7", "D6D9DC", 1.0)
        for _ in range(5):
            dot(slide, bxc + rng.uniform(0.22, 1.53), 1.98 + rng.uniform(0.12, 0.92), 0.1, "9AA0A6")
    dot(slide, bx[0] + 1.62, 2.48, 0.13, "D5D8DB", WARN, 1.6)                            # borderline
    text(slide, "?", bx[0] + 1.7, 2.3, 0.25, 0.25, 11, bold=True, color=WARN)
    text(slide, "borderline patient forced into one bucket", 0.6, 3.12, 4.0, 0.2, 7.5, italic=True, color=WARN)
    text(slide, "one shared model  θ̂  per bucket   →", 0.6, 3.44, 2.7, 0.22, 9, color="55595E")
    for i in range(10):
        box(slide, MSO_SHAPE.RECTANGLE, 3.4 + i * 0.14, 3.44, 0.132, 0.2, "C9CDD1", "FFFFFF", 0.4)
    text(slide, "population-average effect  ·  heterogeneity discarded", 3.4, 3.68, 3.0, 0.2, 8,
         italic=True, color=MUTE)
    text(slide, "needs large, homogeneous cohorts  ·  power vs. personalization tradeoff",
         0.45, 4.0, 5.9, 0.22, 8.5, italic=True, color=MUTE)

    # ---------- RIGHT: contextualized / spectrum ----------
    chip(slide, 6.85, 1.0, 5.95, 0.34, ACCENT, "CONTEXTUALIZED MODELING — PATCH", 11, "FFFFFF")
    text(slide, "every patient on a continuous spectrum of archetypes — a personalized model each",
         6.85, 1.4, 5.95, 0.22, 9, italic=True, color=ACCENT)
    px, py, pw, ph = 6.9, 1.72, 5.9, 1.42
    box(slide, MSO_SHAPE.ROUNDED_RECTANGLE, px, py, pw, ph, "FCFCFD", "E4E6E8", 1.0)
    anc = [(0.15, 0.32), (0.40, 0.2), (0.63, 0.34), (0.3, 0.74), (0.75, 0.66)]
    apos = [(px + ax * pw, py + ay * ph) for ax, ay in anc]
    for _ in range(24):                                                                  # continuous mixtures
        k, j = rng.randrange(5), rng.randrange(5)
        t = rng.uniform(0, 0.5)
        dot(slide, apos[k][0] * (1 - t) + apos[j][0] * t, apos[k][1] * (1 - t) + apos[j][1] * t, 0.07, SUBC[k])
    for k, (axp, ayp) in enumerate(apos):
        dot(slide, axp, ayp, 0.17, SUBC[k], "FFFFFF", 1.2)
    dot(slide, (apos[0][0] + apos[3][0]) / 2, (apos[0][1] + apos[3][1]) / 2, 0.12, "F0F0F0", WARN, 1.6)
    text(slide, "same borderline patient — a natural blend", px + 0.12, py + ph - 0.24, 3.0, 0.2, 7.5,
         italic=True, color=WARN)
    text(slide, "soft membership  Z(C)=softmax(M·C+q)", px + pw - 3.0, py + 0.03, 2.9, 0.2, 7.5,
         italic=True, color=MUTE, align=PP_ALIGN.RIGHT)
    text(slide, "→  a personalized effect  β(C)  per patient", 6.85, 3.26, 3.6, 0.2, 9, color="333333")
    for r in range(3):
        yy = 3.5 + r * 0.19
        dot(slide, 6.98, yy + 0.06, 0.1, SUBC[[0, 3, 1][r]])
        bars(slide, 7.14, yy, 0.11, 0.13, [rng.uniform(-1, 1) for _ in range(9)])
    text(slide, "works from sparse records  ·  borrows strength across archetypes",
         6.85, 4.0, 5.95, 0.22, 8.5, italic=True, color=ACCENT)

    # ---------- BAND 2: architecture ----------
    text(slide, "How — one interpretable model, learned end to end", 0.4, 4.4, 12.6, 0.26, 12, bold=True)
    ay = 4.76
    st = [("Patient\ncontext C", "F3F4F5"), ("Context encoder\nZ(C)=softmax(M·C+q)", "F3F4F5"),
          ("Weighting over\nK=5 archetypes A", "F3F4F5"), ("Per-patient effect\nβ(C)=A·Z(C)", "EAF4F0"),
          ("Logistic outcome\nP(died/failure)", "F3F4F5"), ("Outcome loss\nℓ(θ; Y)", "F3F4F5")]
    cw2, step = 1.8, 2.1
    for i, (t1, fill) in enumerate(st):
        x = 0.4 + i * step
        chip(slide, x, ay, cw2, 0.58, fill, t1, 8.6, INK, "D6D9DC", 1.0)
        if i < 5:
            flow(slide, x + cw2 + 0.02, ay + 0.22)
    for k in range(5):                                                                   # archetype colour dots
        dot(slide, 0.4 + 2 * step + 0.32 + k * 0.26, ay + 0.46, 0.1, SUBC[k])

    # ---------- BAND 3: what each paradigm gives you ----------
    text(slide, "What you get", 0.4, 5.5, 3.0, 0.24, 12, bold=True)
    cD, cC, cP = 0.45, 2.75, 7.55
    wD, wC, wP = 2.2, 4.6, 5.3
    ty = 5.82
    chip(slide, cC, ty, wC, 0.26, "EDEEF0", "Conventional models", 9, "55595E", "D6D9DC", 0.8)
    chip(slide, cP, ty, wP, 0.26, "EAF4F0", "PATCH  (contextualized)", 9, ACCENT, "BFE3D6", 0.8)
    rows = [
        ("Representation", "fixed buckets / subgroups", "continuous spectrum · soft membership Z(C)"),
        ("Per patient", "one shared model", "a context-conditioned coefficient model β(C)"),
        ("Sparse data", "needs large cohorts", "borrows strength — 1,810 params, not ~277,000"),
        ("Estimates", "outcome risk · feature importance", "recorded-regimen associations, drug by drug"),
        ("Trust", "no identifiability signal", "positivity mask (69/130 → trials) + external validation"),
        ("Accuracy", "(a black box)", "matches the black box (~0.79) while staying interpretable"),
    ]
    ry = ty + 0.29
    for i, (d, c, p) in enumerate(rows):
        if i % 2 == 0:
            box(slide, MSO_SHAPE.RECTANGLE, cD, ry - 0.005, 12.4, 0.185, "F7F8F9")
        text(slide, d, cD, ry, wD, 0.185, 8.5, bold=True, color="55595E", anchor=MSO_ANCHOR.MIDDLE)
        text(slide, c, cC + 0.05, ry, wC - 0.1, 0.185, 8.3, color="6B7177", anchor=MSO_ANCHOR.MIDDLE)
        text(slide, p, cP + 0.05, ry, wP - 0.1, 0.185, 8.3, color="1A1A1A", anchor=MSO_ANCHOR.MIDDLE)
        ry += 0.188


SLIDES = [
    {"kind": "tall", "img": A / "fig01_hero.png"},
    {"kind": "leftstack", "title": "Figure 2 — Treatment-response subtypes",
     "p": [A / "fig02_a_rate.png", A / "fig02_b_radars.png", A / "fig02_c_attribution.png"]},
    {"kind": "fig3", "title": "Figure 3 — Contextualization organizes the treatment-response structure and the outcome",
     "p": [A / "fig03m_treatment_umap.png", A / "fig03m_context_umap.png", A / "fig03m_patch_umap.png",
           A / "fig03_ctx_subtype.png", A / "fig03_d_auroc_tiers.png"],
     "caps": ["Treatment (regimen) — flat", "Raw context (335 features)", "PATCH (contextualized)",
              "Latent structure: five subtypes", "Outcome AUROC: treatment < context < PATCH"]},
    {"kind": "fig4", "title": "Figure 4 — Coefficient atlas, subtype coarsening, and post hoc literature triage",
     "atlas": FULL / "fig04_atlas.png", "subtype": A / "fig04_b_subtype_drug.png",
     "heatmap": A / "fig04_c_heatmap.png", "lollipop": A / "fig04_c_lollipop.png"},
    {"kind": "mside", "title": "Figure 5 — External validation on Brazil (n ≈ 298,900)",
     "p": [A / "fig05_a_auroc.png", FULL / "fig05_brazil_replicates.png"]},
    {"kind": "mside", "title": "Figure 6 — Clinical utility (decision-curve analysis)",
     "p": [A / "fig06_a_decision_curve.png", A / "fig06_b_yield.png"]},
    {"kind": "single", "title": "Supplementary — UMAP density by subtype", "img": FULL / "fig03_umap_density.png"},
    {"kind": "single", "title": "Supplementary — Identifiability / positivity map", "img": FULL / "fig_identifiability.png"},
    {"kind": "single", "title": "Supplementary — Model ladder (forest)", "img": FULL / "fig_model_ladder_forest.png"},
    {"kind": "single", "title": "Supplementary — Leave-one-country-out generalization", "img": FULL / "fig_loco_country.png"},
    {"kind": "mside", "title": "Supplementary — Calibration + precision–recall",
     "p": [A / "figcal_a_calibration.png", A / "figcal_b_pr.png"]},
    {"kind": "mside", "title": "Supplementary — Entropy load-balancing penalty",
     "p": [A / "figentropy_a_balance.png", A / "figentropy_b_auroc.png"]},
    {"kind": "single", "title": "Supplementary — Hyperparameter sweep", "img": FULL / "fig_hparam_parcoords.png"},
    {"kind": "single", "title": "Supplementary — Subtype fingerprints (parallel coordinates)", "img": FULL / "fig02_parcoords.png"},
    {"kind": "single", "title": "Supplementary — Coefficient-atlas volcano", "img": FULL / "fig04_volcano.png"},
]

# Fig 4 ledger — display labels map to the evidence taxonomy in docs/EVIDENCE_LEDGER.md.
LEDGER = [
    ("009E73", "Concordant recorded associations", "positive control",
     ["HIV → higher risk", "MDR / XDR → higher risk", "High bacillary burden → higher risk",
      "Rifampicin association with", "drug-sensitive TB"]),
    ("0072B2", "Robust recorded association", "",
     ["Rifampicin ↔ drug-sensitive", "Δβ ≈ −0.43,  q < 0.001", "(sole FDR survivor of 130", "subtype × drug associations)"]),
    ("9AA0A6", "Not identifiable — walled off", "",
     ["69 / 130 cells near-", "deterministic (positivity fails)", "per-drug substitution", "NOT licensed by these data"]),
]


def schematic(slide):
    stages = [
        ("4C566A", "Inputs", ["Multimodal context C", "335 features · 8 modalities:", "resistance · clinical ·", "imaging · genomics · labs", "+ regimen X (26 drugs)"]),
        ("0072B2", "Archetype factoring", ["Z(C) = softmax(M · C + q)", "", "share strength across", "K = 5 learnable", "archetypes (simplex)"]),
        ("CC79A7", "Per-patient coefficients", ["β(C) = Z(C) · A", "", "context-conditioned", "coefficient vector", "+ identifiability audit"]),
        ("333333", "Outcome model", ["logit P(died / fail)", "= X·β(C) + μ(C)", "", "matches a tuned black box;", "geared to interpretability"]),
        ("D55E00", "Outputs", ["5 reproducible", "treatment-response", "subtypes", "", "validated on Brazil", "(SINAN, n = 298,900)"]),
    ]
    bw, gap, x0, by, bh = 2.26, 0.36, 0.42, 2.5, 2.3
    for k, (c, hh, body) in enumerate(stages):
        x = x0 + k * (bw + gap)
        box(slide, MSO_SHAPE.ROUNDED_RECTANGLE, x, by, bw, bh, "F6F8FB", line="E2E8F0")
        box(slide, MSO_SHAPE.OVAL, x + 0.14, by + 0.16, 0.44, 0.44, c)
        text(slide, str(k + 1), x + 0.14, by + 0.16, 0.44, 0.44, 15, bold=True, color="FFFFFF",
             align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
        text(slide, hh, x + 0.66, by + 0.18, bw - 0.74, 0.42, 12.5, bold=True, color=c, anchor=MSO_ANCHOR.MIDDLE)
        multiline(slide, body, x + 0.18, by + 0.74, bw - 0.34, bh - 0.9, 10.5)
        if k < len(stages) - 1:
            box(slide, MSO_SHAPE.RIGHT_ARROW, x + bw + 0.04, by + bh / 2 - 0.11, gap - 0.06, 0.22, "94A0AE")
    text(slide, [("Key finding:  ", {"bold": True, "color": INK}),
                 ("drug resistance organizes who does ", {"color": "333333"}),
                 ("well", {"bold": True, "italic": True, "color": "0072B2"}),
                 (" — treatment ", {"color": "333333"}),
                 ("failure", {"bold": True, "italic": True, "color": "D55E00"}),
                 (" is multimodal (bacillary burden, region, HIV).", {"color": "333333"})],
         0.5, 5.25, 12.3, 0.5, 15, align=PP_ALIGN.CENTER)
    text(slide, "Factoring per-patient coefficients through a few shared archetypes makes them learnable from sparse data and reproducible across cohorts; the built-in audit flags which effects the data actually support.",
         0.9, 6.15, 11.5, 0.7, 12, italic=True, color=MUTE, align=PP_ALIGN.CENTER)


def main() -> int:
    prs = Presentation()
    prs.slide_width = Inches(SW)
    prs.slide_height = Inches(SH)
    blank = prs.slide_layouts[6]
    for i, sp in enumerate(SLIDES):
        s = prs.slides.add_slide(blank)
        k = sp["kind"]
        if k == "fig1":
            fig1_native(s)
        elif k == "schematic":
            header(s, sp["title"])
            schematic(s)
        elif k == "single":
            header(s, sp["title"])
            pic(s, sp["img"], 0.4, 0.95, 12.5, 6.35)
        elif k == "tall":
            pic(s, sp["img"], 0.1, 0.1, 13.13, 7.3)
        elif k == "mside":
            header(s, sp["title"])
            pic(s, sp["p"][0], 0.4, 1.1, 6.1, 6.0)
            pic(s, sp["p"][1], 6.8, 1.1, 6.1, 6.0)
            label(s, "a", 0.45, 1.0)
            label(s, "b", 6.85, 1.0)
        elif k == "fig3":                              # a/b/c three risk surfaces (crescendo) · d subtypes · e AUROC bars
            header(s, sp["title"])
            P, caps = sp["p"], sp["caps"]
            xs, w1, y1, h1 = [0.45, 4.5, 8.55], 3.95, 1.02, 3.05     # row 1: three risk surfaces
            for i in range(3):
                text(s, caps[i], xs[i] + 0.45, y1, w1 - 0.45, 0.26, 10.5, bold=True, color=MUTE)
                pic(s, P[i], xs[i], y1 + 0.28, w1, h1 - 0.28)
                label(s, "abc"[i], xs[i], y1 - 0.05)
            y2, h2 = 4.25, 2.9                                        # row 2: subtypes (d) + AUROC bars (e)
            text(s, caps[3], 0.9, y2, 5.4, 0.26, 11, bold=True, color=MUTE)
            pic(s, P[3], 0.45, y2 + 0.3, 5.9, h2 - 0.3)
            label(s, "d", 0.45, y2 - 0.05)
            text(s, caps[4], 7.15, y2, 5.5, 0.26, 11, bold=True, color=MUTE)
            pic(s, P[4], 6.7, y2 + 0.3, 6.15, h2 - 0.3)
            label(s, "e", 6.7, y2 - 0.05)

        elif k == "fig4":                              # a atlas · b subtype grid · c bordered heatmap · d lollipop
            header(s, sp["title"])
            pic(s, sp["atlas"], 0.32, 1.0, 3.95, 6.15)
            label(s, "a", 0.34, 0.94)
            rx = 4.5
            pic(s, sp["subtype"], rx, 1.02, 8.5, 1.72)
            label(s, "b", rx, 0.94)
            pic(s, sp["heatmap"], rx, 2.94, 8.5, 1.95)
            label(s, "c", rx, 2.88)
            pic(s, sp["lollipop"], rx, 5.02, 8.5, 2.05)
            label(s, "d", rx, 4.96)

        elif k == "leftstack":
            header(s, sp["title"])
            pic(s, sp["p"][0], 0.4, 1.0, 4.7, 6.15)
            pic(s, sp["p"][1], 5.35, 1.0, 7.55, 2.9)
            pic(s, sp["p"][2], 5.35, 4.15, 7.55, 3.0)
            label(s, "a", 0.45, 0.95)
            label(s, "b", 5.4, 0.95)
            label(s, "c", 5.4, 4.1)
        elif k == "matrix":
            header(s, sp["title"])
            cols, rows = sp["cols"], sp["rows"]
            nc, nr = len(cols), len(rows)
            gx, gy, gw, gh, gp = 1.6, 1.25, 11.55, 6.0, 0.1
            cw, rh = (gw - (nc - 1) * gp) / nc, (gh - (nr - 1) * gp) / nr
            for ci, c in enumerate(cols):
                text(s, c, gx + ci * (cw + gp), 0.84, cw, 0.32, 13, bold=True, align=PP_ALIGN.CENTER)
            for ri, r in enumerate(rows):
                text(s, r, 0.04, gy + ri * (rh + gp) + rh / 2 - 0.5, 1.5, 1.0, 11, bold=True,
                     align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
            for ri in range(nr):
                for ci in range(nc):
                    pic(s, sp["p"][ri * nc + ci], gx + ci * (cw + gp), gy + ri * (rh + gp), cw, rh)
        pagenum(s, i + 1)
    prs.save(str(OUT))
    print(f"wrote {OUT}  ({len(SLIDES)} slides)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
