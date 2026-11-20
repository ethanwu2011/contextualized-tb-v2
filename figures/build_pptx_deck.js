const pptxgen = require("pptxgenjs");
const fs = require("fs");
const path = require("path");

const ROOT = path.resolve(__dirname, "..");
const FULL = (process.env.CTB_RESULTS_DIR || ROOT + "/results") + "/figures/plotly";
const A = FULL + "/pptx_assets";

const INK = "1A1A1A", MUTE = "6B7177";

function pngSize(p) { const b = fs.readFileSync(p); return { w: b.readUInt32BE(16), h: b.readUInt32BE(20) }; }
function fit(p, bx, by, bw, bh) {
  const { w: nw, h: nh } = pngSize(p);
  const s = Math.min(bw / nw, bh / nh);
  const w = nw * s, h = nh * s;
  return { path: p, x: bx + (bw - w) / 2, y: by + (bh - h) / 2, w, h };
}

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";
pres.author = "Ethan Wu";
pres.title = "PATCH figures";

const header = (s, t) => s.addText(t, { x: 0.4, y: 0.18, w: 12.5, h: 0.5, fontSize: 19, bold: true,
  color: INK, fontFace: "Arial", margin: 0, valign: "middle" });
const label = (s, l, x, y) => s.addText(l, { x, y, w: 0.45, h: 0.4, fontSize: 22, bold: true, color: INK,
  fontFace: "Arial", margin: 0, valign: "top" });
const pageNum = (s, n) => s.addText(String(n), { x: 12.7, y: 7.05, w: 0.4, h: 0.3, fontSize: 10,
  color: MUTE, fontFace: "Arial", align: "right", margin: 0 });
const L = ["a", "b", "c", "d", "e"];

const slides = [
  { kind: "schematic", title: "Figure 1 — PATCH: per-patient treatment-response models from sparse, multimodal registries" },
  { kind: "leftstack", title: "Figure 2 — Treatment-response subtypes",
    p: [A + "/fig02_a_rate.png", A + "/fig02_b_radars.png", A + "/fig02_c_attribution.png"] },
  { kind: "matrix24", title: "Figure 3 — Outcome organization sharpens treatment → context → PATCH, robust across UMAP/t-SNE/PCA (died/fail rate surface · outcome AUROC)",
    cols: ["UMAP", "t-SNE", "PCA"],
    rows: ["PATCH (contextualized)", "Context only (raw features)", "Treatment only (regimen)"],
    p: ["patch_umap", "patch_tsne", "patch_pca",
        "context_umap", "context_tsne", "context_pca",
        "treatment_umap", "treatment_tsne", "treatment_pca"].map(x => `${A}/fig03m_${x}.png`) },
  { kind: "matrix24", title: "Figure 3 (detail) — same UMAP recoloured: what each space organizes by (kNN accuracy)",
    cols: ["Outcome (died/fail)", "Subtype", "Resistance", "HIV", "Region"],
    rows: ["Contextualized (PATCH)", "Non-contextualized (raw)"],
    p: ["ctx_outcome", "ctx_subtype", "ctx_resistance", "ctx_hiv", "ctx_region",
        "raw_outcome", "raw_subtype", "raw_resistance", "raw_hiv", "raw_region"].map(x => `${A}/fig03_${x}.png`) },
  { kind: "tall", title: "Figure 4 — Context-conditioned coefficient atlas β(C) (clustered)",
    img: FULL + "/fig04_atlas.png" },
  { kind: "mside", title: "Figure 4 (cont.) — Archetypal coarsening + variance decomposition",
    p: [A + "/fig04_b_subtype_drug.png", A + "/fig04_c_variance.png"] },
  { kind: "mside", title: "Figure 5 — External validation on Brazil (n ≈ 298,900)",
    p: [A + "/fig05_a_auroc.png", FULL + "/fig05_brazil_replicates.png"] },
  { kind: "mside", title: "Figure 6 — Clinical utility (decision-curve analysis)",
    p: [A + "/fig06_a_decision_curve.png", A + "/fig06_b_yield.png"] },
  { kind: "single", title: "Figure 3 (alt) — UMAP density by subtype", img: FULL + "/fig03_umap_density.png" },
  { kind: "single", title: "Supplementary — Identifiability / positivity map", img: FULL + "/fig_identifiability.png" },
  { kind: "single", title: "Supplementary — Model ladder (forest)", img: FULL + "/fig_model_ladder_forest.png" },
  { kind: "single", title: "Supplementary — Leave-one-country-out generalization", img: FULL + "/fig_loco_country.png" },
  { kind: "mside", title: "Supplementary — Calibration + precision–recall",
    p: [A + "/figcal_a_calibration.png", A + "/figcal_b_pr.png"] },
  { kind: "mside", title: "Supplementary — Entropy load-balancing penalty",
    p: [A + "/figentropy_a_balance.png", A + "/figentropy_b_auroc.png"] },
  { kind: "single", title: "Supplementary — Hyperparameter sweep", img: FULL + "/fig_hparam_parcoords.png" },
  { kind: "single", title: "Figure 2 (alt) — Subtype fingerprints as parallel coordinates", img: FULL + "/fig02_parcoords.png" },
  { kind: "single", title: "Figure 4 (companion) — Effect-atlas volcano", img: FULL + "/fig04_volcano.png" },
  { kind: "single", title: "Figure 2 (alt) — Modality attribution treemap", img: A + "/fig02_c_treemap.png" },
];

slides.forEach((sp, i) => {
  const slide = pres.addSlide();
  slide.background = { color: "FFFFFF" };

  if (sp.kind === "tall") {                          // portrait hero: full height, no header
    slide.addImage(fit(sp.img, 0.3, 0.18, 12.7, 7.14));

  } else if (sp.kind === "single") {
    header(slide, sp.title);
    slide.addImage(fit(sp.img, 0.4, 0.95, 12.5, 6.35));

  } else if (sp.kind === "mside") {
    header(slide, sp.title);
    slide.addImage(fit(sp.p[0], 0.4, 1.1, 6.1, 6.0));
    slide.addImage(fit(sp.p[1], 6.8, 1.1, 6.1, 6.0));
    label(slide, "a", 0.45, 1.0); label(slide, "b", 6.85, 1.0);

  } else if (sp.kind === "leftstack") {
    header(slide, sp.title);
    slide.addImage(fit(sp.p[0], 0.4, 1.0, 4.7, 6.15));
    slide.addImage(fit(sp.p[1], 5.35, 1.0, 7.55, 2.9));
    slide.addImage(fit(sp.p[2], 5.35, 4.15, 7.55, 3.0));
    label(slide, "a", 0.45, 0.95); label(slide, "b", 5.4, 0.95); label(slide, "c", 5.4, 4.1);

  } else if (sp.kind === "matrix24") {
    header(slide, sp.title);
    const nc = sp.cols.length, nr = sp.rows.length;
    const gx = 1.6, gy = 1.25, gw = 11.55, gh = 6.0, gap = 0.1;
    const cw = (gw - (nc - 1) * gap) / nc, rh = (gh - (nr - 1) * gap) / nr;
    sp.cols.forEach((c, ci) => slide.addText(c, { x: gx + ci * (cw + gap), y: 0.84, w: cw, h: 0.32,
      fontSize: 13, bold: true, color: INK, align: "center", fontFace: "Arial", margin: 0 }));
    sp.rows.forEach((r, ri) => slide.addText(r, { x: 0.04, y: gy + ri * (rh + gap) + rh / 2 - 0.5,
      w: 1.5, h: 1.0, fontSize: 11, bold: true, color: INK, align: "center", valign: "middle",
      fontFace: "Arial", margin: 0 }));
    for (let ri = 0; ri < nr; ri++) for (let ci = 0; ci < nc; ci++) {
      slide.addImage(fit(sp.p[ri * nc + ci], gx + ci * (cw + gap), gy + ri * (rh + gap), cw, rh));
    }
  } else if (sp.kind === "schematic") {
    header(slide, sp.title);
    const stages = [
      { c: "4C566A", h: "Inputs", b: ["Multimodal context C", "335 features · 8 modalities:", "resistance · clinical ·", "imaging · genomics · labs", "+ regimen X (26 drugs)"] },
      { c: "0072B2", h: "Archetype factoring", b: ["Z(C) = softmax(M · C)", "", "share strength across", "K = 5 learnable", "archetypes (simplex)"] },
      { c: "CC79A7", h: "Per-patient coefficients", b: ["β(C) = Z(C) · A", "", "context-conditioned", "coefficient vector", "+ identifiability audit"] },
      { c: "333333", h: "Outcome model", b: ["logit P(died / fail)", "= X·β(C) + μ(C)", "", "matches a tuned black box;", "geared to interpretability"] },
      { c: "D55E00", h: "Outputs", b: ["5 reproducible", "treatment-response", "subtypes", "", "validated on Brazil", "(SINAN, n = 298,900)"] },
    ];
    const bw = 2.26, gap = 0.36, x0 = 0.42, by = 2.5, bh = 2.3;
    stages.forEach((st, k) => {
      const x = x0 + k * (bw + gap);
      slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: by, w: bw, h: bh, fill: { color: "F6F8FB" },
        line: { color: "E2E8F0", width: 1 }, rectRadius: 0.09,
        shadow: { type: "outer", color: "000000", blur: 7, offset: 2, angle: 45, opacity: 0.1 } });
      slide.addShape(pres.shapes.OVAL, { x: x + 0.14, y: by + 0.16, w: 0.44, h: 0.44, fill: { color: st.c } });
      slide.addText(String(k + 1), { x: x + 0.14, y: by + 0.16, w: 0.44, h: 0.44, fontSize: 15, bold: true,
        color: "FFFFFF", align: "center", valign: "middle", margin: 0 });
      slide.addText(st.h, { x: x + 0.66, y: by + 0.18, w: bw - 0.74, h: 0.42, fontSize: 12.5, bold: true,
        color: st.c, align: "left", valign: "middle", margin: 0 });
      slide.addText(st.b.map((t) => ({ text: t, options: { breakLine: true } })),
        { x: x + 0.18, y: by + 0.74, w: bw - 0.34, h: bh - 0.9, fontSize: 10.5, color: "333333",
          align: "left", valign: "top", margin: 0, lineSpacingMultiple: 1.06 });
      if (k < stages.length - 1) slide.addShape(pres.shapes.LINE,
        { x: x + bw + 0.05, y: by + bh / 2, w: gap - 0.1, h: 0,
          line: { color: "94A0AE", width: 2.5, endArrowType: "triangle" } });
    });
    slide.addText([{ text: "Key finding:  ", options: { bold: true, color: "1A1A1A" } },
      { text: "drug resistance organizes who does ", options: { color: "333333" } },
      { text: "well", options: { bold: true, italic: true, color: "0072B2" } },
      { text: " — treatment ", options: { color: "333333" } },
      { text: "failure", options: { bold: true, italic: true, color: "D55E00" } },
      { text: " is multimodal (bacillary burden, region, HIV).", options: { color: "333333" } }],
      { x: 0.5, y: 5.25, w: 12.3, h: 0.5, fontSize: 15, align: "center", margin: 0 });
    slide.addText("Factoring per-patient coefficients through a few shared archetypes makes them learnable from sparse data and reproducible across cohorts; the built-in audit flags which effects the data actually support.",
      { x: 0.5, y: 6.15, w: 12.3, h: 0.7, fontSize: 12, italic: true, color: "6B7177", align: "center", margin: 0 });
  }
  pageNum(slide, i + 1);
});

pres.writeFile({ fileName: "/tmp/pptxbuild/PATCH_figures.pptx" }).then((f) => console.log("wrote", f));
