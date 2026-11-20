"""Export individual sub-panels of the multi-panel Plotly figures as standalone high-res PNGs.

These are the building blocks for the PowerPoint figure deck: each panel is rendered on its own (white
background, its own axes/legend/subtitle, NO figure-level title or a/b letter — those are added in PPT),
so the multi-panel figures can be composed with proper spacing in PowerPoint. Single-panel figures reuse
their existing full renders in ``results/figures/plotly/``. Aggregate tables only (DUA-safe).

Run: ``.venv/bin/python -m figures.pptx_export``
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import scipy.cluster.hierarchy as sch

from figures.palette import (
    MODALITY_COLORS,
    NARRATIVE_ORDER,
    subtype_color,
    subtype_name,
)
from figures.plotly_style import DIVERGING, INK, MUTED, TABLES, hex_to_rgba

SUB = [f"S{i}" for i in NARRATIVE_ORDER]
SNAMES = [subtype_name(i) for i in NARRATIVE_ORDER]
SCOLORS = [subtype_color(i) for i in NARRATIVE_ORDER]
DRUG_LABELS = {"Cotrimoxazol": "Cotrim", "other_second_line": "2nd-line"}

ASSETS = TABLES.parent / "figures" / "plotly" / "pptx_assets"
C_PATCH, C_RES, C_GREY, C_DARK = "#0072B2", "#882255", "#B8B8B8", "#333333"


def _save(fig: go.Figure, name: str, w: int, h: int) -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    fig.update_layout(template="patch", width=w, height=h, paper_bgcolor="white", plot_bgcolor="white",
                      font=dict(family="Arial, Helvetica, sans-serif", size=13, color=INK))
    fig.write_image(ASSETS / f"{name}.png", scale=3)
    try:                                            # vector sibling -> deck embeds SVG (Convert to Shape)
        fig.write_image(ASSETS / f"{name}.svg")
    except Exception as e:
        print(f"  [svg skip] {name}: {e}")
    print(f"  {name}.png (+svg)")


# ---- Fig 6 clinical utility ---------------------------------------------------------------------
def clinical_a() -> None:
    d = pd.read_csv(TABLES / "decision_curve.csv")
    t = d["threshold"] * 100
    prev = float(d["treat_all"].iloc[0])
    fig = go.Figure()
    fig.add_vrect(x0=10, x1=30, fillcolor=C_PATCH, opacity=0.06, line_width=0)
    for name, y, c, wdt, dash in [
        ("PATCH (multimodal context)", d["PATCH_multimodal"], C_PATCH, 3.2, "solid"),
        ("resistance class only (standard)", d["resistance_class_only"], C_RES, 2.3, "solid"),
        ("intervene on all", d["treat_all"], C_GREY, 1.6, "dash"),
        ("intervene on none", d["treat_none"], C_DARK, 1.2, "dot")]:
        fig.add_trace(go.Scatter(x=t, y=y, name=name, mode="lines", line=dict(color=c, width=wdt, dash=dash)))
    fig.add_hline(y=0, line=dict(color="#B0B0B0", width=1, dash="dot"))
    fig.add_vline(x=20, line=dict(color=MUTED, width=1, dash="dash"),
                  annotation_text="yield threshold (20%)", annotation_position="top right",
                  annotation_font=dict(size=10, color=MUTED))
    op = d.iloc[(d["threshold"] * 100 - 20).abs().argmin()]
    fig.add_trace(go.Scatter(x=[20, 20], y=[op["PATCH_multimodal"], op["resistance_class_only"]],
                             mode="markers", showlegend=False,
                             marker=dict(size=8, color=[C_PATCH, C_RES], line=dict(color="white", width=1))))
    fig.update_xaxes(range=[2, 50], title_text="threshold risk for intensified monitoring (%)")
    fig.update_yaxes(range=[-0.01, prev + 0.014], title_text="net benefit")
    fig.update_layout(title=dict(text="Multimodal risk-targeting compared with resistance-class targeting",
                                 x=0.5, xanchor="center", font=dict(size=15)),
                      legend=dict(orientation="h", yanchor="bottom", y=-0.30, xanchor="center", x=0.5),
                      margin=dict(l=66, r=30, t=56, b=96))
    _save(fig, "fig06_a_decision_curve", 760, 560)


def clinical_b() -> None:
    yld = pd.read_csv(TABLES / "decision_curve_yield.csv").set_index("strategy")
    d = pd.read_csv(TABLES / "decision_curve.csv")
    auc_res, auc_patch = float(d["auroc_resistance"].iloc[0]), float(d["auroc_patch"].iloc[0])
    order = ["Monitor all", "Resistance class", "PATCH (multimodal)"]
    labels = ["monitor all", "resistance class", "PATCH"]
    colors = [C_DARK, C_RES, C_PATCH]
    tp = [int(yld.loc[s, "true_highrisk_per_1000"]) for s in order]
    fp = [int(yld.loc[s, "overmonitored_per_1000"]) for s in order]
    fig = go.Figure()
    fig.add_trace(go.Bar(y=labels, x=tp, orientation="h", name="true high-risk flagged (will die/fail)",
                         marker=dict(color=colors, line=dict(color="white", width=1)),
                         text=[str(v) for v in tp], textposition="inside", insidetextanchor="middle",
                         textfont=dict(color="white", size=13)))
    fig.add_trace(go.Bar(y=labels, x=fp, orientation="h", name="unnecessarily intensified",
                         marker=dict(color=[hex_to_rgba(c, 0.26) for c in colors],
                                     line=dict(color="white", width=1))))
    for lab, a, b in zip(labels, tp, fp, strict=True):
        fig.add_annotation(x=a + b + 14, y=lab, text=f"{a + b} flagged", showarrow=False,
                           font=dict(size=11, color=MUTED), xanchor="left")
    fig.update_xaxes(range=[0, 1090], title_text="per 1,000 patients (monitoring threshold 20%)")
    fig.update_yaxes(showgrid=False)
    fig.update_layout(
        barmode="stack",
        title=dict(text=f"True high-risk flagged — PATCH {tp[2]} of {tp[0]} vs {tp[1]} "
                        f"(AUROC {auc_patch:.2f} vs {auc_res:.2f})", x=0.5, xanchor="center",
                   font=dict(size=14)),
        legend=dict(orientation="h", yanchor="bottom", y=-0.32, xanchor="center", x=0.5),
        margin=dict(l=118, r=40, t=56, b=92))
    _save(fig, "fig06_b_yield", 700, 560)


# ---- Calibration --------------------------------------------------------------------------------
def _cal(kind: str, title: str, xt: str, yt: str, name: str, diag: bool, rng: list) -> None:
    c = pd.read_csv(TABLES / "calibration_auprc.csv")
    cols = {"PATCH": "#CC79A7", "context_LR": "#0072B2"}
    lab = {"PATCH": "PATCH", "context_LR": "context LR"}
    fig = go.Figure()
    if diag:
        fig.add_trace(go.Scatter(x=rng, y=rng, mode="lines", showlegend=False,
                                 line=dict(color="#BBBBBB", width=1, dash="dash")))
    for model, g in c[c["kind"] == kind].groupby("model"):
        g = g.sort_values("x")
        metric = f"Brier {g['brier'].iloc[0]:.3f}" if kind == "calibration" else f"AUPRC {g['auprc'].iloc[0]:.3f}"
        fig.add_trace(go.Scatter(x=g["x"], y=g["y"], mode="lines" + ("+markers" if kind == "calibration" else ""),
                                 name=f"{lab[model]} ({metric})", line=dict(color=cols[model], width=2.6),
                                 marker=dict(size=5)))
    fig.update_xaxes(title_text=xt, range=rng)
    fig.update_yaxes(title_text=yt, range=rng if kind == "calibration" else [0, 1])
    fig.update_layout(title=dict(text=title, x=0.5, xanchor="center", font=dict(size=15)),
                      legend=dict(orientation="h", yanchor="bottom", y=-0.28, xanchor="center", x=0.5),
                      margin=dict(l=64, r=28, t=56, b=84))
    _save(fig, name, 600, 540)


def calibration_a() -> None:
    _cal("calibration", "Calibration (reliability)", "predicted probability", "observed frequency",
         "figcal_a_calibration", True, [0, 0.6])


def calibration_b() -> None:
    _cal("pr", "Precision–recall", "recall", "precision", "figcal_b_pr", False, [0, 1])


# ---- Entropy sensitivity ------------------------------------------------------------------------
def entropy_a() -> None:
    prof = pd.read_csv(TABLES / "entropy_sensitivity_profiles.csv")
    lambdas = sorted(prof["lambda"].unique())
    fig = go.Figure()
    for i in range(5):
        sub = prof[prof["subtype"] == f"S{i}"].set_index("lambda").reindex(lambdas)
        fig.add_trace(go.Bar(x=[str(l) for l in lambdas], y=sub["frac%"], name=subtype_name(i),
                             marker=dict(color=subtype_color(i), line=dict(color="white", width=1))))
    fig.update_xaxes(title_text="entropy weight λ", type="category")
    fig.update_yaxes(title_text="population share (%)")
    fig.update_layout(barmode="stack",
                      title=dict(text="Archetype population share vs λ", x=0.5, xanchor="center",
                                 font=dict(size=15)),
                      legend=dict(orientation="h", yanchor="bottom", y=-0.34, xanchor="center", x=0.5),
                      margin=dict(l=60, r=24, t=56, b=110))
    _save(fig, "figentropy_a_balance", 680, 560)


def entropy_b() -> None:
    summ = pd.read_csv(TABLES / "entropy_sensitivity_summary.csv").sort_values("lambda")
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=summ["lambda"], y=summ["in_sample_auroc"], mode="lines+markers",
                             name="in-sample AUROC", line=dict(color="#0072B2", width=2.6),
                             marker=dict(size=7)), secondary_y=False)
    fig.add_trace(go.Scatter(x=summ["lambda"], y=summ["min_frac%"], mode="lines+markers",
                             name="smallest archetype share", line=dict(color="#D55E00", width=2.6, dash="dot"),
                             marker=dict(size=7)), secondary_y=True)
    fig.update_xaxes(title_text="entropy weight λ")
    fig.update_yaxes(title_text="in-sample AUROC", color="#0072B2", range=[0.5, 0.9], secondary_y=False)
    fig.update_yaxes(title_text="smallest archetype share (%)", color="#D55E00", range=[0, 25], secondary_y=True)
    fig.update_layout(title=dict(text="AUROC vs balance vs λ", x=0.5, xanchor="center", font=dict(size=15)),
                      legend=dict(orientation="h", yanchor="bottom", y=-0.30, xanchor="center", x=0.5),
                      margin=dict(l=60, r=60, t=56, b=92))
    _save(fig, "figentropy_b_auroc", 680, 560)


# ---- Fingerprints -------------------------------------------------------------------------------
def fingerprints_a() -> None:
    NAMES = [subtype_name(i) for i in NARRATIVE_ORDER]
    COLORS = [subtype_color(i) for i in NARRATIVE_ORDER]
    feats = [("sensitive%", "Drug-sens"), ("MDR%", "MDR"), ("XDR%", "XDR"), ("HIV%", "HIV+"),
             ("smear+%", "Smear+"), ("diabetes%", "Diabetes"), ("age", "Age"),
             ("Ukraine%", "Ukraine"), ("SSAfrica%", "SS-Africa")]
    df = pd.read_csv(TABLES / "multimodal_subtype_characterization.csv").set_index("subtype").reindex(
        [f"S{i}" for i in NARRATIVE_ORDER])
    unfav = df["unfavorable%"].to_numpy()
    raw = df[[c for c, _ in feats]].to_numpy(float)
    lo, hi = raw.min(0), raw.max(0)
    norm = (raw - lo) / np.where(hi - lo == 0, 1, hi - lo)
    theta = [lbl for _, lbl in feats] + [feats[0][1]]
    titles = [f"{nm}<br><span style='font-size:11px;color:{MUTED}'>{u:.1f}% unfav</span>"
              for nm, u in zip(NAMES, unfav, strict=True)]
    fig = make_subplots(rows=1, cols=5, specs=[[{"type": "polar"}] * 5], subplot_titles=titles,
                        horizontal_spacing=0.075)
    for k in range(5):
        r = list(norm[k]) + [norm[k][0]]
        fig.add_trace(go.Scatterpolar(r=r, theta=theta, fill="toself", showlegend=False,
                                      fillcolor=hex_to_rgba(COLORS[k], 0.30),
                                      line=dict(color=COLORS[k], width=2)), row=1, col=k + 1)
    for k in range(1, 6):
        pax = "polar" if k == 1 else f"polar{k}"
        fig.layout[pax].update(radialaxis=dict(range=[0, 1], showticklabels=False, ticks="",
                                               gridcolor="#E6E6E6"),
                               angularaxis=dict(tickfont=dict(size=8.5, color="#555555"),
                                                gridcolor="#ECECEC", rotation=90, direction="clockwise"),
                               bgcolor="white")
    for ann, c in zip(fig.layout.annotations, COLORS, strict=False):
        if ann.text.split("<")[0] in NAMES:
            ann.font.size, ann.font.color = 13, c
    fig.update_layout(margin=dict(l=46, r=46, t=56, b=10))
    _save(fig, "fig02_b_radars", 1180, 320)


def fingerprints_b() -> None:
    attr = pd.read_csv(TABLES / "multimodal_modality_attribution_ci.csv").sort_values(
        "total_share", ascending=False)
    mods = attr["modality"].tolist()
    tot = (attr["total_share"] * 100).to_numpy()
    pf = (attr["per_feature_share"] * 100).to_numpy()
    nfeat = attr["n_features"].to_numpy()
    labels = ["attribution"] + [m.replace("_", " ") for m in mods]
    parents = [""] + ["attribution"] * len(mods)
    values = [float(tot.sum())] + list(tot)
    colors = [0.0] + list(pf)
    text = [""] + [f"{m.replace('_', ' ')}<br>{t:.0f}% total · {p:.0f}%/feat<br>{int(n)} features"
                   for m, t, p, n in zip(mods, tot, pf, nfeat, strict=True)]
    fig = go.Figure(go.Treemap(
        labels=labels, parents=parents, values=values, branchvalues="total",
        marker=dict(colors=colors, colorscale="Cividis", cmin=0, cmax=float(pf.max()),
                    colorbar=dict(title=dict(text="% per<br>feature", font=dict(size=11)), len=0.9,
                                  thickness=14, tickfont=dict(size=10), outlinewidth=0),
                    line=dict(color="white", width=2)),
        text=text, textinfo="text", textfont=dict(size=13, color="white"), tiling=dict(pad=2)))
    fig.update_layout(margin=dict(l=6, r=70, t=56, b=6),
                      title=dict(text="Modality attribution — area = total share, colour = per-feature share",
                                 x=0.5, xanchor="center", font=dict(size=14)),
                      uniformtext=dict(minsize=9, mode="hide"))
    _save(fig, "fig02_c_treemap", 1180, 470)


# ---- Fig 3 panel d: outcome-AUROC tiering (treatment < context < PATCH, across embeddings) -------
def fig3_auroc_tiers() -> None:
    # AUROC of 2-D position -> died/failure, from figures.fig03_outcome_methods (reproducible run).
    methods = ["UMAP", "t-SNE", "PCA"]
    data = {"Treatment<br>(regimen)": [0.635, 0.644, 0.638],
            "Context<br>(raw features)": [0.703, 0.710, 0.678],
            "PATCH<br>(contextualized)": [0.812, 0.819, 0.761]}
    reps = list(data.keys())
    mcol = ["#9DBED6", "#4C86C6", "#0B4F8A"]
    fig = go.Figure()
    for mi, m in enumerate(methods):
        fig.add_trace(go.Bar(name=m, x=reps, y=[data[r][mi] for r in reps],
                             marker=dict(color=mcol[mi], line=dict(color="white", width=1)),
                             text=[f"{data[r][mi]:.2f}" for r in reps], textposition="outside",
                             textfont=dict(size=9)))
    fig.add_hline(y=0.5, line=dict(color="#C0392B", width=1, dash="dash"),
                  annotation_text="chance", annotation_position="bottom right",
                  annotation_font=dict(size=9, color="#C0392B"))
    fig.update_yaxes(range=[0.5, 0.87], title_text="outcome AUROC (died / failure)", dtick=0.1)
    fig.update_xaxes(tickfont=dict(size=11))
    fig.update_layout(barmode="group", bargap=0.28, bargroupgap=0.06,
                      title=dict(text="Position → outcome: treatment &lt; context &lt; PATCH (robust across embeddings)",
                                 x=0.5, xanchor="center", font=dict(size=13.5)),
                      legend=dict(orientation="h", yanchor="bottom", y=-0.16, xanchor="center", x=0.5),
                      margin=dict(l=64, r=24, t=56, b=70))
    _save(fig, "fig03_d_auroc_tiers", 760, 560)


# ---- Fig 4 panel c: post hoc literature triage of the 23 significant associations ---------------
# Hand-curated after the model fit and shipped with the code (reviewed, hash-pinned source data).
LIT_TABLE = Path(__file__).resolve().parent / "source_data" / "fig4_literature_validation.csv"
DRUG_FULL = {"R": "rifampicin", "H": "isoniazid", "E": "ethambutol", "Z": "pyrazinamide",
             "S": "streptomycin", "Cs": "cycloserine", "Eto": "ethionamide", "Km": "kanamycin",
             "Trd": "terizidone", "Ofx": "ofloxacin", "other_second_line": "other second-line",
             "ART": "antiretroviral therapy", "Pa": "pretomanid", "Cotrimoxazol": "cotrimoxazole"}
CATCOL = {"confirmed": "#00A651", "confounded": "#F2A900", "discordant": "#E8112D", "novel": "#7C3AED"}
CATNAME = {"confirmed": "Concordant background", "confounded": "Confounding by indication",
           "discordant": "Discordant as coded", "novel": "Novel hypothesis"}
SUB_SHORT = {"S0": "DS", "S3": "SR-HIV", "S1": "HB", "S2": "RR", "S4": "MDR"}


CAT_BORDER_W = {"confirmed": 8.0, "confounded": 5.5, "discordant": 5.5, "novel": 5.5}  # green boldest


def fig4_validation_heatmap() -> None:
    order, ds = _drug_order()
    eff = ds.pivot(index="subtype", columns="drug", values="drug_specific").reindex(SUB)[order]
    pval = ds.pivot(index="subtype", columns="drug", values="pval").reindex(SUB)[order]
    sig = (pval < 0.05).to_numpy()
    keep = [j for j in range(len(order)) if sig[:, j].any()]   # drop drug columns with no sig cell
    order = [order[j] for j in keep]
    eff = eff.iloc[:, keep]
    sig = sig[:, keep]
    Z = np.where(sig, eff.to_numpy(float), np.nan)
    vmax = float(np.nanmax(np.abs(Z)))
    drugs = [DRUG_LABELS.get(d, d) for d in order]
    tags = [f'<span style="color:{subtype_color(i)}"><b>S{i}</b></span>' for i in NARRATIVE_ORDER]
    cnt = pd.read_csv(LIT_TABLE)
    catmap = {(r.subtype_short, r.drug): r.category for r in cnt.itertuples()}
    counts = cnt["category"].value_counts()

    fig = go.Figure(go.Heatmap(
        z=Z, x=list(range(len(order))), y=list(range(5)), colorscale=DIVERGING, zmid=0, zmin=-vmax,
        zmax=vmax, xgap=4, ygap=4, hoverongaps=False,
        colorbar=dict(title=dict(text="Δβ<br>(− more favorable recorded association)", font=dict(size=9)), len=0.85,
                      thickness=11, tickfont=dict(size=9), outlinewidth=0)))
    for i, sc in enumerate(SUB):
        for j, dc in enumerate(order):
            if sig[i, j]:
                cat = catmap.get((SUB_SHORT[sc], DRUG_FULL.get(dc, dc)), "novel")
                fig.add_shape(type="rect", x0=j - 0.45, x1=j + 0.45, y0=i - 0.45, y1=i + 0.45,
                              line=dict(color=CATCOL[cat], width=CAT_BORDER_W[cat]), fillcolor="rgba(0,0,0,0)")
    for cat, name in CATNAME.items():
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=f"{name} ({int(counts.get(cat, 0))})",
                                 marker=dict(symbol="square", size=15, color=CATCOL[cat],
                                             line=dict(color="white", width=1))))
    fig.update_xaxes(tickvals=list(range(len(order))), ticktext=drugs, tickangle=-45,
                     tickfont=dict(size=11), showgrid=False, ticks="")
    fig.update_yaxes(tickvals=list(range(5)), ticktext=tags, autorange="reversed",
                     tickfont=dict(size=14), showgrid=False, ticks="")
    fig.update_layout(
        title=dict(text="Significant associations · fill = Δβ (blue more favorable / red less favorable) · "
                        "<b>border colour = evidence status</b>", x=0.5, xanchor="center", font=dict(size=13)),
        legend=dict(orientation="h", yanchor="bottom", y=-0.30, xanchor="center", x=0.5, font=dict(size=11)),
        margin=dict(l=44, r=64, t=52, b=104))
    _save(fig, "fig04_c_heatmap", 1120, 560)


def fig4_validation_lollipop() -> None:
    v = pd.read_csv(LIT_TABLE)
    order = ["confirmed", "confounded", "discordant", "novel"]
    v["cat"] = pd.Categorical(v["category"], categories=order, ordered=True)
    v = v.sort_values(["cat", "dbeta"]).reset_index(drop=True)
    x = list(range(len(v)))
    labels = [f"{d} · {s}" for d, s in zip(v["drug"], v["subtype_short"], strict=True)]
    counts = v["category"].value_counts()
    fig = go.Figure()
    fig.add_hline(y=0, line=dict(color="#B0B0B0", width=1))
    for cat in order:
        idx = [i for i in range(len(v)) if v["category"].iloc[i] == cat]
        for i in idx:
            fig.add_shape(type="line", x0=x[i], x1=x[i], y0=0, y1=v["dbeta"].iloc[i],
                          line=dict(color=CATCOL[cat], width=2))
        fig.add_trace(go.Scatter(x=[x[i] for i in idx], y=[v["dbeta"].iloc[i] for i in idx], mode="markers",
                                 name=f"{CATNAME[cat]} ({int(counts.get(cat, 0))})",
                                 marker=dict(size=15, color=CATCOL[cat], line=dict(color="white", width=1.5)),
                                 text=[labels[i] for i in idx],
                                 hovertemplate="%{text}<br>Δβ %{y:.2f}<extra></extra>"))
    fig.update_xaxes(tickvals=x, ticktext=labels, tickangle=-55, tickfont=dict(size=9), showgrid=False)
    fig.update_yaxes(title_text="drug-specific Δβ  (− more favorable recorded association · + less favorable)",
                     zeroline=False, gridcolor="#EEEEEE")
    fig.update_layout(
        title=dict(text="Same associations ranked — colour = evidence status", x=0.5, xanchor="center", font=dict(size=13)),
        legend=dict(orientation="h", yanchor="bottom", y=-0.52, xanchor="center", x=0.5, font=dict(size=11)),
        margin=dict(l=64, r=24, t=52, b=132))
    _save(fig, "fig04_c_lollipop", 1120, 560)


# ---- Fig 2 panel c: modality attribution (readable grouped bars) --------------------------------
def fig2_attribution() -> None:
    attr = pd.read_csv(TABLES / "multimodal_modality_attribution_ci.csv").sort_values("total_share")
    mods = [m.replace("_", " ") for m in attr["modality"]]
    tot = (attr["total_share"] * 100).to_numpy()
    pf = (attr["per_feature_share"] * 100).to_numpy()
    nf = attr["n_features"].to_numpy()
    cols = [MODALITY_COLORS.get(m, "#BFBFBF") for m in attr["modality"]]
    fig = go.Figure()
    fig.add_trace(go.Bar(y=mods, x=tot, orientation="h", offsetgroup=0, showlegend=False,
                         marker=dict(color=cols, line=dict(color="white", width=1)),
                         hovertemplate="%{y}: %{x:.1f}% total share<extra></extra>"))
    fig.add_trace(go.Bar(y=mods, x=pf, orientation="h", offsetgroup=1, showlegend=False,
                         marker=dict(color=[hex_to_rgba(c, 0.45) for c in cols],
                                     line=dict(color="white", width=1), pattern=dict(shape="/", size=4)),
                         hovertemplate="%{y}: %{x:.1f}% per-feature share<extra></extra>"))
    for m, t, p, n in zip(mods, tot, pf, nf, strict=True):
        fig.add_annotation(x=max(t, p) + 0.6, y=m, text=f"{int(n)} feat", showarrow=False, xanchor="left",
                           font=dict(size=9, color=MUTED))
    # neutral solid-vs-pattern legend (colour already encodes modality)
    fig.add_trace(go.Bar(x=[None], y=[None], name="total share", marker=dict(color="#8A8A8A")))
    fig.add_trace(go.Bar(x=[None], y=[None], name="per-feature share",
                         marker=dict(color=hex_to_rgba("#8A8A8A", 0.5), pattern=dict(shape="/"))))
    fig.update_xaxes(title_text="contribution (%)", range=[0, 42])
    fig.update_yaxes(tickfont=dict(size=11))
    fig.update_layout(barmode="group", bargap=0.3, bargroupgap=0.08,
                      title=dict(text="No single modality dominates: resistance is broad-but-thin, "
                                      "labs/region narrow-but-concentrated", x=0.5, xanchor="center",
                                 font=dict(size=13)),
                      legend=dict(orientation="h", yanchor="bottom", y=-0.27, xanchor="center", x=0.5),
                      margin=dict(l=120, r=70, t=58, b=96))
    _save(fig, "fig02_c_attribution", 780, 520)


# ---- Fig 2 panel a: unfavorable-rate bars -------------------------------------------------------
def fig2_rate() -> None:
    df = pd.read_csv(TABLES / "multimodal_subtype_characterization.csv").set_index("subtype").reindex(SUB)
    n, unfav = df["n"].to_numpy(), df["unfavorable%"].to_numpy()
    k = np.round(n * unfav / 100).astype(int)
    z = 1.96
    p = k / n
    den = 1 + z**2 / n
    ctr = (p + z**2 / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / den
    lo, hi = (ctr - half) * 100, (ctr + half) * 100
    fig = go.Figure(go.Bar(
        y=SNAMES, x=unfav, orientation="h", showlegend=False,
        marker=dict(color=SCOLORS, line=dict(color="white", width=1)),
        error_x=dict(type="data", symmetric=False, array=hi - unfav, arrayminus=unfav - lo,
                     color="#555555", thickness=1.2, width=5),
        text=[f"{u:.1f}%" if u < 10 else f"{u:.0f}%" for u in unfav], textposition="outside",
        textfont=dict(size=12, color=INK), cliponaxis=False))
    fig.update_xaxes(range=[0, 44], title_text="unfavorable outcome (%), Wilson 95% CI")
    fig.update_yaxes(autorange="reversed", tickfont=dict(size=11.5))
    fig.update_layout(title=dict(text="Unfavorable-outcome rate by subtype", x=0.5, xanchor="center",
                                 font=dict(size=15)), margin=dict(l=190, r=44, t=56, b=64))
    _save(fig, "fig02_a_rate", 660, 520)


# ---- Fig 4 panel a: context-conditioned β(C) interaction atlas (≤34 clustered contexts) ----------
def _drug_order():
    ds = pd.read_csv(TABLES / "multimodal_subtype_drugspecific.csv")
    eff = ds.pivot(index="subtype", columns="drug", values="drug_specific").reindex(SUB)
    return eff.abs().max(0).sort_values(ascending=False).index.tolist(), ds


def fig4_atlas() -> None:
    order, _ = _drug_order()
    at = pd.read_csv(TABLES / "heterogeneity_atlas_unfavorable_interaction.csv")
    eff = at.pivot(index="context", columns="drug", values="delta_beta").reindex(columns=order)
    sig = at.pivot(index="context", columns="drug", values="significant").reindex(columns=order).fillna(False)
    keep = sig.index[sig.any(axis=1)]
    if len(keep) > 34:
        keep = (eff.abs() * sig).sum(1).loc[keep].sort_values(ascending=False).head(34).index
    eff, sig = eff.loc[keep], sig.loc[keep]
    if len(eff) >= 3:
        ro = sch.leaves_list(sch.linkage(np.nan_to_num(eff.to_numpy(float)), method="ward"))
        eff, sig = eff.iloc[ro], sig.iloc[ro]
    M, S = eff.to_numpy(float), sig.to_numpy(bool)
    vmax = float(np.nanpercentile(np.abs(M[S]), 98)) if S.any() else 0.1
    ctx = [c.replace("_", " ").replace(">med", " > median") for c in eff.index]
    drugs = [DRUG_LABELS.get(d, d) for d in order]
    fig = go.Figure()
    fig.add_trace(go.Heatmap(z=np.where(S, 0.0, 0.5), x=drugs, y=ctx, colorscale=[[0, "#EDEDED"], [1, "#EDEDED"]],
                             showscale=False, hoverinfo="skip"))  # grey base for n.s.
    fig.add_trace(go.Heatmap(z=np.where(S, M, np.nan), x=drugs, y=ctx, colorscale=DIVERGING, zmid=0,
                             zmin=-vmax, zmax=vmax, xgap=0.5, ygap=0.5,
                             colorbar=dict(title=dict(text="drug-specific<br>Δβ", font=dict(size=10)),
                                           len=0.85, thickness=13, tickfont=dict(size=9), outlinewidth=0),
                             hovertemplate="%{y} · %{x}<br>Δβ %{z:.3f}<extra></extra>"))
    fig.update_xaxes(tickangle=-45, tickfont=dict(size=10), side="bottom", showgrid=False, ticks="")
    fig.update_yaxes(tickfont=dict(size=8.5), showgrid=False, ticks="", autorange="reversed")
    fig.update_layout(title=dict(text="Context-conditioned coefficient map β(C): context × drug "
                                      "(coloured = significant, grey = n.s.)", x=0.5, xanchor="center",
                                 font=dict(size=14)), margin=dict(l=250, r=80, t=56, b=86))
    _save(fig, "fig04_a_atlas", 900, 760)


# ---- Fig 4 panel b: archetypal coarsening — subtype × drug --------------------------------------
def fig4_subtype_drug() -> None:
    order, ds = _drug_order()
    eff = ds.pivot(index="subtype", columns="drug", values="drug_specific").reindex(SUB)[order]
    sig = ds.pivot(index="subtype", columns="drug", values="significant").reindex(SUB)[order]
    M, S = eff.to_numpy(float), sig.to_numpy(bool)
    vmax = max(0.05, float(np.nanmax(np.abs(M))))
    drugs = [DRUG_LABELS.get(d, d) for d in order]
    tags = [f'<span style="color:{c}"><b>{s}</b></span>' for c, s in zip(SCOLORS, SUB, strict=True)]
    fig = go.Figure(go.Heatmap(z=M, x=drugs, y=tags, colorscale=DIVERGING, zmid=0, zmin=-vmax, zmax=vmax,
                               xgap=1, ygap=1,
                               colorbar=dict(title=dict(text="drug-specific<br>Δβ", font=dict(size=10)),
                                             len=0.9, thickness=13, tickfont=dict(size=9), outlinewidth=0),
                               hovertemplate="%{y} · %{x}<br>Δβ %{z:.3f}<extra></extra>"))
    for i in range(5):
        for j in range(len(order)):
            if S[i, j]:
                fig.add_annotation(x=drugs[j], y=tags[i], text="✱", showarrow=False,
                                   font=dict(size=14, color="#111111"))
    fig.update_xaxes(tickangle=-45, tickfont=dict(size=10), showgrid=False, ticks="")
    fig.update_yaxes(tickfont=dict(size=12), showgrid=False, ticks="", autorange="reversed")
    fig.update_layout(title=dict(text="Archetypal coarsening — same drugs pooled into 5 subtypes "
                                      "(✱ = FDR q &lt; 0.05; rifampicin↔drug-sensitive)", x=0.5,
                                 xanchor="center", font=dict(size=13.5)), margin=dict(l=58, r=80, t=56, b=86))
    _save(fig, "fig04_b_subtype_drug", 1080, 360)


# ---- Fig 4 panel c: variance decomposition donut -----------------------------------------------
def fig4_variance() -> None:
    dec = pd.read_csv(TABLES / "heterogeneity_decomposition.csv").set_index("component")["variance_share"]
    grad, drug = float(dec["responsiveness_gradient"]), float(dec["drug_specific"])
    n = int(pd.read_csv(TABLES / "multimodal_subtype_characterization.csv")["n"].sum())
    fig = go.Figure(go.Pie(values=[drug, grad], labels=[f"drug-specific {drug*100:.0f}%",
                                                        f"responsiveness gradient {grad*100:.0f}%"],
                           hole=0.55, sort=False, direction="clockwise", rotation=0,
                           marker=dict(colors=["#4C566A", "#9ECAE1"], line=dict(color="white", width=2)),
                           textinfo="none", hovertemplate="%{label}<extra></extra>"))
    fig.add_annotation(text=f"variance<br>of β(C)<br><span style='font-size:11px;color:{MUTED}'>n={n:,}</span>",
                       x=0.5, y=0.5, showarrow=False, font=dict(size=14, color=INK))
    fig.update_layout(title=dict(text="Coefficient heterogeneity is large", x=0.5, xanchor="center",
                                 font=dict(size=15)),
                      legend=dict(orientation="h", yanchor="bottom", y=-0.12, xanchor="center", x=0.5),
                      margin=dict(l=20, r=20, t=56, b=60))
    _save(fig, "fig04_c_variance", 560, 540)


# ---- Fig 5 panel a: risk-transfer AUROC --------------------------------------------------------
def fig5_auroc() -> None:
    s = pd.read_csv(TABLES / "brazil_external_validation.csv").iloc[0]
    rows = [("Brazil (CV ceiling)", s["auc_brazil_ceiling"], "#2E3440"),
            ("→ Brazil (context LR)", s["auc_transfer_lr"], "#6C7787"),
            ("→ Brazil (PATCH)", s["auc_transfer_patch"], "#4C566A"),
            ("TB Portals (in-sample)", s["auc_tbp_insample"], "#B0B0B0")]
    fig = go.Figure(go.Bar(
        y=[r[0] for r in rows], x=[r[1] for r in rows], orientation="h", showlegend=False,
        marker=dict(color=[r[2] for r in rows], line=dict(color="white", width=1)),
        text=[f"{r[1]:.3f}" for r in rows], textposition="outside", textfont=dict(size=12), cliponaxis=False))
    fig.add_vline(x=0.5, line=dict(color="#999999", width=1, dash="dot"))
    fig.update_xaxes(range=[0.5, 0.83], title_text="AUROC (died / failure)")
    fig.update_yaxes(tickfont=dict(size=11.5))
    fig.update_layout(title=dict(text=f"Risk transfers to Brazil (n = {int(s['n_brazil']):,})", x=0.5,
                                 xanchor="center", font=dict(size=15)), margin=dict(l=180, r=70, t=56, b=60))
    _save(fig, "fig05_a_auroc", 640, 520)


def main() -> int:
    print("rendering pptx panel assets:")
    for fn in (clinical_a, clinical_b, calibration_a, calibration_b, entropy_a, entropy_b,
               fig2_rate, fingerprints_a, fingerprints_b, fig2_attribution, fig3_auroc_tiers,
               fig4_atlas, fig4_subtype_drug, fig4_variance, fig4_validation_heatmap,
               fig4_validation_lollipop, fig5_auroc):
        fn()
    print(f"wrote panels to {ASSETS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
