"""Figure 1, the PATCH hero, with clinician-readable labelled panels (vector SVG, 16:9 full-bleed).

Four labelled panels a clinician reads at a glance (rsvg-convert -> PNG/PDF):
  a  Conventional "one answer per group" — discrete buckets, borderline patient forced in, one shared
     prescription (population-average effect, heterogeneity discarded).
  b  PATCH "your own answer" — every patient on a continuous spectrum of a few reference response-types,
     each with a context-conditioned coefficient profile beta(C).
  c  How PATCH builds a context-conditioned model: episode data -> a mix of 5 reference types -> a coefficient profile
     -> your predicted risk. Plain language leads; math tokens are secondary.
  d  The benefits, in visual form: aligns with a known resistance pattern (positive control), flags what it can't identify
     (positivity), validates on another continent, and matches a black box while staying interpretable.

Drugs are drawn as intuitive coefficient profiles (green bar up = more favorable association, red bar down = less favorable). On-palette.
DUA-safe (illustrative, seeded — no row-level data).

Run: ``.venv/bin/python -m figures.fig01_hero``
"""

from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path

import numpy as np

from ctb.config import FIGURES_DIR

OUT = FIGURES_DIR
ASSETS = OUT / "plotly" / "pptx_assets"
DESKTOP = Path.home() / "Desktop" / "PATCH_plotly_NEW"

INK, MUTE, FAINT = "#1A1A1A", "#5B6167", "#AEB4B9"
LINEC, CARD = "#DFE2E5", "#FFFFFF"
FONT = "Helvetica, Arial, sans-serif"
HELP, HURT, WARN = "#12805B", "#C4362B", "#E07B39"
GREEN = "#009E73"
SUB = ["#009E73", "#0072B2", "#56B4E9", "#E69F00", "#D55E00"]
# Compact tags are used in the wide schematic. Full names live in figures.palette.SUBTYPES.
SUBNM = ["Drug-sensitive responders", "Mixed-spectrum resistance responders", "MDR-predominant responders",
         "High-burden / regional failure", "HIV-enriched severe-resistance failure"]

S: list[str] = []
def add(x): S.append(x)
def esc(s): return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def txt(x, y, s, size=12, w="normal", col=INK, anc="start", it=False):
    st = ' font-style="italic"' if it else ""
    add(f'<text x="{x:.1f}" y="{y:.1f}" font-family="{FONT}" font-size="{size}" font-weight="{w}" '
        f'fill="{col}" text-anchor="{anc}"{st} xml:space="preserve">{esc(s)}</text>')


def title_runs(x, y, parts, size=15):
    add(f'<text x="{x:.1f}" y="{y:.1f}" font-family="{FONT}" font-size="{size}" font-weight="bold" '
        f'xml:space="preserve">')
    for t, c in parts:
        add(f'<tspan fill="{c}" xml:space="preserve">{esc(t)}</tspan>')
    add('</text>')


def plet(x, y, ch):
    txt(x, y, ch, 25, "bold", INK)


def rrect(x, y, w, h, fill=CARD, stroke=LINEC, sw=1.0, rx=11, shadow=False):
    f = ' filter="url(#sh)"' if shadow else ""
    add(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" fill="{fill}" '
        f'stroke="{stroke}" stroke-width="{sw}"{f}/>')


def circle(cx, cy, r, fill="none", stroke="none", sw=1.0):
    add(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.1f}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')


def line(x1, y1, x2, y2, stroke=INK, w=1.0, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    add(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{stroke}" '
        f'stroke-width="{w}" stroke-linecap="round"{d}/>')


def rect(x, y, w, h, fill, rx=1.5, stroke="none", sw=0.0):
    add(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" fill="{fill}" '
        f'stroke="{stroke}" stroke-width="{sw}"/>')


def arrow(x1, y1, x2, y2, w=2.4, col="#9AA0A6"):
    line(x1, y1, x2, y2, col, w)
    add(f'<use href="#ah" x="{x2:.1f}" y="{y2:.1f}" fill="{col}"/>')


def profile(x, y, w, h, vals, mono=None):
    base = y + h / 2
    n = len(vals)
    bw, step = w / n * 0.66, w / n
    line(x, base, x + w, base, "#C9CED3", 1)
    for i, v in enumerate(vals):
        bx = x + i * step + (step - bw) / 2
        bh = abs(v) * (h / 2 - 2)
        col = mono if mono else (HELP if v >= 0 else HURT)
        rect(bx, base - bh if v >= 0 else base, bw, max(bh, 0.9), col, 1.4)


def person(cx, cy, s=1.0, col=MUTE):
    circle(cx, cy - 5 * s, 4 * s, col)
    add(f'<path d="M{cx-6*s:.1f},{cy+6*s:.1f} a{6*s:.1f},{6*s:.1f} 0 0 1 {12*s:.1f},0 z" fill="{col}"/>')


def icon(kind, cx, cy, col=MUTE):
    if kind == "hiv":
        circle(cx, cy, 4.6, "none", col, 1.4)
        for a in range(0, 360, 60):
            dx, dy = math.cos(math.radians(a)), math.sin(math.radians(a))
            line(cx + 4.6 * dx, cy + 4.6 * dy, cx + 6.6 * dx, cy + 6.6 * dy, col, 1.4)
            circle(cx + 7.2 * dx, cy + 7.2 * dy, 1.0, col)
    elif kind == "lungs":
        add(f'<path d="M{cx:.1f},{cy-6:.1f} v7 M{cx-1:.1f},{cy-1:.1f} c-5,0 -6,5 -6,9 c0,2 3,2 4,0 c1,-3 2,-6 2,-9 '
            f'M{cx+1:.1f},{cy-1:.1f} c5,0 6,5 6,9 c0,2 -3,2 -4,0 c-1,-3 -2,-6 -2,-9" '
            f'fill="none" stroke="{col}" stroke-width="1.4" stroke-linejoin="round"/>')
    elif kind == "drop":
        add(f'<path d="M{cx:.1f},{cy-7:.1f} c4,5 5,7 5,10 a5,5 0 0 1 -10,0 c0,-3 1,-5 5,-10 z" '
            f'fill="none" stroke="{col}" stroke-width="1.4"/>')
    elif kind == "dna":
        for k in range(-6, 7, 3):
            add(f'<path d="M{cx-5:.1f},{cy+k:.1f} q5,-3 10,0" fill="none" stroke="{col}" stroke-width="1.2"/>')
        line(cx - 5, cy - 6, cx - 5, cy + 6, col, 1.2)
        line(cx + 5, cy - 6, cx + 5, cy + 6, col, 1.2)


def check(cx, cy, r=6, col=HELP):
    circle(cx, cy, r, col)
    add(f'<path d="M{cx-r*0.45:.1f},{cy:.1f} l{r*0.3:.1f},{r*0.4:.1f} l{r*0.6:.1f},{-r*0.7:.1f}" '
        f'fill="none" stroke="#FFFFFF" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/>')


def donut(cx, cy, r, frac, col, label, sub):
    add(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.1f}" fill="none" stroke="#E7EAEC" stroke-width="9"/>')
    a0 = -math.pi / 2
    a1 = a0 + 2 * math.pi * frac
    large = 1 if frac > 0.5 else 0
    x0, y0 = cx + r * math.cos(a0), cy + r * math.sin(a0)
    x1, y1 = cx + r * math.cos(a1), cy + r * math.sin(a1)
    add(f'<path d="M{x0:.1f},{y0:.1f} A{r:.1f},{r:.1f} 0 {large} 1 {x1:.1f},{y1:.1f}" fill="none" '
        f'stroke="{col}" stroke-width="9" stroke-linecap="round"/>')
    txt(cx, cy + 2, label, 16, "bold", INK, "middle")
    txt(cx, cy + 17, sub, 8.5, "normal", MUTE, "middle")


def gauge(cx, cy, r, frac, label):
    add(f'<path d="M{cx-r:.1f},{cy:.1f} A{r:.1f},{r:.1f} 0 0 1 {cx+r:.1f},{cy:.1f}" fill="none" '
        f'stroke="#E3E6E9" stroke-width="8" stroke-linecap="round"/>')
    a = math.pi * (1 - frac)
    add(f'<path d="M{cx-r:.1f},{cy:.1f} A{r:.1f},{r:.1f} 0 0 1 {cx+r*math.cos(a):.1f},{cy-r*math.sin(a):.1f}" '
        f'fill="none" stroke="{HURT}" stroke-width="8" stroke-linecap="round"/>')
    txt(cx, cy - 4, label, 17, "bold", INK, "middle")


def minibars(x, y, w, h, vals, cols, hi=None):
    n = len(vals)
    bw, step = w / n * 0.62, w / n
    mx = max(vals)
    for i, v in enumerate(vals):
        bh = v / mx * h
        rect(x + i * step + (step - bw) / 2, y + h - bh, bw, bh, cols[i], 1)
        if hi == i:
            add(f'<rect x="{x+i*step+(step-bw)/2-1.5:.1f}" y="{y+h-bh-1.5:.1f}" width="{bw+3:.1f}" '
                f'height="{bh+3:.1f}" rx="1.5" fill="none" stroke="{INK}" stroke-width="1.2"/>')


def regplot(x0, y0, w, h, lines, pts, ptcols, title, sub, titlecol=INK):
    """Small x->y scatter with regression line(s). lines = list of (slope, intercept, colour, width)."""
    rrect(x0, y0, w, h, CARD, LINEC, 1, 8)
    ax0, ay0, aw, ah = x0 + 34, y0 + 22, w - 48, h - 74
    line(ax0, ay0, ax0, ay0 + ah, "#C9CED3", 1.2)
    line(ax0, ay0 + ah, ax0 + aw, ay0 + ah, "#C9CED3", 1.2)
    txt(ax0 - 6, ay0 - 6, "outcome", 8, "normal", MUTE)
    for (pu, pv), c in zip(pts, ptcols):
        circle(ax0 + pu * aw, ay0 + ah - pv * ah, 2.6, c)
    for (m, b0, c, wd) in lines:
        ya, yb = min(1, max(0, b0 + m * 0.04)), min(1, max(0, b0 + m * 0.96))
        line(ax0 + 0.04 * aw, ay0 + ah - ya * ah, ax0 + 0.96 * aw, ay0 + ah - yb * ah, c, wd)
    txt(ax0 + aw / 2, ay0 + ah + 11, "drug exposure →", 8, "normal", MUTE, "middle", it=True)
    txt(x0 + w / 2, y0 + h - 18, title, 10, "bold", titlecol, "middle")
    txt(x0 + w / 2, y0 + h - 6, sub, 8.6, "normal", MUTE, "middle", it=True)


def build() -> str:
    W, H = 1600, 900
    rng = np.random.default_rng(7)
    add(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">')
    add(f'<rect width="{W}" height="{H}" fill="#FFFFFF"/>')
    add('<defs>'
        '<marker id="ah" viewBox="0 0 10 10" refX="7.5" refY="5" markerWidth="7" markerHeight="7" '
        'orient="auto-start-reverse"><path d="M0,1.5 L8.5,5 L0,8.5 z"/></marker>'
        '<filter id="sh" x="-20%" y="-30%" width="140%" height="160%">'
        '<feDropShadow dx="0" dy="1.5" stdDeviation="3" flood-color="#101418" flood-opacity="0.10"/></filter>'
        '<linearGradient id="land" x1="0" y1="0" x2="1" y2="0.5">'
        '<stop offset="0" stop-color="#DFF1EA"/><stop offset="0.5" stop-color="#EAF0DE"/>'
        '<stop offset="1" stop-color="#FBE6CE"/></linearGradient></defs>')

    # ---------- title ----------
    title_runs(30, 46, [("PATCH, a context-conditioned coefficient map for ", INK), ("each patient", GREEN),
                        (", not a group", INK)], 25)
    txt(30, 76, "The clinical question is how recorded regimen associations vary with patient context. Conventional models answer it for a whole "
        "group. PATCH maps each episode onto a spectrum of a few reference response-types and remains usable when some tests are missing.",
        13.5, "normal", MUTE, it=True)

    # shared colour legend — the 5 archetypes (response-types), used in BOTH a and b
    txt(30, 100, "response-types (archetypes)", 11, "bold", "#3A3F44")
    for _k in range(5):
        _x = 254 + _k * 140
        circle(_x, 96, 6, SUB[_k])
        txt(_x + 11, 100, SUBNM[_k], 11, "normal", "#3A3F44")

    # ---------- shared plane: the same patients live on the same context plane in a and b ----------
    arU = [(0.17, 0.34), (0.5, 0.5), (0.82, 0.32), (0.33, 0.74), (0.7, 0.68)]   # 5 archetype anchors (u,v; v up)
    prng = np.random.default_rng(4)
    pats = []
    for _ in range(28):
        pk = int(prng.integers(0, 5))
        pu = min(0.93, max(0.07, arU[pk][0] + prng.normal(0, 0.11)))
        pv = min(0.93, max(0.07, arU[pk][1] + prng.normal(0, 0.11)))
        pats.append((pu, pv, pk))

    def px(lx, ly, lw, lh, u, v):
        left, right = lx + 58 * v, lx + lw - 58 * v
        return left + u * (right - left), (ly + lh) - v * lh

    def blendcol(u, v):
        ws = [math.exp(-((u - au) ** 2 + (v - av) ** 2) / 0.05) for au, av in arU]
        sw = sum(ws)
        r = sum(w / sw * int(c[1:3], 16) for w, c in zip(ws, SUB))
        g = sum(w / sw * int(c[3:5], 16) for w, c in zip(ws, SUB))
        b = sum(w / sw * int(c[5:7], 16) for w, c in zip(ws, SUB))
        return f"#{int(r):02X}{int(g):02X}{int(b):02X}"

    SECT = ["#E4A7B7", "#9FC6DA", "#C7B4E2", "#EAC79B", "#AFDBBE", "#F0B7A0",
            "#B7A6D6", "#D9CE86", "#8FBFB2", "#E7B3C9", "#A9C58E", "#CDB79A"]

    # ================= a — conventional (arbitrary sections; each patient a discrete category) =========
    aX, aY, aW, aH = 24, 116, 792, 292
    plet(aX + 2, aY + 20, "a")
    rrect(aX + 26, aY, aW - 26, aH, "#F7F8FA", LINEC, 1, 12, shadow=True)
    rrect(aX + 26, aY, aW - 26, 34, "#EAECEF", "none", 0, 12)
    rect(aX + 26, aY + 20, aW - 26, 14, "#EAECEF", 0)
    txt(aX + 46, aY + 22, "CONVENTIONAL", 13.5, "bold", "#565B60")
    txt(aX + 200, aY + 22, "the same plane, carved into arbitrary sections", 12.5, "normal", MUTE, it=True)
    lax, lay, law, lah = aX + 50, aY + 54, 404, 172
    nc, nr = 4, 3
    for j in range(nr):
        for i in range(nc):
            u0, u1, v0, v1 = i / nc, (i + 1) / nc, j / nr, (j + 1) / nr
            pcs = [px(lax, lay, law, lah, uu, vv) for uu, vv in [(u0, v0), (u1, v0), (u1, v1), (u0, v1)]]
            add(f'<polygon points="{" ".join(f"{x:.1f},{y:.1f}" for x, y in pcs)}" fill="{SECT[j*nc+i]}" '
                f'fill-opacity="0.42" stroke="#FFFFFF" stroke-width="1.4"/>')
    for (u, v, k) in pats:                                     # each patient = a discrete square, its section's colour
        i, j = min(nc - 1, int(u * nc)), min(nr - 1, int(v * nr))
        xx, yy = px(lax, lay, law, lah, u, v)
        rect(xx - 4.5, yy - 4.5, 9, 9, SECT[j * nc + i], 1.5, "#FFFFFF", 0.6)
    bxx, bxy = px(lax, lay, law, lah, 0.5, 0.5)               # borderline, straddling a section edge
    add(f'<rect x="{bxx-6:.1f}" y="{bxy-6:.1f}" width="12" height="12" rx="2" fill="none" stroke="{WARN}" stroke-width="2.2"/>')
    txt(bxx + 13, bxy - 6, "?", 15, "bold", WARN)
    txt(lax + 4, lay + 14, "sections are arbitrary, not the real structure", 9.5, "normal", MUTE, it=True)
    ra = np.random.default_rng(21)
    pa = [(float(x), float(min(0.94, max(0.06, 0.22 + 0.55 * x + ra.normal(0, 0.07)))))
          for x in ra.uniform(0.06, 0.94, 14)]
    regplot(aX + 486, aY + 52, 226, 152, [(0.55, 0.22, INK, 2.6)], pa, ["#9AA0A6"] * 14,
            "Conventional, one slope for all", "same recorded regimen coefficient β for every patient")
    txt(aX + 252, aY + 250, "each patient becomes one discrete category, its section’s colour",
        11, "bold", WARN, "middle")
    txt(aX + 252, aY + 272, "arbitrary sections · one answer each · the real spectrum is lost",
        11, "normal", MUTE, "middle", it=True)

    # ================= b — PATCH =================
    bX, bW = 832, 744
    plet(bX + 2, aY + 20, "b")
    rrect(bX + 26, aY, bW - 26, aH, "#F4FBF8", "#CDE9DF", 1, 12, shadow=True)
    rrect(bX + 26, aY, bW - 26, 34, GREEN, "none", 0, 12)
    rect(bX + 26, aY + 20, bW - 26, 14, GREEN, 0)
    txt(bX + 46, aY + 22, "PATCH", 13.5, "bold", "#FFFFFF")
    txt(bX + 130, aY + 22, "your own answer, a spectrum, not a box", 12.5, "normal", "#EAF7F1", it=True)
    lx, ly, lw, lh = bX + 46, aY + 54, 400, 172
    add(f'<path d="M{lx},{ly+lh} L{lx+lw},{ly+lh} L{lx+lw-58},{ly} L{lx+58},{ly} Z" fill="url(#land)" '
        f'stroke="#D8DEE2" stroke-width="1"/>')
    for (u, v, k) in pats:                                     # patients as blended-colour dots (soft membership)
        xx, yy = px(lx, ly, lw, lh, u, v)
        circle(xx, yy, 4.2, blendcol(u, v))
    for k, (au, av) in enumerate(arU):                        # the 5 anchors ARE the archetypes
        axp, ayp = px(lx, ly, lw, lh, au, av)
        circle(axp, ayp, 10, "none", SUB[k], 1.2)
        circle(axp, ayp, 6.5, SUB[k], "#FFFFFF", 1.8)
        txt(axp, ayp - 12, "A" + str(k + 1), 8.5, "bold", SUB[k], "middle")
    mbx, mby = px(lx, ly, lw, lh, 0.5, 0.5)                    # same borderline patient as a
    circle(mbx, mby, 6, blendcol(0.5, 0.5), WARN, 2.2)
    txt(lx + 6, ly + 14, "A1–A5 are archetypes, each patient is a blend", 10, "bold", "#0B6B4C")
    txt(lx + 6, ly + 28, "a blue dot is mostly blue, but tinted by its neighbours", 9.5, "normal", "#4B8A76", it=True)
    rb = np.random.default_rng(22)
    lines_b = [(0.12, 0.52, SUB[0]), (0.38, 0.4, SUB[1]), (0.58, 0.3, SUB[2]),
               (0.78, 0.2, SUB[3]), (0.95, 0.1, SUB[4])]
    pb, cb = [], []
    for (m, b0, c) in lines_b:
        for _ in range(3):
            x = float(rb.uniform(0.08, 0.92))
            pb.append((x, float(min(0.95, max(0.05, b0 + m * x + rb.normal(0, 0.035)))))); cb.append(c)
    regplot(bX + 482, aY + 52, 230, 130, [(m, b0, c, 2.2) for (m, b0, c) in lines_b], pb, cb,
            "PATCH, a context-conditioned slope β(C)", "the recorded regimen coefficient varies with context", titlecol="#0B6B4C")
    # call out one episode and its context-conditioned coefficient profile
    hu, hv = 0.8, 0.6
    hx, hy = px(lx, ly, lw, lh, hu, hv)
    circle(hx, hy, 7.5, "none", "#0B6B4C", 2)
    cx2, cy2 = bX + 482, aY + 188
    line(hx, hy, cx2, cy2 + 22, "#9CC8BA", 1.3, dash="3,3")
    rrect(cx2, cy2, 230, 46, CARD, "#CDE9DF", 1.2, 8)
    circle(cx2 + 13, cy2 + 14, 4.5, blendcol(hu, hv))
    txt(cx2 + 23, cy2 + 15, "this episode, its coefficient profile β(C)", 8.2, "bold", "#0B6B4C")
    profile(cx2 + 16, cy2 + 22, 150, 20, [0.9, 0.5, -0.4, 0.7, -0.6, 0.3, -0.2, 0.4])
    txt(cx2 + 176, cy2 + 34, "more / less favorable", 7.5, "normal", MUTE)
    txt(bX + 46, aY + 258, "each patient is a blend of archetypes → a personalized profile  β(C)", 12, "bold", "#0B6B4C")
    txt(bX + 46, aY + 276, "green = more favorable recorded association · red = less favorable · works even with missing tests",
        10.5, "normal", "#4B8A76", it=True)

    # ================= c — how PATCH builds YOUR model =================
    cY = 452
    plet(24, cY - 6, "c")
    txt(58, cY - 6, "How PATCH builds your personal model", 16, "bold", INK)
    txt(58, cY + 13, "one interpretable model, learned end to end, read left to right", 12, "normal", MUTE, it=True)
    sy = cY + 28
    # step 1 — your data
    rrect(24, sy, 250, 176, CARD, LINEC, 1, 12, shadow=True)
    txt(149, sy + 27, "1 · Your data", 13, "bold", INK, "middle")
    txt(149, sy + 44, "whatever we have, even if incomplete", 10, "normal", MUTE, "middle", it=True)
    for i, (ic, lab) in enumerate([("hiv", "HIV status"), ("drop", "labs & comorbidities"),
                                   ("lungs", "imaging & burden"), ("dna", "resistance / DST")]):
        yy = sy + 74 + i * 25
        icon(ic, 52, yy, "#7A8085")
        txt(72, yy + 4, lab, 11, "normal", "#3A3F44")
    arrow(278, sy + 88, 314, sy + 88)
    # step 2 — your mix
    rrect(322, sy, 430, 176, CARD, LINEC, 1, 12, shadow=True)
    txt(537, sy + 27, "2 · Your mix of 5 reference response-types", 13, "bold", INK, "middle")
    txt(537, sy + 44, "how much you resemble each pattern", 10, "normal", MUTE, "middle", it=True)
    mix = [0.58, 0.06, 0.05, 0.24, 0.07]
    for k in range(5):
        cxk = 362 + k * 80
        rrect(cxk - 30, sy + 58, 60, 46, "#FCFCFD", SUB[k], 1.4, 7)
        profile(cxk - 24, sy + 64, 48, 34, list(np.random.default_rng(30 + k).uniform(-1, 1, 5)))
        txt(cxk, sy + 118, SUBNM[k], 8.4, "bold", SUB[k], "middle")
        bh = 30 * mix[k] / max(mix)
        add(f'<rect x="{cxk-10:.1f}" y="{sy+160-bh:.1f}" width="20" height="{bh:.1f}" rx="2" fill="{SUB[k]}" '
            f'opacity="{0.4+0.6*mix[k]/max(mix):.2f}"/>')
        txt(cxk, sy + 172, f"{int(mix[k]*100)}%", 9, "bold", "#3A3F44", "middle")
    txt(560, sy + 140, "your mix  Z(C)", 9.5, "normal", MUTE, "middle", it=True)
    arrow(756, sy + 88, 792, sy + 88)
    # step 3 — your profile
    rrect(800, sy, 290, 176, "#F4FBF8", "#CDE9DF", 1.2, 12, shadow=True)
    txt(945, sy + 27, "3 · Your coefficient profile", 13, "bold", "#0B6B4C", "middle")
    txt(945, sy + 44, "where recorded regimen associations vary with context", 10, "normal", "#4B8A76", "middle", it=True)
    profile(824, sy + 62, 242, 66, [0.9, 0.6, 0.7, -0.4, 0.3, -0.6, 0.2, -0.3, 0.5, -0.2])
    txt(828, sy + 148, "more favorable", 10.5, "bold", HELP)
    txt(1066, sy + 148, "less favorable", 10.5, "bold", HURT, "end")
    txt(945, sy + 168, "β(C) = A · Z(C), a blend of the 5 patterns", 9.5, "normal", MUTE, "middle", it=True)
    arrow(1094, sy + 88, 1130, sy + 88)
    # step 4 — your risk
    rrect(1138, sy, 438, 176, CARD, LINEC, 1, 12, shadow=True)
    txt(1357, sy + 27, "4 · Your predicted risk of treatment failure", 13, "bold", INK, "middle")
    gauge(1230, sy + 118, 60, 0.34, "34%")
    txt(1230, sy + 150, "for your regimen", 10, "normal", "#3A3F44", "middle")
    txt(1420, sy + 74, "…and the honest part.", 11, "bold", INK, "middle")
    txt(1420, sy + 96, "the model also flags which of", 10, "normal", MUTE, "middle")
    txt(1420, sy + 111, "these effects the data can’t yet", 10, "normal", MUTE, "middle")
    txt(1420, sy + 126, "prove, so they route to a trial,", 10, "normal", MUTE, "middle")
    txt(1420, sy + 141, "not to your prescription.", 10, "normal", MUTE, "middle")

    # ================= d — the benefits, visually =================
    dY = 690
    plet(24, dY, "d")
    txt(58, dY, "Why this matters, the benefits in one picture", 16, "bold", INK)
    ty, th = dY + 16, 178
    tw = (1552 - 3 * 16) / 4
    tx = [24 + i * (tw + 16) for i in range(4)]
    for x in tx:
        rrect(x, ty, tw, th, CARD, LINEC, 1, 12, shadow=True)
    # d1 — aligns with a known resistance pattern
    donut(tx[0] + 78, ty + 78, 34, 10 / 23, GREEN, "10/23", "confirmed")
    for i, d in enumerate(["rifampicin", "isoniazid", "fluoroquinolone"]):
        check(tx[0] + 148, ty + 52 + i * 24)
        txt(tx[0] + 160, ty + 56 + i * 24, d, 10, "normal", "#3A3F44")
    txt(tx[0] + tw / 2, ty + 150, "Aligns with known drug biology, as a positive control", 10.5, "bold", INK, "middle")
    txt(tx[0] + tw / 2, ty + 166, "(a built-in positive control)", 9.5, "normal", MUTE, "middle", it=True)
    # d2 — flags what it can't identify (positivity grid)
    g0x, g0y, cs, cg = tx[1] + 28, ty + 24, 24, 3
    for idx in range(130):
        r, c = divmod(idx, 13)
        col = HELP if idx == 0 else ("#BFE3D6" if idx <= 60 else "#D6DADE")
        rect(g0x + c * (cs + cg), g0y + r * (11 + 2), cs, 11, col, 1)
    txt(tx[1] + tw / 2, ty + 150, "Flags what it can’t prove, 69/130 → trials", 10.5, "bold", INK, "middle")
    txt(tx[1] + tw / 2, ty + 166, "green = identified · grey = not yet answerable", 9.5, "normal", MUTE, "middle", it=True)
    # d3 — validates on another continent
    for gi, (gx0, nm) in enumerate([(tx[2] + 40, "TB Portals"), (tx[2] + 200, "Brazil")]):
        minibars(gx0, ty + 30, 120, 84, [0.28, 0.29, 0.34, 0.62, 0.66], SUB, hi=4)
        txt(gx0 + 60, ty + 130, nm, 10, "bold", "#3A3F44", "middle")
    txt(tx[2] + tw / 2, ty + 150, "Validates on another continent", 10.5, "bold", INK, "middle")
    txt(tx[2] + tw / 2, ty + 166, "HIV subtype highest-risk in both · 298,900 patients", 9.5, "normal", MUTE, "middle", it=True)
    # d4 — accurate AND interpretable
    for bi, (lab, val, col) in enumerate([("PATCH", 0.79, GREEN), ("black box", 0.795, "#9AA0A6")]):
        bx = tx[3] + 70 + bi * 120
        bh = (val - 0.5) / 0.35 * 96
        rect(bx, ty + 24 + (96 - bh), 56, bh, col, 2)
        txt(bx + 28, ty + 24 + (96 - bh) - 6, f"{val:.2f}", 11, "bold", INK, "middle")
        txt(bx + 28, ty + 134, lab, 10, "normal", "#3A3F44", "middle")
    txt(tx[3] + tw / 2, ty + 150, "As accurate as a black box, but readable", 10.5, "bold", INK, "middle")
    txt(tx[3] + tw / 2, ty + 166, "AUROC ≈ 0.79, statistically tied", 9.5, "normal", MUTE, "middle", it=True)

    add('</svg>')
    return "\n".join(S)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    ASSETS.mkdir(parents=True, exist_ok=True)
    svg = OUT / "fig01_hero.svg"
    svg.write_text(build())
    (ASSETS / "fig01_hero.svg").write_text(svg.read_text())     # sibling of the PNG for SVG-embed in the deck
    png = ASSETS / "fig01_hero.png"
    subprocess.run(["rsvg-convert", "-w", "2600", str(svg), "-o", str(png)], check=True)
    subprocess.run(["rsvg-convert", "-f", "pdf", str(svg), "-o", str(OUT / "fig01_hero.pdf")], check=True)
    try:
        DESKTOP.mkdir(parents=True, exist_ok=True)
        subprocess.run(["cp", str(png), str(DESKTOP / "fig01_hero.png")], check=False)
    except OSError:
        pass
    print(f"wrote {svg.name}, {png.relative_to(OUT.parents[1])}, fig01_hero.pdf")
    return 0


if __name__ == "__main__":
    sys.exit(main())
