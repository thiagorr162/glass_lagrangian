"""3D explorer: the spectral map seen through three eigenfunctions at a time, as a grid of small 3D panels (one per
triple phi_i x phi_j x phi_k of phi1..phi5, like the pairs of app.py), with the same candidate generation and tables.

- Panels: all 10 triples by default; the "panels" checklist shows fewer (each 3D panel is its own WebGL context;
  browsers allow ~16 per page, so 10 is fine but fewer panels redraw faster). Drag to rotate, scroll to zoom.
- Selection: plotly has no lasso in 3D, so the region is a BOX in phi1..phi5 (standardized), one range slider per
  eigenfunction, shared by all panels: the box is the intersection of the five ranges (sliders left at full range
  do not constrain). Clicking any point in a panel moves the box there along that panel's three axes (keeping its
  size; an unconstrained axis starts at 2 * DEFAULT_HALF wide). "Reset box" selects everything. Glasses outside
  the box are drawn gray (a fixed random 15k of them, enough to show the shape of the cloud).
- Generate candidates: parents = real glasses inside the box; mixtures exactly as in app.py. A mixture is kept if
  Nystrom places it inside the box AND inside the 3D convex hull of the parents in every shown panel that involves
  a constrained axis (translucent blue surfaces; the 3D analogue of the dashed hulls of app.py). Emptiness /
  enclosure are measured in phi1..phi5.
- Tables, filters, Export CSV and typed compositions: as in app.py.

The figure is built once (and again when the panels change); every other change is sent as a Patch of only what
changed: ~58k points x 10 panels is too much to resend on every slider move.

Works with either embedding: --tag m10000_s1 (composition only) or m10000_s1_ri1 (RI-weighted kernel,
spectral_embedding_ri.py).

Run from the project root (needs outputs of spectral_embedding*.py and train_ri_model.py):
  uv run src/app_3d.py [--tag m10000_s1] [--port 8051]
then open http://127.0.0.1:8051
"""

import argparse
import sys
from datetime import datetime
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, Patch, State, ctx, dash_table, dcc, html, no_update
from _plotly_utils.utils import to_typed_array_spec
from plotly.subplots import make_subplots
from scipy.spatial import ConvexHull, Delaunay

sys.path.insert(0, str(Path("src")))
from app import (CELL, COLOR_BY, FLAG_STYLE, HOLE_EMPTINESS, HOLE_ENCLOSURE, N_PHI, PHIS, QUERY_COLOR,  # noqa: E402
                 QUERY_COLS, ROUND, STATUS_COLOR, TABLE_COLS, UNSEL_COLOR, Context, filtered, legend_dot,
                 parse_composition)
from check_candidates import STATUSES  # noqa: E402
from spectral_embedding import DATA_OUT, ensure_dirs  # noqa: E402

TRIPLES = list(combinations(range(1, N_PHI + 1), 3))
BASE, SEL, BOX, HULL, GEN_T, QUERY_T = range(6)  # trace order in each panel
N_TRACES = 6
N_COLS = 5
PANEL_HEIGHT = 330  # px per row of panels
N_BASE = 15_000  # gray points drawn for the glasses outside the box
DEFAULT_HALF = 0.5  # half-width (standardized phi units) of an axis a click starts constraining
GEN, HULLS, QUERY = {}, {}, {}  # last candidates / their hulls per panel / typed compositions (local single-user app)


def tkey(t) -> str:
    return "-".join(map(str, t))


def tlabel(t) -> str:
    return "φ{} × φ{} × φ{}".format(*t)


def shown_panels(keys) -> list[tuple]:
    keys = set(keys or [])
    return [t for t in TRIPLES if tkey(t) in keys]


def scene_name(n: int) -> str:
    return "scene" if n == 0 else f"scene{n + 1}"


# ---------------------------------------------------------------------------- box and hull geometry
def in_box(Z: np.ndarray, box, axes=range(1, N_PHI + 1)) -> np.ndarray:
    """box = [[lo, hi]] for phi1..phi5; only the (1-based) `axes` are tested."""
    m = np.ones(len(Z), bool)
    for k in axes:
        lo, hi = box[k - 1]
        m &= (Z[:, k - 1] >= lo) & (Z[:, k - 1] <= hi)
    return m


def padded(box):
    return [[lo - p, hi + p] for lo, hi in box for p in [0.08 * (hi - lo) if hi > lo else 0.1]]


class Hull3D:
    """Convex hull of 3D points: surface triangles for drawing, and an inside test."""

    def __init__(self, P: np.ndarray):
        self.vertices = P[ConvexHull(P).vertices]
        self.simplices = ConvexHull(self.vertices).simplices
        self.tri = Delaunay(self.vertices)

    def contains(self, Q: np.ndarray) -> np.ndarray:
        return self.tri.find_simplex(Q) >= 0


def wireframe(b3) -> dict:
    (x0, x1), (y0, y1), (z0, z1) = b3
    xs, ys, zs = [], [], []
    for a in (y0, y1):
        for b in (z0, z1):
            xs += [x0, x1, None]; ys += [a, a, None]; zs += [b, b, None]  # noqa: E702
    for a in (x0, x1):
        for b in (z0, z1):
            xs += [a, a, None]; ys += [y0, y1, None]; zs += [b, b, None]  # noqa: E702
    for a in (x0, x1):
        for b in (y0, y1):
            xs += [a, a, None]; ys += [b, b, None]; zs += [z0, z1, None]  # noqa: E702
    return dict(x=xs, y=ys, z=zs)


# ---------------------------------------------------------------------------- context
class Context3D(Context):
    def __init__(self, args):
        super().__init__(args)
        self.lo = np.floor(self.R.min(axis=0) * 10) / 10  # slider limits per phi
        self.hi = np.ceil(self.R.max(axis=0) * 10) / 10
        self.R32 = self.R.astype(np.float32)
        self.ri32 = self.ri.astype(np.float32)
        self.base_idx = np.sort(np.random.default_rng(0).choice(len(self.R), min(N_BASE, len(self.R)), replace=False))

    def full_box(self) -> list[list[float]]:
        return [[float(lo), float(hi)] for lo, hi in zip(self.lo, self.hi)]

    def constrained(self, box) -> list[int]:
        """1-based phi indices along which the box is narrower than everything."""
        return [k for k, ((lo, hi), (flo, fhi)) in enumerate(zip(box, self.full_box()), start=1)
                if lo > flo + 1e-9 or hi < fhi - 1e-9]

    def generate_box(self, box, panels, n: int, seed: int = 0):
        """Candidates for the box (see module docstring). Returns (DataFrame, C, Z, info, hulls)."""
        rng = np.random.default_rng(seed)
        pool = np.flatnonzero(in_box(self.R, box))
        if len(pool) < 2:
            return None, None, None, f"only {len(pool)} real glasses in the box — move or enlarge it", {}
        C, i, p2, p3, W = self.mixtures(pool, n, rng)
        info = (f"{len(pool)} parent glasses in the box · {n} mixtures generated · "
                f"{len(C)} plausible (d_nn ≤ {100 * self.args.dnn_max:.1f} mol%)")
        if len(C) == 0:
            return None, None, None, info, {}
        Z = self.embed(C)
        keep = in_box(Z, box)
        hulls = {}
        con = set(self.constrained(box))
        for t in panels:
            if not con & set(t):
                continue
            cols = [k - 1 for k in t]
            try:
                hulls[tkey(t)] = Hull3D(self.R[pool][:, cols])
            except Exception:  # QhullError: too few / coplanar parents -> no hull in this panel
                continue
            keep &= hulls[tkey(t)].contains(Z[:, cols])
        info += f" · {int(keep.sum())} land inside the box ∩ the 3D hulls of the parents ({len(hulls)} panel(s))"
        if not keep.any():
            return None, None, None, info + " — try a larger box or more mixtures", hulls
        C, Z = C[keep], Z[keep]
        return self.candidates(pool, C, Z, i[keep], p2[keep], p3[keep], W[keep]), C, Z, info, hulls


def color_values(K: Context, key: str):
    if key.startswith("ox:"):  # a single oxide, mol%
        values = 100 * K.X[:, list(K.oxides).index(key[3:])]
    else:
        values = {"ri": K.ri, "formers": K.formers, "intermediates": K.intermediates}[key]
    lo, hi = np.percentile(values, [1, 99]) if key == "ri" else (0.0, float(values.max()))
    return np.clip(values, lo, hi).astype(np.float32), float(lo), float(hi)


# ---------------------------------------------------------------------------- figure
STATUS_CODE = {s: k for k, s in enumerate(STATUSES)}  # candidate color = status, sent as a small int
STATUS_SCALE = [[k / max(len(STATUSES) - 1, 1), STATUS_COLOR[s]] for k, s in enumerate(STATUSES)]
GEN_HOVER = ("candidate · RI≈%{customdata[0]:.3f} · emptiness %{customdata[1]:.1f} · "
             "enclosure %{customdata[2]:.2f} · d_nn %{customdata[3]:.1f} mol%<extra></extra>")


def updates(K: Context3D, panels, st: dict, coords: bool = True, marker: bool = True, extras: bool = True):
    """What depends on the state st (box, color_key, show_gen, zoom): [(trace index, props)], [(scene, axis, props)].
    coords / marker: include the (heavy) coordinates / colors of the glasses inside the box.
    extras: include hulls, candidates and typed compositions (only needed when those, or the zoom, changed)."""
    box = st["box"]
    active = bool(K.constrained(box))
    zoom = st["zoom"] and active
    view = padded(box)
    sel = in_box(K.R, box)
    if marker:
        values, cmin, cmax = color_values(K, st["color_key"])
    traces, scenes = [], []
    for n, t in enumerate(panels):
        cols = [k - 1 for k in t]
        i0 = N_TRACES * n
        traces.append((i0 + BASE, dict(visible=active and not zoom)))
        if coords:
            R = K.R32[sel][:, cols]
            traces.append((i0 + SEL, dict(x=R[:, 0], y=R[:, 1], z=R[:, 2], customdata=K.ri32[sel])))
        if marker:
            traces.append((i0 + SEL, dict(marker=dict(
                size=2, color=values[sel], colorscale="Viridis", cmin=cmin, cmax=cmax, opacity=0.9,
                showscale=(n == 0), colorbar=dict(title=dict(text=COLOR_BY[st["color_key"]]), thickness=10, len=0.5,
                                                  y=1, yanchor="top")))))
        traces.append((i0 + BOX, wireframe([box[k] for k in cols]) if active else dict(x=[], y=[], z=[])))
        for a, k in zip(("xaxis", "yaxis", "zaxis"), t):
            scenes.append((scene_name(n), a, dict(range=view[k - 1], autorange=False) if zoom else dict(autorange=True)))
        if not extras:
            continue
        h = HULLS.get(tkey(t))
        traces.append((i0 + HULL, dict(x=h.vertices[:, 0], y=h.vertices[:, 1], z=h.vertices[:, 2],
                                       i=h.simplices[:, 0], j=h.simplices[:, 1], k=h.simplices[:, 2])
                       if h is not None else dict(x=[], y=[], z=[], i=[], j=[], k=[])))
        for which, store, on in ((GEN_T, GEN, st["show_gen"] and "df" in GEN), (QUERY_T, QUERY, bool(QUERY))):
            if not on:
                traces.append((i0 + which, dict(x=[], y=[], z=[], customdata=[], hovertext=[], text=[],
                                                marker=dict(color=[]))))
                continue
            m = in_box(store["Z"], view, t) if zoom else np.ones(len(store["Z"]), bool)
            Z = store["Z"][m][:, cols].astype(np.float32)
            props = dict(x=Z[:, 0], y=Z[:, 1], z=Z[:, 2])
            if which == GEN_T:  # numbers + status codes: ~10x lighter than per-point hover strings in 10 panels
                props["customdata"] = store["custom"][m]
                props["marker"] = dict(size=3, symbol="diamond", color=store["codes"][m], colorscale=STATUS_SCALE,
                                       cmin=0, cmax=len(STATUSES) - 1)
            else:
                props["text"] = list(np.array(store["labels"], dtype=object)[m])
                props["hovertext"] = list(np.array(store["hover"], dtype=object)[m])
                props["marker"] = dict(size=6, color=QUERY_COLOR, line=dict(width=1, color="white"))
            traces.append((i0 + which, props))
    return traces, scenes


def full_figure(K: Context3D, panels, st: dict) -> go.Figure:
    cols = min(N_COLS, len(panels))
    rows = int(np.ceil(len(panels) / cols))
    fig = make_subplots(rows=rows, cols=cols, specs=[[{"type": "scene"}] * cols for _ in range(rows)],
                        subplot_titles=[tlabel(t) for t in panels], horizontal_spacing=0.005, vertical_spacing=0.03)
    b = K.base_idx
    for n, t in enumerate(panels):
        r, c = divmod(n, cols)
        kw = dict(row=r + 1, col=c + 1)
        Rb = K.R32[b][:, [k - 1 for k in t]]
        fig.add_trace(go.Scatter3d(x=Rb[:, 0], y=Rb[:, 1], z=Rb[:, 2], mode="markers", name="outside box",
                                   marker=dict(size=1.2, color=UNSEL_COLOR, opacity=0.12),
                                   hovertemplate="outside the box<extra></extra>"), **kw)
        fig.add_trace(go.Scatter3d(mode="markers", name="real",
                                   hovertemplate="real · RI=%{customdata:.4f}<extra></extra>"), **kw)
        fig.add_trace(go.Scatter3d(mode="lines", line=dict(color="#222", width=2), hoverinfo="skip"), **kw)
        fig.add_trace(go.Mesh3d(color="#1f77b4", opacity=0.12, hoverinfo="skip", name="hull of parents"), **kw)
        fig.add_trace(go.Scatter3d(mode="markers", name="generated", hovertemplate=GEN_HOVER), **kw)
        fig.add_trace(go.Scatter3d(mode="markers+text", name="query", textfont=dict(color=QUERY_COLOR, size=12),
                                   hovertemplate="%{hovertext}<extra></extra>"), **kw)
        axis = dict(title=dict(font=dict(size=11)), tickfont=dict(size=8))
        fig.layout[scene_name(n)].update(aspectmode="cube", **{f"{a}axis": dict(axis, title_text=f"φ{k}")
                                                               for a, k in zip("xyz", t)})
    traces, scenes = updates(K, panels, st)
    for i, props in traces:
        fig.data[i].update(props)
    for s, a, props in scenes:
        fig.layout[s][a].update(props)
    fig.update_annotations(font_size=12)
    # uirevision: keep every camera through Patches; a new panel set starts fresh
    fig.update_layout(showlegend=False, template="plotly_white", height=PANEL_HEIGHT * rows,
                      margin=dict(l=0, r=0, t=22, b=0), uirevision="3d-" + "_".join(map(tkey, panels)))
    return fig


def typed(v):
    """numpy arrays -> plotly.js typed-array spec (base64), recursively. A figure gets this from plotly, but a Patch
    would send plain JSON lists (~3x larger, and slower to parse)."""
    if isinstance(v, np.ndarray) and v.dtype != object:
        return to_typed_array_spec(np.ascontiguousarray(v))
    if isinstance(v, dict):
        return {k: typed(x) for k, x in v.items()}
    return v


def figure_patch(K: Context3D, panels, st: dict, coords: bool, marker: bool, extras: bool) -> Patch:
    patch = Patch()
    traces, scenes = updates(K, panels, st, coords=coords, marker=marker, extras=extras)
    for i, props in traces:
        for k, v in props.items():
            patch["data"][i][k] = typed(v)
    for s, a, props in scenes:
        for k, v in props.items():
            patch["layout"][s][a][k] = v
    return patch


def box_after_click(K: Context3D, box, t, point) -> list[list[float]]:
    """Move the box to the clicked point along the panel's axes t, keeping its size (DEFAULT_HALF if unconstrained)."""
    full, con = K.full_box(), set(K.constrained(box))
    box = [list(b) for b in box]
    for k, c in zip(t, (point["x"], point["y"], point["z"])):
        lo, hi = box[k - 1]
        half = (hi - lo) / 2 if k in con else DEFAULT_HALF
        box[k - 1] = [round(max(full[k - 1][0], c - half), 2), round(min(full[k - 1][1], c + half), 2)]
    return box


# ---------------------------------------------------------------------------- app
def build_app(K: Context3D) -> Dash:
    args = K.args
    full = K.full_box()
    kernel = "RI-weighted kernel" if K.ri_weighted else "composition kernel"
    app = Dash(__name__, title="Glass spectral explorer 3D")
    sliders = [html.Div(style={"display": "flex", "alignItems": "center", "gap": "6px"}, children=[
        html.Span(f"φ{k}", style={"width": "24px", "fontWeight": "600", "fontSize": "13px"}),
        html.Div(dcc.RangeSlider(id=f"slider-{k}", min=full[k - 1][0], max=full[k - 1][1], value=full[k - 1],
                                 step=0.01, marks=None, updatemode="mouseup", allowCross=False,
                                 tooltip={"placement": "bottom", "always_visible": True}),
                 style={"flex": "1"})]) for k in range(1, N_PHI + 1)]

    app.layout = html.Div(style={"fontFamily": "system-ui, sans-serif", "padding": "8px 16px"}, children=[
        html.Div(style={"display": "flex", "alignItems": "center", "gap": "14px", "flexWrap": "wrap"}, children=[
            html.H3(f"Glass spectral explorer 3D · {args.tag} ({kernel})", style={"margin": "4px 0"}),
            html.Button("Reset box", id="reset-box"),
            html.Button("Generate candidates", id="gen-btn",
                        style={"fontWeight": "600", "background": "#1f77b4", "color": "white",
                               "border": "none", "padding": "4px 10px", "borderRadius": "4px"}),
            html.Span(["mixtures: ", dcc.Input(id="gen-n", type="number", value=args.gen_n, min=1000, step=1000,
                                               style={"width": "80px"})], style={"fontSize": "13px"}),
            html.Button("Export CSV", id="export"),
            html.Span(["color by: ", dcc.Dropdown(id="color-by", value="ri", clearable=False, searchable=False,
                                                  options=[{"label": v, "value": k} for k, v in COLOR_BY.items()],
                                                  style={"width": "260px", "display": "inline-block",
                                                         "verticalAlign": "middle"})],
                      style={"fontSize": "13px"}),
            dcc.Checklist(id="show-gen", options=[{"label": " show candidates", "value": "on"}],
                          value=["on"], inline=True),
            dcc.Checklist(id="zoom", options=[{"label": " zoom to box", "value": "on"}], value=[], inline=True),
            html.Span(id="export-msg", style={"color": "#555"}),
        ]),
        html.Div(["panels: ", dcc.Checklist(id="panels", inline=True, value=[tkey(t) for t in TRIPLES],
                                            options=[{"label": f" {tlabel(t)}  ", "value": tkey(t)} for t in TRIPLES],
                                            style={"display": "inline-block"}),
                  html.Span(id="panels-msg", style={"color": "#b00", "marginLeft": "8px"})],
                 style={"fontSize": "13px", "margin": "4px 0"}),
        html.Div(["Set the box with the sliders, or click a point in any panel to move the box there along that "
                  "panel's axes; then Generate. Candidates: ", *[legend_dot(STATUS_COLOR[s], s) for s in STATUSES]],
                 style={"color": "#555", "fontSize": "13px", "marginBottom": "4px"}),
        html.Div(sliders, style={"display": "grid", "gridTemplateColumns": "repeat(auto-fit, minmax(380px, 1fr))",
                                 "columnGap": "24px", "rowGap": "2px", "margin": "4px 0 10px"}),
        html.Div(style={"display": "flex", "gap": "8px", "alignItems": "flex-start", "flexWrap": "wrap",
                        "margin": "4px 0"}, children=[
            dcc.Textarea(id="query-text", placeholder="compositions in mol%, one per line, e.g.\nTeO2 75 PbO 25\n"
                                                      "70TeO2-20ZnO-10Na2O",
                         style={"width": "380px", "height": "64px", "fontFamily": "monospace", "fontSize": "12px"}),
            html.Div([html.Button("Show compositions", id="query-btn"), html.Br(),
                      html.Button("Clear (back to normal)", id="query-clear", style={"marginTop": "4px"})]),
            html.Span(id="query-msg", style={"color": "#b00", "fontSize": "13px"}),
        ]),
        dash_table.DataTable(id="query-table", columns=[{"id": c, "name": n} for c, n in QUERY_COLS],
                             style_cell=CELL, style_header={"fontWeight": "600"},
                             style_data_conditional=FLAG_STYLE),
        html.Div(id="summary", style={"margin": "6px 0", "fontSize": "14px", "padding": "6px 10px",
                                      "background": "#f6f6f6", "borderRadius": "6px"}),
        dcc.Loading(dcc.Graph(id="graph", config={"displaylogo": False}), delay_show=800),
        dcc.Loading(html.Div(id="gen-msg", style={"fontSize": "13px", "margin": "4px 0", "minHeight": "18px"})),
        dcc.Checklist(id="table-opts", inline=True, value=["empty"], options=[
            {"label": f" only interior holes (emptiness ≥ {HOLE_EMPTINESS:g}, enclosure ≥ {HOLE_ENCLOSURE:g})",
             "value": "empty"},
            {"label": " only new chemistry (new system / outside range)", "value": "new"}]),
        dash_table.DataTable(
            id="table", columns=[{"id": c, "name": n} for c, n in TABLE_COLS], page_size=25,
            sort_action="native", filter_action="native", style_cell=CELL, style_header={"fontWeight": "600"},
            style_data_conditional=FLAG_STYLE),
        dcc.Store(id="gen-version", data=0),
        dcc.Store(id="query-version", data=0),
    ])

    slider_ids = [f"slider-{k}" for k in range(1, N_PHI + 1)]
    slider_states = [State(s, "value") for s in slider_ids]

    # ---- the box: Reset -> everything; click in a panel -> move the box there along that panel's axes
    @app.callback(*[Output(s, "value") for s in slider_ids],
                  Input("reset-box", "n_clicks"), Input("graph", "clickData"), *slider_states,
                  State("panels", "value"), prevent_initial_call=True)
    def on_box(_reset, click, *rest):
        box, keys = list(rest[:N_PHI]), rest[N_PHI]
        if ctx.triggered_id == "reset-box":
            return tuple(full)
        p = (click or {}).get("points", [{}])[0]
        panels = shown_panels(keys)
        n = p.get("curveNumber", -1) // N_TRACES
        if not all(c in p for c in "xyz") or not 0 <= n < len(panels):
            return (no_update,) * N_PHI
        return tuple(box_after_click(K, box, panels[n], p))

    # ---- draw: full figure at start / when the panels change; Patch otherwise
    @app.callback(Output("graph", "figure"), Output("summary", "children"), Output("panels-msg", "children"),
                  *[Input(s, "value") for s in slider_ids], Input("color-by", "value"), Input("show-gen", "value"),
                  Input("zoom", "value"), Input("gen-version", "data"), Input("query-version", "data"),
                  Input("panels", "value"))
    def render(*inputs):
        box = list(inputs[:N_PHI])
        color_key, show_gen, zoom, _gv, _qv, keys = inputs[N_PHI:]
        panels = shown_panels(keys)
        if not panels:
            return no_update, no_update, "pick at least one panel"
        st = dict(box=box, color_key=color_key, show_gen=bool(show_gen), zoom=bool(zoom))
        trig = {t["prop_id"].split(".")[0] for t in ctx.triggered}
        if ctx.triggered_id is None or "panels" in trig:
            fig = full_figure(K, panels, st)
        else:
            moved = bool(trig & set(slider_ids))
            extras = bool(trig & {"gen-version", "query-version", "show-gen", "zoom"}) or (moved and st["zoom"])
            fig = figure_patch(K, panels, st, coords=moved, marker=moved or "color-by" in trig, extras=extras)
        con = K.constrained(box)
        if not con:
            return fig, "No box: all glasses. Narrow the sliders, or click a point to put a box around it.", ""
        sel = in_box(K.R, box)
        ri_sel = K.ri[sel]
        ri_txt = (f"RI {np.median(ri_sel):.3f} (median), {np.percentile(ri_sel, 5):.3f}–"
                  f"{np.percentile(ri_sel, 95):.3f} (p5–p95)" if len(ri_sel) else "empty box")
        dims = ", ".join(f"φ{k} ∈ [{box[k - 1][0]:.2f}, {box[k - 1][1]:.2f}]" for k in con)
        return fig, html.Div([html.B(f"Box {dims}: {int(sel.sum())} real glasses"), f" · {ri_txt}"]), ""

    # ---- generate candidates for the box
    @app.callback(Output("gen-msg", "children"), Output("gen-version", "data"),
                  Input("gen-btn", "n_clicks"), *slider_states, State("panels", "value"), State("gen-n", "value"),
                  State("gen-version", "data"), prevent_initial_call=True)
    def on_generate(_clicks, *rest):
        box, (keys, n, version) = list(rest[:N_PHI]), rest[N_PHI:]
        if not K.constrained(box):
            return "narrow the box first (sliders, or click a point)", no_update
        gen, C, Z, info, hulls = K.generate_box(box, shown_panels(keys), int(n or args.gen_n), seed=int(version or 0))
        GEN.clear()
        HULLS.clear()
        HULLS.update(hulls)
        if gen is None:
            return info, (version or 0) + 1
        custom = gen[["ri_interp", "emptiness", "enclosure", "d_nn_pct"]].to_numpy(np.float32)  # status = color
        GEN.update(df=gen, C=C, Z=Z, custom=custom, codes=gen["status"].map(STATUS_CODE).to_numpy(np.int8))
        st = gen["status"].value_counts()
        n_holes = int(((gen["emptiness"] >= HOLE_EMPTINESS) & (gen["enclosure"] >= HOLE_ENCLOSURE)).sum())
        msg = [info, html.Br(), html.B(f"{len(gen)} candidates in {gen['family'].nunique()} families · "),
               ", ".join(f"{s}: {st.get(s, 0)}" for s in STATUSES), f" · interior holes: {n_holes}"]
        return msg, (version or 0) + 1

    @app.callback(Output("table", "data"), Input("gen-version", "data"), Input("table-opts", "value"))
    def render_table(_version, opts):
        if "df" not in GEN:
            return []
        gen = filtered(GEN["df"], opts or [])
        if gen.empty:
            return []
        reps = gen.sort_values("emptiness", ascending=False).drop_duplicates("family")
        reps = reps.assign(n=reps["family"].map(gen["family"].value_counts()).to_numpy()).head(500)
        reps = reps.assign(rank=np.arange(1, len(reps) + 1))
        return reps[[c for c, _ in TABLE_COLS]].round(ROUND).to_dict("records")

    # ---- typed compositions
    @app.callback(Output("query-table", "data"), Output("query-msg", "children"), Output("query-version", "data"),
                  Input("query-btn", "n_clicks"), Input("query-clear", "n_clicks"), State("query-text", "value"),
                  State("query-version", "data"), prevent_initial_call=True)
    def on_query(_show, _clear, text, version):
        lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
        if ctx.triggered_id == "query-clear" or not lines:
            QUERY.clear()
            return [], "", (version or 0) + 1
        comps, errors = [], []
        for k, ln in enumerate(lines, start=1):
            try:
                comps.append(parse_composition(ln, K.oxides))
            except ValueError as e:
                errors.append(f"line {k}: {e}")
        if errors:
            return no_update, "; ".join(errors), no_update
        C = np.array(comps)
        Z = K.embed(C)
        info = K.describe(C, Z)
        info["label"] = [str(k) for k in range(1, len(C) + 1)]
        hover = [f"#{lab} · {c} · RI≈{r:.3f} · {s}"
                 for lab, c, r, s in zip(info["label"], info["composition"], info["ri_interp"], info["status"])]
        QUERY.clear()
        QUERY.update(Z=Z, labels=info["label"].tolist(), hover=hover)
        return info[[c for c, _ in QUERY_COLS]].round(ROUND).to_dict("records"), "", (version or 0) + 1

    @app.callback(Output("export-msg", "children"), Input("export", "n_clicks"), State("table-opts", "value"),
                  prevent_initial_call=True)
    def on_export(_clicks, opts):
        if "df" not in GEN:
            return "generate candidates first"
        gen = filtered(GEN["df"], opts or [])
        if gen.empty:
            return "no candidates with the current filters"
        comp = pd.DataFrame(100 * GEN["C"][gen.index], columns=K.oxides).round(3)
        comp = comp.loc[:, (comp > 0).any()]
        Z = pd.DataFrame(GEN["Z"][gen.index], columns=PHIS)
        out = pd.concat([gen.drop(columns=["composition"]).reset_index(drop=True), Z, comp], axis=1)
        out = out.sort_values(["family", "emptiness"], ascending=[True, False])
        path = DATA_OUT / f"candidates_3d_{args.tag}_{datetime.now():%Y%m%d_%H%M%S}.csv"
        out.to_csv(path, index=False)
        return f"wrote {path} ({len(out)} candidates, {out['family'].nunique()} families)"

    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="m10000_s1", help="m10000_s1 (composition) or m10000_s1_ri1 (RI-weighted)")
    ap.add_argument("--gen-n", type=int, default=20_000, help="mixtures generated per click")
    ap.add_argument("--dnn-max", type=float, default=0.073,
                    help="max distance to nearest real glass, in fractions (0.073 = 7.3 mol%%, p90 of real spacing)")
    ap.add_argument("--family-radius", type=float, default=1.0, help="mol%% radius for grouping similar candidates")
    ap.add_argument("--ri-disagree", type=float, default=0.03, help="flag if |RI interp - RI hgb| exceeds this")
    ap.add_argument("--port", type=int, default=8051)
    args = ap.parse_args()
    ensure_dirs()
    K = Context3D(args)
    build_app(K).run(debug=False, port=args.port)


if __name__ == "__main__":
    main()
