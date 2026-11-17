"""Dash explorer: linked panels of spectral coordinates, select (even empty) regions, get candidate glasses.

- 10 panels = all pairs of phi1..phi5 (standardized), as subplots of ONE figure (one WebGL context; ten
  separate WebGL graphs exceed the browser's context limit and panels go blank). Gray/colored dots: real glasses (color = RI).
  Orange dots (drawn underneath the real glasses, so they show only where no real glass is): mixture-cloud
  compositions (from src/mixture_cloud.py) that land far from real glasses. "show mixtures" toggles them.
- Lasso/box in any panel; selections in different panels combine by INTERSECTION. A new selection in a
  panel replaces that panel's previous one; the × buttons in "Active selections" drop one; "Clear" drops all.
  The store is the single source of truth: plotly's native selection is cleared after each lasso and the
  region is redrawn as an outline shape, so every panel (including the one with the lasso) shows the intersection.
- Candidates = mixtures inside every selection. Each shows its composition, parents + weights, d_nn
  (distance to nearest real glass, composition space), novelty (distance to nearest real glass in phi1..5,
  in units of the typical real-real spacing), and two RI estimates:
    RI interp - distance-weighted mean RI of the k nearest real glasses (harmonic extension on the kNN graph)
    RI hgb    - gradient boosting on composition (outputs/models/ri_model.joblib)
  If they disagree by more than --ri-disagree, the candidate is flagged.
- "Export CSV" writes the full compositions (oxide %, same units as the input data) of the current candidates to outputs/data/.

Run from the project root (needs outputs from spectral_embedding.py, mixture_cloud.py, train_ri_model.py):
  uv run src/app.py [--tag m10000_s1] [--dnn-max 0.073] [--novelty-min 5.0] [--types pair triplet] [--max-mix 30000]
then open http://127.0.0.1:8050
"""

import argparse
import sys
from datetime import datetime
from itertools import combinations
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import ALL, Dash, Input, Output, Patch, State, ctx, dash_table, dcc, html, no_update
from matplotlib.path import Path as MplPath
from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors

sys.path.insert(0, str(Path("src")))
from mixture_cloud import compositions  # noqa: E402
from spectral_embedding import DATA, DATA_OUT, MODELS, ensure_dirs  # noqa: E402
from train_ri_model import RI_RANGE  # noqa: E402

N_PHI = 5
PHIS = [f"phi{k}" for k in range(1, N_PHI + 1)]
PAIRS = list(combinations(range(1, N_PHI + 1), 2))
MIX_COLOR = "#e4572e"
UNSEL_COLOR, UNSEL_OPACITY = "#9e9e9e", 0.15  # points outside the selection: gray, transparent
MIX_TRACE, REAL_TRACE = 0, 1  # trace order in each panel (mixtures drawn underneath)


def top_oxides(C: np.ndarray, names: np.ndarray, top: int = 6) -> list[str]:
    order = np.argsort(-C, axis=1)[:, :top]
    return [" ".join(f"{names[j].upper()} {100 * C[i, j]:.1f}" for j in order[i] if C[i, j] > 5e-4)
            for i in range(len(C))]


# ---------------------------------------------------------------------------- data
def load_data(args):
    df = pd.read_parquet(DATA)
    oxides = df.columns.drop("refractive_index")
    X = df[oxides].to_numpy() / 100.0
    ri = df["refractive_index"].to_numpy()
    ids = df.index.to_numpy()

    real = pd.read_parquet(DATA_OUT / f"spectral_embedding_{args.tag}.parquet")
    assert (real["ID"].to_numpy() == ids).all(), "embedding rows are not aligned with the data"
    mix = pd.read_parquet(DATA_OUT / f"mixture_cloud_{args.tag}.parquet")

    mu, sd = real[PHIS].mean(), real[PHIS].std()
    R = ((real[PHIS] - mu) / sd).to_numpy()
    M = ((mix[PHIS] - mu) / sd).to_numpy()

    print("novelty of mixtures in phi1..5 ...")
    nn = NearestNeighbors(n_neighbors=2).fit(R)
    spacing = np.median(nn.kneighbors(R)[0][:, 1])
    mix["novelty"] = nn.kneighbors(M, n_neighbors=1)[0][:, 0] / spacing

    sel = mix["type"].isin(args.types) & (mix["d_nn"] <= args.dnn_max) & (mix["novelty"] >= args.novelty_min)
    print(f"mixtures passing filters: {sel.sum()} of {len(mix)}")
    mix, M = mix[sel.to_numpy()].reset_index(drop=True), M[sel.to_numpy()]
    if len(mix) > args.max_mix:
        pick = np.sort(np.random.default_rng(0).choice(len(mix), args.max_mix, replace=False))
        mix, M = mix.iloc[pick].reset_index(drop=True), M[pick]
    C = compositions(mix, X)

    ok = (ri >= RI_RANGE[0]) & (ri <= RI_RANGE[1])
    mix["ri_interp"] = KNeighborsRegressor(10, weights="distance").fit(X[ok], ri[ok]).predict(C)
    model_path = MODELS / "ri_model.joblib"
    mix["ri_hgb"] = joblib.load(model_path).predict(C) if model_path.exists() else np.nan
    mix["ri_flag"] = (mix["ri_interp"] - mix["ri_hgb"]).abs() > args.ri_disagree
    mix["composition"] = top_oxides(C, np.array(oxides))
    mix["parents"] = [
        " + ".join(f"{w:.2f}·{ids[p]}" for p, w in zip(ps, ws) if p >= 0)
        for ps, ws in zip(mix[["p1", "p2", "p3"]].to_numpy(), mix[["w1", "w2", "w3"]].to_numpy())
    ]
    return dict(R=R, M=M, ri=ri, ids=ids, mix=mix, C=C, oxides=np.array(oxides))


# ---------------------------------------------------------------------------- figures
N_COLS = 5


def axis_ids(n: int) -> tuple[str, str]:
    """Plotly axis ids of the n-th panel (0-based): ('x', 'y'), ('x2', 'y2'), ..."""
    return ("x", "y") if n == 0 else (f"x{n + 1}", f"y{n + 1}")


def make_grid(D) -> go.Figure:
    """All panels in ONE figure: a single WebGL context. Ten separate scattergl graphs exceed the browser's
    WebGL context limit and the oldest panels go blank."""
    # float32 halves the payload (10 panels x ~70k points) with no visible loss
    R, M, mix = D["R"].astype(np.float32), D["M"].astype(np.float32), D["mix"]
    ri = D["ri"].astype(np.float32)
    cmin, cmax = np.percentile(ri, [1, 99])
    rows = int(np.ceil(len(PAIRS) / N_COLS))
    fig = make_subplots(rows=rows, cols=N_COLS, subplot_titles=[f"φ{i} × φ{j}" for i, j in PAIRS],
                        horizontal_spacing=0.035, vertical_spacing=0.09)
    mix_cd = np.c_[mix["ri_interp"], mix["d_nn"]].astype(np.float32)
    for n, (i, j) in enumerate(PAIRS):
        r, c = divmod(n, N_COLS)
        # mixtures first (drawn underneath): orange only shows where there is no real glass on top
        fig.add_trace(go.Scattergl(
            x=M[:, i - 1], y=M[:, j - 1], mode="markers", name="mixtures",
            marker=dict(size=3, color=MIX_COLOR, opacity=0.8),
            selected=dict(marker=dict(opacity=1, color=MIX_COLOR)),
            unselected=dict(marker=dict(color=UNSEL_COLOR, opacity=UNSEL_OPACITY)),
            customdata=mix_cd,
            hovertemplate="mixture · RI≈%{customdata[0]:.4f} · d_nn=%{customdata[1]:.3f}<extra></extra>"),
            row=r + 1, col=c + 1)
        fig.add_trace(go.Scattergl(
            x=R[:, i - 1], y=R[:, j - 1], mode="markers", name="real",
            marker=dict(size=3, color=np.clip(ri, cmin, cmax), colorscale="Viridis", cmin=cmin, cmax=cmax,
                        showscale=(n == 0), colorbar=dict(title="RI", len=0.5, y=1, yanchor="top", thickness=10)),
            selected=dict(marker=dict(opacity=1)),
            unselected=dict(marker=dict(color=UNSEL_COLOR, opacity=UNSEL_OPACITY)),
            customdata=ri, hovertemplate="real · RI=%{customdata:.4f}<extra></extra>"),
            row=r + 1, col=c + 1)
    fig.update_annotations(font_size=13)
    fig.update_layout(dragmode="lasso", showlegend=False, template="plotly_white", uirevision="keep",
                      margin=dict(l=30, r=10, t=30, b=25), height=340 * rows)
    return fig


def trace_index(n: int, which: int) -> int:
    return 2 * n + which


# ---------------------------------------------------------------------------- selection logic
def geometry(sel: dict | None) -> tuple[str, dict] | None:
    """Plotly selectedData -> (panel key "i-j", geometry with plain x/y keys). Keys of range/lassoPoints are
    axis ids ('x2', 'y2', ...), which tell which subplot the selection was drawn in."""
    if not sel:
        return None
    kind = "lassoPoints" if sel.get("lassoPoints") else "range" if sel.get("range") else None
    if kind is None:
        return None
    g = sel[kind]
    xk = next(k for k in g if k.startswith("x"))
    yk = next(k for k in g if k.startswith("y"))
    n = 0 if xk == "x" else int(xk[1:]) - 1
    i, j = PAIRS[n]
    return f"{i}-{j}", {kind: {"x": g[xk], "y": g[yk]}}


def inside(geom: dict, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    if "lassoPoints" in geom:
        poly = np.c_[geom["lassoPoints"]["x"], geom["lassoPoints"]["y"]]
        return MplPath(poly).contains_points(np.c_[x, y])
    (x0, x1), (y0, y1) = geom["range"]["x"], geom["range"]["y"]
    return (x >= min(x0, x1)) & (x <= max(x0, x1)) & (y >= min(y0, y1)) & (y <= max(y0, y1))


def outline(geom: dict, n: int) -> dict:
    """Selection region drawn as a plain shape on panel n (does not interact with plotly's selection)."""
    xr, yr = axis_ids(n)
    style = dict(xref=xr, yref=yr, line=dict(color="#222", width=1.5, dash="dot"), fillcolor="rgba(0,0,0,0)")
    if "lassoPoints" in geom:
        xs, ys = geom["lassoPoints"]["x"], geom["lassoPoints"]["y"]
        path = "M " + " L ".join(f"{x},{y}" for x, y in zip(xs, ys)) + " Z"
        return dict(type="path", path=path, **style)
    (x0, x1), (y0, y1) = geom["range"]["x"], geom["range"]["y"]
    return dict(type="rect", x0=x0, x1=x1, y0=y0, y1=y1, **style)


def masks(D, store: dict):
    r = np.ones(len(D["R"]), bool)
    m = np.ones(len(D["M"]), bool)
    for key, geom in store.items():
        i, j = map(int, key.split("-"))
        r &= inside(geom, D["R"][:, i - 1], D["R"][:, j - 1])
        m &= inside(geom, D["M"][:, i - 1], D["M"][:, j - 1])
    return r, m


# ---------------------------------------------------------------------------- app
TABLE_COLS = [("rank", "#"), ("type", "type"), ("ri_interp", "RI interp"), ("ri_hgb", "RI hgb"),
              ("ri_flag", "RI disagree"), ("d_nn", "d_nn"), ("novelty", "novelty"),
              ("composition", "composition (oxide %, top oxides)"), ("parents", "parents (weight·ID)")]


def build_app(D, args) -> Dash:
    app = Dash(__name__, title="Glass spectral explorer")
    app.layout = html.Div(style={"fontFamily": "system-ui, sans-serif", "padding": "8px 16px"}, children=[
        html.Div(style={"display": "flex", "alignItems": "center", "gap": "16px", "flexWrap": "wrap"}, children=[
            html.H3("Glass spectral explorer", style={"margin": "4px 0"}),
            html.Button("Clear selections", id="clear"),
            html.Button("Export CSV", id="export"),
            dcc.Checklist(id="show-mix", options=[{"label": " show mixtures", "value": "on"}], value=["on"],
                          inline=True),
            html.Span(id="export-msg", style={"color": "#555"}),
        ]),
        html.Div(f"{len(D['R'])} real glasses · {len(D['M'])} mixtures shown "
                 f"(types={args.types}, d_nn ≤ {args.dnn_max}, novelty ≥ {args.novelty_min}). "
                 "Lasso/box in any panel; selections in different panels combine by intersection; "
                 "a new selection in a panel replaces that panel's; × removes one.",
                 style={"color": "#555", "fontSize": "13px", "marginBottom": "4px"}),
        html.Div(id="active-bar", style={"display": "flex", "gap": "8px", "alignItems": "center", "flexWrap": "wrap",
                                         "minHeight": "28px"}),
        html.Div(id="summary", children="No selection.",
                 style={"margin": "4px 0 8px", "fontSize": "14px", "padding": "6px 10px", "background": "#f6f6f6",
                        "borderRadius": "6px"}),
        dcc.Graph(id="grid", figure=make_grid(D), config={"displaylogo": False, "scrollZoom": True}),
        dash_table.DataTable(
            id="table", columns=[{"id": c, "name": n} for c, n in TABLE_COLS], page_size=25,
            sort_action="native", filter_action="native",
            style_cell={"fontSize": "12px", "textAlign": "left", "padding": "3px 6px", "whiteSpace": "normal",
                        "maxWidth": "420px"},
            style_header={"fontWeight": "600"},
            style_data_conditional=[{"if": {"filter_query": "{ri_flag} = true"}, "backgroundColor": "#fff3cd"}]),
        dcc.Store(id="sel-store", data={}),
    ])

    @app.callback(
        Output("grid", "figure"),
        Output("sel-store", "data"), Output("summary", "children"), Output("table", "data"),
        Output("active-bar", "children"),
        Input("grid", "selectedData"),
        Input("clear", "n_clicks"),
        Input({"type": "rm", "key": ALL}, "n_clicks"),
        State("sel-store", "data"),
        prevent_initial_call=True)
    def on_select(_sel, _clicks, _rm_clicks, store):
        store = dict(store or {})

        def skip():
            return (no_update,) * 5

        trig = ctx.triggered_id
        value = ctx.triggered[0]["value"] if ctx.triggered else None
        if trig == "clear":
            store = {}
        elif isinstance(trig, dict) and trig.get("type") == "rm":
            if not value:  # fired because the button was just created, not clicked
                return skip()
            store.pop(trig["key"], None)
        else:
            parsed = geometry(value)
            if parsed is None:
                # deselect events also fire when we clear plotly's native selection below: ignore them;
                # selections are removed with × or "Clear selections"
                return skip()
            key, geom = parsed
            store[key] = geom

        # The store is the single source of truth. Plotly's native selection is cleared in every panel
        # (otherwise a panel with a lasso keeps highlighting only its own selection, ignoring the others),
        # and each panel's own region is drawn back as a plain outline shape.
        r_in, m_in = masks(D, store)
        r_idx = np.flatnonzero(r_in).tolist() if store else None
        m_idx = np.flatnonzero(m_in).tolist() if store else None
        patch = Patch()
        for n, (i, j) in enumerate(PAIRS):
            patch["data"][trace_index(n, REAL_TRACE)]["selectedpoints"] = r_idx
            patch["data"][trace_index(n, MIX_TRACE)]["selectedpoints"] = m_idx
        patch["layout"]["selections"] = []
        patch["layout"]["shapes"] = [outline(store[f"{i}-{j}"], n) for n, (i, j) in enumerate(PAIRS)
                                     if f"{i}-{j}" in store]

        bar = [html.Span("Active selections:", style={"fontSize": "13px", "color": "#555"})] + [
            html.Button(f"φ{k.replace('-', ' × φ')}  ×", id={"type": "rm", "key": k},
                        style={"fontSize": "12px", "padding": "2px 8px", "borderRadius": "12px",
                               "border": "1px solid #999", "background": "white", "cursor": "pointer"})
            for k in store] if store else []
        if not store:
            return patch, store, "No selection.", [], bar

        mix = D["mix"].iloc[m_idx].sort_values("d_nn")
        ri_sel = D["ri"][r_in]
        ri_txt = (f"RI of selected real glasses: {np.median(ri_sel):.3f} (median), "
                  f"{np.percentile(ri_sel, 5):.3f}–{np.percentile(ri_sel, 95):.3f} (p5–p95)" if len(ri_sel) else
                  "no real glass inside — empty region")
        cand_txt = (f"{len(mix)} candidate mixtures ({', '.join(f'{t}: {n}' for t, n in mix['type'].value_counts().items())}); "
                    f"RI interp {mix['ri_interp'].min():.3f}–{mix['ri_interp'].max():.3f}; "
                    f"{int(mix['ri_flag'].sum())} with RI disagreement" if len(mix) else "no candidate mixtures inside")
        summary = html.Div([
            html.B(f"Intersection of {len(store)} selection(s): {int(r_in.sum())} real glasses · {len(mix)} candidates"),
            html.Div(ri_txt),
            html.Div(cand_txt),
        ])
        rows = mix.head(500).assign(rank=np.arange(1, min(len(mix), 500) + 1))
        rows = rows[[c for c, _ in TABLE_COLS]].round({"ri_interp": 4, "ri_hgb": 4, "d_nn": 4, "novelty": 2})
        return patch, store, summary, rows.to_dict("records"), bar

    @app.callback(Output("grid", "figure", allow_duplicate=True),
                  Input("show-mix", "value"), prevent_initial_call=True)
    def on_toggle_mixtures(value):
        patch = Patch()
        for n in range(len(PAIRS)):
            patch["data"][trace_index(n, MIX_TRACE)]["visible"] = bool(value)
        return patch

    @app.callback(Output("export-msg", "children"), Input("export", "n_clicks"), State("sel-store", "data"),
                  prevent_initial_call=True)
    def on_export(_clicks, store):
        if not store:
            return "nothing selected"
        _, m_in = masks(D, store)
        if not m_in.any():
            return "no candidates in selection"
        mix = D["mix"][m_in]
        comp = pd.DataFrame(100 * D["C"][m_in], columns=D["oxides"]).round(3)
        comp = comp.loc[:, (comp > 0).any()]
        out = pd.concat([mix[["type", "ri_interp", "ri_hgb", "ri_flag", "d_nn", "novelty", "parents"]
                             + PHIS].reset_index(drop=True), comp], axis=1).sort_values("d_nn")
        path = DATA_OUT / f"candidates_{datetime.now():%Y%m%d_%H%M%S}.csv"
        out.to_csv(path, index=False)
        return f"wrote {path} ({len(out)} candidates)"

    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="m10000_s1")
    ap.add_argument("--dnn-max", type=float, default=0.073, help="max distance to nearest real glass (p90 of real spacing)")
    ap.add_argument("--novelty-min", type=float, default=5.0,
                    help="min phi-space distance to real glasses, in units of the median real-real spacing "
                         "(real glasses themselves: p90 = 4.2)")
    ap.add_argument("--types", nargs="+", default=["pair", "triplet"], choices=["pair", "triplet", "random"])
    ap.add_argument("--max-mix", type=int, default=30_000)
    ap.add_argument("--ri-disagree", type=float, default=0.03, help="flag if |RI interp - RI hgb| exceeds this")
    ap.add_argument("--port", type=int, default=8050)
    args = ap.parse_args()
    ensure_dirs()
    D = load_data(args)
    build_app(D, args).run(debug=False, port=args.port)


if __name__ == "__main__":
    main()
