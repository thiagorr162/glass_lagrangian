"""Dash explorer: select a region of the spectral map (e.g. a hole inside a dense region) and generate candidate
glasses for it on demand.

Compositions are mol% throughout.

Panels
- 10 panels = all pairs of phi1..phi5 (standardized), as subplots of ONE figure (one WebGL context; ten
  separate WebGL graphs exceed the browser's context limit and panels go blank). Real glasses colored by RI.
- Lasso/box in any panel; selections in different panels combine by INTERSECTION. A new selection in a panel
  replaces that panel's; the × chips drop one; "Clear selections" drops all. The selection store is the single
  source of truth: plotly's native selection is cleared after each lasso and redrawn as an outline shape.

Generate candidates (button)
- Select a populated region that has holes in it. Parents: the real glasses inside the selection.
- Region: the intersection of the selections, made concrete as the convex hull of the selected glasses in each
  panel that has a lasso (drawn dashed blue after generating): tighter than the lasso, and keeps the holes inside.
- Mixtures (convex, so valid compositions): parent + 1 or 2 of its 10 nearest real glasses in composition, and
  random pairs of parents. Each is placed on the map with the Nystrom extension; kept if it lands inside the
  region and is within --dnn-max of a real glass in composition.
- Each candidate gets: emptiness and enclosure (src/holes.py: how empty its spot is relative to the local
  spacing of real glasses, and how much real glasses surround it), status w.r.t. the DB (src/check_candidates.py:
  new system / outside the known range of its system / known range), nearest real glass, and two RI estimates:
    RI interp - distance-weighted (1/d) mean RI of the 10 nearest real glasses in composition (a kNN; CV MAE 0.016)
    RI hgb    - gradient boosting on composition (outputs/models/ri_model.joblib; CV MAE 0.017)
- Candidates within --family-radius mol% of each other form a family; the table shows one row per family.
  The composition column is the full composition normalized to 100, in the format of the composition box
  (click the cell, Ctrl+C, paste). "Export CSV" writes all of them.

Edit mode (checkbox): click ANYWHERE on a panel (src/assets/edit_click.js catches clicks on empty space too),
or lasso/box a region, and get compositions that land there; they are drawn in every panel. A spot of one panel
fixes only 2 of the 5 coordinates, so its "equivalent" in the other panels is a set of compositions, not a point:
each solution is a convex mixture of 5 real glasses near the spot in that panel, with weights fitted (damped
Gauss-Newton on the simplex, analytic gradient of the Nystrom map) so that it lands exactly on the spot (within
0.03 standardized units). A drawn region is filled with solutions colored by their position in it (hue: left ->
right, lightness: bottom -> top), so the same colors in the other panels show where each part of the region goes.
Clicks: magenta. Dark gray: farther than --dnn-max from every real glass (implausible). Real glasses turn gray. Spots far from
every real glass in that panel are unreachable (the Nystrom map is a weighted average of the landmarks). In edit
mode a lasso does not filter: the selection store is unchanged. The results replace the candidates in the table
(sorted by d_nn) and can be exported the same way.

"show candidates" hides/shows the generated candidates on the panels.
"color by" colors the real glasses by RI, by each network former alone (mol%), by network formers (SiO2 P2O5 B2O3 GeO2 TeO2, mol%) or by
intermediates (ZrO2 Al2O3 Ta2O5 La2O3 Ga2O3 TiO2 Nb2O5 PbO WO3 Sb2O3 Bi2O3 As2O3 MoO3 V2O5 ..., mol%); both sums are
also shown on hover and as table columns.
"Zoom to selection" fits every panel to the real glasses left by the selection (8% margin); "Reset zoom" undoes it.

Query compositions: type compositions, one per line ("TeO2 75 PbO 25", "75TeO2-25PbO", "teo2=75, pbo=25");
they are marked with a red dot in every panel, with their status and RI.
"Clear" removes them.


With a model fitted with the RI kernel (spectral_embedding.py --ri-scale ..., e.g. --tag m10000_s1_ri1), new
compositions are placed using their kNN-interpolated RI.

Run from the project root (needs outputs of spectral_embedding.py and train_ri_model.py):
  uv run src/app.py
then open http://127.0.0.1:8050
"""

import argparse
import re
import sys
from datetime import datetime
from itertools import combinations
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import ALL, Dash, Input, Output, Patch, State, ctx, dash_table, dcc, html, no_update
from matplotlib.path import Path as MplPath
from scipy.spatial import ConvexHull
from plotly.subplots import make_subplots
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors

sys.path.insert(0, str(Path("src")))
from check_candidates import STATUSES, SystemIndex, families  # noqa: E402
from holes import HoleScorer  # noqa: E402
from spectral_embedding import DATA, DATA_OUT, MODELS, SpectralEmbedding, ensure_dirs  # noqa: E402
from train_ri_model import RI_RANGE  # noqa: E402

N_PHI = 5
PHIS = [f"phi{k}" for k in range(1, N_PHI + 1)]
PAIRS = list(combinations(range(1, N_PHI + 1), 2))
STATUS_COLOR = {"new system": "#d62728", "outside range": "#ff7f0e", "known range": "#e8c547"}
UNSEL_COLOR, UNSEL_OPACITY = "#9e9e9e", 0.15  # points outside the selection: gray, transparent
REAL_TRACE, GEN_TRACE, QUERY_TRACE = range(3)  # trace order in each panel
N_TRACES = 3
NEW_CHEM = ["new system", "outside range"]
QUERY_COLOR = "#d62728"
# oxide roles used to color the map and summarize compositions
NETWORK_FORMERS = ["SiO2", "P2O5", "B2O3", "GeO2", "TeO2"]
INTERMEDIATES = ["ZrO2", "Al2O3", "Ta2O5", "La2O3", "Ga2O3", "TiO2", "Nb2O5", "PbO", "WO3", "Sb2O3", "Bi2O3", "As2O3",
                 "MoO3", "V2O5", "As2O5", "VO6", "V2O3", "Nb2O3", "Ta2O3"]
COLOR_BY = {"ri": "RI", "formers": "network formers, total (mol%)", "intermediates": "intermediates, total (mol%)",
            **{f"ox:{o.lower()}": f"{o} (mol%)" for o in NETWORK_FORMERS}}  # plus each network former alone


def oxide_formula(name: str) -> str:
    """'nb2o5' -> 'Nb2O5', 'sio2' -> 'SiO2', 'coo' -> 'CoO' (as typed in the composition box)."""
    m = re.fullmatch(r"([a-z]+?)(\d*)o(\d*)", name)
    return f"{m.group(1).capitalize()}{m.group(2)}O{m.group(3)}" if m else name


def full_compositions(Cpct: np.ndarray, names: np.ndarray) -> list[str]:
    """Every oxide (>= 0.01 mol%), normalized to 100, largest first; paste-ready for the composition box."""
    formulas = [oxide_formula(o) for o in names]
    Cn = 100 * Cpct / Cpct.sum(axis=1, keepdims=True)
    order = np.argsort(-Cn, axis=1)
    return [" ".join(f"{formulas[j]} {Cn[i, j]:.2f}" for j in order[i] if Cn[i, j] >= 0.005)
            for i in range(len(Cn))]


# ---------------------------------------------------------------------------- data / models
class Context:
    """Real glasses, models and helpers to describe any composition (generated candidate or typed query)."""

    def __init__(self, args):
        df = pd.read_parquet(DATA)
        self.oxides = np.array(df.columns.drop("refractive_index"))
        self.X = df[self.oxides].to_numpy() / 100.0
        self.ri = df["refractive_index"].to_numpy()
        self.ids = df.index.to_numpy()
        self.args = args

        real = pd.read_parquet(DATA_OUT / f"spectral_embedding_{args.tag}.parquet")
        assert (real["ID"].to_numpy() == self.ids).all(), "embedding rows are not aligned with the data"
        self.mu, self.sd = real[PHIS].mean().to_numpy(), real[PHIS].std().to_numpy()
        self.R = (real[PHIS].to_numpy() - self.mu) / self.sd
        self.holes = HoleScorer(self.R, k=30)

        self.se = SpectralEmbedding.load(MODELS / f"model_{args.tag}.npz")
        print("composition neighbors of real glasses ...")
        self.nn_db = NearestNeighbors(n_neighbors=11).fit(self.X)
        self.nbrs = self.nn_db.kneighbors(self.X, return_distance=False)[:, 1:]
        ok = (self.ri >= RI_RANGE[0]) & (self.ri <= RI_RANGE[1])
        self.ri_knn = KNeighborsRegressor(10, weights="distance").fit(self.X[ok], self.ri[ok])
        model_path = MODELS / "ri_model.joblib"
        self.ri_model = joblib.load(model_path) if model_path.exists() else None
        self.systems = SystemIndex(100 * self.X, self.oxides)
        idx = {o: k for k, o in enumerate(self.oxides)}
        self.former_idx = [idx[o.lower()] for o in NETWORK_FORMERS if o.lower() in idx]
        self.inter_idx = [idx[o.lower()] for o in INTERMEDIATES if o.lower() in idx]
        self.formers = 100 * self.X[:, self.former_idx].sum(axis=1)
        self.intermediates = 100 * self.X[:, self.inter_idx].sum(axis=1)
        # Nystrom weights of phi1..phi5: phi(x) = p(x) @ W, p = kernel row normalized to 1 (see transform)
        se = self.se
        self.W = se.V_[:, 1:N_PHI + 1] / np.sqrt(se.d_)[:, None] / se.mu_[1:N_PHI + 1][None, :]
        self._panel_nn = {}

    def embed(self, C: np.ndarray) -> np.ndarray:
        # with the property-aware kernel the RI of a new composition is unknown: use the kNN-interpolated RI
        y = self.ri_knn.predict(C) if self.se.uses_ri else None
        return (self.se.transform(C, y)[:, :N_PHI] - self.mu) / self.sd

    def describe(self, C: np.ndarray, Z: np.ndarray) -> pd.DataFrame:
        """Composition fractions C and their map coordinates Z -> everything shown in the tables."""
        out = self.systems.compare(100 * C)
        dist, idx = self.nn_db.kneighbors(C, n_neighbors=1)
        out["d_nn_pct"] = 100 * dist[:, 0]
        out["nn1"] = [f"{self.ids[j]} · RI {self.ri[j]:.3f}" for j in idx[:, 0]]
        out["emptiness"], out["enclosure"] = self.holes.score(Z)
        out["ri_interp"] = self.ri_knn.predict(C)
        out["ri_hgb"] = self.ri_model.predict(C) if self.ri_model is not None else np.nan
        out["ri_flag"] = (out["ri_interp"] - out["ri_hgb"]).abs() > self.args.ri_disagree
        out["composition"] = full_compositions(100 * C, self.oxides)
        tot = C.sum(axis=1)
        out["formers"] = 100 * C[:, self.former_idx].sum(axis=1) / tot
        out["intermediates"] = 100 * C[:, self.inter_idx].sum(axis=1) / tot
        return out

    def generate(self, store: dict, n: int, seed: int = 0):
        """Candidates for the selected region (see module docstring). Returns (DataFrame, C, Z, info)."""
        rng = np.random.default_rng(seed)
        pool = np.flatnonzero(points_inside(store, self.R))  # parents: the real glasses of the selected region
        if len(pool) < 2:
            return None, None, None, f"only {len(pool)} real glasses in the selection — select a region with glasses", None

        k = self.nbrs.shape[1]
        m = n // 3
        i = rng.choice(pool, size=n)
        p2 = np.r_[self.nbrs[i[:m], rng.integers(k, size=m)],                      # parent + composition neighbor
                   self.nbrs[i[m:2 * m], rng.integers(k, size=m)],                 # parent + 2 neighbors
                   rng.choice(pool, size=n - 2 * m)]                               # two parents of the region
        a = rng.integers(k, size=m)
        p3 = np.full(n, -1)
        p3[m:2 * m] = self.nbrs[i[m:2 * m], (a + rng.integers(1, k, size=m)) % k]
        W = np.zeros((n, 3))
        W[:, :2] = rng.dirichlet([1, 1], size=n)
        W[m:2 * m] = rng.dirichlet([1, 1, 1], size=m)
        C = W[:, :1] * self.X[i] + W[:, 1:2] * self.X[p2]
        C[m:2 * m] += W[m:2 * m, 2:3] * self.X[p3[m:2 * m]]

        d_nn = self.nn_db.kneighbors(C, n_neighbors=1)[0][:, 0]
        plausible = d_nn <= self.args.dnn_max
        C, i, p2, p3, W, d_nn = C[plausible], i[plausible], p2[plausible], p3[plausible], W[plausible], d_nn[plausible]
        Z = self.embed(C)
        # the region is the intersection of the selections, made concrete as the convex hull of the selected
        # glasses in each panel that has a lasso (tighter than the lasso itself; keeps the holes inside)
        region = hull_region(self.R[pool], store.keys())
        keep = points_inside(store, Z) & points_inside(region, Z)
        info = (f"{len(pool)} parent glasses in the selection · {n} mixtures generated · "
                f"{int(plausible.sum())} plausible (d_nn ≤ {100 * self.args.dnn_max:.1f} mol%) · "
                f"{int(keep.sum())} land inside the region (hull of the selected glasses in the {len(region)} lassoed panel(s))")
        if not keep.any():
            return None, None, None, info + " — try a larger selection or more mixtures", region
        C, Z, i, p2, p3, W = C[keep], Z[keep], i[keep], p2[keep], p3[keep], W[keep]

        gen = self.describe(C, Z)
        gen["type"] = np.where(p3 >= 0, "triplet", np.where(np.isin(p2, pool), "pair (region)", "pair"))
        gen["parents"] = [" + ".join(f"{w:.2f}·{self.ids[q]}" for q, w in zip(qs, ws) if q >= 0)
                          for qs, ws in zip(np.c_[i, p2, p3], W)]
        gen["family"] = families(100 * C, -gen["emptiness"].to_numpy(), self.args.family_radius)
        return gen, C, Z, info, region

    # ------------------------------------------------------------ edit mode: compositions that land on a spot
    def phi_and_jacobian(self, x: np.ndarray, cols: list[int]):
        """Standardized phi_cols(x) and its gradient w.r.t. the composition x (B x len(cols) x n_oxides).

        phi(x) = sum_l p_l(x) W_l with p_l = K_l / sum K, K_l = exp(-|x - x_l|^2 / 2 sigma^2), so
        d phi / dx = (1/sigma^2) sum_l p_l W_l (x_l - xbar),  xbar = sum_l p_l x_l  (a p-weighted covariance).
        With the RI kernel the RI factor is held fixed at the interpolated RI of x (its gradient is ignored)."""
        se = self.se
        dist2 = euclidean_distances(x, se.X_, squared=True)
        Kx = np.exp(-dist2 / (2 * se.sigma_**2))
        if se.uses_ri:
            Kx *= se._ri_factor(np.clip(self.ri_knn.predict(x), se.y_lo_, se.y_hi_), se.y_)
        Kx[dist2 < 1e-12] = 0.0
        p = Kx / Kx.sum(axis=1, keepdims=True)
        Wc = self.W[:, cols]
        phi = p @ Wc
        xbar = p @ se.X_
        J = np.stack([(p * Wc[:, c]) @ se.X_ - phi[:, c:c + 1] * xbar for c in range(len(cols))], axis=1)
        sd = self.sd[cols]
        return (phi - self.mu[cols]) / sd, J / se.sigma_**2 / sd[None, :, None]

    def preimage(self, i: int, j: int, T: np.ndarray, per_target: int, seed: int = 0, n_iter: int = 40,
                 pool_size: int = 200, n_seed: int = 50):
        """Compositions whose Nystrom coordinates (phi_i, phi_j) equal each target in T (standardized units).

        A point of one panel fixes 2 of the 5 coordinates, so its "equivalent" in the other panels is a whole set
        of compositions. Each solution is a convex mixture (always a valid composition) of 5 anchor glasses:
        a seed among the 50 real glasses nearest to the target IN THAT PANEL, its 2 nearest glasses in composition
        and 2 random glasses, both among the 200 nearest in that panel. Different seeds give different
        solutions. The mixture weights are fitted by damped Gauss-Newton on the simplex until the mixture lands on
        the target."""
        rng = np.random.default_rng(seed)
        cols = [i - 1, j - 1]
        if (i, j) not in self._panel_nn:
            self._panel_nn[(i, j)] = NearestNeighbors().fit(self.R[:, cols])
        B = len(T) * per_target
        rows = np.arange(B)
        tgt = np.repeat(T, per_target, axis=0)
        pool = self._panel_nn[(i, j)].kneighbors(T, n_neighbors=pool_size, return_distance=False)
        pool = np.repeat(pool, per_target, axis=0)
        seed_g = pool[rows, rng.integers(n_seed, size=B)]
        d = np.linalg.norm(self.X[pool] - self.X[seed_g][:, None, :], axis=2)
        d[d < 0.005] = np.inf  # skip the seed itself and its near-duplicates (they add no direction)
        near = pool[rows[:, None], np.argsort(d, axis=1)[:, :2]]
        rand = pool[rows[:, None], rng.integers(pool_size, size=(B, 2))]
        A = np.c_[seed_g, near, rand]
        CA = self.X[A]  # B x 5 anchors x oxides
        w = rng.dirichlet(np.ones(A.shape[1]), size=B)
        eye = np.eye(2)
        act = rows  # solutions still moving
        best = np.full(B, np.inf)
        best_w = w.copy()
        stall = np.zeros(B, int)
        for _ in range(n_iter):
            z, J = self.phi_and_jacobian(np.einsum("ba,bad->bd", w[act], CA[act]), cols)
            r = tgt[act] - z
            res = np.linalg.norm(r, axis=1)
            better = res < 0.99 * best[act]
            best_w[act[better]] = w[act[better]]
            best[act[better]] = res[better]
            stall[act] = np.where(better, 0, stall[act] + 1)
            # stop a solution once it is on target, or after 5 steps without progress (target out of reach)
            moving = (res > 1e-3) & (stall[act] < 5)
            act, r, J = act[moving], r[moving], J[moving]
            if len(act) == 0:
                break
            Jw = np.einsum("bkd,bad->bka", J, CA[act])
            Jw -= Jw.mean(axis=2, keepdims=True)  # moves that keep sum(w) = 1
            G = Jw @ Jw.transpose(0, 2, 1)
            G += (1e-9 + 1e-3 * np.trace(G, axis1=1, axis2=2))[:, None, None] * eye
            step = np.einsum("bka,bk->ba", Jw, np.linalg.solve(G, r[..., None])[..., 0])
            step *= np.minimum(1.0, 0.5 / np.maximum(np.abs(step).max(axis=1, keepdims=True), 1e-12))
            w[act] = project_simplex(w[act] + step)
        z, _ = self.phi_and_jacobian(np.einsum("ba,bad->bd", w, CA), cols)  # keep the last step if it was best
        last_better = np.linalg.norm(tgt - z, axis=1) < best
        best_w[last_better] = w[last_better]
        w = best_w
        C = np.einsum("ba,bad->bd", w, CA)
        Z = self.embed(C)  # exact placement (all 5 coordinates)
        resid = np.linalg.norm(Z[:, cols] - tgt, axis=1)
        return C, Z, resid, A, w


# ---------------------------------------------------------------------------- selection geometry
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


def points_inside(store: dict, Z: np.ndarray) -> np.ndarray:
    m = np.ones(len(Z), bool)
    for key, geom in store.items():
        i, j = map(int, key.split("-"))
        m &= inside(geom, Z[:, i - 1], Z[:, j - 1])
    return m


def hull_region(P: np.ndarray, panels) -> dict:
    """Convex hull of the points P (standardized phi1..5) in the given panels ("i-j" keys), as lasso-like
    polygons. Panels where the points are degenerate (fewer than 3, or collinear) are left unconstrained."""
    region = {}
    for i, j in (map(int, k.split("-")) for k in panels):
        pts = P[:, [i - 1, j - 1]]
        try:
            hull = ConvexHull(pts)
        except Exception:  # QhullError: too few / collinear points
            continue
        v = pts[hull.vertices]
        region[f"{i}-{j}"] = {"lassoPoints": {"x": v[:, 0].tolist(), "y": v[:, 1].tolist()}}
    return region


def project_simplex(V: np.ndarray) -> np.ndarray:
    """Euclidean projection of each row onto the probability simplex."""
    U = -np.sort(-V, axis=1)
    css = np.cumsum(U, axis=1) - 1
    k = np.arange(1, V.shape[1] + 1)
    rho = (U - css / k > 0).sum(axis=1)
    theta = css[np.arange(len(V)), rho - 1] / rho
    return np.maximum(V - theta[:, None], 0.0)


def sample_inside(geom: dict, n: int, rng: np.random.Generator) -> np.ndarray:
    """n points uniformly inside a lasso/box geometry (rejection sampling in its bounding box)."""
    xs, ys = ((geom["lassoPoints"]["x"], geom["lassoPoints"]["y"]) if "lassoPoints" in geom
              else (geom["range"]["x"], geom["range"]["y"]))
    lo, hi = np.array([min(xs), min(ys)]), np.array([max(xs), max(ys)])
    out = np.zeros((0, 2))
    for _ in range(50):
        P = lo + (hi - lo) * rng.random((4 * n, 2))
        out = np.r_[out, P[inside(geom, P[:, 0], P[:, 1])]]
        if len(out) >= n:
            break
    return out[:n]


OUTLINE_STYLE = {"selection": dict(color="#222", width=1.5, dash="dot"),
                 "hull": dict(color="#1f77b4", width=1.5, dash="dash"),
                 "edit": dict(color="#d6008f", width=2, dash="solid")}
EDIT_COLOR, EDIT_FAR_COLOR = "#d6008f", "#7a7a7a"
EDIT_BG_COLOR = "#d9d9d9"  # real glasses while in edit mode  # edit-mode solutions: plausible / far from every real glass


def region_colors(P: np.ndarray, geom: dict) -> np.ndarray:
    """2D color map over the bounding box of a region: hue from x (left -> right), lightness from y."""
    import colorsys
    xs, ys = ((geom["lassoPoints"]["x"], geom["lassoPoints"]["y"]) if "lassoPoints" in geom
              else (geom["range"]["x"], geom["range"]["y"]))
    u = (P[:, 0] - min(xs)) / max(max(xs) - min(xs), 1e-12)
    v = (P[:, 1] - min(ys)) / max(max(ys) - min(ys), 1e-12)
    rgb = [colorsys.hls_to_rgb(0.8 * a, 0.25 + 0.45 * b, 0.9) for a, b in zip(np.clip(u, 0, 1), np.clip(v, 0, 1))]
    return np.array(["#%02x%02x%02x" % tuple(int(255 * c) for c in col) for col in rgb])


def target_marker(n: int, x: float, y: float) -> dict:
    """Ring of fixed pixel size around a clicked spot of panel n."""
    xr, yr = axis_ids(n)
    return dict(type="circle", xref=xr, yref=yr, xsizemode="pixel", ysizemode="pixel", xanchor=x, yanchor=y,
                x0=-9, x1=9, y0=-9, y1=9, line=OUTLINE_STYLE["edit"], fillcolor="rgba(0,0,0,0)")


def outline(geom: dict, n: int, kind: str = "selection") -> dict:
    """Region drawn as a plain shape on panel n (does not interact with plotly's selection)."""
    xr, yr = axis_ids(n)
    style = dict(xref=xr, yref=yr, line=OUTLINE_STYLE[kind], fillcolor="rgba(0,0,0,0)")
    if "lassoPoints" in geom:
        xs, ys = geom["lassoPoints"]["x"], geom["lassoPoints"]["y"]
        path = "M " + " L ".join(f"{x},{y}" for x, y in zip(xs, ys)) + " Z"
        return dict(type="path", path=path, **style)
    (x0, x1), (y0, y1) = geom["range"]["x"], geom["range"]["y"]
    return dict(type="rect", x0=x0, x1=x1, y0=y0, y1=y1, **style)


# ---------------------------------------------------------------------------- figure
N_COLS = 5


def axis_ids(n: int) -> tuple[str, str]:
    """Plotly axis ids of the n-th panel (0-based): ('x', 'y'), ('x2', 'y2'), ..."""
    return ("x", "y") if n == 0 else (f"x{n + 1}", f"y{n + 1}")


def trace_index(n: int, which: int) -> int:
    return N_TRACES * n + which


def make_grid(K: Context) -> go.Figure:
    """All panels in ONE figure: a single WebGL context."""
    R = K.R.astype(np.float32)  # float32 halves the payload
    ri = K.ri.astype(np.float32)
    cmin, cmax = np.percentile(ri, [1, 99])
    rows = int(np.ceil(len(PAIRS) / N_COLS))
    fig = make_subplots(rows=rows, cols=N_COLS, subplot_titles=[f"φ{i} × φ{j}" for i, j in PAIRS],
                        horizontal_spacing=0.035, vertical_spacing=0.09)
    for n, (i, j) in enumerate(PAIRS):
        r, c = divmod(n, N_COLS)
        kw = dict(row=r + 1, col=c + 1)
        fig.add_trace(go.Scattergl(
            x=R[:, i - 1], y=R[:, j - 1], mode="markers", name="real",
            marker=dict(size=3, color=np.clip(ri, cmin, cmax), colorscale="Viridis", cmin=cmin, cmax=cmax,
                        showscale=(n == 0), colorbar=dict(title="RI", len=0.5, y=1, yanchor="top", thickness=10)),
            # color is switched by the "color by" selector (RI / network formers / intermediates)
            selected=dict(marker=dict(opacity=1)),
            unselected=dict(marker=dict(color=UNSEL_COLOR, opacity=UNSEL_OPACITY)),
            customdata=np.c_[ri, K.formers, K.intermediates].astype(np.float32),
            hovertemplate="real · RI=%{customdata[0]:.4f} · formers %{customdata[1]:.1f}% · "
                          "intermediates %{customdata[2]:.1f}%<extra></extra>"), **kw)
        fig.add_trace(go.Scattergl(x=[], y=[], mode="markers", name="generated",
                                   marker=dict(size=5, symbol="diamond"),
                                   unselected=dict(marker=dict(opacity=1)),
                                   hovertemplate="%{hovertext}<extra></extra>"), **kw)
        # WebGL like the other traces: SVG traces are drawn UNDER the WebGL canvas. Last trace = on top.
        fig.add_trace(go.Scattergl(x=[], y=[], mode="markers+text", name="query", textposition="top right",
                                 textfont=dict(color=QUERY_COLOR, size=14),
                                 marker=dict(symbol="circle", size=13, color=QUERY_COLOR, line=dict(width=1.5, color="white")),
                                 unselected=dict(marker=dict(opacity=1)),
                                 hovertemplate="%{hovertext}<extra></extra>"), **kw)
    fig.update_annotations(font_size=13)
    fig.update_layout(dragmode="lasso", showlegend=False, template="plotly_white", uirevision="keep",
                      margin=dict(l=30, r=10, t=30, b=25), height=340 * rows)
    return fig


def marker_patch(patch: Patch, which: int, Z: np.ndarray, hover: list[str], text=None, color=None):
    for n, (i, j) in enumerate(PAIRS):
        t = patch["data"][trace_index(n, which)]
        t["x"] = Z[:, i - 1].tolist()
        t["y"] = Z[:, j - 1].tolist()
        t["hovertext"] = hover
        if text is not None:
            t["text"] = text
        if color is not None:
            t["marker"]["color"] = color


# ---------------------------------------------------------------------------- query compositions
OX_RE = re.compile(r"([A-Za-z][A-Za-z0-9]*)\s*[:=]?\s*(\d+(?:[.,]\d+)?)|(\d+(?:[.,]\d+)?)\s*([A-Za-z][A-Za-z0-9]*)")


def parse_composition(line: str, oxides: np.ndarray) -> np.ndarray:
    """'TeO2 75 PbO 25' / '75TeO2-25PbO' / 'teo2=75, pbo=25' -> fractions over `oxides` (normalized to 1)."""
    index = {o: k for k, o in enumerate(oxides)}
    c = np.zeros(len(oxides))
    unknown = []
    for m in OX_RE.finditer(line):
        name, val = (m.group(1), m.group(2)) if m.group(1) else (m.group(4), m.group(3))
        if name.lower() not in index:
            unknown.append(name)
            continue
        c[index[name.lower()]] += float(val.replace(",", "."))
    if unknown:
        raise ValueError(f"unknown oxide(s): {', '.join(unknown)}")
    if c.sum() <= 0:
        raise ValueError("no oxide amounts found")
    return c / c.sum()


# ---------------------------------------------------------------------------- app
TABLE_COLS = [("rank", "#"), ("n", "n similar"), ("emptiness", "emptiness"), ("enclosure", "enclosure"),
              ("status", "status"), ("system", "system"), ("n_same_system", "DB glasses in system"),
              ("max_out", "mol% outside known range"), ("ri_interp", "RI interp"), ("ri_hgb", "RI hgb"),
              ("ri_flag", "RI disagree"), ("d_nn_pct", "d_nn (mol%)"), ("formers", "formers (mol%)"),
              ("intermediates", "intermediates (mol%)"),
              ("composition", "composition (mol%, full, copy-ready)"), ("nn1", "nearest real glass"),
              ("type", "type"), ("parents", "parents (weight·ID)")]
QUERY_COLS = [("label", "#"), ("composition", "composition (mol%, normalized)"), ("emptiness", "emptiness"),
              ("enclosure", "enclosure"), ("status", "status"), ("system", "system"),
              ("n_same_system", "DB glasses in system"), ("max_out", "mol% outside known range"),
              ("ri_interp", "RI interp"), ("ri_hgb", "RI hgb"), ("ri_flag", "RI disagree"),
              ("d_nn_pct", "d_nn (mol%)"), ("formers", "formers (mol%)"), ("intermediates", "intermediates (mol%)"),
              ("nn1", "nearest real glass")]
ROUND = {"ri_interp": 4, "ri_hgb": 4, "d_nn_pct": 2, "max_out": 2, "emptiness": 2, "enclosure": 2,
         "formers": 1, "intermediates": 1}
CELL = {"fontSize": "12px", "textAlign": "left", "padding": "3px 6px", "whiteSpace": "normal", "maxWidth": "380px"}
FLAG_STYLE = [{"if": {"filter_query": "{ri_flag} = true"}, "backgroundColor": "#fff3cd"}]
HOLE_EMPTINESS, HOLE_ENCLOSURE = 2.0, 0.4  # interior hole: emptier than ~98% of real glasses, and surrounded
GEN = {}  # last generated candidates (local single-user app): df, C, Z


def legend_dot(color: str, label: str):
    return html.Span([html.Span("◆", style={"color": color, "fontSize": "15px"}), f" {label}"],
                     style={"marginRight": "12px"})


def filtered(gen: pd.DataFrame, opts: list) -> pd.DataFrame:
    if "new" in opts:
        gen = gen[gen["status"].isin(NEW_CHEM)]
    if "empty" in opts:
        gen = gen[(gen["emptiness"] >= HOLE_EMPTINESS) & (gen["enclosure"] >= HOLE_ENCLOSURE)]
    return gen


def build_app(K: Context) -> Dash:
    args = K.args
    # src/assets/edit_click.js: in edit mode, a click anywhere in a panel (not only on a point) is sent here
    app = Dash(__name__, title="Glass spectral explorer", assets_folder=str(Path("src/assets").resolve()))
    app.layout = html.Div(style={"fontFamily": "system-ui, sans-serif", "padding": "8px 16px"}, children=[
        html.Div(style={"display": "flex", "alignItems": "center", "gap": "14px", "flexWrap": "wrap"}, children=[
            html.H3("Glass spectral explorer", style={"margin": "4px 0"}),
            html.Button("Clear selections", id="clear"),
            html.Button("Zoom to selection", id="zoom-btn"),
            html.Button("Reset zoom", id="zoom-reset"),
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
            html.Span(style={"fontSize": "13px", "padding": "2px 8px", "border": f"1px solid {EDIT_COLOR}",
                             "borderRadius": "6px"}, children=[
                dcc.Checklist(id="edit-mode", options=[{"label": " edit mode", "value": "on"}], value=[],
                              inline=True, style={"display": "inline-block", "fontWeight": "600",
                                                  "color": EDIT_COLOR}),
                " solutions: ", dcc.Input(id="edit-n", type="number", value=600, min=10, max=2000, step=50,
                                          style={"width": "60px"})]),
            html.Span(id="export-msg", style={"color": "#555"}),
        ]),
        html.Div(["Edit mode: lasso/box a region of a panel to see where it goes in the other panels (it is "
                  "filled with compositions that land inside it, colored by their position in it), or click a "
                  "single spot. Click: ",
                  legend_dot(EDIT_COLOR, f"within {100 * args.dnn_max:.1f} mol% of a real glass"),
                  legend_dot(EDIT_FAR_COLOR, "farther (implausible)"),
                  "· the table lists them (most plausible first)."],
                 id="edit-help", style={"display": "none"}),
        html.Div(["Select a region (lasso/box; panels combine by intersection), then Generate. Candidates: ",
                  *[legend_dot(STATUS_COLOR[s], s) for s in STATUSES],
                  "· click a composition cell and Ctrl+C to copy it into the composition box."],
                 style={"color": "#555", "fontSize": "13px", "marginBottom": "4px"}),
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
        html.Div(id="active-bar", style={"display": "flex", "gap": "8px", "alignItems": "center", "flexWrap": "wrap",
                                         "minHeight": "28px", "marginTop": "6px"}),
        html.Div(id="summary", children="No selection.",
                 style={"margin": "4px 0", "fontSize": "14px", "padding": "6px 10px", "background": "#f6f6f6",
                        "borderRadius": "6px"}),
        dcc.Graph(id="grid", figure=make_grid(K), config={"displaylogo": False, "scrollZoom": True}),
        dcc.Loading(html.Div(id="gen-msg", style={"fontSize": "13px", "margin": "4px 0", "minHeight": "18px"})),
        html.Div(style={"display": "flex", "gap": "16px", "alignItems": "center"}, children=[
            dcc.Checklist(id="table-opts", inline=True, value=["empty"], options=[
                {"label": f" only interior holes (emptiness ≥ {HOLE_EMPTINESS:g}, enclosure ≥ {HOLE_ENCLOSURE:g})",
                 "value": "empty"},
                {"label": " only new chemistry (new system / outside range)", "value": "new"}]),
        ]),
        dash_table.DataTable(
            id="table", columns=[{"id": c, "name": n} for c, n in TABLE_COLS], page_size=25,
            sort_action="native", filter_action="native", style_cell=CELL, style_header={"fontWeight": "600"},
            style_data_conditional=FLAG_STYLE),
        dcc.Store(id="sel-store", data={}),
        dcc.Store(id="gen-version", data=0),
        dcc.Store(id="edit-click"),   # written by src/assets/edit_click.js
        dcc.Store(id="edit-region"),  # a lasso/box drawn in edit mode
    ])

    # ---- selection
    @app.callback(
        Output("grid", "figure"), Output("sel-store", "data"), Output("summary", "children"),
        Output("active-bar", "children"), Output("edit-region", "data"),
        Input("grid", "selectedData"), Input("clear", "n_clicks"), Input({"type": "rm", "key": ALL}, "n_clicks"),
        State("sel-store", "data"), State("edit-mode", "value"), prevent_initial_call=True)
    def on_select(_sel, _clicks, _rm_clicks, store, edit_mode):
        store = dict(store or {})
        trig = ctx.triggered_id
        value = ctx.triggered[0]["value"] if ctx.triggered else None
        edit_region = no_update
        if trig == "clear":
            store = {}
        elif isinstance(trig, dict) and trig.get("type") == "rm":
            if not value:  # fired because the button was just created, not clicked
                return (no_update,) * 5
            store.pop(trig["key"], None)
        else:
            parsed = geometry(value)
            if parsed is None:  # deselect events also fire when we clear plotly's native selection: ignore
                return (no_update,) * 5
            key, geom = parsed
            if edit_mode:  # the lasso is a region to fill with solutions, not a filter: the store is unchanged
                edit_region = {"key": key, "geom": geom, "t": datetime.now().timestamp()}
            else:
                store[key] = geom

        # the store is the single source of truth: clear plotly's native selection, redraw outlines
        r_in = points_inside(store, K.R)
        r_idx = np.flatnonzero(r_in).tolist() if store else None
        patch = Patch()
        for n in range(len(PAIRS)):
            patch["data"][trace_index(n, REAL_TRACE)]["selectedpoints"] = r_idx
        patch["layout"]["selections"] = []
        patch["layout"]["shapes"] = [outline(store[f"{i}-{j}"], n) for n, (i, j) in enumerate(PAIRS)
                                     if f"{i}-{j}" in store]
        bar = [html.Span("Active selections:", style={"fontSize": "13px", "color": "#555"})] + [
            html.Button(f"φ{k.replace('-', ' × φ')}  ×", id={"type": "rm", "key": k},
                        style={"fontSize": "12px", "padding": "2px 8px", "borderRadius": "12px",
                               "border": "1px solid #999", "background": "white", "cursor": "pointer"})
            for k in store] if store else []
        if not store:
            return patch, store, "No selection.", bar, edit_region
        ri_sel = K.ri[r_in]
        ri_txt = (f"RI {np.median(ri_sel):.3f} (median), {np.percentile(ri_sel, 5):.3f}–"
                  f"{np.percentile(ri_sel, 95):.3f} (p5–p95)" if len(ri_sel) else "empty region")
        summary = html.Div([html.B(f"Intersection of {len(store)} selection(s): {int(r_in.sum())} real glasses"),
                            f" · {ri_txt}"])
        return patch, store, summary, bar, edit_region

    # ---- color the real glasses by RI / network formers / intermediates
    @app.callback(Output("grid", "figure", allow_duplicate=True), Input("color-by", "value"),
                  Input("edit-mode", "value"), prevent_initial_call=True)
    def on_color_by(key, edit_mode):
        patch = Patch()
        if edit_mode:  # gray background, so the colors of the edit-mode solutions can be read
            for n in range(len(PAIRS)):
                patch["data"][trace_index(n, REAL_TRACE)]["marker"]["color"] = EDIT_BG_COLOR
            patch["data"][trace_index(0, REAL_TRACE)]["marker"]["showscale"] = False
            return patch
        patch["data"][trace_index(0, REAL_TRACE)]["marker"]["showscale"] = True
        if key.startswith("ox:"):  # a single oxide, mol%
            values = 100 * K.X[:, list(K.oxides).index(key[3:])]
        else:
            values = {"ri": K.ri, "formers": K.formers, "intermediates": K.intermediates}[key]
        lo, hi = (np.percentile(values, [1, 99]) if key == "ri" else (0.0, float(values.max())))
        colors = np.round(np.clip(values, lo, hi), 3).tolist()  # rounded: smaller payload (58k values x 10 panels)
        for n in range(len(PAIRS)):
            m = patch["data"][trace_index(n, REAL_TRACE)]["marker"]
            m["color"] = colors
            m["cmin"], m["cmax"] = float(lo), float(hi)
        patch["data"][trace_index(0, REAL_TRACE)]["marker"]["colorbar"]["title"]["text"] = COLOR_BY[key]
        return patch

    # ---- zoom every panel to the glasses left by the selection
    @app.callback(Output("grid", "figure", allow_duplicate=True),
                  Input("zoom-btn", "n_clicks"), Input("zoom-reset", "n_clicks"), State("sel-store", "data"),
                  prevent_initial_call=True)
    def on_zoom(_zoom, _reset, store):
        patch = Patch()
        # a new uirevision makes plotly apply these ranges even if the user zoomed/panned by hand before
        patch["layout"]["uirevision"] = f"zoom-{datetime.now().timestamp()}"
        sel = points_inside(store or {}, K.R) if store else np.zeros(len(K.R), bool)
        reset = ctx.triggered_id == "zoom-reset" or not sel.any()
        for n, (i, j) in enumerate(PAIRS):
            xa, ya = (("xaxis", "yaxis") if n == 0 else (f"xaxis{n + 1}", f"yaxis{n + 1}"))
            if reset:
                patch["layout"][xa]["autorange"] = True
                patch["layout"][ya]["autorange"] = True
                continue
            for ax, k in ((xa, i), (ya, j)):
                lo, hi = K.R[sel, k - 1].min(), K.R[sel, k - 1].max()
                pad = 0.08 * (hi - lo) if hi > lo else 0.1
                patch["layout"][ax]["range"] = [float(lo - pad), float(hi + pad)]
                patch["layout"][ax]["autorange"] = False  # "Reset zoom" sets autorange=True, which would override range
        return patch

    # ---- generate candidates for the current selection
    @app.callback(Output("grid", "figure", allow_duplicate=True), Output("gen-msg", "children"),
                  Output("gen-version", "data"),
                  Input("gen-btn", "n_clicks"), State("sel-store", "data"), State("gen-n", "value"),
                  State("gen-version", "data"), prevent_initial_call=True)
    def on_generate(_clicks, store, n, version):
        if not store:
            return no_update, "select a region first", no_update
        gen, C, Z, info, region = K.generate(store, int(n or args.gen_n), seed=int(version or 0))
        patch = Patch()
        # lasso outlines (dotted black) + the effective region: hull of the selected glasses (dashed blue)
        patch["layout"]["shapes"] = (
            [outline(store[f"{i}-{j}"], n) for n, (i, j) in enumerate(PAIRS) if f"{i}-{j}" in store]
            + [outline(region[f"{i}-{j}"], n, "hull") for n, (i, j) in enumerate(PAIRS) if region and f"{i}-{j}" in region])
        if gen is None:
            GEN.clear()
            marker_patch(patch, GEN_TRACE, np.zeros((0, N_PHI)), [], color=[])
            return patch, info, (version or 0) + 1
        GEN.clear()
        GEN.update(df=gen, C=C, Z=Z)
        hover = [f"{c} · RI≈{r:.3f} · {s} · emptiness {e:.1f}" for c, r, s, e in
                 zip(gen["composition"], gen["ri_interp"], gen["status"], gen["emptiness"])]
        marker_patch(patch, GEN_TRACE, Z, hover, color=gen["status"].map(STATUS_COLOR).tolist())
        st = gen["status"].value_counts()
        msg = [info, html.Br(), html.B(f"{len(gen)} candidates in {gen['family'].nunique()} families · "),
               ", ".join(f"{s}: {st.get(s, 0)}" for s in STATUSES),
               f" · interior holes: {int(((gen['emptiness'] >= HOLE_EMPTINESS) & (gen['enclosure'] >= HOLE_ENCLOSURE)).sum())}"]
        return patch, msg, (version or 0) + 1

    @app.callback(Output("table", "data"), Input("gen-version", "data"), Input("table-opts", "value"))
    def render_table(_version, opts):
        if not GEN:
            return []
        gen = filtered(GEN["df"], opts or [])
        if gen.empty:
            return []
        by, asc = GEN.get("sort", ("emptiness", False))  # edit mode: most plausible (smallest d_nn) first
        reps = gen.sort_values(by, ascending=asc).drop_duplicates("family")
        reps = reps.assign(n=reps["family"].map(gen["family"].value_counts()).to_numpy()).head(500)
        reps = reps.assign(rank=np.arange(1, len(reps) + 1))
        return reps[[c for c, _ in TABLE_COLS]].round(ROUND).to_dict("records")

    # ---- edit mode: compositions that land on a clicked spot / inside a drawn region
    @app.callback(Output("edit-help", "style"), Input("edit-mode", "value"))
    def on_edit_mode(mode):
        return ({"color": "#555", "fontSize": "13px", "marginBottom": "4px", "padding": "4px 8px",
                 "background": "#fdf0f8", "borderRadius": "6px"} if mode else {"display": "none"})

    @app.callback(Output("grid", "figure", allow_duplicate=True),
                  Output("gen-msg", "children", allow_duplicate=True),
                  Output("gen-version", "data", allow_duplicate=True), Output("table-opts", "value"),
                  Output("show-gen", "value", allow_duplicate=True),
                  Input("edit-click", "data"), Input("edit-region", "data"),
                  State("edit-mode", "value"), State("edit-n", "value"), State("gen-version", "data"),
                  State("sel-store", "data"), prevent_initial_call=True)
    def on_edit(click, region, mode, n_sol, version, store):
        if not mode:
            return (no_update,) * 5
        n_sol = int(n_sol or 300)
        rng = np.random.default_rng(int(version or 0))
        shapes = [outline(store[f"{i}-{j}"], n) for n, (i, j) in enumerate(PAIRS) if f"{i}-{j}" in (store or {})]
        if ctx.triggered_id == "edit-click":
            if not click:
                return (no_update,) * 5
            panel = 0 if click["xaxis"] == "x" else int(click["xaxis"][1:]) - 1
            T = np.array([[click["x"], click["y"]]])
            per_target = n_sol
            shapes.append(target_marker(panel, click["x"], click["y"]))
            where = f"the spot ({click['x']:.2f}, {click['y']:.2f})"
            kind = "click"
        else:
            if not region:
                return (no_update,) * 5
            i0, j0 = map(int, region["key"].split("-"))
            panel = PAIRS.index((i0, j0))
            per_target = 2
            T = sample_inside(region["geom"], max(1, n_sol // per_target), rng)
            if len(T) == 0:
                return no_update, "edit mode: the drawn region is empty", no_update, no_update, no_update
            shapes.append(outline(region["geom"], panel, "edit"))
            where = f"{len(T)} spots spread over the drawn region"
            kind = "region"
        i, j = PAIRS[panel]
        C, Z, resid, A, w = K.preimage(i, j, T, per_target, seed=int(version or 0))
        tol = 0.03  # standardized units (~0.5% of a panel's width)
        hit = resid <= tol
        patch = Patch()
        patch["layout"]["shapes"] = shapes
        head = f"edit mode · φ{i} × φ{j} · {where} · {len(C)} mixtures fitted · {int(hit.sum())} land on target"
        if not hit.any():
            GEN.clear()
            marker_patch(patch, GEN_TRACE, np.zeros((0, N_PHI)), [], color=[])
            return (patch, head + " — unreachable: too far from every real glass in this panel",
                    (version or 0) + 1, [], no_update)
        C, Z, A, w = C[hit], Z[hit], A[hit], w[hit]
        src = np.repeat(T, per_target, axis=0)[hit]  # where each solution sits in the drawn panel
        gen = K.describe(C, Z)
        gen["type"] = np.where(gen["d_nn_pct"] < 0.1, f"edit {kind} φ{i}×φ{j} (= a real glass)",
                               f"edit {kind} φ{i}×φ{j}")
        gen["parents"] = [" + ".join(f"{wk:.2f}·{K.ids[q]}" for q, wk in zip(qs, ws) if wk >= 0.005)
                          for qs, ws in zip(A, w)]
        gen["family"] = families(100 * C, gen["d_nn_pct"].to_numpy(), args.family_radius)
        GEN.clear()
        GEN.update(df=gen, C=C, Z=Z, sort=("d_nn_pct", True))
        near = gen["d_nn_pct"].to_numpy() <= 100 * args.dnn_max
        hover = [f"{c} · RI≈{r:.3f} · {s} · d_nn {d:.1f} mol%" for c, r, s, d in
                 zip(gen["composition"], gen["ri_interp"], gen["status"], gen["d_nn_pct"])]
        if kind == "region":  # color = position inside the drawn region, carried to every other panel
            colors = np.where(near, region_colors(src, region["geom"]), EDIT_FAR_COLOR)
        else:
            colors = np.where(near, EDIT_COLOR, EDIT_FAR_COLOR)
        marker_patch(patch, GEN_TRACE, Z, hover, color=colors.tolist())
        others = [k for k in range(N_PHI) if k not in (i - 1, j - 1)]
        spread = ", ".join(f"φ{k + 1} {np.percentile(Z[near, k], 5):.2f}…{np.percentile(Z[near, k], 95):.2f}"
                           for k in others) if near.sum() >= 2 else "—"
        msg = [head, html.Br(),
               html.B(f"{int(near.sum())} plausible (d_nn ≤ {100 * args.dnn_max:.1f} mol%) in "
                      f"{gen.loc[near, 'family'].nunique()} families"),
               f" · {int((~near).sum())} farther (gray) · spread in the other eigenfunctions (p5…p95): {spread}"]
        if kind == "region":
            msg += [html.Br(), f"Color = position inside the drawn region of φ{i} × φ{j} (left → right: hue, "
                               "bottom → top: dark → light). The same colors in the other panels show where each "
                               "part of the region goes."]
        return patch, msg, (version or 0) + 1, [], ["on"]

    # ---- typed compositions
    @app.callback(Output("grid", "figure", allow_duplicate=True), Output("query-table", "data"),
                  Output("query-msg", "children"),
                  Input("query-btn", "n_clicks"), Input("query-clear", "n_clicks"), State("query-text", "value"),
                  prevent_initial_call=True)
    def on_query(_show, _clear, text):
        patch = Patch()
        lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
        if ctx.triggered_id == "query-clear" or not lines:
            marker_patch(patch, QUERY_TRACE, np.zeros((0, N_PHI)), [], [])
            return patch, [], ""
        comps, errors = [], []
        for k, ln in enumerate(lines, start=1):
            try:
                comps.append(parse_composition(ln, K.oxides))
            except ValueError as e:
                errors.append(f"line {k}: {e}")
        if errors:
            return no_update, no_update, "; ".join(errors)
        C = np.array(comps)
        Z = K.embed(C)
        info = K.describe(C, Z)
        info["label"] = [str(k) for k in range(1, len(C) + 1)]
        hover = [f"#{lab} · {c} · RI≈{r:.3f} · {s}"
                 for lab, c, r, s in zip(info["label"], info["composition"], info["ri_interp"], info["status"])]
        marker_patch(patch, QUERY_TRACE, Z, hover, info["label"].tolist())
        return patch, info[[c for c, _ in QUERY_COLS]].round(ROUND).to_dict("records"), ""

    @app.callback(Output("grid", "figure", allow_duplicate=True),
                  Input("show-gen", "value"), prevent_initial_call=True)
    def on_toggle_candidates(value):
        patch = Patch()
        for n in range(len(PAIRS)):
            patch["data"][trace_index(n, GEN_TRACE)]["visible"] = bool(value)
        return patch

    @app.callback(Output("export-msg", "children"), Input("export", "n_clicks"), State("table-opts", "value"),
                  prevent_initial_call=True)
    def on_export(_clicks, opts):
        if not GEN:
            return "generate candidates first"
        gen = filtered(GEN["df"], opts or [])
        if gen.empty:
            return "no candidates with the current filters"
        comp = pd.DataFrame(100 * GEN["C"][gen.index], columns=K.oxides).round(3)
        comp = comp.loc[:, (comp > 0).any()]
        Z = pd.DataFrame(GEN["Z"][gen.index], columns=PHIS)
        out = pd.concat([gen.drop(columns=["composition"]).reset_index(drop=True), Z, comp], axis=1)
        out = out.sort_values(["family", "emptiness"], ascending=[True, False])
        path = DATA_OUT / f"candidates_{datetime.now():%Y%m%d_%H%M%S}.csv"
        out.to_csv(path, index=False)
        return f"wrote {path} ({len(out)} candidates, {out['family'].nunique()} families)"

    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="m10000_s1")
    ap.add_argument("--gen-n", type=int, default=20_000, help="mixtures generated per click")
    ap.add_argument("--dnn-max", type=float, default=0.073,
                    help="max distance to nearest real glass, in fractions (0.073 = 7.3 mol%%, p90 of real spacing)")
    ap.add_argument("--family-radius", type=float, default=1.0, help="mol%% radius for grouping similar candidates")
    ap.add_argument("--ri-disagree", type=float, default=0.03, help="flag if |RI interp - RI hgb| exceeds this")
    ap.add_argument("--port", type=int, default=8050)
    args = ap.parse_args()
    ensure_dirs()
    K = Context(args)
    build_app(K).run(debug=False, port=args.port)


if __name__ == "__main__":
    main()
