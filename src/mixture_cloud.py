"""Mixture cloud: which empty regions of the spectral map can be reached by mixing real glasses?

Convex mixtures of valid compositions are valid compositions (>= 0, sum 100%). We sample mixtures,
embed them with the Nystrom extension and see where they land compared with the real glasses.

Mixture types:
  pair     - glass i + one of its k nearest neighbors in composition space, w ~ U(0, 1)
  triplet  - glass i + two of its neighbors, w ~ Dirichlet(1, 1, 1)
  random   - two random glasses (bolder: mixes unrelated chemistries), w ~ U(0, 1)

Each mixture stores its parents and weights (composition = sum_a w_a x_a), its phi coordinates,
the kernel density d(x) and the distance to the nearest real glass (plausibility).

Run from the project root (needs outputs/models/model_<tag>.npz from src/spectral_embedding.py):
  uv run src/mixture_cloud.py [--tag m10000_s1] [--n 100000] [--knn 10]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, str(Path("src")))
from spectral_embedding import DATA, DATA_OUT, MODELS, VIZ, SpectralEmbedding, ensure_dirs  # noqa: E402

TYPES = ["pair", "triplet", "random"]


def sample_mixtures(n: int, nbrs: np.ndarray, rng: np.random.Generator) -> pd.DataFrame:
    N, k = nbrs.shape
    parts = []
    # pairs of composition neighbors
    i = rng.integers(N, size=n)
    j = nbrs[i, rng.integers(k, size=n)]
    w = rng.random(n)
    parts.append(pd.DataFrame({"type": "pair", "p1": i, "p2": j, "p3": -1, "w1": w, "w2": 1 - w, "w3": 0.0}))
    # triplets: i + two distinct neighbors
    i = rng.integers(N, size=n)
    a = rng.integers(k, size=n)
    b = (a + rng.integers(1, k, size=n)) % k
    W = rng.dirichlet(np.ones(3), size=n)
    parts.append(pd.DataFrame({"type": "triplet", "p1": i, "p2": nbrs[i, a], "p3": nbrs[i, b],
                               "w1": W[:, 0], "w2": W[:, 1], "w3": W[:, 2]}))
    # random pairs
    i, j = rng.integers(N, size=n), rng.integers(N, size=n)
    w = rng.random(n)
    parts.append(pd.DataFrame({"type": "random", "p1": i, "p2": j, "p3": -1, "w1": w, "w2": 1 - w, "w3": 0.0}))
    return pd.concat(parts, ignore_index=True)


def compositions(mix: pd.DataFrame, X: np.ndarray) -> np.ndarray:
    C = mix["w1"].to_numpy()[:, None] * X[mix["p1"]] + mix["w2"].to_numpy()[:, None] * X[mix["p2"]]
    has3 = mix["p3"].to_numpy() >= 0
    C[has3] += mix["w3"].to_numpy()[has3, None] * X[mix["p3"].to_numpy()[has3]]
    return C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="m10000_s1", help="which fitted model in outputs/models/ to use")
    ap.add_argument("--n", type=int, default=100_000, help="mixtures per type")
    ap.add_argument("--knn", type=int, default=10, help="composition neighbors used for pair/triplet")
    ap.add_argument("--bins", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ensure_dirs()
    se = SpectralEmbedding.load(MODELS / f"model_{args.tag}.npz")
    real = pd.read_parquet(DATA_OUT / f"spectral_embedding_{args.tag}.parquet")
    df = pd.read_parquet(DATA)
    X = df.drop(columns="refractive_index").to_numpy() / 100.0

    print(f"kNN (k={args.knn}) in composition space over {len(X)} glasses")
    nn = NearestNeighbors(n_neighbors=args.knn + 1).fit(X)
    nbrs = nn.kneighbors(X, return_distance=False)[:, 1:]

    rng = np.random.default_rng(args.seed)
    mix = sample_mixtures(args.n, nbrs, rng)
    C = compositions(mix, X)

    phi, dens = se.transform(C, return_density=True)
    dnn = nn.kneighbors(C, n_neighbors=1)[0][:, 0]
    for k in range(phi.shape[1]):
        mix[f"phi{k + 1}"] = phi[:, k]
    mix["density"], mix["d_nn"] = dens, dnn
    out = DATA_OUT / f"mixture_cloud_{args.tag}.parquet"
    mix.to_parquet(out, index=False)

    # real-glass reference for plausibility
    _, dens_real = se.transform(X[:: 10], return_density=True)
    print(f"\ndensity of real glasses: p5={np.percentile(dens_real, 5):.1f}  median={np.median(dens_real):.1f}")
    print(mix.groupby("type")[["density", "d_nn"]].describe(percentiles=[0.05, 0.5, 0.95])
          .round(3).T.to_string())

    # ---- occupancy on (phi1, phi2): where do mixtures land that real glasses don't?
    xr, yr = real["phi1"].to_numpy(), real["phi2"].to_numpy()
    lo_x, hi_x = np.percentile(np.r_[xr, mix["phi1"]], [0.05, 99.95])
    lo_y, hi_y = np.percentile(np.r_[yr, mix["phi2"]], [0.05, 99.95])
    ex, ey = np.linspace(lo_x, hi_x, args.bins + 1), np.linspace(lo_y, hi_y, args.bins + 1)
    H_real, *_ = np.histogram2d(xr, yr, bins=[ex, ey])
    occupied = H_real > 0
    print(f"\n(phi1, phi2) grid {args.bins}x{args.bins}: {occupied.sum()} cells with real glasses")

    fig = make_subplots(rows=1, cols=4, shared_yaxes=True, horizontal_spacing=0.02,
                        subplot_titles=["real glasses"] + [f"{t}: new cells (no real glass)" for t in TYPES])
    cx, cy = (ex[:-1] + ex[1:]) / 2, (ey[:-1] + ey[1:]) / 2
    fig.add_trace(go.Heatmap(x=cx, y=cy, z=np.where(occupied, np.log10(np.maximum(H_real, 1)), np.nan).T,
                             colorscale="Greys", showscale=False, hovertemplate="log10 n=%{z:.2f}<extra></extra>"),
                  1, 1)
    for c, t in enumerate(TYPES, start=2):
        m = mix[mix["type"] == t]
        H, *_ = np.histogram2d(m["phi1"], m["phi2"], bins=[ex, ey])
        new = (H > 0) & ~occupied
        print(f"  {t:8s}: lands in {((H > 0) & occupied).sum()} occupied cells, "
              f"{new.sum()} NEW cells ({H[new].sum():.0f} mixtures = {H[new].sum() / len(m):.2%})")
        fig.add_trace(go.Heatmap(x=cx, y=cy, z=np.where(occupied, 0.0, np.nan).T,
                                 colorscale=[[0, "#dddddd"], [1, "#dddddd"]], showscale=False, hoverinfo="skip"),
                      1, c)
        fig.add_trace(go.Heatmap(x=cx, y=cy, z=np.where(new, np.log10(np.maximum(H, 1)), np.nan).T, colorscale="Inferno_r",
                                 showscale=(c == 4), colorbar=dict(title="log10 mixtures"),
                                 hovertemplate="log10 n=%{z:.2f}<extra></extra>"), 1, c)
    fig.update_layout(title=f"Mixture cloud on (φ₁, φ₂) — gray: cells with real glasses; color: mixtures "
                            f"landing where no real glass is ({args.n} per type)",
                      template="plotly_white", height=550)
    fig.update_xaxes(title_text="φ₁")
    fig.update_yaxes(title_text="φ₂", col=1)
    html = VIZ / f"mixture_cloud_{args.tag}.html"
    fig.write_html(html, include_plotlyjs=True)
    print(f"\nwrote {out} and {html}")


if __name__ == "__main__":
    main()
