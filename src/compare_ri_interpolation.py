"""Can RI be predicted by interpolating on the learned manifold? Compare against a plain regressor.

Same 5-fold split for all methods (RI restricted to [1.3, 3.0]):
  hgb_composition   - gradient boosting on the 73 oxide fractions (reference, = train_ri_model.py)
  linear_phi10      - linear regression on the 10 spectral coordinates (global smooth function on the manifold)
  hgb_phi10         - gradient boosting on the 10 spectral coordinates
  knn_composition   - distance-weighted mean RI of the k nearest glasses in composition (local interpolation)
  harmonic_graph    - harmonic extension on the kNN graph (Zhu, Ghahramani & Lafferty 2003): unknown RI
                      minimizes sum_ij w_ij (f_i - f_j)^2 with known RI fixed -> interpolation along the graph

Run from the project root:
  uv run src/compare_ri_interpolation.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors
from tqdm import tqdm

sys.path.insert(0, str(Path("src")))
from spectral_embedding import DATA, DATA_OUT  # noqa: E402

RI_RANGE = (1.3, 3.0)
K = 10


def knn_graph(X: np.ndarray, k: int) -> sp.csr_matrix:
    dist, idx = NearestNeighbors(n_neighbors=k + 1).fit(X).kneighbors(X)
    dist, idx = dist[:, 1:], idx[:, 1:]
    sigma = np.median(dist[:, -1])  # global bandwidth (local sigma broke the graph before)
    w = np.exp(-(dist**2) / (2 * sigma**2))
    n = len(X)
    W = sp.csr_matrix((w.ravel(), (np.repeat(np.arange(n), k), idx.ravel())), shape=(n, n))
    return W.maximum(W.T).tocsr()


def harmonic(W: sp.csr_matrix, y: np.ndarray, known: np.ndarray) -> np.ndarray:
    """Solve L_uu f_u = W_ul y_l  (L = D - W)."""
    u, l = np.flatnonzero(~known), np.flatnonzero(known)
    d = np.asarray(W.sum(axis=1)).ravel()
    L_uu = sp.diags(d[u]) - W[u][:, u]
    rhs = W[u][:, l] @ y[l]
    return spsolve(L_uu.tocsc(), rhs)


def main():
    df = pd.read_parquet(DATA)
    emb = pd.read_parquet(DATA_OUT / "spectral_embedding_m10000_s1.parquet")
    y_all = df["refractive_index"].to_numpy()
    keep = (y_all >= RI_RANGE[0]) & (y_all <= RI_RANGE[1])
    X = df.drop(columns="refractive_index").to_numpy()[keep] / 100.0
    P = emb[[f"phi{k}" for k in range(1, 11)]].to_numpy()[keep]
    P = (P - P.mean(0)) / P.std(0)
    y = y_all[keep]
    W = knn_graph(X, K)

    preds = {m: np.empty_like(y) for m in
             ["hgb_composition", "linear_phi10", "hgb_phi10", "knn_composition", "harmonic_graph"]}
    folds = list(KFold(5, shuffle=True, random_state=0).split(X))
    for tr, te in tqdm(folds, desc="folds"):
        hgb = dict(max_iter=800, learning_rate=0.08, max_leaf_nodes=63, l2_regularization=1e-3, random_state=0)
        preds["hgb_composition"][te] = HistGradientBoostingRegressor(**hgb).fit(X[tr], y[tr]).predict(X[te])
        preds["linear_phi10"][te] = LinearRegression().fit(P[tr], y[tr]).predict(P[te])
        preds["hgb_phi10"][te] = HistGradientBoostingRegressor(**hgb).fit(P[tr], y[tr]).predict(P[te])
        preds["knn_composition"][te] = KNeighborsRegressor(K, weights="distance").fit(X[tr], y[tr]).predict(X[te])
        known = np.zeros(len(y), bool)
        known[tr] = True
        preds["harmonic_graph"][te] = harmonic(W, y, known)

    rows = []
    for m, p in preds.items():
        e = np.abs(p - y)
        rows.append({"method": m, "MAE": e.mean(), "median AE": np.median(e), "p95 AE": np.percentile(e, 95),
                     "R²": 1 - ((p - y) ** 2).sum() / ((y - y.mean()) ** 2).sum()})
    print(pd.DataFrame(rows).set_index("method").round(4).to_string())


if __name__ == "__main__":
    main()
