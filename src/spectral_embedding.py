"""Classic spectral embedding (Ng-Jordan-Weiss affinity) + Nystrom extension, colored by refractive index.

Pipeline:
  1. compositions (73 oxides, % -> fractions)
  2. random subsample of m glasses ("landmarks")
  3. dense Gaussian affinity  A_ij = exp(-|x_i - x_j|^2 / 2 sigma^2),  A_ii = 0,
     sigma = median pairwise distance (times --sigma-scale)
  4. normalized Laplacian  L = I - D^{-1/2} A D^{-1/2}  -> smallest eigenpairs
     (= largest of M = D^{-1/2} A D^{-1/2})
  5. Nystrom extension: embed every glass (and later any new composition) via `SpectralEmbedding.transform`
  6. interactive Plotly HTML (2D / 3D, WebGL), colored by RI

Run from the project root:
  uv run src/spectral_embedding.py [--m 10000] [--sigma-scale 1.0] [--n-components 10] [--alpha 0]
(alpha != 0 adds _a<alpha> to the tag, e.g. m10000_s1_a1; see SpectralEmbedding for what alpha does)
Outputs: outputs/viz (html), outputs/data (parquet), outputs/models (npz)
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.linalg import eigh
from sklearn.metrics.pairwise import euclidean_distances
from tqdm import tqdm

DATA = Path("data/refractive_index.parquet")
OUT = Path("outputs")
VIZ = OUT / "viz"        # figures / interactive plots (html, png)
DATA_OUT = OUT / "data"  # derived tables (parquet)
MODELS = OUT / "models"  # fitted models (npz, npy)
LOGS = OUT / "logs"


def ensure_dirs():
    for d in (VIZ, DATA_OUT, MODELS, LOGS):
        d.mkdir(parents=True, exist_ok=True)


@dataclass
class SpectralEmbedding:
    """NJW spectral embedding fitted on landmarks, with out-of-sample (Nystrom) extension.

    Coordinates are phi_k = D^{-1/2} v_k for the non-trivial eigenvectors v_k of M
    (the trivial one, v_0 ∝ sqrt(d), carries only degree information and is dropped).

    alpha (Coifman-Lafon alpha-normalization): with q_i = sum_j A_ij (a kernel density estimate), the kernel is
    first reweighted A_ij <- A_ij / (q_i^alpha q_j^alpha), then normalized as above. In the small-sigma limit the
    random walk's generator is  Delta f + 2(1 - alpha) grad(log p) . grad f  (p = sampling density):
      alpha = 0   : graph Laplacian / NJW (the default; previous results), density pulls the walk to dense regions
      alpha = 1/2 : Fokker-Planck diffusion with equilibrium p
      alpha = 1   : Laplace-Beltrami, geometry only: how many glasses sit in a region no longer shapes the map
    """

    sigma_scale: float = 1.0
    n_components: int = 10
    sigma: float | None = None  # fixed bandwidth; if None, sigma_scale * median pairwise distance
    alpha: float = 0.0

    def fit(self, X: np.ndarray) -> "SpectralEmbedding":
        self.X_ = X
        m = X.shape[0]
        pbar = tqdm(total=3, desc=f"fit (m={m}, alpha={self.alpha:g})")
        pbar.set_postfix_str("distances")
        dist = euclidean_distances(X)
        self.sigma_ = self.sigma if self.sigma is not None else (
            self.sigma_scale * np.median(dist[np.triu_indices_from(dist, k=1)]))
        pbar.update()
        pbar.set_postfix_str("kernel + normalization")
        A = np.exp(-(dist**2) / (2 * self.sigma_**2))
        del dist
        np.fill_diagonal(A, 0.0)
        return self._spectral(A, pbar)

    def _spectral(self, A: np.ndarray, pbar) -> "SpectralEmbedding":
        """Kernel A (zero diagonal; modified in place) -> alpha-reweighting, normalization, eigenpairs."""
        m = A.shape[0]
        self.q_ = A.sum(axis=1)  # density estimate at the landmarks (= the degrees when alpha = 0)
        if self.alpha:
            w = self.q_ ** -self.alpha
            A *= w[:, None]
            A *= w[None, :]
        d = A.sum(axis=1)
        s = 1.0 / np.sqrt(d)
        M = A * s[:, None] * s[None, :]
        del A
        pbar.update()
        pbar.set_postfix_str("eigh (slowest step)")
        k = self.n_components + 1
        mu, V = eigh(M, subset_by_index=[m - k, m - 1])  # largest eigenvalues of M
        pbar.update()
        pbar.close()
        order = np.argsort(-mu)
        self.mu_, self.V_ = mu[order], V[:, order]
        self.eigenvalues_ = 1.0 - self.mu_  # eigenvalues of L
        self.d_ = d
        self.embedding_ = self.V_[:, 1:] * s[:, None]  # drop trivial, phi = D^{-1/2} v
        return self

    def transform(self, X_new: np.ndarray, chunk: int = 4000, return_density: bool = False):
        """Nystrom: v_k(x) = (1/mu_k) sum_i a(x,x_i) v_k(x_i) / sqrt(d(x) d_i);  phi_k(x) = v_k(x)/sqrt(d(x)).

        With return_density=True also returns q(x) = sum_i a(x, x_i) (kernel density w.r.t. the landmarks).
        """
        return self._nystrom(X_new, chunk, return_density)

    def _nystrom(self, X_new: np.ndarray, chunk: int, return_density: bool, factor=None):
        """factor(a, b): optional extra kernel factor for rows a:b of X_new vs the landmarks (used by subclasses).

        With alpha > 0 the reweighted kernel is a(x, x_i) / (q(x)^alpha q_i^alpha). The q(x)^alpha cancels between
        the kernel and d(x), so only the landmarks' q_i^alpha is needed."""
        W = self.V_[:, 1:] / np.sqrt(self.d_)[:, None] / self.mu_[1:][None, :]
        w = self.q_ ** -self.alpha if self.alpha else None
        out = np.empty((X_new.shape[0], self.n_components))
        dens = np.empty(X_new.shape[0])
        for a in tqdm(range(0, X_new.shape[0], chunk), desc=f"Nyström (n={X_new.shape[0]})"):
            dist2 = euclidean_distances(X_new[a:a + chunk], self.X_, squared=True)
            K = np.exp(-dist2 / (2 * self.sigma_**2))
            if factor is not None:
                K *= factor(a, a + chunk)
            # a landmark does not count itself (A_ii = 0); tolerance because the GEMM-based distance isn't exactly 0
            K[dist2 < 1e-12] = 0.0
            dens[a:a + chunk] = K.sum(axis=1)
            if w is not None:
                K *= w[None, :]
            out[a:a + chunk] = (K @ W) / K.sum(axis=1)[:, None]
        return (out, dens) if return_density else out

    def save(self, path: Path, **extra):
        np.savez(path, X=self.X_, sigma=self.sigma_, mu=self.mu_, V=self.V_, d=self.d_,
                 n_components=self.n_components, alpha=self.alpha, q=self.q_, **extra)

    @classmethod
    def load(cls, path: Path) -> "SpectralEmbedding":
        z = np.load(path)
        alpha = float(z["alpha"]) if "alpha" in z.files else 0.0  # models saved before alpha existed: 0
        se = cls(n_components=int(z["n_components"]), sigma=float(z["sigma"]), alpha=alpha)
        se.X_, se.sigma_, se.mu_, se.V_, se.d_ = z["X"], float(z["sigma"]), z["mu"], z["V"], z["d"]
        se.q_ = z["q"] if "q" in z.files else se.d_  # equal when alpha = 0
        se.eigenvalues_ = 1.0 - se.mu_
        se.embedding_ = se.V_[:, 1:] / np.sqrt(se.d_)[:, None]
        return se


def effective_support(phi: np.ndarray) -> np.ndarray:
    """Participation ratio per column: ~n if spread over all points, small if localized on few."""
    p = phi / np.linalg.norm(phi, axis=0)
    return 1.0 / (p**4).sum(axis=0)


def hover_text(ids, comps: pd.DataFrame, ri: np.ndarray, is_landmark: np.ndarray, top: int = 4) -> list[str]:
    M = comps.to_numpy()
    names = np.array(comps.columns)
    order = np.argsort(-M, axis=1)[:, :top]
    out = []
    for i in range(M.shape[0]):
        parts = [f"{names[j].upper()} {M[i, j]:.1f}%" for j in order[i] if M[i, j] > 0]
        tag = "landmark" if is_landmark[i] else "Nyström"
        out.append(f"<b>RI = {ri[i]:.4f}</b><br>{ids[i]} ({tag})<br>" + "<br>".join(parts))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=10_000, help="landmarks for the dense kernel")
    ap.add_argument("--sigma-scale", type=float, default=1.0, help="sigma = scale * median pairwise distance")
    ap.add_argument("--n-components", type=int, default=10, help="non-trivial eigenvectors to keep")
    ap.add_argument("--alpha", type=float, default=0.0,
                    help="Coifman-Lafon normalization: 0 = NJW (default), 1 = remove the sampling density")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    ensure_dirs()

    df = pd.read_parquet(DATA)
    comps = df.drop(columns="refractive_index")
    ri = df["refractive_index"].to_numpy()
    ids = df.index.to_numpy()
    X = comps.to_numpy() / 100.0
    n = X.shape[0]

    rng = np.random.default_rng(args.seed)
    land = rng.choice(n, size=min(args.m, n), replace=False)
    is_landmark = np.zeros(n, dtype=bool)
    is_landmark[land] = True

    se = SpectralEmbedding(sigma_scale=args.sigma_scale, n_components=args.n_components,
                           alpha=args.alpha).fit(X[land])
    print(f"{n} glasses, {len(land)} landmarks, sigma = {se.sigma_:.4f}, alpha = {args.alpha:g}")
    print("smallest eigenvalues of L:", np.round(se.eigenvalues_, 5))
    print("effective support of phi_1..: ", np.round(effective_support(se.embedding_)).astype(int),
          f"(out of {len(land)})")

    emb = se.transform(X)
    emb[land] = se.embedding_  # exact values for landmarks
    tag = f"m{len(land)}_s{args.sigma_scale:g}" + (f"_a{args.alpha:g}" if args.alpha else "")  # alpha=0: old tag
    se.save(MODELS / f"model_{tag}.npz")
    np.save(MODELS / f"landmarks_{tag}.npy", land)

    cols = {f"phi{k + 1}": emb[:, k] for k in range(emb.shape[1])}
    pd.DataFrame({"ID": ids, "refractive_index": ri, "landmark": is_landmark, **cols}).to_parquet(
        DATA_OUT / f"spectral_embedding_{tag}.parquet", index=False)

    # color clipped to p1-p99: RI has absurd outliers (min 0.56, max 301)
    cmin, cmax = np.percentile(ri, [1, 99])
    text = hover_text(ids, comps, ri, is_landmark)
    marker = dict(color=np.clip(ri, cmin, cmax), colorscale="Viridis", cmin=cmin, cmax=cmax,
                  colorbar=dict(title="RI"), opacity=0.8)
    title = (f"Spectral embedding (Gaussian σ={se.sigma_:.3f}, α={args.alpha:g}) — {len(land)} landmarks, "
             f"{n} glasses via Nyström — color = RI (clipped p1–p99)")

    fig2 = go.Figure(go.Scattergl(x=emb[:, 0], y=emb[:, 1], mode="markers", text=text, hoverinfo="text",
                                  marker=dict(size=4, **marker)))
    fig2.update_layout(title=title, xaxis_title="φ₁", yaxis_title="φ₂", template="plotly_white", height=800)
    fig2.write_html(VIZ / f"spectral_2d_{tag}.html", include_plotlyjs=True)

    fig3 = go.Figure(go.Scatter3d(x=emb[:, 0], y=emb[:, 1], z=emb[:, 2], mode="markers", text=text,
                                  hoverinfo="text", marker=dict(size=2, **marker)))
    fig3.update_layout(title=title, template="plotly_white", height=850,
                       scene=dict(xaxis_title="φ₁", yaxis_title="φ₂", zaxis_title="φ₃"))
    fig3.write_html(VIZ / f"spectral_3d_{tag}.html", include_plotlyjs=True)

    print(f"wrote {VIZ}/spectral_{{2d,3d}}_{tag}.html, {DATA_OUT}/spectral_embedding_{tag}.parquet, {MODELS}/model_{tag}.npz")


if __name__ == "__main__":
    main()
