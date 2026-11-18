"""RI-weighted spectral embedding: the NJW affinity of spectral_embedding.py times a refractive-index factor,

    A_ij = exp(-|x_i - x_j|^2 / 2 sigma^2) * exp(-(n_i - n_j)^2 / 2 sigma_n^2),    A_ii = 0,

with n = refractive index. (In the form exp(-(n_i - n_j)^2 / s), s = 2 sigma_n^2.)
  sigma   = --sigma-scale    * median pairwise composition distance (as in spectral_embedding.py)
  sigma_n = --ri-sigma-scale * median |n_i - n_j| over pairs of landmarks
Two glasses are now strongly connected only if they are close in composition AND in RI, so the eigenfunctions
separate regions of different RI as well as different chemistry.

In the RI factor, RI is clipped to robust percentiles of the landmarks' RI (--ri-clip-pct: p0.5-p99.5 by default,
about 1.46-2.40). A glass whose RI is far from every other glass is almost disconnected by the RI factor, and
the first eigenvectors become indicators of such tiny clusters. Without this clip, 5 glasses at RI ~ 3.0 (likely
DB errors; p99 is 2.30) had degree ~4 vs a median of ~3900 and were all of phi1 (effective support 5 out of
10000, with 99% of the glasses squeezed into one point). Clipped glasses are treated as "as high (low) as the
p99.5 (p0.5) glass" in the kernel; their stored refractive_index is untouched.

Nystrom extension needs RI(x) for any new composition x. Real glasses use their measured RI. Candidates and typed
compositions in the apps use the kNN estimate ("RI interp"), so they are placed where their *predicted* RI puts them.

Same landmarks as spectral_embedding.py for the same --m and --seed, so the two maps can be compared.

Run from the project root:
  uv run src/spectral_embedding_ri.py [--m 10000] [--sigma-scale 1.0] [--ri-sigma-scale 1.0] [--n-components 10]
Writes the same outputs as spectral_embedding.py, tagged m<m>_s<sigma-scale>_ri<ri-sigma-scale>. Explore with
  uv run src/app.py --tag m10000_s1_ri1        or        uv run src/app_3d.py --tag m10000_s1_ri1
"""

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.linalg import eigh
from sklearn.metrics.pairwise import euclidean_distances
from tqdm import tqdm

sys.path.insert(0, str(Path("src")))
from spectral_embedding import (DATA, DATA_OUT, MODELS, VIZ, SpectralEmbedding, effective_support,  # noqa: E402
                                ensure_dirs, hover_text)
from train_ri_model import RI_RANGE  # noqa: E402


@dataclass
class RISpectralEmbedding(SpectralEmbedding):
    """SpectralEmbedding whose kernel also multiplies by exp(-(n_i - n_j)^2 / 2 sigma_n^2). fit/transform take RI."""

    ri_sigma_scale: float = 1.0
    ri_sigma: float | None = None  # fixed RI bandwidth; if None, ri_sigma_scale * median |n_i - n_j|
    ri_clip_pct: float = 0.5  # RI clipped to [p, 100 - p] percentiles of the landmarks (see module docstring)

    def clip_ri(self, ri) -> np.ndarray:
        return np.clip(np.asarray(ri, dtype=float), *self.ri_clip_)

    def ri_factor(self, ri_a: np.ndarray, ri_b: np.ndarray) -> np.ndarray:
        return np.exp(-(ri_a[:, None] - ri_b[None, :]) ** 2 / (2 * self.ri_sigma_**2))

    def fit(self, X: np.ndarray, ri: np.ndarray) -> "RISpectralEmbedding":
        ri = np.clip(np.asarray(ri, dtype=float), *RI_RANGE)  # drop the absurd values (0.56 ... 301) first
        self.ri_clip_ = tuple(np.percentile(ri, [self.ri_clip_pct, 100 - self.ri_clip_pct]))
        self.X_, self.ri_ = X, self.clip_ri(ri)
        m = X.shape[0]
        pbar = tqdm(total=3, desc=f"fit RI-weighted (m={m})")
        pbar.set_postfix_str("distances")
        dist = euclidean_distances(X)
        self.sigma_ = self.sigma if self.sigma is not None else (
            self.sigma_scale * np.median(dist[np.triu_indices_from(dist, k=1)]))
        if self.ri_sigma is not None:
            self.ri_sigma_ = self.ri_sigma
        else:  # median over 2M random pairs: indistinguishable from all m^2/2, without an m x m temporary
            a, b = np.random.default_rng(0).integers(m, size=(2, 2_000_000))
            self.ri_sigma_ = self.ri_sigma_scale * np.median(np.abs(self.ri_[a[a != b]] - self.ri_[b[a != b]]))
        pbar.update()
        pbar.set_postfix_str("kernel + normalization")
        A = np.exp(-(dist**2) / (2 * self.sigma_**2))
        del dist
        for s in range(0, m, 2000):  # RI factor by row blocks (a full m x m factor would double the memory)
            A[s:s + 2000] *= self.ri_factor(self.ri_[s:s + 2000], self.ri_)
        np.fill_diagonal(A, 0.0)
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
        self.eigenvalues_ = 1.0 - self.mu_
        self.d_ = d
        self.embedding_ = self.V_[:, 1:] * s[:, None]
        return self

    def transform(self, X_new: np.ndarray, ri_new: np.ndarray | None = None, chunk: int = 4000,
                  return_density: bool = False):
        """Nystrom as in SpectralEmbedding.transform, with the RI factor in the kernel. ri_new: RI of X_new."""
        if ri_new is None:
            raise ValueError("the RI-weighted embedding needs the RI of the new compositions (ri_new)")
        ri_new = self.clip_ri(ri_new)
        W = self.V_[:, 1:] / np.sqrt(self.d_)[:, None] / self.mu_[1:][None, :]
        out = np.empty((X_new.shape[0], self.n_components))
        dens = np.empty(X_new.shape[0])
        for a in tqdm(range(0, X_new.shape[0], chunk), desc=f"Nyström RI (n={X_new.shape[0]})"):
            dist2 = euclidean_distances(X_new[a:a + chunk], self.X_, squared=True)
            K = np.exp(-dist2 / (2 * self.sigma_**2)) * self.ri_factor(ri_new[a:a + chunk], self.ri_)
            K[dist2 < 1e-12] = 0.0  # a landmark does not count itself (A_ii = 0)
            dx = K.sum(axis=1)
            out[a:a + chunk] = (K @ W) / dx[:, None]
            dens[a:a + chunk] = dx
        return (out, dens) if return_density else out

    def save(self, path: Path):
        np.savez(path, X=self.X_, sigma=self.sigma_, mu=self.mu_, V=self.V_, d=self.d_,
                 n_components=self.n_components, ri=self.ri_, ri_sigma=self.ri_sigma_, ri_clip=self.ri_clip_)

    @classmethod
    def load(cls, path: Path) -> "RISpectralEmbedding":
        z = np.load(path)
        se = cls(n_components=int(z["n_components"]), sigma=float(z["sigma"]), ri_sigma=float(z["ri_sigma"]))
        se.X_, se.sigma_, se.mu_, se.V_, se.d_ = z["X"], float(z["sigma"]), z["mu"], z["V"], z["d"]
        se.ri_, se.ri_sigma_, se.ri_clip_ = z["ri"], float(z["ri_sigma"]), tuple(z["ri_clip"])
        se.eigenvalues_ = 1.0 - se.mu_
        se.embedding_ = se.V_[:, 1:] / np.sqrt(se.d_)[:, None]
        return se


def load_model(path: Path) -> SpectralEmbedding:
    """Plain or RI-weighted model, whichever was saved at `path`."""
    with np.load(path) as z:
        ri_weighted = "ri_sigma" in z.files
    return (RISpectralEmbedding if ri_weighted else SpectralEmbedding).load(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=10_000, help="landmarks for the dense kernel")
    ap.add_argument("--sigma-scale", type=float, default=1.0, help="sigma = scale * median pairwise distance")
    ap.add_argument("--ri-sigma-scale", type=float, default=1.0,
                    help="sigma_n = scale * median |RI_i - RI_j|; smaller = RI matters more")
    ap.add_argument("--ri-clip-pct", type=float, default=0.5,
                    help="clip RI to [p, 100-p] percentiles in the kernel; 0 = only the [1.3, 3.0] sanity range")
    ap.add_argument("--n-components", type=int, default=10, help="non-trivial eigenvectors to keep")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    ensure_dirs()

    df = pd.read_parquet(DATA)
    comps = df.drop(columns="refractive_index")
    ri = df["refractive_index"].to_numpy()
    ids = df.index.to_numpy()
    X = comps.to_numpy() / 100.0
    n = X.shape[0]

    rng = np.random.default_rng(args.seed)  # same draw as spectral_embedding.py
    land = rng.choice(n, size=min(args.m, n), replace=False)
    is_landmark = np.zeros(n, dtype=bool)
    is_landmark[land] = True

    se = RISpectralEmbedding(sigma_scale=args.sigma_scale, ri_sigma_scale=args.ri_sigma_scale,
                             ri_clip_pct=args.ri_clip_pct, n_components=args.n_components).fit(X[land], ri[land])
    print(f"{n} glasses, {len(land)} landmarks, sigma = {se.sigma_:.4f}, sigma_n (RI) = {se.ri_sigma_:.4f}"
          f"  [exp(-dn^2/s) with s = {2 * se.ri_sigma_**2:.4f}], RI clipped to "
          f"[{se.ri_clip_[0]:.3f}, {se.ri_clip_[1]:.3f}] in the kernel")
    print("smallest eigenvalues of L:", np.round(se.eigenvalues_, 5))
    support = effective_support(se.embedding_)
    print("effective support of phi_1..: ", np.round(support).astype(int), f"(out of {len(land)})")
    print(f"min degree {se.d_.min():.3g} (median {np.median(se.d_):.3g})")
    if (support[:5] < 0.01 * len(land)).any():
        print("WARNING: some of phi1..phi5 live on < 1% of the landmarks (localized on a few isolated glasses); "
              "try a larger --ri-sigma-scale or --ri-clip-pct")

    emb = se.transform(X, ri)
    emb[land] = se.embedding_  # exact values for landmarks
    tag = f"m{len(land)}_s{args.sigma_scale:g}_ri{args.ri_sigma_scale:g}"
    se.save(MODELS / f"model_{tag}.npz")
    np.save(MODELS / f"landmarks_{tag}.npy", land)

    cols = {f"phi{k + 1}": emb[:, k] for k in range(emb.shape[1])}
    out = pd.DataFrame({"ID": ids, "refractive_index": ri, "landmark": is_landmark, **cols})
    out.to_parquet(DATA_OUT / f"spectral_embedding_{tag}.parquet", index=False)
    ok = (ri >= RI_RANGE[0]) & (ri <= RI_RANGE[1])
    rho = out.loc[ok, list(cols)[:5]].corrwith(out.loc[ok, "refractive_index"], method="spearman")
    print("Spearman correlation with RI, phi1..phi5:", np.round(rho.to_numpy(), 3))

    cmin, cmax = np.percentile(ri, [1, 99])
    text = hover_text(ids, comps, ri, is_landmark)
    marker = dict(color=np.clip(ri, cmin, cmax), colorscale="Viridis", cmin=cmin, cmax=cmax,
                  colorbar=dict(title="RI"), opacity=0.8)
    title = (f"RI-weighted spectral embedding (σ={se.sigma_:.3f}, σ_n={se.ri_sigma_:.3f}) — {len(land)} landmarks, "
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

    print(f"wrote {VIZ}/spectral_{{2d,3d}}_{tag}.html, {DATA_OUT}/spectral_embedding_{tag}.parquet, "
          f"{MODELS}/model_{tag}.npz")


if __name__ == "__main__":
    main()
