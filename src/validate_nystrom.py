"""How well does the Nystrom extension place glasses that were NOT landmarks?

Test 1 (out-of-sample accuracy):
  fit on L (m landmarks) -> Nystrom on held-out H
  fit exactly on L ∪ H   -> "true" coordinates of H
  align the two fits with a linear map estimated on L only, then compare on H.
Test 2 (landmark-sampling stability):
  fit on two disjoint landmark sets, Nystrom-embed everything, compare after alignment.

Run from the project root:
  uv run src/validate_nystrom.py [--m 10000] [--h 4000] [--k 3]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path("src")))
from spectral_embedding import DATA, SpectralEmbedding  # noqa: E402


def align(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Least-squares linear map src -> dst (handles sign flips, scale and mixing of near-degenerate eigvecs)."""
    T, *_ = np.linalg.lstsq(src, dst, rcond=None)
    return T


def report(name: str, pred: np.ndarray, true: np.ndarray):
    r2 = 1 - ((pred - true) ** 2).sum(0) / ((true - true.mean(0)) ** 2).sum(0)
    corr = [np.corrcoef(pred[:, j], true[:, j])[0, 1] for j in range(true.shape[1])]
    rel = np.linalg.norm(pred - true, axis=1) / np.linalg.norm(true, axis=1).mean()
    print(f"{name:42s} R²={np.round(r2, 4)}  corr={np.round(corr, 4)}  "
          f"rel.err median={np.median(rel):.4f} p95={np.percentile(rel, 95):.4f}")
    return rel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=10_000)
    ap.add_argument("--h", type=int, default=4_000)
    ap.add_argument("--k", type=int, default=3, help="components to evaluate")
    ap.add_argument("--sigma-scale", type=float, default=1.0)
    args = ap.parse_args()

    df = pd.read_parquet(DATA)
    X = df.drop(columns="refractive_index").to_numpy() / 100.0
    n = X.shape[0]
    perm = np.random.default_rng(0).permutation(n)
    L, H = perm[: args.m], perm[args.m: args.m + args.h]
    k = args.k

    # ---- test 1: out-of-sample accuracy
    se_L = SpectralEmbedding(args.sigma_scale, k).fit(X[L])
    sigma = se_L.sigma_
    # same sigma in both fits, so the only difference is the extra points
    se_LH = SpectralEmbedding(n_components=k, sigma=sigma).fit(X[np.r_[L, H]])
    print(f"sigma fit L = {sigma:.4f}, sigma fit L∪H = {se_LH.sigma_:.4f}")
    print("eigenvalues L  :", np.round(se_L.eigenvalues_, 4))
    print("eigenvalues L∪H:", np.round(se_LH.eigenvalues_, 4))

    true_L, true_H = se_LH.embedding_[: args.m], se_LH.embedding_[args.m:]
    T = align(se_L.embedding_, true_L)
    print("\n-- Test 1: Nystrom on held-out vs exact fit including them (aligned on landmarks only)")
    report("landmarks (in-sample, reference)", se_L.embedding_ @ T, true_L)
    rel = report("held-out via Nystrom", se_L.transform(X[H]) @ T, true_H)

    # where does it fail? relate error to how close the point is to the landmarks
    from sklearn.metrics.pairwise import euclidean_distances
    dmin = euclidean_distances(X[H], X[L]).min(axis=1)  # fast (4k x 10k), no progress bar needed
    q = np.quantile(dmin, [0, 0.5, 0.9, 0.99, 1])
    print("  error vs distance to nearest landmark:")
    for a, b in zip(q[:-1], q[1:]):
        sel = (dmin >= a) & (dmin <= b)
        print(f"    d_nn in [{a:.4f}, {b:.4f}]: median rel.err={np.median(rel[sel]):.4f}  "
              f"p95={np.percentile(rel[sel], 95):.4f}  (n={sel.sum()})")

    # ---- test 2: stability w.r.t. which landmarks were drawn
    print("\n-- Test 2: two disjoint landmark sets, Nystrom on all glasses")
    L2 = perm[args.m + args.h: 2 * args.m + args.h]
    se_L2 = SpectralEmbedding(n_components=k, sigma=sigma).fit(X[L2])
    E1, E2 = se_L.transform(X), se_L2.transform(X)
    T12 = align(E2, E1)
    report("set B aligned to set A (all 58k)", E2 @ T12, E1)


if __name__ == "__main__":
    main()
