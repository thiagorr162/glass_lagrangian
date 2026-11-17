"""Interior holes of the spectral map: empty spots surrounded by real glasses.

Idea: dense regions of the map with a few holes in the middle are good places to look for glasses nobody has
made yet. The chemistry around them is known to form glass, and interpolation (of composition and of RI) is
backed by data on all sides. Empty space at the EDGE of the cloud is more likely a real glass-forming limit.

For a point q (a mixture, in standardized phi1..phi5) and its k nearest real glasses:
  emptiness  = distance to the nearest real glass / local spacing,
               local spacing = median, over those k real glasses, of their distance to their 5th nearest real
               glass (floored at its global 10th percentile; a local ruler, so small holes inside dense regions
               are found too, robust to series of near-identical glasses)
  enclosure  = 1 - |mean of the unit vectors from q to its k nearest real glasses|
               ~1 when the real glasses surround q on all sides, ~0 when they all lie on one side (edge)
A mixture is an interior hole if emptiness >= --emptiness-min and enclosure >= --enclosure-min.

Reference values are computed the same way for the real glasses themselves (leaving each one out).

Run from the project root (needs spectral_embedding.py and mixture_cloud.py outputs):
  uv run src/holes.py [--tag m10000_s1] [--k 30]
Writes outputs/data/holes_<tag>.parquet (one row per mixture, aligned with mixture_cloud_<tag>.parquet).
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, str(Path("src")))
from spectral_embedding import DATA_OUT, ensure_dirs  # noqa: E402

PHIS = [f"phi{k}" for k in range(1, 6)]


class HoleScorer:
    """emptiness and enclosure w.r.t. the real glasses R (standardized phi coordinates); fitted once."""

    def __init__(self, R: np.ndarray, k: int = 30):
        self.R, self.k = R, k
        self.nn = NearestNeighbors(n_neighbors=k + 1).fit(R)
        # local ruler of each real glass: distance to its 5th nearest other real glass. The 1st neighbor is
        # useless here: composition series of near-identical glasses make it ~0 and every gap looks huge.
        # Floored at the global 10th percentile for the same reason.
        d5 = self.nn.kneighbors(R, n_neighbors=6)[0][:, 5]
        self.d_rr = np.maximum(d5, np.percentile(d5, 10))

    def score(self, Q: np.ndarray, exclude_self: bool = False, chunk: int = 20000):
        k = self.k
        emptiness = np.empty(len(Q))
        enclosure = np.empty(len(Q))
        for a in range(0, len(Q), chunk):
            dist, idx = self.nn.kneighbors(Q[a:a + chunk], n_neighbors=k + 1)
            dist, idx = (dist[:, 1:], idx[:, 1:]) if exclude_self else (dist[:, :k], idx[:, :k])
            local = np.median(self.d_rr[idx], axis=1)
            emptiness[a:a + chunk] = dist[:, 0] / np.maximum(local, 1e-12)
            V = self.R[idx] - Q[a:a + chunk, None, :]
            U = V / np.maximum(np.linalg.norm(V, axis=2, keepdims=True), 1e-12)
            enclosure[a:a + chunk] = 1 - np.linalg.norm(U.mean(axis=1), axis=1)
        return emptiness, enclosure


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="m10000_s1")
    ap.add_argument("--k", type=int, default=30, help="real neighbors used for the local ruler and enclosure")
    args = ap.parse_args()
    ensure_dirs()

    real = pd.read_parquet(DATA_OUT / f"spectral_embedding_{args.tag}.parquet")
    mix = pd.read_parquet(DATA_OUT / f"mixture_cloud_{args.tag}.parquet", columns=["type", "d_nn"] + PHIS)
    mu, sd = real[PHIS].mean(), real[PHIS].std()
    R = ((real[PHIS] - mu) / sd).to_numpy()
    Q = ((mix[PHIS] - mu) / sd).to_numpy()

    scorer = HoleScorer(R, args.k)
    e_r, c_r = scorer.score(R, exclude_self=True)
    print("real glasses (reference, leave-one-out):")
    print(f"  emptiness p50={np.median(e_r):.2f} p90={np.percentile(e_r, 90):.2f} p99={np.percentile(e_r, 99):.2f}")
    print(f"  enclosure p10={np.percentile(c_r, 10):.2f} p50={np.median(c_r):.2f} p90={np.percentile(c_r, 90):.2f}")

    e_m, c_m = scorer.score(Q)
    out = pd.DataFrame({"emptiness": e_m, "enclosure": c_m})
    path = DATA_OUT / f"holes_{args.tag}.parquet"
    out.to_parquet(path, index=False)
    print(f"mixtures: emptiness p50={np.median(e_m):.2f} p90={np.percentile(e_m, 90):.2f}; "
          f"enclosure p50={np.median(c_m):.2f}")
    for e_min, c_min in [(2, 0.5), (3, 0.5), (3, 0.6), (5, 0.5)]:
        sel = (e_m >= e_min) & (c_m >= c_min) & mix["type"].isin(["pair", "triplet"]).to_numpy() & (mix["d_nn"] <= 0.073).to_numpy()
        print(f"  holes with emptiness>={e_min}, enclosure>={c_min} (pair/triplet, d_nn<=0.073): {sel.sum()}")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
