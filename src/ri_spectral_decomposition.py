"""Can RI be written as a combination of the Laplacian eigenfunctions? Composition-only vs. RI-aware kernel.

Both maps are refitted on the same landmarks as the saved models (outputs/models/landmarks_m10000_s1.npy), with
--n-components eigenfunctions. Every glass is placed with Nystrom (the RI-aware map uses each glass's true RI:
this describes RI on the map, it is not a prediction). Then, for growing k, ordinary least squares of RI on
phi1..phik gives the fraction of RI variance carried by the first k eigenfunctions, and the eigenfunctions that
add the most to it are listed. RI restricted to [1.3, 3.0] (a few database values are absurd, e.g. 301).

Run from the project root:
  uv run src/ri_spectral_decomposition.py [--n-components 200] [--ri-scale 2.0]
Writes outputs/data/ri_spectral_decomposition.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path("src")))
from spectral_embedding import DATA, DATA_OUT, MODELS, SpectralEmbedding, ensure_dirs  # noqa: E402

RI_RANGE = (1.3, 3.0)


def r2_curve(P: np.ndarray, y: np.ndarray) -> np.ndarray:
    """R² of least squares of y on [1, phi1..phik], for every k (QR: adding phi_k only adds its new direction)."""
    Q, _ = np.linalg.qr(np.c_[np.ones(len(y)), P])
    yc = y - y.mean()
    gain = (Q[:, 1:].T @ yc) ** 2  # variance added by each phi_k beyond phi1..phi(k-1)
    return np.cumsum(gain) / (yc @ yc), gain / (yc @ yc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-components", type=int, default=200)
    ap.add_argument("--ri-scale", type=float, default=2.0)
    args = ap.parse_args()
    ensure_dirs()

    df = pd.read_parquet(DATA)
    X = df.drop(columns="refractive_index").to_numpy() / 100.0
    ri = df["refractive_index"].to_numpy()
    land = np.load(MODELS / "landmarks_m10000_s1.npy")
    ok = (ri >= RI_RANGE[0]) & (ri <= RI_RANGE[1])

    maps = {"composition": SpectralEmbedding(n_components=args.n_components).fit(X[land]),
            f"composition × RI (ri_scale {args.ri_scale:g})":
                SpectralEmbedding(n_components=args.n_components, ri_scale=args.ri_scale).fit(X[land], ri[land])}
    ks = [k for k in [1, 2, 3, 5, 10, 20, 50, 100, 150, 200] if k <= args.n_components]
    table = {}
    for name, se in maps.items():
        P = se.transform(X[ok], ri[ok] if se.uses_ri else None)
        cum, gain = r2_curve(P, ri[ok])
        table[name] = cum[np.array(ks) - 1]
        top = np.argsort(-gain)[:8]
        print(f"\n{name}" + (f"  (tau = {se.tau_:.3f})" if se.uses_ri else ""))
        print("  eigenfunctions adding the most RI variance: "
              + ", ".join(f"φ{k + 1} {100 * gain[k]:.1f}%" for k in top))
    out = pd.DataFrame(table, index=pd.Index(ks, name="k (φ1..φk)")).round(3)
    print("\nR² of RI as a linear combination of φ1..φk (all glasses with RI in [1.3, 3.0]):")
    print(out.to_string())
    path = DATA_OUT / "ri_spectral_decomposition.csv"
    out.to_csv(path)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
