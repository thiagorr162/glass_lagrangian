"""Do the Laplacian eigenfunctions separate the glass families?

Each real glass gets a family from its dominant network former (mol%):
  silicate (SiO2), borate (B2O3), phosphate (P2O5), germanate (GeO2), tellurite (TeO2);
  "non-conventional" if the network formers add up to less than --min-formers mol%.
Then:
  1. kNN classification of the family from phi1..phiK alone (5-fold CV) vs. the majority baseline;
  2. which families occupy the two ends (top / bottom 2%) of each eigenfunction;
  3. Spearman correlation of each eigenfunction with each former's mol%;
  4. an interactive plot (outputs/viz/families_phi_<tag>.html) colored by family.

Run from the project root:
  uv run src/families_vs_eigenfunctions.py [--tag m10000_s1] [--k 5]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import spearmanr
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier

sys.path.insert(0, str(Path("src")))
from spectral_embedding import DATA, DATA_OUT, VIZ, ensure_dirs  # noqa: E402

FORMERS = {"sio2": "silicate", "b2o3": "borate", "p2o5": "phosphate", "geo2": "germanate", "teo2": "tellurite"}
COLORS = {"silicate": "#1f77b4", "borate": "#ff7f0e", "phosphate": "#2ca02c", "germanate": "#d62728",
          "tellurite": "#9467bd", "non-conventional": "#7f7f7f"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="m10000_s1")
    ap.add_argument("--k", type=int, default=5, help="eigenfunctions used (phi1..phik)")
    ap.add_argument("--min-formers", type=float, default=20.0)
    args = ap.parse_args()
    ensure_dirs()

    df = pd.read_parquet(DATA)
    emb = pd.read_parquet(DATA_OUT / f"spectral_embedding_{args.tag}.parquet")
    assert (emb["ID"].to_numpy() == df.index.to_numpy()).all()
    F = df[list(FORMERS)]
    family = F.idxmax(axis=1).map(FORMERS).where(F.sum(axis=1) >= args.min_formers, "non-conventional")
    phis = [f"phi{i}" for i in range(1, args.k + 1)]
    P = emb[phis].to_numpy()
    P = (P - P.mean(0)) / P.std(0)
    y = family.to_numpy()

    print("families:", family.value_counts().to_dict())

    # 1. classification from the eigenfunctions only
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    base = family.value_counts(normalize=True).iloc[0]
    print(f"\n1. kNN (k=15) family from eigenfunctions, 5-fold CV accuracy   (majority baseline {base:.1%})")
    for k in [2, 3, args.k]:
        acc = cross_val_score(KNeighborsClassifier(15), P[:, :k], y, cv=cv).mean()
        print(f"   phi1..phi{k}: {acc:.1%}")
    from sklearn.model_selection import cross_val_predict
    pred = cross_val_predict(KNeighborsClassifier(15), P, y, cv=cv)
    rec = pd.Series(pred == y).groupby(y).mean()
    print("   per-family recall (phi1..phi%d): " % args.k + ", ".join(f"{f} {r:.0%}" for f, r in rec.items()))

    # 2. who sits at the ends of each eigenfunction
    print("\n2. families at the ends of each eigenfunction (top / bottom 2%)")
    for i, p in enumerate(phis):
        v = P[:, i]
        lo, hi = np.quantile(v, [0.02, 0.98])
        def top(mask):
            s = family[mask].value_counts(normalize=True).head(2)
            return ", ".join(f"{f} {x:.0%}" for f, x in s.items())
        print(f"   {p}:  low end -> {top(v <= lo):38s} | high end -> {top(v >= hi)}")

    # 3. correlation with each former
    print("\n3. Spearman correlation of eigenfunctions with former content (mol%)")
    corr = pd.DataFrame({o.upper(): [spearmanr(P[:, i], df[o])[0] for i in range(args.k)] for o in FORMERS},
                        index=phis)
    print(corr.round(2).to_string())

    # 4. plot
    fig = make_subplots(rows=1, cols=3, subplot_titles=["φ1 × φ2", "φ1 × φ3", "φ3 × φ4"])
    for c, (a, b) in enumerate([(0, 1), (0, 2), (2, 3)], start=1):
        for f, col in COLORS.items():
            m = y == f
            fig.add_trace(go.Scattergl(x=P[m, a], y=P[m, b], mode="markers", name=f, legendgroup=f,
                                       showlegend=(c == 1), marker=dict(size=2.5, color=col, opacity=0.6)), 1, c)
    fig.update_layout(title="Real glasses colored by dominant network former", template="plotly_white", height=520,
                      legend=dict(itemsizing="constant"))
    out = VIZ / f"families_phi_{args.tag}.html"
    fig.write_html(out, include_plotlyjs=True)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
