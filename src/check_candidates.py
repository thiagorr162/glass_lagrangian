"""Check candidates against the database: is there something similar already measured?

Compositions are mol% (checked on PbO-SiO2: RI vs PbO matches mol% literature values).

For each candidate:
  system          - oxides with >= min_pct mol% (sorted by amount), e.g. TEO2-PBO-TIO2
  n_same_system   - real glasses whose oxide set (>= min_pct) is exactly the same system
  max_out         - largest excursion (mol%) outside the min..max range of any oxide among same-system glasses
  status          - "new system"     : no real glass in the same system
                    "outside range"  : same system exists, but max_out > range_tol
                    "known range"    : inside the measured range of its system (interpolation of known chemistry)
  family          - candidates within `radius` mol% of each other (greedy leader clustering; the leader is the
                    candidate closest to a real glass) share a family id

The functions are reused by src/app.py. As a script, it also adds the 3 nearest real glasses and the number of
real glasses in sub-systems, and writes outputs/data/<input name>_check.csv:
  uv run src/check_candidates.py outputs/data/candidates_<timestamp>.csv [--min-pct 0.5]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, str(Path("src")))
from spectral_embedding import DATA, DATA_OUT  # noqa: E402

STATUSES = ["new system", "outside range", "known range"]


def system_keys(present: np.ndarray) -> np.ndarray:
    """Boolean (n, n_oxides) presence matrix -> one hashable key (bytes) per row."""
    packed = np.packbits(present, axis=1)
    return np.array([r.tobytes() for r in packed], dtype=object)


def system_labels(Xpct: np.ndarray, oxides, min_pct: float) -> list[str]:
    names = np.array([o.upper() for o in oxides])
    out = []
    for row in Xpct:
        idx = np.flatnonzero(row >= min_pct)
        out.append("-".join(names[idx[np.argsort(-row[idx])]]))
    return out


class SystemIndex:
    """Per-system statistics of the real glasses (count, min and max of every oxide)."""

    def __init__(self, Xpct_db: np.ndarray, oxides, min_pct: float = 0.5, range_tol: float = 1.0):
        self.oxides, self.min_pct, self.range_tol = list(oxides), min_pct, range_tol
        keys = system_keys(Xpct_db >= min_pct)
        df = pd.DataFrame(Xpct_db, columns=self.oxides)
        df["_key"] = keys
        g = df.groupby("_key")
        self.count = g.size().to_dict()
        lo, hi = g.min(), g.max()
        self.lo = {k: lo.loc[k].to_numpy() for k in lo.index}
        self.hi = {k: hi.loc[k].to_numpy() for k in hi.index}

    def compare(self, Xpct: np.ndarray) -> pd.DataFrame:
        present = Xpct >= self.min_pct
        keys = system_keys(present)
        n_same = np.array([self.count.get(k, 0) for k in keys])
        max_out = np.full(len(Xpct), np.nan)
        for i, k in enumerate(keys):
            if n_same[i]:
                p = present[i]
                out = np.maximum(self.lo[k][p] - Xpct[i, p], 0) + np.maximum(Xpct[i, p] - self.hi[k][p], 0)
                max_out[i] = out.max()
        status = np.where(n_same == 0, "new system", np.where(max_out > self.range_tol, "outside range", "known range"))
        return pd.DataFrame({"system": system_labels(Xpct, self.oxides, self.min_pct),
                             "n_same_system": n_same, "max_out": max_out, "status": status})


def families(Xpct: np.ndarray, priority: np.ndarray, radius: float = 1.0) -> np.ndarray:
    """Greedy leader clustering: in order of `priority` (lowest first), each unassigned candidate becomes a
    leader and claims every unassigned candidate within `radius` (Euclidean, mol%)."""
    neigh = NearestNeighbors(radius=radius).fit(Xpct).radius_neighbors(Xpct, return_distance=False)
    fam = np.full(len(Xpct), -1)
    fid = 0
    for i in np.argsort(priority, kind="stable"):
        if fam[i] >= 0:
            continue
        members = neigh[i][fam[neigh[i]] < 0]
        fam[members] = fid
        fam[i] = fid
        fid += 1
    return fam


def comp_str(row: pd.Series, top: int = 6) -> str:
    r = row[row >= 0.05].sort_values(ascending=False).head(top)
    return " ".join(f"{k.upper()} {v:.1f}" for k, v in r.items())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("candidates", type=Path)
    ap.add_argument("--min-pct", type=float, default=0.5, help="oxides below this mol% are ignored for the system")
    ap.add_argument("--range-tol", type=float, default=1.0, help="mol% outside the known range to count as outside")
    ap.add_argument("--family-radius", type=float, default=1.0, help="mol% radius for grouping similar candidates")
    args = ap.parse_args()

    db = pd.read_parquet(DATA)
    oxides = list(db.columns.drop("refractive_index"))
    Xdb = db[oxides]
    cand = pd.read_csv(args.candidates)
    Xc = cand.reindex(columns=oxides, fill_value=0.0)  # exported CSV keeps only oxides used by candidates

    rep = SystemIndex(Xdb.to_numpy(), oxides, args.min_pct, args.range_tol).compare(Xc.to_numpy())
    rep.insert(1, "composition", [comp_str(Xc.iloc[k]) for k in range(len(Xc))])
    for c in ["ri_interp", "ri_hgb", "d_nn", "novelty"]:
        rep[c] = cand[c]
    rep["family"] = families(Xc.to_numpy(), cand["d_nn"].to_numpy(), args.family_radius)

    present_db = (Xdb >= args.min_pct).to_numpy()
    present_c = Xc.to_numpy() >= args.min_pct
    rep["n_subsystem"] = [int((~present_db[:, ~p]).all(axis=1).sum()) for p in present_c]

    dist, idx = NearestNeighbors(n_neighbors=3).fit(Xdb.to_numpy()).kneighbors(Xc.to_numpy())
    for r in range(3):
        rep[f"nn{r + 1}_id"] = db.index[idx[:, r]]
        rep[f"nn{r + 1}_dist"] = dist[:, r]
        rep[f"nn{r + 1}_ri"] = db["refractive_index"].to_numpy()[idx[:, r]]
        rep[f"nn{r + 1}_comp"] = [comp_str(Xdb.iloc[j]) for j in idx[:, r]]

    out_path = DATA_OUT / f"{args.candidates.stem}_check.csv"
    rep.to_csv(out_path, index=False)
    print(f"{len(rep)} candidates, {rep['system'].nunique()} systems, {rep['family'].nunique()} families "
          f"(radius {args.family_radius} mol%)")
    print("  status:", rep["status"].value_counts().to_dict())
    print(f"  distance to nearest real glass (mol%): median {rep.nn1_dist.median():.2f}, "
          f"p90 {rep.nn1_dist.quantile(.9):.2f}, max {rep.nn1_dist.max():.2f}")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
