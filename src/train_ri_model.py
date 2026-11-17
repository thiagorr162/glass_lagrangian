"""Refractive-index regressor on compositions, used to estimate RI of proposed glasses.

Drops physically implausible RI values (outside [1.3, 3.0]: 0.56, 301, ... are data errors),
reports 5-fold CV error, then fits on everything and saves outputs/models/ri_model.joblib.

Caveat: the CV split is random, and the databases contain composition series (near-duplicates),
so CV error is optimistic for compositions far from known glasses.

Run from the project root:
  uv run src/train_ri_model.py
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_predict

sys.path.insert(0, str(Path("src")))
from spectral_embedding import DATA, MODELS, ensure_dirs  # noqa: E402

RI_RANGE = (1.3, 3.0)


def main():
    ensure_dirs()
    df = pd.read_parquet(DATA)
    y = df["refractive_index"].to_numpy()
    keep = (y >= RI_RANGE[0]) & (y <= RI_RANGE[1])
    X = df.drop(columns="refractive_index").to_numpy()[keep] / 100.0
    y = y[keep]
    print(f"training on {keep.sum()} glasses (dropped {(~keep).sum()} with RI outside {RI_RANGE})")

    model = HistGradientBoostingRegressor(max_iter=800, learning_rate=0.08, max_leaf_nodes=63,
                                          l2_regularization=1e-3, random_state=0)
    pred = cross_val_predict(model, X, y, cv=KFold(5, shuffle=True, random_state=0), n_jobs=5, verbose=1)
    err = np.abs(pred - y)
    r2 = 1 - ((pred - y) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    print(f"5-fold CV: MAE={err.mean():.4f}  median AE={np.median(err):.4f}  p95 AE={np.percentile(err, 95):.4f}  "
          f"R²={r2:.4f}")

    model.fit(X, y)
    out = MODELS / "ri_model.joblib"
    joblib.dump(model, out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
