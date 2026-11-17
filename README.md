# glass_lagrangian

Spectral embedding (graph Laplacian eigenmaps) of ~58k oxide-glass compositions, colored by refractive index,
plus an interactive explorer to select a populated region of the map that has holes in it and generate candidate
glasses that fill those holes.

## Pipeline

| Step | Script | What it does | Time |
|---|---|---|---|
| 1 | `src/spectral_embedding.py` | Fits the spectral embedding on 10k landmark glasses and extends it to all glasses with Nyström. Writes 2D/3D plots, coordinates and the fitted model. | ~1 min |
| 2 | `src/validate_nystrom.py` | Checks how well Nyström places glasses that were not landmarks, and how stable the map is to the landmark draw. | ~3 min |
| 3 | `src/train_ri_model.py` | Trains a refractive-index regressor (gradient boosting on composition) and reports 5-fold CV error. | ~30 s |
| 4 | `src/compare_ri_interpolation.py` | Compares the regressor with interpolating RI from neighboring glasses. | ~2 min |
| 5 | `src/app.py` | Dash explorer: select a region, generate candidates, inspect, export. | ~5 s to start |
| — | `src/holes.py` | Hole scores (emptiness, enclosure), used by the app; as a script, prints reference distributions. | ~10 s |
| — | `src/check_candidates.py` | Compares candidates with the database (system, known range, families), used by the app; as a script, checks an exported CSV. | ~3 s |
| — | `src/mixture_cloud.py` | Earlier approach: 300k mixtures precomputed over the whole map. Not used by the app anymore. | ~2 min |

## Setup

Requires [uv](https://docs.astral.sh/uv/). Always run from the repository root.

```bash
uv sync
```

### Data (not included)

The dataset is **not in this repository**. It merges SciGlass with INTERGLAD, a commercial database, and patent
data, so it is not redistributed here. Place it at `data/refractive_index.parquet` with:

- index `ID` (string);
- one column per oxide (`sio2`, `b2o3`, `na2o`, … 73 in total), in **mol%**, summing to 100 per row;
- `refractive_index` (float).

The units were checked on PbO–SiO₂ glasses. Their RI versus PbO content matches mol% literature values (n ≈ 1.91 at
50% PbO; wt% would give ≈ 1.69). Their PbO content also stops at ~70%, the known mol% glass-forming limit.

### Run

```bash
uv run src/spectral_embedding.py        # options: --m 10000 --sigma-scale 1.0 --n-components 10
uv run src/validate_nystrom.py --k 5
uv run src/train_ri_model.py
uv run src/compare_ri_interpolation.py
uv run src/app.py                       # then open http://127.0.0.1:8050
```

Outputs go to `outputs/` (git-ignored):

```
outputs/
├── viz/      interactive plots (html)
├── data/     embeddings, exported candidates (parquet, csv)
├── models/   fitted spectral model (npz), landmark indices, RI regressor (joblib)
└── logs/     run logs
```

## Method

### Spectral embedding

Classic Ng–Jordan–Weiss affinity on a random subsample of *m* = 10,000 landmark glasses. Compositions are taken as
fractions.

- Gaussian affinity: `A_ij = exp(−‖x_i − x_j‖² / 2σ²)`, with `A_ii = 0`. σ is the median pairwise distance
  (0.7255).
- Normalized Laplacian: `L = I − D^{-1/2} A D^{-1/2}`. We take its smallest eigenpairs.
- Coordinates: `φ_k = D^{-1/2} v_k` for the non-trivial eigenvectors. The trivial eigenvector is dropped. Rows are
  **not** normalized to unit length (as NJW clustering does), because that would put the 2D embedding on a circle.

**Why not a kNN graph with local bandwidths?** A first attempt used self-tuning σ_i, the distance to the 7th
neighbor. In dense series of near-identical compositions σ_i → 0, edge weights underflowed (min ≈ 1e-199) and the
eigenvectors localized on clusters of 9–24 points.

### Nyström extension

Any composition `x`, not only landmarks, gets coordinates

```
φ_k(x) = (1/μ_k) · Σ_i p_i(x) · φ_k(x_i),    p_i(x) = K(x, x_i) / Σ_j K(x, x_j)
```

where `μ_k = 1 − λ_k`. This is `SpectralEmbedding.transform`, and it is what places generated candidates and typed
compositions on the map. A typed composition of a real glass lands exactly on its stored coordinates.

Validation (`validate_nystrom.py --k 5`):

| | φ₁ | φ₂ | φ₃ | φ₄ | φ₅ |
|---|---|---|---|---|---|
| R², held-out glasses (Nyström) vs. exact fit including them | 0.9999 | 0.9999 | 1.000 | 0.9998 | 0.9998 |
| R², two disjoint landmark draws | 1.000 | 0.9998 | 0.9995 | 0.9998 | 0.995 |

The median relative error is about 1.2%, and it does not grow with distance to the nearest landmark. φ₃ and φ₄
have close eigenvalues (0.953 and 0.961), so different fits may rotate them into each other. Compare individual
axes only within the same saved model.

### Refractive index of candidates

RI varies smoothly with composition. Averaging the RI of neighboring glasses is as accurate as a trained regressor.
5-fold CV, RI restricted to [1.3, 3.0]:

| Method | MAE | Median AE | R² |
|---|---|---|---|
| Gradient boosting on composition | 0.0172 | 0.0082 | 0.961 |
| Harmonic extension on the kNN graph | 0.0163 | 0.0066 | 0.957 |
| Distance-weighted mean of 10 nearest glasses (kNN) | 0.0160 | 0.0053 | 0.956 |
| Gradient boosting on φ₁…φ₁₀ | 0.0246 | 0.0119 | 0.937 |
| Linear on φ₁…φ₁₀ | 0.0638 | 0.0427 | 0.757 |

The explorer shows two RI estimates per candidate:

- `RI interp`: the kNN mean (10 nearest real glasses **in composition**, weights 1/distance). It does not use the
  eigenfunctions.
- `RI hgb`: the regressor.

Candidates where the two differ by more than 0.03 are highlighted. The random CV split is optimistic, because the
databases contain composition series of near-duplicates. Some database RI values look wrong (e.g. a TeO₂–ZnO–Sm₂O₃
series at 2.3–2.65); they are kept, on the assumption that they average out.

## Explorer (`src/app.py`)

1. **Select** a populated region that has holes, with lasso or box, in any of the 10 panels (all pairs of φ₁…φ₅).
   - Selections in different panels combine by **intersection**.
   - The × chips remove one selection; **Clear selections** removes all.
   - **Zoom to selection** fits every panel to the glasses left by the selection; **Reset zoom** undoes it.
2. Click **Generate candidates**. In ~9 s for 20k mixtures:
   - **Parents:** the real glasses inside the selection.
   - **Mixtures:** convex mixtures, so always valid compositions summing to 100. They are a parent plus 1 or 2 of
     its 10 nearest glasses in composition, and random pairs of parents.
   - **Region:** in each panel with a lasso, the convex hull of the selected glasses, drawn dashed blue. It is
     tighter than the lasso and keeps the holes inside. A mixture is kept if Nyström places it inside the lassos
     and those hulls, and it is within 7.3 mol% of a real glass in composition.
3. **Inspect** the table: one row per family (candidates within 1 mol% of each other). Columns:
   - **emptiness:** distance to the nearest real glass divided by the local spacing. The local spacing is the
     median, over the 30 nearest real glasses, of their distance to their 5th nearest glass. Real glasses
     themselves have median 0.49 and p99 2.86.
   - **enclosure:** 1 − |mean unit vector to the 30 nearest real glasses|. Near 1 means surrounded, near 0 means
     at the edge. Real glasses have p10 0.33 and median 0.55.
   - **status:** *new system* (no real glass with the same set of oxides ≥ 0.5 mol%), *outside range* (exceeds
     the measured range of its system by > 1 mol%) or *known range*.
   - **d_nn (mol%):** distance to the nearest real glass in composition. About 2–6 mol% is the useful range.
     Below that, the glass practically exists; above, it is extrapolation.
   - **RI interp, RI hgb, nearest real glass, parents.**
   - **composition:** the full composition, normalized to 100 and paste-ready.

   **Only interior holes** (default on) keeps candidates with emptiness ≥ 2 and enclosure ≥ 0.4. **Only new
   chemistry** keeps *new system* and *outside range*.
4. **Export CSV** writes the candidates with their full mol% compositions to `outputs/data/candidates_*.csv`.
5. **Show compositions:** paste compositions, one per line (`SiO2 70 Na2O 15 CaO 15`, `75TeO2-25PbO`, …).
   - Each is plotted as a red dot in every panel, with everything else made faint.
   - The composition table shows the same columns for them.
   - **Clear** restores the normal view.

All panels are subplots of one figure. Ten separate WebGL figures exceed the browser's WebGL context limit, and the
oldest panels go blank.

## Caveats

- **Glass formation.** The databases only contain compositions that formed glass. A hole may be a hole because
  those compositions crystallize. Candidates are "plausible by similarity", not guaranteed glass formers. Check
  them against the literature: glass-forming limits, coloring by TiO₂/WO₃/Bi₂O₃, toxic oxides.
- **Patents.** Many high-index regions (lanthanum borates, Nb–Ti silicates) are covered by patents whose claimed
  ranges go beyond their measured examples.
- **No dispersion.** Only nd is predicted. Optical glasses are chosen by (nd, Abbe number), and the dataset has no
  Abbe number.
- **No extreme RI from interpolation.** Interpolated RI lies between the RI values of the neighbors.
- **Map vs. composition.** A hole in the map may be a region where the map stretches. Look at `d_nn` as well, and
  remember that only φ₁…φ₅ are used.
- **Many-component glasses.** With 9 or more oxides almost every candidate is a *new system*; rely on `d_nn`
  instead.
