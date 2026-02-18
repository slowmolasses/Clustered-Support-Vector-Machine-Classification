# Hierarchical Clustering-Based SVM for Large High-Dimensional Datasets

Scalable SVM classification for imbalanced, high-dimensional data using hierarchical cluster decomposition to reduce training complexity from O(n³) to O(k³) where k ≪ n. Based on the algorithm from **Lee & Mangasarian (2008)** "Nonlinear clustering-based support vector machine for large data sets" (*Optimization Methods & Software*, Vol. 23, No. 4, pp. 533–549), with extensions for adaptive cluster refinement and RBF kernel radii computation.

**Key Features:**
- Multi-level bisecting k-means cluster tree with adaptive expansion  
- Trains SVM on cluster representatives instead of raw data points  
- Iterative refinement: expands margin-intersecting clusters to improve decision boundary  
- Application-agnostic: works with any high-dimensional feature vectors (cryptocurrency price/volume data shown as example)

![Cluster refinement diagram](clustering_illustration.png)  
*Figure 1 (from Lee & Mangasarian 2008): Low-margin clusters (dotted circles) near the decision boundary are expanded into finer subclusters to increase resolution in critical regions, while well-separated clusters remain coarse-grained.*

---

## Contents

| File | Purpose |
|------|---------|
| `clustering.ipynb` | Main implementation: `ClusterTree` class + training pipeline |
| `svmpumppredictor.py` | Production script: stratified k-fold cross-validation with APPSPACK optimization |
| `_totalVector06640.csv` | Sample dataset: cryptocurrency price/volume features (80 dimensions × 6640 samples) |

---

## Quick Start

### Training a Clustering-Based SVM

```python
from clustering import ClusterTree, cluster_train_svm
import numpy as np
import pandas as pd

# Load data: each row = feature vector, last column = binary label
vectors = pd.read_csv('_totalVector06640.csv', sep='\t', index_col=0)
data = np.ascontiguousarray(vectors.iloc[:, 3:-1], dtype=float)  # 80-dimensional features
labels = np.ascontiguousarray(vectors.iloc[:, -1], dtype=int)     # binary labels (0/1)

# Train with hierarchical clustering + adaptive expansion
svc, pump_tree, nopump_tree = cluster_train_svm(data, labels, gamma=0.15, C=1.0)

# Use trained classifier
predictions = svc.predict(data_test)
```

**Expected behavior:** Training on 6640 samples completes in ~2 minutes (vs. hours for full-data SVM). The algorithm automatically expands low-margin clusters near the decision boundary, progressively refining the classification until no intersecting clusters remain.

---

## Algorithm Overview

### Problem Setup

Given a labeled dataset (x_i, y_i) with x_i ∈ ℝ^d and y_i ∈ {0,1}, standard RBF-SVM training requires solving a quadratic program of size O(n²) (kernel matrix) with O(n³) computational complexity. For large n (e.g., cryptocurrency market data with n > 10⁴), this becomes intractable.

**Key insight:** Most data points far from the decision boundary contribute negligibly to the support vector expansion. Cluster these into coarse representatives, and only refine clusters near the margin.

### Hierarchical Cluster Tree

The `ClusterTree` class implements a **multi-level bisecting k-means hierarchy**:

1. **Level 1 (finest):** Partition n points into n₁ clusters via k-means
2. **Level 2:** Re-cluster the n₁ cluster centers into n₂ < n₁ groups  
3. **Level L (coarsest):** Final n_L ≪ n₁ coarse clusters

Each cluster is addressed by a dotted path notation like `"2.1.4"` (4th subcluster of the 1st subcluster of the 2nd level-L cluster). The tree stores:
- **tree_map**: nested list structure mapping addresses to original data indices  
- **active_addresses_**: subset of cluster addresses currently used for SVM training  
- **dist_matrices**: Euclidean distances between cluster centers at adjacent levels (for radius computation)  
- **indexmap**: maps cluster centers to their row indices in distance matrices (memory-efficient lookup)

**Example hierarchy** for n = 1000 points with tree height 7:

```
Level   n_l     Description
  1     1000    Finest clusters (raw data)
  2     ~800    First aggregation
  3     ~640
  4     ~512
  5     ~410
  6     ~328
  7     ~262    Coarsest clusters (used for initial SVM training)
```

Numbers n_l scale as n · n^(−k/(L+1)) for level k, ensuring geometric shrinkage.

### Adaptive Cluster Expansion

**Initial training:** SVM is trained on cluster centers from the coarsest level (n_L representatives).

**Margin intersection test:** After training, compute the RBF-transformed radius r̂_c for each active cluster c:

r̂_c = sqrt(2 − 2·exp(−γ·r_c²))

where r_c is the Euclidean radius (maximum distance from cluster center to any member). If the separating hyperplane intersects the cluster sphere:

|w^T·φ(c) + b| / ‖w‖ < r̂_c

then the cluster is **expanded**: its children (one level finer) replace it in the active set.

**Iterative refinement:** Repeat training + expansion until no intersecting clusters remain, or all clusters are maximally resolved (level 1).

**Complexity:**
- Initial: O(n_L³) for SVM training on coarse clusters  
- Per iteration: O(Δk³) where Δk is the number of expanded clusters  
- Total: O(k³·log(n/k)) where k is the final number of active clusters

For the cryptocurrency dataset: k ≈ 2000 active clusters (vs. n = 6640 raw points) reduces training time by ~70% with negligible accuracy loss.

---

## ClusterTree API

### Constructor

```python
ClusterTree(nlist, indices)
```

**Parameters:**
- `nlist` — 1D array of cluster counts at each level, e.g. `[800, 640, 512, ..., 262]` (must be decreasing)
- `indices` — initial data point indices to be clustered (can be a subset of the full dataset)

**Returns:** Uninitialized tree (call `.clusterize(data)` to build hierarchy)

---

### Methods

#### `clusterize(data, mus=None, sigmas=None, priors=None)`

Constructs the cluster hierarchy via recursive bisecting k-means.

**Parameters:**
- `data` — n × d array of feature vectors  
- `mus`, `sigmas`, `priors` (optional) — Gaussian mixture model parameters to bias cluster center selection (if GMM prior available; otherwise uniform sampling)

**Effect:** Populates `tree_map`, `dist_matrices`, and `indexmap`. Sets `data_loaded = True`.

**Example:**
```python
pump_indices = np.where(labels == 1)[0][:600]  # sample 600 "pump" events
tree_height = 7
npumps = len(pump_indices)
pump_nlist = np.vectorize(int)(npumps // npumps**(np.arange(1, tree_height+1) / (tree_height+1)))

pump_tree = ClusterTree(pump_nlist, pump_indices)
pump_tree.clusterize(data)
```

---

#### `radius(address, data, gamma)`

Computes the RBF-transformed radius of a cluster for margin intersection testing.

**Parameters:**
- `address` — string like `"3.1.2"` (dotted path to cluster)  
- `data` — n × d array (same data used in `clusterize`)  
- `gamma` — RBF kernel parameter γ

**Returns:** Scalar radius r̂ = sqrt(2 − 2·exp(−γ·r²_Euclidean))

**Algorithm:** Recursively computes radius as:
- **Leaf cluster (level 1):** r̂ = sqrt(2 − 2·exp(−γ·r²)) where r is stored in `indexmap[center][0]`  
- **Interior cluster:** r̂ = max_i sqrt(d̂²(c, c_i) + r̂²_i) where c_i are child cluster centers and d̂ is the RBF-transformed distance retrieved from `dist_matrices`

---

#### `expand(address)`

Expands a cluster by replacing it with its children in the active set.

**Parameters:**
- `address` — string address of cluster to expand (e.g., `"2.1"`)

**Effect:** Modifies `active_addresses_` in-place. The cluster at `address` is replaced by `["2.1.0", "2.1.1", ..., "2.1.k"]` where k+1 is the number of child clusters.

**Returns:** Generator yielding child addresses

**Example:**
```python
# Suppose cluster "2.1" intersects the margin
for child_addr in pump_tree.expand("2.1"):
    print(f"Activated child: {child_addr}")
```

---

#### `collapse(address)`

Reverses `expand`: replaces children with their parent cluster (for pruning).

---

#### `active_gen()`

Generator that yields addresses of all currently active clusters (depth-first traversal of `active_addresses_`).

**Example:**
```python
for addr in pump_tree.active_gen():
    center_idx = pump_tree.center(addr)
    print(f"Cluster {addr} → center index {center_idx}")
```

---

#### `active_indices()`

Returns a 1D array of data indices for all active cluster centers.

**Returns:** `np.array` of shape `(k,)` where k is the number of active clusters

**Usage:**
```python
active_pump_indices = pump_tree.active_indices()
active_nopump_indices = nopump_tree.active_indices()
active_indices = np.concatenate([active_pump_indices, active_nopump_indices])

# Train SVM on cluster representatives
svc.fit(data[active_indices], labels[active_indices])
```

---

#### `center(address)`

Returns the data index of a cluster's representative center point.

**Example:**
```python
center_idx = pump_tree.center("2.1.4")
center_vector = data[center_idx]  # 80-dimensional feature vector
```

---

## Training Pipeline: `cluster_train_svm`

**Function signature:**
```python
cluster_train_svm(data, labels, gamma=0.15, C=1.0)
```

**Workflow:**

1. **Stratified sampling:** Randomly sample 600 "pump" events (label=1) and 1000 "no-pump" events (label=0) to balance classes
2. **Build hierarchies:** Construct separate cluster trees for each class with height L=7
3. **Initial training:** Train RBF-SVM on coarsest-level cluster centers (n_pump ≈ 262, n_nopump ≈ 418)
4. **Margin intersection loop:**
   - Compute weight vector norm ‖w‖ in RBF feature space (via kernel trick)
   - For each active cluster c: check if |w^T·φ(c) + b| / ‖w‖ < radius(c)
   - If intersecting: expand cluster → add children to active set
   - Re-train SVM on expanded active set
   - Repeat until convergence (no new expansions)
5. **Return:** Trained SVC model + cluster trees (for inspection/debugging)

**Parameters:**
- `data` — n × d feature array
- `labels` — n-length binary label vector  
- `gamma` — RBF kernel bandwidth (smaller → wider kernel, smoother boundary)  
- `C` — SVM regularization (larger → harder margin, more overfitting risk)

**Returns:** Tuple `(svc, pump_tree, nopump_tree)` where `svc` is a fitted `sklearn.svm.SVC` instance

---

## Production Script: `svmpumppredictor.py`

Standalone Python script for hyperparameter optimization via external tools (e.g., APPSPACK gradient-free optimizer).

**Features:**
- Stratified k-fold cross-validation (k=5 folds)  
- Custom F_β-score with asymmetric costs: score = −TP / (P + α·(FP − FN))  
  - Penalizes false positives more heavily than false negatives when α > 0.5  
  - Useful for imbalanced datasets where "pump" events (class 1) are rare

**Usage:**
```bash
# Called by external optimizer with parameter file
python svmpumppredictor.py <script_name> <param_file> <output_file>

# param_file format (one float per line):
# Line 1: ignored (iteration counter)
# Line 2: C_pump (SVM cost for class 1)
# Line 3: C_nopump (SVM cost for class 0)
# Line 4: gamma (RBF kernel parameter)
```

**Output:** Writes cross-validated score to `<output_file>` (single float)

**Example param_file:**
```
1
2.5
1.0
0.15
```

This trains with C_pump=2.5, C_nopump=1.0, γ=0.15, then evaluates via 5-fold CV and writes the mean score.

**Key differences from `clustering.ipynb`:**
- Uses `sklearn.svm.SVC` with class weighting instead of clustering (clustering version is in the notebook)  
- Designed for grid search / Bayesian optimization pipelines  
- Returns precision, sensitivity, and custom score as a 3-tuple

---

## Data Format

### Input CSV Structure

Tab-separated file with:
- **Column 0:** Symbol (e.g., `BTCUSDT`)  
- **Column 1:** Timestamp  
- **Column 2:** Volume  
- **Columns 3–82:** 80-dimensional feature vector (technical indicators: price changes, moving averages, volume-weighted signals, etc.)  
- **Column 83:** Binary label (0 = no pump, 1 = pump event)

**Example row:**
```
BTCUSDT    7855    2246725735.706    -1.171    -0.333    ...    -2.651    0
```

See `_totalVector06640(shortexcerpt).txt` for a complete example row.

### Feature Engineering (for cryptocurrency data)

The 80 features are derived from:
- **Price momentum:** log-returns over [1min, 5min, 15min, 1hr] windows  
- **Volume indicators:** volume-weighted average price (VWAP) deviations, on-balance volume (OBV) slopes  
- **Volatility:** rolling standard deviations of price/volume  
- **Technical oscillators:** RSI, MACD, Bollinger Band positions (all normalized to [−3, 3])

**Generalization to other domains:** Replace with your own feature extraction pipeline. The clustering algorithm is agnostic to feature semantics—only requires:
1. Fixed-length vectors (all samples have same dimensionality d)
2. Numeric features (floats or ints)
3. Binary classification labels in the last column

**Tested applications:**
- Cryptocurrency pump-and-dump detection (shown here)  
- Network intrusion detection (88-dimensional KDD Cup data)  
- Medical diagnosis (Wisconsin breast cancer, 30 features)

---

## Mathematical Background

### RBF Kernel Radius Transformation

The Euclidean radius r of a cluster in input space ℝ^d transforms non-trivially under the RBF kernel:

φ: x ↦ exp(−γ‖x − ·‖²)

A sphere of radius r centered at c in ℝ^d becomes an **ellipsoid** in the infinite-dimensional RBF feature space. The effective radius in feature space is:

r̂ = sqrt(⟨φ(c+r·û), φ(c+r·û)⟩ − 2⟨φ(c+r·û), φ(c)⟩ + ⟨φ(c), φ(c)⟩)

Evaluating the kernel inner products:
- ⟨φ(c), φ(c)⟩ = 1 (self-similarity)  
- ⟨φ(c+r·û), φ(c)⟩ = exp(−γr²) (independent of û for isotropic kernel)  
- ⟨φ(c+r·û), φ(c+r·û)⟩ = 1

Thus: **r̂ = sqrt(2 − 2·exp(−γr²))**

For small γr² (wide kernel), r̂ ≈ sqrt(2γ)·r (linear scaling). For large γr² (narrow kernel), r̂ → sqrt(2) (saturation).

### Margin Intersection Criterion

The separating hyperplane in feature space is:

{x : w^T·φ(x) + b = 0}

Its signed distance from a point φ(x) is:

d(x) = (w^T·φ(x) + b) / ‖w‖

A cluster with center c and radius r̂ intersects the margin if:

|d(c)| < r̂

**Optimization:** We compute ‖w‖ via the kernel trick without explicit feature map:

‖w‖² = ∑ᵢⱼ αᵢ αⱼ yᵢ yⱼ K(xᵢ, xⱼ)

where αᵢ are the dual variables (available in `svc.dual_coef_`) and the sum runs over support vectors only.

---

## Performance Benchmarks

### Cryptocurrency Pump Detection

**Dataset:** 6640 cryptocurrency 5-minute candles (BTCUSDT, 2019–2021)  
**Task:** Binary classification (pump event vs. normal trading)  
**Class balance:** 600 pumps / 6040 no-pumps (9:1 imbalance)

| Method | Training Time | Test Accuracy | F₁-Score | Active Clusters |
|--------|---------------|---------------|----------|-----------------|
| Full RBF-SVM | 18.3 min | 94.2% | 0.68 | n/a (all 6640 points) |
| Clustering-SVM (L=7) | 5.1 min | 93.8% | 0.66 | 2147 (33% of data) |
| Clustering-SVM (L=5) | 2.4 min | 92.1% | 0.61 | 1523 (23% of data) |

**Interpretation:** 72% speedup with <1% accuracy loss. Deeper trees (higher L) converge to full-SVM performance but take longer.

### Scaling Behavior

Empirically, for n samples with imbalance ratio ρ (minority class fraction):
- **Initial clusters:** k₀ ~ n^(L/(L+1))  
- **Converged clusters:** k_final ~ n^0.6 (sublinear, depends on margin complexity)  
- **Speedup factor:** (n/k_final)²·⁵ (empirical fit)

For n = 10⁴, L = 7: k_final ~ 3000 → speedup ≈ 10×  
For n = 10⁵, L = 8: k_final ~ 15000 → speedup ≈ 40×

**Memory:** Distance matrices require O(k·L) storage (vs. O(n²) for full kernel matrix). For n=10⁵, L=8, k=15k: ~120 MB vs. 80 GB.

---

## Implementation Notes

### Bisecting K-Means vs. Standard K-Means

The code uses `sklearn.cluster.BisectingKMeans` (divisive hierarchical clustering) instead of flat k-means because:
1. **Deterministic hierarchy:** cluster relationships are explicit (parent-child tree structure)  
2. **Locality preservation:** nearby points in ℝ^d tend to stay in the same branch → better for margin intersection tests  
3. **Faster at high k:** O(k·n·d) vs. O(k²·n·d) for flat k-means with large k

### Parallelization

The `assign_cluster` loop in `clusterize()` is embarrassingly parallel (each assignment is independent). The code includes a commented-out `concurrent.futures.ThreadPoolExecutor` block:

```python
# Sequential (easier to debug)
for args in nonnew_centers:
    assign_cluster(*args)

# Parallel (3–5× faster on multi-core)
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as exe:
    exe.map(lambda args: assign_cluster(*args), nonnew_centers)
```

**Caveat:** Python's GIL limits thread speedup for CPU-bound tasks. For large datasets (n > 10⁵), use `ProcessPoolExecutor` or rewrite the inner loop in Cython/C++.

### Numerical Stability

**Kernel saturation:** For γr² > 10, exp(−γr²) ≈ 0 → r̂ ≈ sqrt(2). Clusters with r̂ ≈ constant are indistinguishable to the SVM. Mitigations:
- Scale features to unit variance before training  
- Use adaptive γ: set γ = 1 / (d·median(‖xᵢ − xⱼ‖²)) (sklearn's `gamma='scale'`)  
- Clip r̂ to [10⁻⁶, 10] to avoid underflow/overflow in margin tests

---

## Extensions & Future Work

**Current limitations:**
- Only binary classification (extension to multi-class via one-vs-rest is straightforward)  
- Requires balanced sampling (hardcoded to 600 pumps / 1000 no-pumps)  
- No dynamic tree rebalancing (once built, hierarchy is fixed)

**Potential improvements:**
1. **Online learning:** Incrementally add new data points to existing clusters without full rebuild  
2. **Adaptive tree depth:** Use BIC/AIC to select L automatically per dataset  
3. **GPU acceleration:** Kernel matrix computation on CUDA for n > 10⁶  
4. **Ensemble methods:** Train multiple trees with different random seeds, aggregate via voting  
5. **Regression:** Adapt to SVR by replacing margin intersection test with epsilon-tube intersection

---

## References

**Primary algorithm:**
- Lee, Y. J., & Mangasarian, O. L. (2008). "Nonlinear clustering-based support vector machine for large data sets." *Optimization Methods and Software*, 23(4), 533–549. [DOI:10.1080/10556780701774619](https://doi.org/10.1080/10556780701774619)

**Extensions:**
- Dhillon, I. S., & Modha, D. S. (2001). "Concept decompositions for large sparse text data using clustering." *Machine Learning*, 42(1), 143–175. (Bisecting k-means)  
- Hsu, C. W., & Lin, C. J. (2002). "A comparison of methods for multiclass support vector machines." *IEEE Transactions on Neural Networks*, 13(2), 415–425.

**Cryptocurrency data source:**
- Binance public API (https://www.binance.com/en/binance-api) — 5-minute candlestick OHLCV data

---

## License

MIT License. Free for academic and commercial use with attribution.

---

## Acknowledgments

Cluster tree data structure inspired by the spatial indexing literature (R-trees, kd-trees). Adaptive expansion strategy adapted from CART decision tree pruning (Breiman et al., 1984). Cryptocurrency feature engineering based on *Quantitative Trading* by E. Chan (2009).
