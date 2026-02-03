# Low-Rank-SVD-Reduction
The script generates a synthetic low-rank matrix with noise, computes the optimal rank-k approximation, reports reconstruction error, and visualizes the singular value spectrum.
Demonstrates the **optimal low-rank approximation** of matrices via truncated SVD — and how Gaussian noise degrades recovery of the underlying signal. Implements full, truncated, and randomized SVD methods with numerical safeguards, error analysis, and publication-quality visualizations.

## Mathematical Background

Given clean low-rank signal **L** ∈ ℝ^{m × n} with rank *r*, and additive Gaussian noise **E** ∼ 𝒩(0, σ² I),

**A** = **L** + **E**

The Eckart–Young–Mirsky theorem states that the **best rank-*k* approximation** Âₖ (in Frobenius or operator norm) is given by the truncated SVD keeping the *k* largest singular values/vectors.

We study reconstruction error

**err_F(k, σ) = ‖A − Âₖ‖_F / ‖A‖_F**

and observe the typical U-shaped curve: error decreases until *k* ≈ *r*, then rises as noise overfitting dominates.

## Features

- Comparison of SVD methods: full (`numpy.linalg.svd`), truncated (`scipy.sparse.linalg.svds`), randomized (`sklearn.utils.extmath.randomized_svd`)
- Defensive numerics: `np.finfo(float).eps` instead of hardcoded 1e-12, rank safety checks (loud failure/warning if requested rank > min(m,n))
- Reproducible experiments with seeded noise
- Visualizations: side-by-side original / approx / residual heatmaps, singular value spectrum, error-vs-rank curves for multiple σ
- pytest suite checking **mathematical invariants** (rotation invariance, near-idempotence of projection, expected noise scaling)
