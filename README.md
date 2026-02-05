# Low-Rank-SVD-Reduction

**An interactive, high-fidelity environment for real-time Singular Value Decomposition (SVD) analysis and topological signal recovery.**

This project demonstrates the **optimal low-rank approximation** of matrices via truncated SVD, illustrating how Gaussian noise degrades the recovery of the underlying signal. It implements **full, truncated, and randomized SVD methods** with numerical safeguards and error analysis.

---

## Mathematical Background

### Eckart–Young–Mirsky Theorem
The theorem guarantees that the **best rank-*k* approximation** of a matrix \( A \) (in Frobenius or spectral norm) is given by the **truncated SVD**, retaining only the *k* largest singular values and vectors.

- **Optimal Compression**: No other rank-*k* matrix can approximate \( A \) with lower error.
- **Denoising**: Truncating the SVD discards noise-dominated components, recovering the true signal.

---

### Reconstruction Error Dynamics
We analyze the **reconstruction error** \( \|A - \hat{A}_k\| \) and observe:

1. **Underfitting** (*k < true rank*): Error decreases as more signal components are included.
2. **Optimal Recovery** (*k ≈ true rank*): Error is minimized, and the signal is fully recovered.
3. **Overfitting** (*k > true rank*): Error relative to the **clean signal** increases as noise components are included, leading to a **U-shaped error curve** when comparing to the ground truth.

---

## Key Features

- **Real-Time 3D Visualization**: Interactive exploration of rank-*k* approximations.
- **Error Analysis**: Quantitative metrics (MSE, Frobenius norm) to assess reconstruction quality.
- **Numerical Safeguards**: Robust handling of large matrices (1000×1000) with optimized striding for performance.

---

## Applications

- **Dimensionality Reduction**: Compress large datasets while preserving structure.
- **Denoising**: Filter out Gaussian noise from signals.
- **Latent Feature Extraction**: Identify dominant patterns in high-dimensional data.

---

## Implementation

- **Full SVD**: Baseline spectral analysis.
- **Truncated SVD**: Optimal hard thresholding for denoising.
- **Randomized SVD**: High-speed approximation for massive datasets.

