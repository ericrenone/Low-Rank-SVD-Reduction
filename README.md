# Low-Rank-SVD-Reduction
The script generates a synthetic low-rank matrix with noise, computes the optimal rank-k approximation, reports reconstruction error, and visualizes the singular value spectrum.

Demonstrates the **optimal low-rank approximation** of matrices via truncated SVD and how Gaussian noise degrades recovery of the underlying signal. 
Implements full, truncated, and randomized SVD methods with numerical safeguards/error analysis.

## Mathematical Background

The Eckart–Young–Mirsky theorem states that the **best rank-*k* approximation** Âₖ (in Frobenius or operator norm) is given by the truncated SVD keeping the *k* largest singular values/vectors.

We study reconstruction error and observe the typical U-shaped curve: error decreases until *k* ≈ *r*, then rises as noise overfitting dominates.
