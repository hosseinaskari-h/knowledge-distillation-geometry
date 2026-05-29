"""
Intrinsic Dimension Estimators
"""
import torch
import numpy as np
from typing import Union, Optional


class IntrinsicDimensionEstimator:
    """Estimate intrinsic dimensionality of point clouds"""

    def __init__(self, method: str = 'mle'):
        """
        Args:
            method: Estimation method ('mle', 'correlation', 'pca')
        """
        self.method = method

    def estimate(self, embeddings: Union[torch.Tensor, np.ndarray],
                 k: Optional[int] = None) -> float:
        """
        Estimate intrinsic dimension

        Args:
            embeddings: Point cloud [N, D]
            k: Number of neighbors for k-NN methods

        Returns:
            Estimated intrinsic dimension
        """
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.detach().cpu().numpy()

        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)

        n = len(embeddings)
        if k is None:
            k = min(10, n - 1)

        if n < 3:
            return 0.0

        if self.method == 'mle':
            return self._mle_estimator(embeddings, k)
        elif self.method == 'correlation':
            return self._correlation_dimension(embeddings)
        elif self.method == 'pca':
            return self._pca_dimension(embeddings)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def _mle_estimator(self, X: np.ndarray, k: int) -> float:
        """Maximum Likelihood Estimator (Levina-Bickel)"""
        try:
            from sklearn.neighbors import NearestNeighbors

            if k < 2:
                return 0.0

            nbrs = NearestNeighbors(n_neighbors=k+1).fit(X)
            distances, _ = nbrs.kneighbors(X)
            distances = distances[:, 1:]  # Exclude self

            m_k = distances[:, -1]  # Distance to k-th neighbor

            dims = []
            for i in range(len(X)):
                if m_k[i] > 1e-10:
                    log_ratios = np.log(m_k[i] / (distances[i, :-1] + 1e-10))
                    if np.sum(log_ratios) != 0:
                        dim = (k - 1) / np.sum(log_ratios)
                        if np.isfinite(dim) and dim > 0 and dim < 1000:
                            dims.append(dim)

            return np.mean(dims) if dims else 0.0

        except Exception:
            return 0.0

    def _correlation_dimension(self, X: np.ndarray, max_pairs: int = 5000) -> float:
        """Grassberger-Procaccia correlation dimension"""
        try:
            n = len(X)

            # Random sampling for efficiency
            if n * (n - 1) // 2 > max_pairs:
                indices = np.random.choice(n, size=int(np.sqrt(2 * max_pairs)), replace=False)
                X_sample = X[indices]
            else:
                X_sample = X

            # Compute pairwise distances
            from scipy.spatial.distance import pdist
            distances = pdist(X_sample)

            if len(distances) == 0:
                return 0.0

            # Range of scales
            r_min = np.percentile(distances, 5)
            r_max = np.percentile(distances, 95)

            if r_min <= 0 or r_max <= r_min:
                return 0.0

            # Correlation integral
            radii = np.logspace(np.log10(r_min), np.log10(r_max), 20)
            correlations = np.array([np.mean(distances < r) for r in radii])

            # Filter valid points
            valid = (correlations > 0) & (correlations < 1)
            if np.sum(valid) < 5:
                return 0.0

            log_r = np.log(radii[valid])
            log_c = np.log(correlations[valid])

            # Linear regression
            slope, _ = np.polyfit(log_r, log_c, 1)

            return max(0.0, slope)

        except Exception:
            return 0.0

    def _pca_dimension(self, X: np.ndarray, threshold: float = 0.95) -> float:
        """PCA-based dimension (number of components for threshold variance)"""
        try:
            from sklearn.decomposition import PCA

            # Center data
            X_centered = X - X.mean(axis=0)

            # Compute covariance eigenvalues
            cov = np.cov(X_centered.T)
            eigenvalues = np.linalg.eigvalsh(cov)
            eigenvalues = np.sort(eigenvalues)[::-1]
            eigenvalues = np.maximum(eigenvalues, 0)

            # Cumulative variance
            total_var = eigenvalues.sum()
            if total_var == 0:
                return 0.0

            cumvar = np.cumsum(eigenvalues) / total_var

            # Dimension for threshold
            dim = np.searchsorted(cumvar, threshold) + 1

            return float(dim)

        except Exception:
            return 0.0
