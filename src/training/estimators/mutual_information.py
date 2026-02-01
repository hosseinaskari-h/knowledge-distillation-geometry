"""
Mutual Information Estimators
"""
import torch
import numpy as np
from typing import Union


class MutualInformationEstimator:
    """Estimate mutual information between high-dimensional embeddings"""

    def __init__(self, method: str = 'kraskov'):
        """
        Args:
            method: Estimation method ('kraskov', 'correlation', 'binned')
        """
        self.method = method

    def estimate(self, x: Union[torch.Tensor, np.ndarray],
                 y: Union[torch.Tensor, np.ndarray]) -> float:
        """
        Estimate MI between x and y

        Args:
            x: First embedding [N, D] or [D]
            y: Second embedding [N, D] or [D]

        Returns:
            Estimated mutual information in nats
        """
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        if isinstance(y, torch.Tensor):
            y = y.detach().cpu().numpy()

        # Handle single vectors
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if y.ndim == 1:
            y = y.reshape(1, -1)

        if self.method == 'correlation':
            return self._correlation_mi(x, y)
        elif self.method == 'kraskov':
            return self._kraskov_mi(x, y)
        elif self.method == 'binned':
            return self._binned_mi(x, y)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def _correlation_mi(self, x: np.ndarray, y: np.ndarray) -> float:
        """Fast correlation-based MI approximation"""
        # Flatten to vectors for correlation
        x_flat = x.flatten()
        y_flat = y.flatten()

        # Normalize
        x_norm = (x_flat - x_flat.mean()) / (x_flat.std() + 1e-10)
        y_norm = (y_flat - y_flat.mean()) / (y_flat.std() + 1e-10)

        # Correlation
        rho = np.mean(x_norm * y_norm)

        # MI approximation for Gaussian: -0.5 * log(1 - rho^2)
        if abs(rho) < 0.999:
            mi = -0.5 * np.log(1 - rho**2)
        else:
            mi = 10.0  # Cap at high value

        return max(0.0, mi)

    def _kraskov_mi(self, x: np.ndarray, y: np.ndarray) -> float:
        """Kraskov k-NN based MI estimator"""
        try:
            from sklearn.neighbors import NearestNeighbors
            from scipy.special import digamma

            # Combine for joint space
            xy = np.hstack([x, y])

            n = len(xy)
            k = min(3, n - 1)

            if k < 1:
                return 0.0

            # k-NN in joint space
            nbrs_xy = NearestNeighbors(n_neighbors=k+1, metric='chebyshev').fit(xy)
            distances, _ = nbrs_xy.kneighbors(xy)
            eps = distances[:, k]  # Distance to k-th neighbor

            # Count neighbors in marginal spaces within eps
            nbrs_x = NearestNeighbors(metric='chebyshev').fit(x)
            nbrs_y = NearestNeighbors(metric='chebyshev').fit(y)

            nx = np.array([
                len(nbrs_x.radius_neighbors([x[i]], eps[i], return_distance=False)[0]) - 1
                for i in range(n)
            ])
            ny = np.array([
                len(nbrs_y.radius_neighbors([y[i]], eps[i], return_distance=False)[0]) - 1
                for i in range(n)
            ])

            # Kraskov estimator
            mi = digamma(k) - np.mean(digamma(nx + 1) + digamma(ny + 1)) + digamma(n)

            return max(0.0, mi)

        except Exception:
            # Fallback to correlation method
            return self._correlation_mi(x, y)

    def _binned_mi(self, x: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
        """Simple binned MI estimator (for reference)"""
        try:
            # Use first principal components for binning
            from sklearn.decomposition import PCA

            if x.shape[1] > 1:
                pca = PCA(n_components=1)
                x_1d = pca.fit_transform(x).flatten()
                y_1d = pca.fit_transform(y).flatten()
            else:
                x_1d = x.flatten()
                y_1d = y.flatten()

            # Joint histogram
            hist_2d, _, _ = np.histogram2d(x_1d, y_1d, bins=bins)
            hist_2d = hist_2d / hist_2d.sum()

            # Marginals
            px = hist_2d.sum(axis=1)
            py = hist_2d.sum(axis=0)

            # MI
            mi = 0.0
            for i in range(bins):
                for j in range(bins):
                    if hist_2d[i, j] > 0 and px[i] > 0 and py[j] > 0:
                        mi += hist_2d[i, j] * np.log(hist_2d[i, j] / (px[i] * py[j]))

            return max(0.0, mi)

        except Exception:
            return 0.0
