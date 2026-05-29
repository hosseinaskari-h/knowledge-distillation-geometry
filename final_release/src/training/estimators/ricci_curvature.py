"""
Ricci Curvature Estimators
"""
import torch
import numpy as np
from typing import Union, Optional, Tuple


class RicciCurvatureEstimator:
    """Estimate Ricci curvature on point clouds"""

    def __init__(self, method: str = 'ollivier'):
        """
        Args:
            method: Estimation method ('ollivier', 'forman', 'approx')
        """
        self.method = method

    def estimate(self, embeddings: Union[torch.Tensor, np.ndarray],
                 k: Optional[int] = None) -> Tuple[float, np.ndarray]:
        """
        Estimate Ricci curvature

        Args:
            embeddings: Point cloud [N, D]
            k: Number of neighbors

        Returns:
            mean_curvature: Mean curvature across edges
            curvatures: Per-edge curvature values
        """
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.detach().cpu().numpy()

        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)

        n = len(embeddings)
        if k is None:
            k = min(5, n - 1)

        if n < 3:
            return 0.0, np.array([])

        if self.method == 'ollivier':
            return self._ollivier_ricci(embeddings, k)
        elif self.method == 'forman':
            return self._forman_ricci(embeddings, k)
        elif self.method == 'approx':
            return self._approx_ricci(embeddings, k)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def _ollivier_ricci(self, X: np.ndarray, k: int) -> Tuple[float, np.ndarray]:
        """
        Ollivier-Ricci curvature
        Uses Wasserstein distance between neighborhood distributions
        """
        try:
            from sklearn.neighbors import NearestNeighbors
            from scipy.stats import wasserstein_distance

            n = len(X)
            nbrs = NearestNeighbors(n_neighbors=k+1).fit(X)
            distances, indices = nbrs.kneighbors(X)

            curvatures = []

            # Sample edges
            num_edges = min(100, n * k)
            for _ in range(num_edges):
                i = np.random.randint(n)
                j = indices[i, np.random.randint(1, k+1)]

                # Neighborhood distributions (simplified: uniform over neighbors)
                neighbors_i = indices[i, 1:]
                neighbors_j = indices[j, 1:]

                # Distance between centers
                d_ij = np.linalg.norm(X[i] - X[j])

                if d_ij < 1e-10:
                    continue

                # Wasserstein between neighborhoods (1D projection)
                proj_i = np.linalg.norm(X[neighbors_i] - X[i], axis=1)
                proj_j = np.linalg.norm(X[neighbors_j] - X[j], axis=1)

                w_dist = wasserstein_distance(proj_i, proj_j)

                # Ollivier curvature: 1 - W(mu_x, mu_y) / d(x,y)
                kappa = 1 - w_dist / d_ij
                curvatures.append(kappa)

            curvatures = np.array(curvatures)
            return np.mean(curvatures) if len(curvatures) > 0 else 0.0, curvatures

        except Exception:
            return self._approx_ricci(X, k)

    def _forman_ricci(self, X: np.ndarray, k: int) -> Tuple[float, np.ndarray]:
        """
        Forman-Ricci curvature
        Combinatorial curvature based on edge triangles
        """
        try:
            from sklearn.neighbors import NearestNeighbors

            n = len(X)
            nbrs = NearestNeighbors(n_neighbors=k+1).fit(X)
            distances, indices = nbrs.kneighbors(X)

            curvatures = []

            # For each edge, count triangles
            for i in range(n):
                for j_idx in range(1, k+1):
                    j = indices[i, j_idx]

                    if j <= i:
                        continue

                    # Count common neighbors (triangles)
                    neighbors_i = set(indices[i, 1:])
                    neighbors_j = set(indices[j, 1:])
                    common = len(neighbors_i & neighbors_j)

                    # Forman curvature for edge
                    # kappa = 4 - degree(i) - degree(j) + 3 * triangles
                    kappa = 4 - k - k + 3 * common
                    curvatures.append(kappa)

            curvatures = np.array(curvatures)
            return np.mean(curvatures) if len(curvatures) > 0 else 0.0, curvatures

        except Exception:
            return self._approx_ricci(X, k)

    def _approx_ricci(self, X: np.ndarray, k: int) -> Tuple[float, np.ndarray]:
        """
        Fast approximate Ricci curvature
        Based on local neighborhood density variation
        """
        try:
            from sklearn.neighbors import NearestNeighbors

            n = len(X)
            nbrs = NearestNeighbors(n_neighbors=k+1).fit(X)
            distances, _ = nbrs.kneighbors(X)

            curvatures = []
            for i in range(n):
                neighbor_dists = distances[i, 1:]

                # Positive curvature = low variance (clustered)
                # Negative curvature = high variance (spread out)
                if np.var(neighbor_dists) > 0:
                    curvature = 1.0 / (1.0 + np.var(neighbor_dists))
                else:
                    curvature = 1.0

                curvatures.append(curvature)

            curvatures = np.array(curvatures)
            return np.mean(curvatures), curvatures

        except Exception:
            return 0.0, np.array([])
