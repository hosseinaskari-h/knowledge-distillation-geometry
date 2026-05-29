"""
Geometric analysis tools
"""
import numpy as np
from typing import Dict, List, Tuple
from sklearn.neighbors import NearestNeighbors
import logging

logger = logging.getLogger(__name__)


class GeometricAnalyzer:
    """Analyze geometric properties of embedding spaces"""

    def __init__(self, embeddings: np.ndarray):
        """
        Args:
            embeddings: [N, D] array of embeddings
        """
        self.embeddings = embeddings
        self.n_samples, self.dim = embeddings.shape

    def ricci_curvature(self, k: int = 5, n_samples: int = 100) -> Dict:
        """
        Estimate Ricci curvature

        Returns distribution of curvatures across edges
        """
        from scipy.stats import wasserstein_distance

        if self.n_samples < k + 1:
            return {'mean': 0.0, 'std': 0.0, 'curvatures': []}

        nbrs = NearestNeighbors(n_neighbors=k+1).fit(self.embeddings)
        distances, indices = nbrs.kneighbors(self.embeddings)

        curvatures = []
        n_edges = min(n_samples, self.n_samples * k)

        for _ in range(n_edges):
            i = np.random.randint(self.n_samples)
            j = indices[i, np.random.randint(1, k+1)]

            neighbors_i = indices[i, 1:]
            neighbors_j = indices[j, 1:]

            d_ij = np.linalg.norm(self.embeddings[i] - self.embeddings[j])
            if d_ij < 1e-10:
                continue

            # 1D projections
            proj_i = np.linalg.norm(self.embeddings[neighbors_i] - self.embeddings[i], axis=1)
            proj_j = np.linalg.norm(self.embeddings[neighbors_j] - self.embeddings[j], axis=1)

            w_dist = wasserstein_distance(proj_i, proj_j)
            kappa = 1 - w_dist / d_ij
            curvatures.append(kappa)

        return {
            'mean': np.mean(curvatures) if curvatures else 0.0,
            'std': np.std(curvatures) if curvatures else 0.0,
            'curvatures': curvatures,
        }

    def local_density(self, k: int = 10) -> np.ndarray:
        """Compute local density at each point"""
        nbrs = NearestNeighbors(n_neighbors=k+1).fit(self.embeddings)
        distances, _ = nbrs.kneighbors(self.embeddings)

        # Density as inverse of mean k-NN distance
        mean_dist = distances[:, 1:].mean(axis=1)
        density = 1.0 / (mean_dist + 1e-10)

        return density

    def cluster_analysis(self, method: str = 'kmeans', n_clusters: int = None) -> Dict:
        """Analyze clustering structure"""
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score

        if n_clusters is None:
            # Estimate number of clusters
            n_clusters = max(2, int(np.sqrt(self.n_samples / 2)))

        if self.n_samples < n_clusters + 1:
            return {'n_clusters': 0, 'silhouette': 0.0}

        kmeans = KMeans(n_clusters=n_clusters, n_init=10)
        labels = kmeans.fit_predict(self.embeddings)

        silhouette = silhouette_score(self.embeddings, labels)

        # Cluster sizes
        unique, counts = np.unique(labels, return_counts=True)

        return {
            'n_clusters': n_clusters,
            'silhouette': silhouette,
            'cluster_sizes': counts.tolist(),
            'labels': labels,
        }

    def fractal_dimension(self, method: str = 'box_counting') -> float:
        """Estimate fractal dimension"""
        if method == 'box_counting':
            return self._box_counting_dimension()
        else:
            return self._correlation_dimension()

    def _box_counting_dimension(self) -> float:
        """Box counting dimension estimate"""
        # Normalize to unit cube
        data = self.embeddings.copy()
        data = (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0) + 1e-10)

        # Multiple scales
        scales = np.logspace(-1, 0, 10)
        counts = []

        for scale in scales:
            # Count occupied boxes
            boxes = np.floor(data / scale).astype(int)
            unique_boxes = len(set(map(tuple, boxes)))
            counts.append(unique_boxes)

        # Linear regression
        log_scales = np.log(scales)
        log_counts = np.log(counts)

        # Fractal dimension is negative slope
        slope, _ = np.polyfit(log_scales, log_counts, 1)
        return -slope

    def _correlation_dimension(self) -> float:
        """Correlation dimension"""
        from scipy.spatial.distance import pdist

        if self.n_samples > 500:
            idx = np.random.choice(self.n_samples, 500, replace=False)
            sample = self.embeddings[idx]
        else:
            sample = self.embeddings

        distances = pdist(sample)

        if len(distances) == 0:
            return 0.0

        r_min = np.percentile(distances, 5)
        r_max = np.percentile(distances, 95)

        if r_min <= 0 or r_max <= r_min:
            return 0.0

        radii = np.logspace(np.log10(r_min), np.log10(r_max), 20)
        correlations = np.array([np.mean(distances < r) for r in radii])

        valid = (correlations > 0) & (correlations < 1)
        if np.sum(valid) < 5:
            return 0.0

        log_r = np.log(radii[valid])
        log_c = np.log(correlations[valid])

        slope, _ = np.polyfit(log_r, log_c, 1)
        return max(0.0, slope)

    def full_analysis(self) -> Dict:
        """Run complete geometric analysis"""
        logger.info("Running geometric analysis...")

        results = {
            'n_samples': self.n_samples,
            'dim': self.dim,
        }

        # Ricci curvature
        ricci = self.ricci_curvature()
        results['ricci_curvature'] = {
            'mean': ricci['mean'],
            'std': ricci['std'],
        }

        # Local density stats
        density = self.local_density()
        results['density'] = {
            'mean': np.mean(density),
            'std': np.std(density),
            'min': np.min(density),
            'max': np.max(density),
        }

        # Clustering
        results['clustering'] = self.cluster_analysis()

        # Fractal dimension
        results['fractal_dimension'] = self.fractal_dimension()

        logger.info(f"Analysis complete: Ricci = {ricci['mean']:.3f}")

        return results
