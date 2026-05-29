"""
Dimensional analysis tools
"""
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import logging

logger = logging.getLogger(__name__)


class DimensionalAnalyzer:
    """Analyze dimensionality of embedding spaces"""

    def __init__(self, embeddings: np.ndarray):
        """
        Args:
            embeddings: [N, D] array of embeddings
        """
        self.embeddings = embeddings
        self.n_samples, self.ambient_dim = embeddings.shape

    def pca_analysis(self, n_components: Optional[int] = None) -> Dict:
        """PCA-based dimensionality analysis"""
        if n_components is None:
            n_components = min(50, self.ambient_dim, self.n_samples)

        pca = PCA(n_components=n_components)
        transformed = pca.fit_transform(self.embeddings)

        # Cumulative variance
        cumvar = np.cumsum(pca.explained_variance_ratio_)

        # Effective dimension (95% variance)
        effective_dim_95 = np.searchsorted(cumvar, 0.95) + 1

        # Participation ratio
        eigenvalues = pca.explained_variance_
        participation_ratio = (np.sum(eigenvalues) ** 2) / np.sum(eigenvalues ** 2)

        return {
            'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
            'cumulative_variance': cumvar.tolist(),
            'effective_dim_95': effective_dim_95,
            'participation_ratio': participation_ratio,
            'transformed': transformed,
        }

    def mle_intrinsic_dimension(self, k: int = 10) -> float:
        """MLE-based intrinsic dimension estimator"""
        from sklearn.neighbors import NearestNeighbors

        if self.n_samples < k + 1:
            return 0.0

        nbrs = NearestNeighbors(n_neighbors=k+1).fit(self.embeddings)
        distances, _ = nbrs.kneighbors(self.embeddings)
        distances = distances[:, 1:]  # Exclude self

        m_k = distances[:, -1]

        dims = []
        for i in range(self.n_samples):
            if m_k[i] > 1e-10:
                log_ratios = np.log(m_k[i] / (distances[i, :-1] + 1e-10))
                if np.sum(log_ratios) != 0:
                    dim = (k - 1) / np.sum(log_ratios)
                    if np.isfinite(dim) and 0 < dim < 1000:
                        dims.append(dim)

        return np.mean(dims) if dims else 0.0

    def correlation_dimension(self, n_pairs: int = 5000) -> float:
        """Grassberger-Procaccia correlation dimension"""
        from scipy.spatial.distance import pdist

        # Sample for efficiency
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

    def tsne_projection(self, n_components: int = 2, perplexity: float = 30.0) -> np.ndarray:
        """t-SNE projection for visualization"""
        if self.n_samples < perplexity:
            perplexity = max(5, self.n_samples - 1)

        tsne = TSNE(n_components=n_components, perplexity=perplexity)
        return tsne.fit_transform(self.embeddings)

    def full_analysis(self) -> Dict:
        """Run complete dimensional analysis"""
        logger.info("Running dimensional analysis...")

        results = {
            'n_samples': self.n_samples,
            'ambient_dim': self.ambient_dim,
        }

        # PCA
        pca_results = self.pca_analysis()
        results['pca'] = {
            'effective_dim_95': pca_results['effective_dim_95'],
            'participation_ratio': pca_results['participation_ratio'],
        }

        # MLE
        results['mle_dimension'] = self.mle_intrinsic_dimension()

        # Correlation dimension
        results['correlation_dimension'] = self.correlation_dimension()

        logger.info(f"Analysis complete: MLE dim = {results['mle_dimension']:.2f}")

        return results
