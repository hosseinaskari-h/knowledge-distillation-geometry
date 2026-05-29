"""
Information-theoretic analysis tools
"""
import numpy as np
from typing import Dict, List, Tuple
from scipy.special import digamma
import logging

logger = logging.getLogger(__name__)


class InformationAnalyzer:
    """Information-theoretic analysis of embedding sequences"""

    def __init__(self, embeddings: np.ndarray):
        """
        Args:
            embeddings: [N, D] array of embeddings (in sequence order)
        """
        self.embeddings = embeddings
        self.n_samples, self.dim = embeddings.shape

    def mutual_information_pairs(self, lag: int = 1) -> List[float]:
        """Compute MI between consecutive pairs with given lag"""
        mi_values = []

        for i in range(self.n_samples - lag):
            x = self.embeddings[i]
            y = self.embeddings[i + lag]
            mi = self._correlation_mi(x, y)
            mi_values.append(mi)

        return mi_values

    def _correlation_mi(self, x: np.ndarray, y: np.ndarray) -> float:
        """Fast correlation-based MI approximation"""
        x_norm = (x - x.mean()) / (x.std() + 1e-10)
        y_norm = (y - y.mean()) / (y.std() + 1e-10)

        rho = np.mean(x_norm * y_norm)

        if abs(rho) < 0.999:
            mi = -0.5 * np.log(1 - rho**2)
        else:
            mi = 10.0

        return max(0.0, mi)

    def entropy_rate(self, k: int = 3) -> float:
        """Estimate entropy rate of the sequence"""
        if self.n_samples < k + 1:
            return 0.0

        # Use differential entropy approximation
        from sklearn.neighbors import NearestNeighbors

        # Create k-history vectors
        histories = np.array([
            self.embeddings[i:i+k].flatten()
            for i in range(self.n_samples - k)
        ])

        if len(histories) < k + 1:
            return 0.0

        # kNN entropy estimator
        nbrs = NearestNeighbors(n_neighbors=k).fit(histories)
        distances, _ = nbrs.kneighbors(histories)

        # Kozachenko-Leonenko estimator
        log_distances = np.log(distances[:, -1] + 1e-10)
        entropy = np.mean(log_distances) + np.log(len(histories)) + digamma(k)

        return entropy

    def channel_capacity(self, method: str = 'waterfilling') -> float:
        """
        Estimate channel capacity of the communication

        Uses water-filling algorithm on covariance eigenvalues
        """
        # Covariance of embeddings
        cov = np.cov(self.embeddings.T)
        eigenvalues = np.linalg.eigvalsh(cov)
        eigenvalues = np.maximum(eigenvalues, 0)
        eigenvalues = eigenvalues[::-1]  # Descending

        if method == 'waterfilling':
            # Water-filling for Gaussian channel
            power = 1.0  # Total power constraint
            noise = 0.1  # Noise level

            # Water level
            water_level = (power + np.sum(noise / (eigenvalues + 1e-10))) / len(eigenvalues)

            # Capacity
            capacity = 0.0
            for eig in eigenvalues:
                if eig > 0:
                    alloc = max(0, water_level - noise / eig)
                    if alloc > 0:
                        capacity += 0.5 * np.log2(1 + alloc * eig / noise)

            return capacity
        else:
            # Simple capacity estimate
            return 0.5 * np.sum(np.log2(1 + eigenvalues / 0.1))

    def information_bottleneck_analysis(self) -> Dict:
        """Analyze compression-relevance tradeoff"""
        # Split into input (t) and output (t+1)
        X = self.embeddings[:-1]
        Y = self.embeddings[1:]

        # Compute mutual information I(X;Y)
        mi_xy = np.mean([self._correlation_mi(x, y) for x, y in zip(X, Y)])

        # Estimate compression via PCA dimension
        from sklearn.decomposition import PCA
        pca = PCA(n_components=min(50, self.dim))
        pca.fit(X)
        cumvar = np.cumsum(pca.explained_variance_ratio_)
        compressed_dim = np.searchsorted(cumvar, 0.95) + 1

        return {
            'mutual_information': mi_xy,
            'compressed_dimension': compressed_dim,
            'compression_ratio': compressed_dim / self.dim,
        }

    def full_analysis(self) -> Dict:
        """Run complete information analysis"""
        logger.info("Running information analysis...")

        results = {
            'n_samples': self.n_samples,
            'dim': self.dim,
        }

        # MI analysis
        mi_lag1 = self.mutual_information_pairs(lag=1)
        mi_lag2 = self.mutual_information_pairs(lag=2)

        results['mutual_information'] = {
            'lag1_mean': np.mean(mi_lag1),
            'lag1_std': np.std(mi_lag1),
            'lag2_mean': np.mean(mi_lag2),
            'lag2_std': np.std(mi_lag2),
        }

        # Entropy rate
        results['entropy_rate'] = self.entropy_rate()

        # Channel capacity
        results['channel_capacity'] = self.channel_capacity()

        # Information bottleneck
        results['bottleneck'] = self.information_bottleneck_analysis()

        logger.info(f"Analysis complete: capacity = {results['channel_capacity']:.2f} bits")

        return results
