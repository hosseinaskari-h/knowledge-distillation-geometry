"""
Temporal analysis tools
"""
import numpy as np
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)


class TemporalAnalyzer:
    """Analyze temporal dynamics of embedding sequences"""

    def __init__(self, embeddings: np.ndarray, episode_boundaries: List[int] = None):
        """
        Args:
            embeddings: [N, D] array of embeddings in temporal order
            episode_boundaries: List of indices where new episodes start
        """
        self.embeddings = embeddings
        self.n_samples, self.dim = embeddings.shape
        self.episode_boundaries = episode_boundaries or []

    def velocity_analysis(self) -> Dict:
        """Analyze embedding space velocity (rate of change)"""
        if self.n_samples < 2:
            return {'mean': 0.0, 'std': 0.0, 'velocities': []}

        # Compute velocities
        velocities = np.linalg.norm(
            self.embeddings[1:] - self.embeddings[:-1],
            axis=1
        )

        return {
            'mean': np.mean(velocities),
            'std': np.std(velocities),
            'min': np.min(velocities),
            'max': np.max(velocities),
            'velocities': velocities,
        }

    def acceleration_analysis(self) -> Dict:
        """Analyze embedding space acceleration"""
        if self.n_samples < 3:
            return {'mean': 0.0, 'std': 0.0, 'accelerations': []}

        # Velocities
        v = self.embeddings[1:] - self.embeddings[:-1]

        # Accelerations
        a = v[1:] - v[:-1]
        accelerations = np.linalg.norm(a, axis=1)

        return {
            'mean': np.mean(accelerations),
            'std': np.std(accelerations),
            'accelerations': accelerations,
        }

    def autocorrelation(self, max_lag: int = 50) -> Dict:
        """Compute autocorrelation function"""
        max_lag = min(max_lag, self.n_samples // 2)

        # Use first principal component for autocorrelation
        from sklearn.decomposition import PCA
        pca = PCA(n_components=1)
        x = pca.fit_transform(self.embeddings).flatten()

        # Normalize
        x = (x - x.mean()) / (x.std() + 1e-10)

        # Autocorrelation
        acf = []
        for lag in range(max_lag):
            if lag == 0:
                acf.append(1.0)
            else:
                corr = np.corrcoef(x[:-lag], x[lag:])[0, 1]
                acf.append(corr if np.isfinite(corr) else 0.0)

        # Decorrelation time (first zero crossing)
        decorr_time = max_lag
        for i, val in enumerate(acf):
            if val < 0:
                decorr_time = i
                break

        return {
            'acf': acf,
            'decorrelation_time': decorr_time,
        }

    def phase_analysis(self, window_size: int = 1000) -> Dict:
        """
        Analyze developmental phases

        Based on thesis findings:
        - Exploration (0-10k): High variance, unstable
        - Consolidation (10k-30k): Decreasing variance
        - Maturation (30k+): Stable, low variance
        """
        if self.n_samples < window_size:
            window_size = self.n_samples // 3

        n_windows = self.n_samples // window_size

        # Metrics per window
        variances = []
        velocities = []
        dimensions = []

        for i in range(n_windows):
            start = i * window_size
            end = start + window_size
            window = self.embeddings[start:end]

            # Variance
            var = np.var(window)
            variances.append(var)

            # Velocity
            if len(window) > 1:
                v = np.mean(np.linalg.norm(window[1:] - window[:-1], axis=1))
                velocities.append(v)
            else:
                velocities.append(0.0)

            # Dimension estimate (simplified)
            from sklearn.decomposition import PCA
            pca = PCA(n_components=min(10, window.shape[1]))
            pca.fit(window)
            cumvar = np.cumsum(pca.explained_variance_ratio_)
            dim = np.searchsorted(cumvar, 0.95) + 1
            dimensions.append(dim)

        return {
            'n_windows': n_windows,
            'window_size': window_size,
            'variances': variances,
            'velocities': velocities,
            'dimensions': dimensions,
            'variance_trend': np.polyfit(range(n_windows), variances, 1)[0] if n_windows > 1 else 0.0,
        }

    def stationarity_test(self) -> Dict:
        """Test for stationarity of the sequence"""
        try:
            from scipy import stats

            # Use first PC
            from sklearn.decomposition import PCA
            pca = PCA(n_components=1)
            x = pca.fit_transform(self.embeddings).flatten()

            # Split into halves
            mid = len(x) // 2
            first_half = x[:mid]
            second_half = x[mid:]

            # Compare distributions
            ks_stat, ks_pvalue = stats.ks_2samp(first_half, second_half)

            # Compare means (t-test)
            t_stat, t_pvalue = stats.ttest_ind(first_half, second_half)

            return {
                'ks_statistic': ks_stat,
                'ks_pvalue': ks_pvalue,
                't_statistic': t_stat,
                't_pvalue': t_pvalue,
                'is_stationary': ks_pvalue > 0.05 and t_pvalue > 0.05,
            }
        except Exception:
            return {'is_stationary': None}

    def full_analysis(self) -> Dict:
        """Run complete temporal analysis"""
        logger.info("Running temporal analysis...")

        results = {
            'n_samples': self.n_samples,
            'dim': self.dim,
        }

        # Velocity
        results['velocity'] = self.velocity_analysis()

        # Acceleration
        results['acceleration'] = self.acceleration_analysis()

        # Autocorrelation
        acf_results = self.autocorrelation()
        results['autocorrelation'] = {
            'decorrelation_time': acf_results['decorrelation_time'],
        }

        # Phase analysis
        results['phases'] = self.phase_analysis()

        # Stationarity
        results['stationarity'] = self.stationarity_test()

        logger.info(f"Analysis complete: decorr time = {acf_results['decorrelation_time']}")

        return results
