"""
Statistical testing tools
"""
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy import stats
import logging

logger = logging.getLogger(__name__)


class StatisticalTests:
    """Statistical tests for comparing experiments"""

    @staticmethod
    def compare_means(group1: np.ndarray, group2: np.ndarray,
                     test: str = 'welch') -> Dict:
        """
        Compare means of two groups

        Args:
            group1, group2: Arrays of values
            test: 'welch' (default), 'student', or 'mann_whitney'
        """
        if test == 'welch':
            stat, pvalue = stats.ttest_ind(group1, group2, equal_var=False)
        elif test == 'student':
            stat, pvalue = stats.ttest_ind(group1, group2, equal_var=True)
        elif test == 'mann_whitney':
            stat, pvalue = stats.mannwhitneyu(group1, group2, alternative='two-sided')
        else:
            raise ValueError(f"Unknown test: {test}")

        # Effect size (Cohen's d)
        pooled_std = np.sqrt((np.var(group1) + np.var(group2)) / 2)
        cohens_d = (np.mean(group1) - np.mean(group2)) / (pooled_std + 1e-10)

        return {
            'statistic': stat,
            'pvalue': pvalue,
            'group1_mean': np.mean(group1),
            'group2_mean': np.mean(group2),
            'group1_std': np.std(group1),
            'group2_std': np.std(group2),
            'cohens_d': cohens_d,
            'significant_05': pvalue < 0.05,
            'significant_01': pvalue < 0.01,
        }

    @staticmethod
    def compare_distributions(group1: np.ndarray, group2: np.ndarray) -> Dict:
        """Compare full distributions using KS test"""
        stat, pvalue = stats.ks_2samp(group1, group2)

        return {
            'ks_statistic': stat,
            'ks_pvalue': pvalue,
            'same_distribution': pvalue > 0.05,
        }

    @staticmethod
    def correlation_test(x: np.ndarray, y: np.ndarray,
                        method: str = 'pearson') -> Dict:
        """Test correlation between two variables"""
        if method == 'pearson':
            r, pvalue = stats.pearsonr(x, y)
        elif method == 'spearman':
            r, pvalue = stats.spearmanr(x, y)
        elif method == 'kendall':
            r, pvalue = stats.kendalltau(x, y)
        else:
            raise ValueError(f"Unknown method: {method}")

        return {
            'correlation': r,
            'pvalue': pvalue,
            'significant': pvalue < 0.05,
        }

    @staticmethod
    def bootstrap_ci(data: np.ndarray, statistic: str = 'mean',
                    n_bootstrap: int = 1000, ci: float = 0.95) -> Dict:
        """
        Bootstrap confidence interval

        Args:
            data: Array of values
            statistic: 'mean', 'median', 'std'
            n_bootstrap: Number of bootstrap samples
            ci: Confidence level
        """
        stat_fn = {
            'mean': np.mean,
            'median': np.median,
            'std': np.std,
        }[statistic]

        bootstrapped = []
        n = len(data)

        for _ in range(n_bootstrap):
            sample = np.random.choice(data, size=n, replace=True)
            bootstrapped.append(stat_fn(sample))

        bootstrapped = np.array(bootstrapped)

        alpha = (1 - ci) / 2
        lower = np.percentile(bootstrapped, alpha * 100)
        upper = np.percentile(bootstrapped, (1 - alpha) * 100)

        return {
            'point_estimate': stat_fn(data),
            'ci_lower': lower,
            'ci_upper': upper,
            'ci_level': ci,
            'se': np.std(bootstrapped),
        }

    @staticmethod
    def anova(groups: List[np.ndarray]) -> Dict:
        """One-way ANOVA across multiple groups"""
        stat, pvalue = stats.f_oneway(*groups)

        # Effect size (eta squared)
        all_data = np.concatenate(groups)
        grand_mean = np.mean(all_data)
        ss_between = sum(len(g) * (np.mean(g) - grand_mean)**2 for g in groups)
        ss_total = np.sum((all_data - grand_mean)**2)
        eta_squared = ss_between / (ss_total + 1e-10)

        return {
            'f_statistic': stat,
            'pvalue': pvalue,
            'eta_squared': eta_squared,
            'n_groups': len(groups),
            'significant': pvalue < 0.05,
        }

    @staticmethod
    def normality_test(data: np.ndarray) -> Dict:
        """Test for normality"""
        # Shapiro-Wilk (best for small samples)
        if len(data) <= 5000:
            sw_stat, sw_pvalue = stats.shapiro(data)
        else:
            # Sample for large datasets
            sample = np.random.choice(data, 5000, replace=False)
            sw_stat, sw_pvalue = stats.shapiro(sample)

        # D'Agostino-Pearson
        if len(data) >= 20:
            dp_stat, dp_pvalue = stats.normaltest(data)
        else:
            dp_stat, dp_pvalue = None, None

        return {
            'shapiro_statistic': sw_stat,
            'shapiro_pvalue': sw_pvalue,
            'dagostino_statistic': dp_stat,
            'dagostino_pvalue': dp_pvalue,
            'is_normal': sw_pvalue > 0.05,
        }

    @staticmethod
    def compare_experiments(exp1_metrics: Dict[str, List[float]],
                          exp2_metrics: Dict[str, List[float]]) -> Dict:
        """Compare two experiments across all metrics"""
        results = {}

        common_metrics = set(exp1_metrics.keys()) & set(exp2_metrics.keys())

        for metric in common_metrics:
            data1 = np.array(exp1_metrics[metric])
            data2 = np.array(exp2_metrics[metric])

            if len(data1) < 2 or len(data2) < 2:
                continue

            # Mean comparison
            mean_test = StatisticalTests.compare_means(data1, data2)

            # Distribution comparison
            dist_test = StatisticalTests.compare_distributions(data1, data2)

            results[metric] = {
                'mean_test': mean_test,
                'distribution_test': dist_test,
                'exp1_summary': {
                    'mean': np.mean(data1),
                    'std': np.std(data1),
                    'n': len(data1),
                },
                'exp2_summary': {
                    'mean': np.mean(data2),
                    'std': np.std(data2),
                    'n': len(data2),
                },
            }

        return results
