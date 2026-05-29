"""
Metrics tracking utilities
"""
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any
from collections import defaultdict
import time


class MetricsTracker:
    """Track and aggregate metrics during training"""

    def __init__(self, log_dir: Optional[Path] = None):
        self.log_dir = Path(log_dir) if log_dir else None
        self.metrics: Dict[str, List[float]] = defaultdict(list)
        self.step_metrics: Dict[str, List[Dict]] = defaultdict(list)
        self.start_time = time.time()

        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)

    def log(self, metrics: Dict[str, float], step: Optional[int] = None):
        """
        Log metrics

        Args:
            metrics: Dict of metric name -> value
            step: Optional step number
        """
        timestamp = time.time() - self.start_time

        for name, value in metrics.items():
            self.metrics[name].append(value)

            if step is not None:
                self.step_metrics[name].append({
                    'step': step,
                    'value': value,
                    'time': timestamp,
                })

    def get(self, name: str) -> List[float]:
        """Get all values for a metric"""
        return self.metrics.get(name, [])

    def get_latest(self, name: str) -> Optional[float]:
        """Get latest value for a metric"""
        values = self.metrics.get(name, [])
        return values[-1] if values else None

    def get_mean(self, name: str, last_n: Optional[int] = None) -> float:
        """Get mean of metric values"""
        values = self.metrics.get(name, [])
        if not values:
            return 0.0
        if last_n:
            values = values[-last_n:]
        return np.mean(values)

    def get_std(self, name: str, last_n: Optional[int] = None) -> float:
        """Get std of metric values"""
        values = self.metrics.get(name, [])
        if not values:
            return 0.0
        if last_n:
            values = values[-last_n:]
        return np.std(values)

    def get_summary(self, last_n: Optional[int] = None) -> Dict[str, Dict[str, float]]:
        """Get summary statistics for all metrics"""
        summary = {}
        for name in self.metrics:
            values = self.metrics[name]
            if last_n:
                values = values[-last_n:]
            if values:
                summary[name] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'last': values[-1],
                    'count': len(values),
                }
        return summary

    def save(self, filename: str = "metrics.json"):
        """Save metrics to file"""
        if not self.log_dir:
            return

        filepath = self.log_dir / filename
        data = {
            'metrics': dict(self.metrics),
            'step_metrics': dict(self.step_metrics),
            'elapsed_time': time.time() - self.start_time,
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def load(self, filename: str = "metrics.json"):
        """Load metrics from file"""
        if not self.log_dir:
            return

        filepath = self.log_dir / filename
        if not filepath.exists():
            return

        with open(filepath, 'r') as f:
            data = json.load(f)

        self.metrics = defaultdict(list, data.get('metrics', {}))
        self.step_metrics = defaultdict(list, data.get('step_metrics', {}))

    def format_latest(self, names: Optional[List[str]] = None) -> str:
        """Format latest metrics as string"""
        if names is None:
            names = list(self.metrics.keys())

        parts = []
        for name in names:
            value = self.get_latest(name)
            if value is not None:
                if abs(value) < 0.01:
                    parts.append(f"{name}: {value:.2e}")
                else:
                    parts.append(f"{name}: {value:.4f}")

        return " | ".join(parts)


class RunningMean:
    """Compute running mean with exponential smoothing"""

    def __init__(self, alpha: float = 0.1):
        """
        Args:
            alpha: Smoothing factor (0-1), higher = more weight on recent
        """
        self.alpha = alpha
        self.value = None

    def update(self, x: float) -> float:
        """Update with new value and return smoothed value"""
        if self.value is None:
            self.value = x
        else:
            self.value = self.alpha * x + (1 - self.alpha) * self.value
        return self.value

    def get(self) -> Optional[float]:
        """Get current smoothed value"""
        return self.value

    def reset(self):
        """Reset state"""
        self.value = None
