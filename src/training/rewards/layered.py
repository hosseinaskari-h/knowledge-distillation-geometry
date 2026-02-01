"""
Layered non-anthropocentric reward
Combines information theory, geometry, and dynamics
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict

from src.core.agent import AgentOutput
from .base import RewardFunction


class EmbeddingPredictor(nn.Module):
    """Small MLP to predict next embedding"""

    def __init__(self, hidden_dim: int, predictor_hidden: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim * 3, predictor_hidden),
            nn.ReLU(),
            nn.Linear(predictor_hidden, hidden_dim)
        )

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            context: [3, hidden_dim] - last 3 embeddings
        Returns:
            predicted: [hidden_dim] - predicted next embedding
        """
        x = context.flatten()
        return self.net(x)


class LayeredNonAnthropocentricReward(RewardFunction):
    """
    Multi-level reward with NO language interface

    Level 1: Information theory (MI, entropy)
    Level 2: Geometry (dimensionality, curvature)
    Level 3: Dynamics (predictability)
    """

    def __init__(self, config: dict):
        super().__init__(config)

        # Weights
        self.info_weight = config.get('info_weight', 0.4)
        self.geom_weight = config.get('geom_weight', 0.3)
        self.dynamics_weight = config.get('dynamics_weight', 0.3)

        # Geometry targets (from thesis findings - but we'll test if they emerge)
        self.target_dimension = config.get('target_dimension', 6.0)
        self.target_ricci = config.get('target_ricci', 0.514)

        # Buffers
        self.embedding_buffer = []
        self.buffer_size = config.get('buffer_size', 100)

        # Predictor
        hidden_dim = config.get('hidden_dim', 768)
        predictor_hidden = config.get('predictor_hidden', 512)

        # Determine device
        self._device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')

        self.predictor = EmbeddingPredictor(hidden_dim, predictor_hidden).to(self._device)
        self.predictor_optimizer = torch.optim.Adam(
            self.predictor.parameters(),
            lr=config.get('predictor_lr', 1e-4)
        )

    def compute_reward(self,
                      episode_history: List[AgentOutput],
                      current_output: AgentOutput) -> Tuple[float, Dict]:

        if len(episode_history) < 2:
            return 0.0, {}

        breakdown = {}

        # === LEVEL 1: INFORMATION ===
        prev_emb = episode_history[-1].get_last_embedding()
        curr_emb = current_output.get_last_embedding()

        # Mutual information
        mi = self._estimate_mutual_information(prev_emb, curr_emb)

        # Entropy
        entropy = self._estimate_entropy(curr_emb)

        # Info reward: high MI, moderate entropy (not too high, not too low)
        info_reward = mi - 0.1 * abs(entropy - 1.0)  # Target entropy around 1.0
        breakdown['mutual_information'] = mi
        breakdown['entropy'] = entropy
        breakdown['info_reward'] = info_reward

        # === LEVEL 2: GEOMETRY ===
        self.embedding_buffer.append(curr_emb.float().detach().clone())  # Cast to float32
        if len(self.embedding_buffer) > self.buffer_size:
            self.embedding_buffer.pop(0)

        geom_reward = 0.0
        if len(self.embedding_buffer) >= 20:
            embs = torch.stack(self.embedding_buffer)

            # Intrinsic dimensionality
            dim = self._intrinsic_dimensionality(embs)
            dim_reward = -abs(dim - self.target_dimension)

            # Ricci curvature approximation
            ricci = self._approximate_ricci(embs)
            ricci_reward = -abs(ricci - self.target_ricci)

            geom_reward = 0.5 * dim_reward + 0.5 * ricci_reward

            breakdown['dimension'] = dim
            breakdown['ricci'] = ricci
            breakdown['geom_reward'] = geom_reward

        # === LEVEL 3: DYNAMICS ===
        dynamics_reward = 0.0
        if len(episode_history) >= 3:
            context = torch.stack([
                episode_history[-3].get_last_embedding(),
                episode_history[-2].get_last_embedding(),
                episode_history[-1].get_last_embedding()
            ]).float().to(self._device)  # Cast to float32

            # Predict
            predicted = self.predictor(context)
            target = curr_emb.float().to(self._device)  # Cast to float32
            prediction_error = F.mse_loss(predicted, target)

            # Reward for predictability
            dynamics_reward = -prediction_error.item()

            # Update predictor
            self.predictor_optimizer.zero_grad()
            prediction_error.backward()
            self.predictor_optimizer.step()

            breakdown['prediction_error'] = prediction_error.item()
            breakdown['dynamics_reward'] = dynamics_reward

        # === COMBINE ===
        total_reward = (
            self.info_weight * info_reward +
            self.geom_weight * geom_reward +
            self.dynamics_weight * dynamics_reward
        )

        breakdown['total'] = total_reward

        return total_reward, breakdown

    def _estimate_mutual_information(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """Correlation-based MI estimate"""
        x = x.float()
        y = y.float()

        # Handle dimension mismatch (different model architectures)
        if x.shape[-1] != y.shape[-1]:
            min_dim = min(x.shape[-1], y.shape[-1])
            x = x[..., :min_dim]  # Truncate to smaller dimension
            y = y[..., :min_dim]

        x_norm = F.normalize(x.unsqueeze(0), dim=-1)
        y_norm = F.normalize(y.unsqueeze(0), dim=-1)
        correlation = (x_norm * y_norm).sum().item()

        if abs(correlation) < 0.999:
            mi = -0.5 * np.log(1 - correlation**2)
        else:
            mi = 10.0

        return max(0.0, mi)

    def _estimate_entropy(self, x: torch.Tensor) -> float:
        """Estimate entropy of embedding using variance as proxy"""
        return torch.var(x.float()).item()  # Cast to float32

    def _intrinsic_dimensionality(self, embeddings: torch.Tensor) -> float:
        """MLE-based intrinsic dimension"""
        try:
            from sklearn.neighbors import NearestNeighbors

            X = embeddings.cpu().numpy()
            k = min(10, len(X) - 1)

            if k < 2:
                return 0.0

            nbrs = NearestNeighbors(n_neighbors=k+1).fit(X)
            distances, _ = nbrs.kneighbors(X)
            distances = distances[:, 1:]
            m_k = distances[:, -1]

            dims = []
            for i in range(len(X)):
                if m_k[i] > 1e-10:
                    log_ratios = np.log(m_k[i] / (distances[i, :-1] + 1e-10))
                    if np.sum(log_ratios) != 0:
                        dim = (k - 1) / np.sum(log_ratios)
                        if np.isfinite(dim) and dim > 0:
                            dims.append(dim)

            return np.mean(dims) if dims else 0.0
        except Exception:
            return 0.0

    def _approximate_ricci(self, embeddings: torch.Tensor) -> float:
        """Fast Ricci curvature approximation"""
        try:
            from sklearn.neighbors import NearestNeighbors

            X = embeddings.cpu().numpy()
            k = min(5, len(X) - 1)

            if k < 2:
                return 0.0

            nbrs = NearestNeighbors(n_neighbors=k+1).fit(X)
            distances, _ = nbrs.kneighbors(X)

            curvatures = []
            for i in range(len(X)):
                neighbor_dists = distances[i, 1:]
                # Positive curvature = low variance (clustered)
                if np.var(neighbor_dists) > 0:
                    curvature = 1.0 / (1.0 + np.var(neighbor_dists))
                else:
                    curvature = 1.0
                curvatures.append(curvature)

            return np.mean(curvatures)
        except Exception:
            return 0.0

    def reset(self):
        """Reset internal state"""
        self.embedding_buffer = []
