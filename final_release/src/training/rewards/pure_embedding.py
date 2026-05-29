"""
Pure embedding space reward - NO language evaluation
"""
import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict

from src.core.agent import AgentOutput
from .base import RewardFunction


class PureEmbeddingSpaceReward(RewardFunction):
    """
    Reward based ONLY on embedding space dynamics
    Zero human/language interpretation
    """

    def __init__(self, config: dict):
        super().__init__(config)

        # Weights for components
        self.predictability_weight = config.get('predictability_weight', 0.3)
        self.information_weight = config.get('information_weight', 0.3)
        self.compression_weight = config.get('compression_weight', 0.2)
        self.diversity_weight = config.get('diversity_weight', 0.1)
        self.temporal_weight = config.get('temporal_weight', 0.1)

    def compute_reward(self,
                      episode_history: List[AgentOutput],
                      current_output: AgentOutput) -> Tuple[float, Dict]:
        """Compute reward from embedding space only"""

        if len(episode_history) < 2:
            return 0.0, {}

        # Extract embeddings
        embeddings = torch.stack([h.get_last_embedding().float() for h in episode_history])
        current_emb = current_output.get_last_embedding().float()

        breakdown = {}

        # === Component 1: Predictability ===
        if len(embeddings) >= 3:
            context = embeddings[-3:]
            predicted = self._linear_extrapolation(context)
            prediction_error = F.mse_loss(predicted, current_emb)
            predictability_reward = -prediction_error.item()
        else:
            predictability_reward = 0.0
        breakdown['predictability'] = predictability_reward

        # === Component 2: Mutual Information ===
        prev_emb = embeddings[-1]
        mi = self._estimate_mutual_information(prev_emb, current_emb)
        breakdown['mutual_information'] = mi

        # === Component 3: Compression (Intrinsic Dimensionality) ===
        all_embs = torch.cat([embeddings, current_emb.unsqueeze(0)])
        if len(all_embs) >= 10:
            intrinsic_dim = self._estimate_intrinsic_dimension(all_embs)
            compression_reward = -intrinsic_dim  # Reward for low dim
        else:
            intrinsic_dim = 0.0
            compression_reward = 0.0
        breakdown['intrinsic_dim'] = intrinsic_dim
        breakdown['compression'] = compression_reward

        # === Component 4: Diversity (Anti-Collapse) ===
        diversity = self._embedding_diversity(all_embs)
        breakdown['diversity'] = diversity

        # === Component 5: Temporal Coherence ===
        if len(embeddings) >= 2:
            prev_transition = embeddings[-1] - embeddings[-2]
            current_transition = current_emb - embeddings[-1]
            transition_consistency = F.cosine_similarity(
                prev_transition.unsqueeze(0),
                current_transition.unsqueeze(0),
                dim=1
            ).item()
            temporal_reward = transition_consistency
        else:
            temporal_reward = 0.0
        breakdown['temporal'] = temporal_reward

        # === Combine ===
        total_reward = (
            self.predictability_weight * predictability_reward +
            self.information_weight * mi +
            self.compression_weight * compression_reward +
            self.diversity_weight * diversity +
            self.temporal_weight * temporal_reward
        )

        breakdown['total'] = total_reward

        return total_reward, breakdown

    def _linear_extrapolation(self, context: torch.Tensor) -> torch.Tensor:
        """Predict next embedding from context"""
        if len(context) == 2:
            # Simple: e_t = 2*e_{t-1} - e_{t-2}
            return 2 * context[-1] - context[-2]
        elif len(context) >= 3:
            # Momentum-based
            v1 = context[-1] - context[-2]
            v2 = context[-2] - context[-3]
            avg_velocity = 0.7 * v1 + 0.3 * v2
            return context[-1] + avg_velocity
        else:
            return context[-1]

    def _estimate_mutual_information(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """
        Estimate MI between two embeddings
        Simplified correlation-based estimator
        """
        x_norm = F.normalize(x.unsqueeze(0), dim=-1)
        y_norm = F.normalize(y.unsqueeze(0), dim=-1)
        correlation = (x_norm * y_norm).sum().item()

        # MI approximation: -0.5 * log(1 - rho^2)
        if abs(correlation) < 0.999:
            mi = -0.5 * np.log(1 - correlation**2)
        else:
            mi = 10.0

        return mi

    def _estimate_intrinsic_dimension(self, embeddings: torch.Tensor) -> float:
        """
        Estimate intrinsic dimensionality using MLE
        Lower = more efficient compression
        """
        try:
            from sklearn.neighbors import NearestNeighbors

            X = embeddings.detach().cpu().numpy()
            k = min(10, len(X) - 1)

            if k < 2:
                return 0.0

            nbrs = NearestNeighbors(n_neighbors=k+1).fit(X)
            distances, _ = nbrs.kneighbors(X)

            # MLE estimator
            distances = distances[:, 1:]  # exclude self
            m_k = distances[:, -1]

            dims = []
            for i in range(len(X)):
                if m_k[i] > 1e-10:
                    dim = (k - 1) / np.sum(np.log(m_k[i] / (distances[i, :-1] + 1e-10)))
                    if np.isfinite(dim) and dim > 0:
                        dims.append(dim)

            return np.mean(dims) if dims else 0.0
        except Exception:
            return 0.0

    def _embedding_diversity(self, embeddings: torch.Tensor) -> float:
        """Measure diversity to prevent collapse"""
        if len(embeddings) < 2:
            return 0.0

        # Pairwise distances
        dists = torch.cdist(embeddings, embeddings)
        mask = ~torch.eye(len(embeddings), dtype=torch.bool, device=embeddings.device)
        mean_dist = dists[mask].mean().item()

        return mean_dist
