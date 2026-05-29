"""
Base reward function interface
"""
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict
from src.core.agent import AgentOutput


class RewardFunction(ABC):
    """Base class for reward functions"""

    def __init__(self, config: dict):
        self.config = config

    @abstractmethod
    def compute_reward(self,
                      episode_history: List[AgentOutput],
                      current_output: AgentOutput) -> Tuple[float, Dict]:
        """
        Compute reward for current step

        Args:
            episode_history: All previous outputs in episode
            current_output: Current agent output

        Returns:
            reward: Scalar reward value
            breakdown: Dict of individual reward components
        """
        pass

    def reset(self):
        """Reset any internal state"""
        pass
