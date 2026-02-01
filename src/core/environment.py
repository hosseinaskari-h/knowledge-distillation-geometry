"""
Base environment interface
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple, Dict, TYPE_CHECKING
import torch

if TYPE_CHECKING:
    from .agent import Agent, AgentOutput


@dataclass
class Turn:
    """Single turn in conversation"""
    agent_name: str
    output: 'AgentOutput'
    step: int

    def to_dict(self) -> dict:
        return {
            'agent_name': self.agent_name,
            'output': self.output.to_dict(),
            'step': self.step,
        }


@dataclass
class Episode:
    """Complete episode of multi-agent communication"""
    id: str
    turns: List[Turn]
    initial_prompt: str
    metadata: Dict

    def __len__(self) -> int:
        return len(self.turns)

    def get_embeddings(self) -> torch.Tensor:
        """Get all embeddings as tensor [n_turns, hidden_dim]"""
        return torch.stack([turn.output.get_last_embedding() for turn in self.turns])

    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'turns': [turn.to_dict() for turn in self.turns],
            'initial_prompt': self.initial_prompt,
            'metadata': self.metadata,
        }


class Environment(ABC):
    """Base environment for agent interaction"""

    def __init__(self, config: dict):
        self.config = config

    @abstractmethod
    def reset(self) -> str:
        """Reset environment, return initial prompt"""
        pass

    @abstractmethod
    def step(self, agent_output: 'AgentOutput') -> Tuple[str, bool, Dict]:
        """
        Execute one step
        Returns: (next_prompt, done, info)
        """
        pass

    @abstractmethod
    def run_episode(self, agents: Dict[str, 'Agent']) -> Episode:
        """Run complete episode with given agents"""
        pass
