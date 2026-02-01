"""
Base agent interfaces and data structures
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
import torch


@dataclass
class AgentOutput:
    """Structured output from agent generation"""
    text: str
    tokens: torch.Tensor  # [seq_len]
    embeddings: torch.Tensor  # [seq_len, hidden_dim] or [hidden_dim] for last token
    logits: Optional[torch.Tensor] = None  # [seq_len, vocab_size]
    hidden_states: Optional[tuple] = None  # tuple of [n_layers, seq_len, hidden_dim]
    attentions: Optional[tuple] = None  # tuple of [n_layers, n_heads, seq_len, seq_len]
    metadata: Dict = field(default_factory=dict)

    def get_last_embedding(self) -> torch.Tensor:
        """Get embedding of last generated token"""
        if self.embeddings.dim() == 1:
            return self.embeddings
        else:
            return self.embeddings[-1]

    def to_dict(self) -> dict:
        """Convert to serializable dict"""
        return {
            'text': self.text,
            'tokens': self.tokens.cpu().tolist(),
            'embeddings': self.embeddings.cpu().numpy(),
            'metadata': self.metadata,
        }


class Agent(ABC):
    """Base agent interface"""

    def __init__(self, name: str, config: dict):
        self.name = name
        self.config = config

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> AgentOutput:
        """Generate response to prompt"""
        pass

    @abstractmethod
    def generate_with_log_prob(self, prompt: str, **kwargs) -> Tuple[AgentOutput, torch.Tensor]:
        """Generate with log probability for RL training"""
        pass

    @abstractmethod
    def get_embeddings(self, text: str) -> torch.Tensor:
        """Get embeddings for text without generation"""
        pass

    @abstractmethod
    def save_checkpoint(self, path: str):
        """Save agent state"""
        pass

    @abstractmethod
    def load_checkpoint(self, path: str):
        """Load agent state"""
        pass

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Dimensionality of embedding space"""
        pass

    @property
    def device(self) -> str:
        """Device agent is on"""
        return self.config.get('device', 'cpu')
