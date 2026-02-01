"""
Base agent class with common functionality
"""
from abc import ABC
from src.core.agent import Agent


class BaseAgent(Agent, ABC):
    """Base class with common agent functionality"""

    def __init__(self, name: str, config: dict):
        super().__init__(name, config)
        self.generation_count = 0

    def _increment_generation(self):
        """Track generation count"""
        self.generation_count += 1
