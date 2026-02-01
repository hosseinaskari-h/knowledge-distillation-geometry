"""
Core data structures for emergent communication experiments
"""
from .agent import Agent, AgentOutput
from .environment import Environment, Episode, Turn

__all__ = [
    'Agent',
    'AgentOutput',
    'Environment',
    'Episode',
    'Turn',
]
