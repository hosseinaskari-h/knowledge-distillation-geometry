"""
Agent registry for creating agents from configs
"""
from typing import Dict
from src.core.agent import Agent
from .hf_agent import HuggingFaceAgent

AGENT_REGISTRY = {
    'hf_agent': HuggingFaceAgent,
    'huggingface': HuggingFaceAgent,
    'hf_causal': HuggingFaceAgent,
}


def register_agent(name: str, agent_class):
    """Register new agent type"""
    AGENT_REGISTRY[name] = agent_class


def create_agent(name: str, config: dict) -> Agent:
    """Create agent from config"""
    agent_type = config.get('type', 'hf_agent')

    if agent_type not in AGENT_REGISTRY:
        raise ValueError(f"Unknown agent type: {agent_type}")

    agent_class = AGENT_REGISTRY[agent_type]
    return agent_class(name, config)
