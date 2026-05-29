"""
Reward function registry
"""
from .base import RewardFunction
from .pure_embedding import PureEmbeddingSpaceReward
from .layered import LayeredNonAnthropocentricReward

REWARD_REGISTRY = {
    'pure_embedding': PureEmbeddingSpaceReward,
    'layered': LayeredNonAnthropocentricReward,
}


def register_reward(name: str, reward_class):
    """Register reward function"""
    REWARD_REGISTRY[name] = reward_class


def create_reward_function(config: dict) -> RewardFunction:
    """Create reward function from config"""
    reward_type = config.get('type', 'layered')

    if reward_type not in REWARD_REGISTRY:
        raise ValueError(f"Unknown reward type: {reward_type}")

    reward_class = REWARD_REGISTRY[reward_type]
    return reward_class(config)
