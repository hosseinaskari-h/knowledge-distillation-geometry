"""
Environment registry
"""
from src.core.environment import Environment
from .freeform import FreeformEnvironment

ENVIRONMENT_REGISTRY = {
    'freeform': FreeformEnvironment,
}


def register_environment(name: str, env_class):
    """Register environment type"""
    ENVIRONMENT_REGISTRY[name] = env_class


def create_environment(config: dict) -> Environment:
    """Create environment from config"""
    env_type = config.get('type', 'freeform')

    if env_type not in ENVIRONMENT_REGISTRY:
        raise ValueError(f"Unknown environment type: {env_type}")

    env_class = ENVIRONMENT_REGISTRY[env_type]
    return env_class(config)
