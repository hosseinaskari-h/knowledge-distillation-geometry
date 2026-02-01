from .rl_trainer import MARLTrainer
from .value_network import ValueNetwork
from .vector_dynamics import (
    VectorDynamicsTrainer,
    DynamicsState,
    DynamicsTrajectory,
    CouplingStrategy,
    DirectCoupling,
    BlendedCoupling,
    ProjectedCoupling,
    DecayCoupling,
    create_coupling_strategy,
    compute_synchronization_index,
    compute_lyapunov_exponent,
    compute_attractor_dimension,
    analyze_trajectory,
)

__all__ = [
    # Branch 1: Text-based RL
    'MARLTrainer',
    'ValueNetwork',
    # Branch 2: Vector Dynamics
    'VectorDynamicsTrainer',
    'DynamicsState',
    'DynamicsTrajectory',
    'CouplingStrategy',
    'DirectCoupling',
    'BlendedCoupling',
    'ProjectedCoupling',
    'DecayCoupling',
    'create_coupling_strategy',
    'compute_synchronization_index',
    'compute_lyapunov_exponent',
    'compute_attractor_dimension',
    'analyze_trajectory',
]
