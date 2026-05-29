"""
Vector Dynamics Trainer

Implementation of coupled dynamical systems for transformer networks.
Facilitates direct hidden state exchange between agents to study geometric alignment
and attractor dynamics without linguistic tokenization.
"""

import torch
import torch.nn as nn
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass, field
from pathlib import Path
import time

from src.agents.hf_agent import HuggingFaceAgent

logger = logging.getLogger(__name__)


@dataclass
class DynamicsState:
    """Snapshot of the coupled system state"""
    step: int
    state_a: torch.Tensor
    state_b: torch.Tensor

    # Optional metrics computed at this step
    cosine_similarity: Optional[float] = None
    l2_distance: Optional[float] = None
    state_a_norm: Optional[float] = None
    state_b_norm: Optional[float] = None


@dataclass
class DynamicsTrajectory:
    """Full trajectory of a vector dynamics run"""
    states: List[DynamicsState] = field(default_factory=list)
    config: Dict = field(default_factory=dict)

    # Computed after run
    synchronization_index: Optional[float] = None
    largest_lyapunov: Optional[float] = None
    attractor_dimension: Optional[float] = None

    def to_tensors(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert to tensors for analysis: [num_steps, hidden_dim]"""
        states_a = torch.stack([s.state_a.cpu().float() for s in self.states])
        states_b = torch.stack([s.state_b.cpu().float() for s in self.states])
        return states_a, states_b

    def get_joint_trajectory(self) -> torch.Tensor:
        """Get concatenated trajectory: [num_steps, common_dim * 2]"""
        states_a, states_b = self.to_tensors()

        # Handle dimension mismatch by truncating to common dimension
        dim_a, dim_b = states_a.shape[1], states_b.shape[1]
        if dim_a != dim_b:
            min_dim = min(dim_a, dim_b)
            states_a = states_a[:, :min_dim]
            states_b = states_b[:, :min_dim]

        return torch.cat([states_a, states_b], dim=1)


class CouplingStrategy:
    """Base class for coupling strategies between agents"""

    def couple(self,
               state_a: torch.Tensor,
               state_b: torch.Tensor,
               agent_a: HuggingFaceAgent,
               agent_b: HuggingFaceAgent) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Given current states, compute next states.

        Returns:
            (new_state_a, new_state_b)
        """
        raise NotImplementedError


import torch.optim as optim

class DirectCoupling(CouplingStrategy):
    """
    Direct coupling: each agent receives the other's state directly.

    state_a(t+1) = F_a(state_b(t))
    state_b(t+1) = F_b(state_a(t))

     each agent transforms the other's state.
    """

    def __init__(self, force_projection: bool = False):
        self.proj_a_to_b = None
        self.proj_b_to_a = None
        self.force_projection = force_projection
        self.is_trained = False

    def train_projections(self, agent_a, agent_b, steps=100):
        """Train projection layers to minimize reconstruction error."""
        if self.is_trained:
            return

        dim_a = agent_a.embedding_dim
        dim_b = agent_b.embedding_dim
        device = agent_a.device

        # Init layers if not already done
        if self.proj_a_to_b is None:
            self.proj_a_to_b = nn.Linear(dim_a, dim_b).to(device)
            self.proj_b_to_a = nn.Linear(dim_b, dim_a).to(device)
            
            # Initialize near-identity for overlapping dims + small noise
            with torch.no_grad():
                self.proj_a_to_b.weight.data.normal_(0, 0.01)
                self.proj_b_to_a.weight.data.normal_(0, 0.01)
                self.proj_a_to_b.bias.data.zero_()
                self.proj_b_to_a.bias.data.zero_()
                
                min_dim = min(dim_a, dim_b)
                self.proj_a_to_b.weight.data[:min_dim, :min_dim] += torch.eye(min_dim).to(device)
                self.proj_b_to_a.weight.data[:min_dim, :min_dim] += torch.eye(min_dim).to(device)

        optimizer = optim.Adam(
            list(self.proj_a_to_b.parameters()) + list(self.proj_b_to_a.parameters()),
            lr=1e-3
        )

        logger.info(f"Training projections for {steps} steps...")
        
        # Training loop
        for _ in range(steps):
            # Sample random vectors from N(0, 0.01I)
            # Batch size 50 as per paper
            h_a = torch.randn(50, dim_a, device=device) * 0.1
            h_b = torch.randn(50, dim_b, device=device) * 0.1
            
            loss = 0
            
            # Cycle A: a -> b -> a
            rec_a = self.proj_b_to_a(self.proj_a_to_b(h_a))
            loss += (h_a - rec_a).pow(2).mean()
            
            # Cycle B: b -> a -> b
            rec_b = self.proj_a_to_b(self.proj_b_to_a(h_b))
            loss += (h_b - rec_b).pow(2).mean()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        self.is_trained = True
        logger.info("Projection training complete.")

    def couple(self, state_a, state_b, agent_a, agent_b):
        dim_a = agent_a.embedding_dim
        dim_b = agent_b.embedding_dim

        # Handle dimension mismatch or forced projection
        if dim_a != dim_b or self.force_projection:
            device = state_a.device
            
            # Train if needed 
            if not self.is_trained:
                self.train_projections(agent_a, agent_b)

            # Project states to match target agent's dimension
            input_for_a = self.proj_b_to_a(state_b.float())
            input_for_b = self.proj_a_to_b(state_a.float())
        else:
            input_for_a = state_b
            input_for_b = state_a

        # A processes B's state, B processes A's state (simultaneously)
        new_state_a = agent_a.forward_from_hidden(input_for_a)
        new_state_b = agent_b.forward_from_hidden(input_for_b)
        return new_state_a, new_state_b


class BlendedCoupling(CouplingStrategy):
    """
    Blended coupling: agents receive a mix of own state and other's state.

    input_a = alpha * state_a + (1-alpha) * state_b
    state_a(t+1) = F_a(input_a)

    alpha = 1.0: no coupling (isolated dynamics)
    alpha = 0.5: equal blend
    alpha = 0.0: pure exchange (like DirectCoupling)
    """

    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha
        self.proj_a_to_b = None
        self.proj_b_to_a = None

    def couple(self, state_a, state_b, agent_a, agent_b):
        dim_a = agent_a.embedding_dim
        dim_b = agent_b.embedding_dim

        # Handle dimension mismatch
        if dim_a != dim_b:
            device = state_a.device
            if self.proj_a_to_b is None:
                self.proj_a_to_b = nn.Linear(dim_a, dim_b).to(device)
                self.proj_b_to_a = nn.Linear(dim_b, dim_a).to(device)
                with torch.no_grad():
                    min_dim = min(dim_a, dim_b)
                    self.proj_a_to_b.weight[:min_dim, :min_dim] = torch.eye(min_dim)
                    self.proj_b_to_a.weight[:min_dim, :min_dim] = torch.eye(min_dim)

            # Project other's state to own dimension for blending
            state_b_in_a_space = self.proj_b_to_a(state_b.float())
            state_a_in_b_space = self.proj_a_to_b(state_a.float())

            input_a = self.alpha * state_a + (1 - self.alpha) * state_b_in_a_space
            input_b = self.alpha * state_b + (1 - self.alpha) * state_a_in_b_space
        else:
            input_a = self.alpha * state_a + (1 - self.alpha) * state_b
            input_b = self.alpha * state_b + (1 - self.alpha) * state_a

        new_state_a = agent_a.forward_from_hidden(input_a)
        new_state_b = agent_b.forward_from_hidden(input_b)
        return new_state_a, new_state_b


class ProjectedCoupling(CouplingStrategy):
    """
    Projected coupling: states pass through learned projection before exchange.

    This allows the coupling interface to be learned/optimized.
    """

    def __init__(self, hidden_dim: int, projection_dim: Optional[int] = None, device: str = 'cuda'):
        self.hidden_dim = hidden_dim
        self.projection_dim = projection_dim or hidden_dim

        # Learnable projections
        self.proj_a_to_b = nn.Linear(hidden_dim, self.projection_dim).to(device)
        self.proj_b_to_a = nn.Linear(hidden_dim, self.projection_dim).to(device)

        # If dimensions differ, need to project back
        if self.projection_dim != hidden_dim:
            self.proj_back_a = nn.Linear(self.projection_dim, hidden_dim).to(device)
            self.proj_back_b = nn.Linear(self.projection_dim, hidden_dim).to(device)
        else:
            self.proj_back_a = nn.Identity()
            self.proj_back_b = nn.Identity()

    def couple(self, state_a, state_b, agent_a, agent_b):
        # Project states through coupling interface
        msg_a_to_b = self.proj_a_to_b(state_a)
        msg_b_to_a = self.proj_b_to_a(state_b)

        # Project back to hidden dim if needed
        input_a = self.proj_back_b(msg_b_to_a)
        input_b = self.proj_back_a(msg_a_to_b)

        new_state_a = agent_a.forward_from_hidden(input_a)
        new_state_b = agent_b.forward_from_hidden(input_b)
        return new_state_a, new_state_b


class DecayCoupling(CouplingStrategy):
    """
    Coupling with memory decay: states blend with history.

    input_a = beta * prev_state_a + (1-beta) * state_b

    This adds temporal smoothing / momentum to the dynamics.
    """

    def __init__(self, decay: float = 0.1):
        self.decay = decay  # How much of previous own state to keep
        self.prev_a = None
        self.prev_b = None
        self.proj_a_to_b = None
        self.proj_b_to_a = None

    def couple(self, state_a, state_b, agent_a, agent_b):
        dim_a = agent_a.embedding_dim
        dim_b = agent_b.embedding_dim

        if self.prev_a is None:
            self.prev_a = state_a.clone()
            self.prev_b = state_b.clone()

        # Handle dimension mismatch
        if dim_a != dim_b:
            device = state_a.device
            if self.proj_a_to_b is None:
                self.proj_a_to_b = nn.Linear(dim_a, dim_b).to(device)
                self.proj_b_to_a = nn.Linear(dim_b, dim_a).to(device)
                with torch.no_grad():
                    min_dim = min(dim_a, dim_b)
                    self.proj_a_to_b.weight[:min_dim, :min_dim] = torch.eye(min_dim)
                    self.proj_b_to_a.weight[:min_dim, :min_dim] = torch.eye(min_dim)

            state_b_in_a_space = self.proj_b_to_a(state_b.float())
            state_a_in_b_space = self.proj_a_to_b(state_a.float())

            input_a = self.decay * self.prev_a + (1 - self.decay) * state_b_in_a_space
            input_b = self.decay * self.prev_b + (1 - self.decay) * state_a_in_b_space
        else:
            input_a = self.decay * self.prev_a + (1 - self.decay) * state_b
            input_b = self.decay * self.prev_b + (1 - self.decay) * state_a

        new_state_a = agent_a.forward_from_hidden(input_a)
        new_state_b = agent_b.forward_from_hidden(input_b)

        # Update memory
        self.prev_a = new_state_a.clone()
        self.prev_b = new_state_b.clone()

        return new_state_a, new_state_b

    def reset(self):
        self.prev_a = None
        self.prev_b = None


class VectorDynamicsTrainer:
    """
    Trainer for vector dynamics experiments.

    Executes the coupled dynamics protocol:
        for step in range(num_steps):
            state_a, state_b = coupling_strategy.couple(state_a, state_b, agent_a, agent_b)
            trajectory.append((state_a, state_b))
    """

    def __init__(self,
                 agent_a: HuggingFaceAgent,
                 agent_b: HuggingFaceAgent,
                 coupling_strategy: CouplingStrategy,
                 config: dict):

        self.agent_a = agent_a
        self.agent_b = agent_b
        self.coupling = coupling_strategy
        self.config = config

        # Settings
        self.record_every = config.get('record_every', 1)
        self.compute_metrics_every = config.get('compute_metrics_every', 10)

        # Check for dimension mismatch
        dim_a = agent_a.embedding_dim
        dim_b = agent_b.embedding_dim
        self.dim_mismatch = dim_a != dim_b

        if self.dim_mismatch:
            logger.warning(f"Dimension mismatch: Agent A={dim_a}, Agent B={dim_b}. Using projection layers.")
            # Create projection layers to common dimension (use smaller)
            self.common_dim = min(dim_a, dim_b)
            device = agent_a.device

            # Projections for comparing states (for metrics)
            self.proj_a = nn.Linear(dim_a, self.common_dim).to(device)
            self.proj_b = nn.Linear(dim_b, self.common_dim).to(device)

            # Initialize with identity-like projection for the matching dimensions
            with torch.no_grad():
                nn.init.eye_(self.proj_a.weight[:, :self.common_dim])
                nn.init.zeros_(self.proj_a.bias)
                nn.init.eye_(self.proj_b.weight[:, :self.common_dim])
                nn.init.zeros_(self.proj_b.bias)
        else:
            self.common_dim = dim_a
            self.proj_a = None
            self.proj_b = None

        logger.info(f"VectorDynamicsTrainer initialized with {type(coupling_strategy).__name__}")

    def run(self,
            num_steps: int,
            init_mode: str = 'random',
            init_text_a: Optional[str] = None,
            init_text_b: Optional[str] = None,
            callback: Optional[Callable[[int, DynamicsState], None]] = None) -> DynamicsTrajectory:
        """
        Run vector dynamics for num_steps.

        Args:
            num_steps: Number of coupling iterations
            init_mode: 'random', 'text', or 'zero'
            init_text_a: Initial text for agent A (if init_mode='text')
            init_text_b: Initial text for agent B (if init_mode='text')
            callback: Optional function called each step with (step, state)

        Returns:
            DynamicsTrajectory containing the full run
        """
        # Initialize states
        if init_mode == 'random':
            state_a = self.agent_a.get_random_hidden_state()
            state_b = self.agent_b.get_random_hidden_state()
        elif init_mode == 'text':
            state_a = self.agent_a.get_hidden_from_text(init_text_a or "Hello")
            state_b = self.agent_b.get_hidden_from_text(init_text_b or "Hello")
        elif init_mode == 'zero':
            state_a = torch.zeros(self.agent_a.embedding_dim, device=self.agent_a.device)
            state_b = torch.zeros(self.agent_b.embedding_dim, device=self.agent_b.device)
        elif init_mode == 'same':
            # Both start from same random state
            state_a = self.agent_a.get_random_hidden_state()
            state_b = state_a.clone().to(self.agent_b.device)
        else:
            raise ValueError(f"Unknown init_mode: {init_mode}")

        trajectory = DynamicsTrajectory(config={
            'num_steps': num_steps,
            'init_mode': init_mode,
            'coupling': type(self.coupling).__name__,
            'agent_a': self.agent_a.name,
            'agent_b': self.agent_b.name,
        })

        logger.info(f"Starting vector dynamics run: {num_steps} steps, init={init_mode}")
        start_time = time.time()

        for step in range(num_steps):
            # Record state (before coupling step)
            if step % self.record_every == 0:
                ds = DynamicsState(
                    step=step,
                    state_a=state_a.detach().clone(),
                    state_b=state_b.detach().clone(),
                )

                # Compute metrics periodically
                if step % self.compute_metrics_every == 0:
                    ds.cosine_similarity = self._cosine_sim(state_a, state_b)
                    ds.l2_distance = self._l2_dist(state_a, state_b)
                    ds.state_a_norm = state_a.norm().item()
                    ds.state_b_norm = state_b.norm().item()

                trajectory.states.append(ds)

                if callback:
                    callback(step, ds)

            # Coupling step: the core of vector dynamics
            state_a, state_b = self.coupling.couple(
                state_a, state_b,
                self.agent_a, self.agent_b
            )

            # Check for numerical issues
            if torch.isnan(state_a).any() or torch.isnan(state_b).any():
                logger.warning(f"NaN detected at step {step}, stopping early")
                break

            if step > 0 and step % 100 == 0:
                elapsed = time.time() - start_time
                logger.info(f"Step {step}/{num_steps} ({elapsed:.1f}s elapsed)")

        # Record final state
        trajectory.states.append(DynamicsState(
            step=num_steps,
            state_a=state_a.detach().clone(),
            state_b=state_b.detach().clone(),
            cosine_similarity=self._cosine_sim(state_a, state_b),
            l2_distance=self._l2_dist(state_a, state_b),
            state_a_norm=state_a.norm().item(),
            state_b_norm=state_b.norm().item(),
        ))

        elapsed = time.time() - start_time
        logger.info(f"Vector dynamics run complete: {len(trajectory.states)} states recorded in {elapsed:.1f}s")

        return trajectory

    def _cosine_sim(self, a: torch.Tensor, b: torch.Tensor) -> float:
        """Compute cosine similarity between two states"""
        # Handle dimension mismatch by projecting to common space
        if self.dim_mismatch:
            with torch.no_grad():
                a = self.proj_a(a.float())
                b = self.proj_b(b.float())

        a_norm = a / (a.norm() + 1e-8)
        b_norm = b / (b.norm() + 1e-8)
        return (a_norm * b_norm).sum().item()

    def _l2_dist(self, a: torch.Tensor, b: torch.Tensor) -> float:
        """Compute L2 distance between two states"""
        # Handle dimension mismatch by projecting to common space
        if self.dim_mismatch:
            with torch.no_grad():
                a = self.proj_a(a.float())
                b = self.proj_b(b.float())

        return (a - b).norm().item()


# Dynamics Analysis Functions

def compute_synchronization_index(trajectory: DynamicsTrajectory) -> float:
    """
    Compute synchronization index between the two agents.

    High values indicate the agents are moving in lockstep.
    Based on phase synchronization measures.
    """
    states_a, states_b = trajectory.to_tensors()

    # Handle dimension mismatch by projecting to common dimension
    dim_a, dim_b = states_a.shape[1], states_b.shape[1]
    if dim_a != dim_b:
        min_dim = min(dim_a, dim_b)
        states_a = states_a[:, :min_dim]
        states_b = states_b[:, :min_dim]

    # Compute velocity (difference between consecutive states)
    vel_a = states_a[1:] - states_a[:-1]
    vel_b = states_b[1:] - states_b[:-1]

    # Normalize velocities
    vel_a_norm = vel_a / (vel_a.norm(dim=1, keepdim=True) + 1e-8)
    vel_b_norm = vel_b / (vel_b.norm(dim=1, keepdim=True) + 1e-8)

    # Average cosine similarity of velocities
    sync_index = (vel_a_norm * vel_b_norm).sum(dim=1).mean().item()

    return sync_index


def compute_lyapunov_exponent(trajectory: DynamicsTrajectory,
                               num_neighbors: int = 5) -> float:
    """
    Estimate largest Lyapunov exponent from trajectory.

    Positive: chaotic dynamics
    Zero: periodic or quasi-periodic
    Negative: converging to attractor
    """
    try:
        from sklearn.neighbors import NearestNeighbors
    except ImportError:
        logger.warning("sklearn required for Lyapunov computation")
        return 0.0

    joint = trajectory.get_joint_trajectory().numpy()
    n_points = len(joint)

    if n_points < 100:
        return 0.0

    # Find nearest neighbors for each point
    nbrs = NearestNeighbors(n_neighbors=num_neighbors + 1).fit(joint)
    distances, indices = nbrs.kneighbors(joint)

    # Track divergence of nearby trajectories
    divergences = []

    for i in range(n_points - 10):
        for j in range(1, num_neighbors + 1):
            neighbor_idx = indices[i, j]
            initial_dist = distances[i, j]

            if initial_dist < 1e-8:
                continue

            # Check how far apart they are after some steps
            future_step = min(i + 10, n_points - 1)
            future_neighbor = min(neighbor_idx + 10, n_points - 1)

            future_dist = np.linalg.norm(joint[future_step] - joint[future_neighbor])

            if future_dist > 1e-8:
                divergence = np.log(future_dist / initial_dist) / 10
                divergences.append(divergence)

    if len(divergences) > 0:
        return np.mean(divergences)
    return 0.0


def compute_attractor_dimension(trajectory: DynamicsTrajectory,
                                 skip_initial: int = 100) -> float:
    """
    Estimate intrinsic dimension of the attractor.

    Uses correlation dimension estimation.
    """
    joint = trajectory.get_joint_trajectory().numpy()

    # Skip initial transient
    if len(joint) > skip_initial:
        joint = joint[skip_initial:]

    if len(joint) < 50:
        return 0.0

    # Correlation dimension via box-counting approximation
    try:
        from sklearn.metrics import pairwise_distances

        # Subsample for efficiency
        n_samples = min(500, len(joint))
        indices = np.random.choice(len(joint), n_samples, replace=False)
        points = joint[indices]

        # Compute pairwise distances
        dists = pairwise_distances(points).flatten()
        dists = dists[dists > 1e-8]  # Remove zeros

        if len(dists) < 100:
            return 0.0

        # Correlation sum at different scales
        epsilons = np.logspace(np.log10(dists.min()), np.log10(dists.max()), 20)
        C_eps = []

        for eps in epsilons:
            C = np.mean(dists < eps)
            if C > 0:
                C_eps.append((np.log(eps), np.log(C)))

        if len(C_eps) < 5:
            return 0.0

        # Linear fit to get dimension
        C_eps = np.array(C_eps)
        # Use middle portion for better estimate
        mid_start = len(C_eps) // 4
        mid_end = 3 * len(C_eps) // 4

        if mid_end - mid_start < 3:
            return 0.0

        slope, _ = np.polyfit(C_eps[mid_start:mid_end, 0],
                              C_eps[mid_start:mid_end, 1], 1)

        return max(0, slope)

    except Exception as e:
        logger.warning(f"Attractor dimension computation failed: {e}")
        return 0.0


def analyze_trajectory(trajectory: DynamicsTrajectory) -> Dict:
    """
    Comprehensive analysis of a dynamics trajectory.

    Returns dict with all computed metrics.
    """
    results = {}

    # Basic statistics
    states_a, states_b = trajectory.to_tensors()

    results['num_steps'] = len(trajectory.states)
    results['final_cosine_similarity'] = trajectory.states[-1].cosine_similarity
    results['final_l2_distance'] = trajectory.states[-1].l2_distance

    # Synchronization
    results['synchronization_index'] = compute_synchronization_index(trajectory)

    # Lyapunov exponent
    results['lyapunov_exponent'] = compute_lyapunov_exponent(trajectory)

    # Attractor dimension
    results['attractor_dimension'] = compute_attractor_dimension(trajectory)

    # State norm statistics
    norms_a = [s.state_a_norm for s in trajectory.states if s.state_a_norm is not None]
    norms_b = [s.state_b_norm for s in trajectory.states if s.state_b_norm is not None]

    if norms_a:
        results['mean_norm_a'] = np.mean(norms_a)
        results['std_norm_a'] = np.std(norms_a)
    if norms_b:
        results['mean_norm_b'] = np.mean(norms_b)
        results['std_norm_b'] = np.std(norms_b)

    # Cosine similarity over time
    cos_sims = [s.cosine_similarity for s in trajectory.states if s.cosine_similarity is not None]
    if cos_sims:
        results['mean_cosine_similarity'] = np.mean(cos_sims)
        results['final_minus_initial_cos'] = cos_sims[-1] - cos_sims[0] if len(cos_sims) > 1 else 0

    # Interpretation
    lyap = results.get('lyapunov_exponent', 0)
    if lyap > 0.1:
        results['dynamics_type'] = 'chaotic'
    elif lyap < -0.1:
        results['dynamics_type'] = 'converging'
    else:
        results['dynamics_type'] = 'periodic_or_quasiperiodic'

    sync = results.get('synchronization_index', 0)
    if sync > 0.8:
        results['synchronization_type'] = 'strong'
    elif sync > 0.5:
        results['synchronization_type'] = 'moderate'
    elif sync > 0.2:
        results['synchronization_type'] = 'weak'
    else:
        results['synchronization_type'] = 'none'

    # Compute CKA if available
    try:
        from src.analysis.cka import CKAAnalyzer
        cka_analyzer = CKAAnalyzer(device=states_a.device)
        results['cka_linear'] = cka_analyzer.linear_cka(states_a, states_b)
        results['cka_kernel'] = cka_analyzer.kernel_cka(states_a, states_b)
    except ImportError:
        pass
    except Exception as e:
        logger.warning(f"CKA computation failed: {e}")

    return results


# Convenience Functions

def create_coupling_strategy(name: str, config: dict) -> CouplingStrategy:
    """Factory function for coupling strategies"""
    if name == 'direct':
        force_proj = config.get('force_projection', False)
        return DirectCoupling(force_projection=force_proj)
    elif name == 'blended':
        alpha = config.get('alpha', 0.5)
        return BlendedCoupling(alpha=alpha)
    elif name == 'projected':
        hidden_dim = config.get('hidden_dim', 768)
        proj_dim = config.get('projection_dim', hidden_dim)
        device = config.get('device', 'cuda')
        return ProjectedCoupling(hidden_dim, proj_dim, device)
    elif name == 'decay':
        decay = config.get('decay', 0.1)
        return DecayCoupling(decay=decay)
    else:
        raise ValueError(f"Unknown coupling strategy: {name}")
