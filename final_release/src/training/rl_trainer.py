"""
Multi-Agent RL Trainer
"""
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import logging
from typing import Dict, Tuple, List
from pathlib import Path

from src.core.agent import Agent, AgentOutput
from src.core.environment import Environment, Episode, Turn
from .value_network import ValueNetwork
from .rewards.base import RewardFunction

logger = logging.getLogger(__name__)


class MARLTrainer:
    """Multi-Agent RL trainer for emergent communication"""

    def __init__(self,
                 agents: Dict[str, Agent],
                 environment: Environment,
                 reward_function: RewardFunction,
                 config: dict):

        self.agents = agents
        self.environment = environment
        self.reward_fn = reward_function
        self.config = config

        # Optimizers for each agent
        self.optimizers = {}
        for name, agent in agents.items():
            if hasattr(agent, 'model'):
                self.optimizers[name] = optim.Adam(
                    agent.model.parameters(),
                    lr=config.get('lr', 1e-4)
                )

        # Value networks (critics)
        self.value_networks = {}
        self.value_optimizers = {}
        for name, agent in agents.items():
            vnet = ValueNetwork(agent.embedding_dim).to(agent.device)
            self.value_networks[name] = vnet
            self.value_optimizers[name] = optim.Adam(
                vnet.parameters(),
                lr=config.get('lr', 1e-4)
            )

        # Training params
        self.gamma = config.get('gamma', 0.99)
        self.value_coef = config.get('value_coef', 0.5)
        self.entropy_coef = config.get('entropy_coef', 0.01)
        self.grad_clip = config.get('grad_clip', 0.5)
        self.warmup_steps = config.get('warmup_steps', 5)

        self.episode_counter = 0

        logger.info("MARL Trainer initialized")

    def train_episode(self) -> Tuple[Episode, Dict]:
        """Train one episode"""

        # Reset
        prompt = self.environment.reset()
        agent_names = list(self.agents.keys())

        episode_history = []
        rewards = []
        log_probs = []
        values = []

        done = False
        current_agent_idx = 0
        step = 0

        # === WARMUP PHASE (no gradient) ===
        for _ in range(self.warmup_steps):
            agent_name = agent_names[current_agent_idx]
            agent = self.agents[agent_name]

            with torch.no_grad():
                output = agent.generate(prompt)

            turn = Turn(agent_name=agent_name, output=output, step=step)
            episode_history.append(turn)
            prompt = output.text

            current_agent_idx = (current_agent_idx + 1) % len(agent_names)
            step += 1

        # === RL TRAINING PHASE ===
        max_turns = self.environment.config.get('max_turns', 20)
        while not done and step < max_turns:
            agent_name = agent_names[current_agent_idx]
            agent = self.agents[agent_name]

            # Generate with log prob
            output, log_prob = agent.generate_with_log_prob(prompt)

            # Compute value estimate
            embedding = output.get_last_embedding().float()  # Cast to float32 for value network compatibility
            value = self.value_networks[agent_name](embedding.to(agent.device))

            # Compute reward (based on embedding space only)
            history_outputs = [turn.output for turn in episode_history]
            reward, reward_breakdown = self.reward_fn.compute_reward(
                history_outputs,
                output
            )

            # Store
            turn = Turn(agent_name=agent_name, output=output, step=step)
            episode_history.append(turn)
            rewards.append(reward)
            log_probs.append(log_prob)
            values.append(value)

            # Next step
            prompt, done, info = self.environment.step(output)
            current_agent_idx = (current_agent_idx + 1) % len(agent_names)
            step += 1

        # === UPDATE AGENTS ===
        if len(rewards) > 0:
            returns = self._compute_returns(rewards, self.gamma)

            # Update each agent
            for agent_name in agent_names:
                # Get this agent's transitions
                training_turns = episode_history[self.warmup_steps:]
                agent_indices = [
                    i for i, turn in enumerate(training_turns)
                    if turn.agent_name == agent_name
                ]

                if len(agent_indices) == 0:
                    continue

                agent_log_probs = [log_probs[i] for i in agent_indices if i < len(log_probs)]
                agent_values = [values[i] for i in agent_indices if i < len(values)]
                agent_returns = [returns[i] for i in agent_indices if i < len(returns)]

                if len(agent_log_probs) == 0:
                    continue

                # Compute advantages
                advantages = [
                    r - v.item()
                    for r, v in zip(agent_returns, agent_values)
                ]

                # Policy loss
                policy_loss = []
                for log_prob, advantage in zip(agent_log_probs, advantages):
                    policy_loss.append(-log_prob * advantage)

                if len(policy_loss) > 0:
                    policy_loss = torch.stack(policy_loss).mean()

                    # Value loss
                    value_loss = torch.stack([
                        F.mse_loss(v, torch.tensor(r, device=v.device, dtype=torch.float32))
                        for v, r in zip(agent_values, agent_returns)
                    ]).mean()

                    # Total loss
                    loss = policy_loss + self.value_coef * value_loss

                    # Update agent
                    if agent_name in self.optimizers:
                        self.optimizers[agent_name].zero_grad()
                        loss.backward(retain_graph=True)

                        # Gradient clipping
                        if hasattr(self.agents[agent_name], 'model'):
                            torch.nn.utils.clip_grad_norm_(
                                self.agents[agent_name].model.parameters(),
                                self.grad_clip
                            )

                        self.optimizers[agent_name].step()

                    # Update value network
                    self.value_optimizers[agent_name].zero_grad()
                    value_loss.backward()
                    self.value_optimizers[agent_name].step()

        # Create episode object
        episode = Episode(
            id=f"ep_{self.episode_counter:06d}",
            turns=episode_history,
            initial_prompt=episode_history[0].output.text if episode_history else "",
            metadata={
                'rewards': rewards,
                'total_reward': sum(rewards) if rewards else 0.0,
                'num_turns': len(episode_history),
            }
        )

        metrics = {
            'episode': self.episode_counter,
            'total_reward': sum(rewards) if rewards else 0.0,
            'avg_reward': np.mean(rewards) if rewards else 0.0,
            'num_turns': len(episode_history),
            'num_training_turns': len(rewards),
        }

        self.episode_counter += 1

        return episode, metrics

    def _compute_returns(self, rewards: List[float], gamma: float) -> List[float]:
        """Compute discounted returns"""
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)
        return returns

    def save_checkpoint(self, path: Path):
        """Save trainer state"""
        checkpoint = {
            'episode_counter': self.episode_counter,
            'value_networks': {
                name: vnet.state_dict()
                for name, vnet in self.value_networks.items()
            },
            'optimizers': {
                name: opt.state_dict()
                for name, opt in self.optimizers.items()
            },
            'value_optimizers': {
                name: opt.state_dict()
                for name, opt in self.value_optimizers.items()
            },
        }
        torch.save(checkpoint, path)
        logger.info(f"Saved trainer checkpoint to {path}")

    def load_checkpoint(self, path: Path):
        """Load trainer state"""
        checkpoint = torch.load(path)
        self.episode_counter = checkpoint['episode_counter']

        for name, state_dict in checkpoint['value_networks'].items():
            if name in self.value_networks:
                self.value_networks[name].load_state_dict(state_dict)

        for name, state_dict in checkpoint['optimizers'].items():
            if name in self.optimizers:
                self.optimizers[name].load_state_dict(state_dict)

        for name, state_dict in checkpoint['value_optimizers'].items():
            if name in self.value_optimizers:
                self.value_optimizers[name].load_state_dict(state_dict)

        logger.info(f"Loaded trainer checkpoint from {path}")
