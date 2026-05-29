"""
Freeform conversation environment
"""
import logging
from typing import Dict, Tuple
import uuid

from src.core.environment import Environment, Episode, Turn
from src.core.agent import Agent, AgentOutput
from .prompt_bank import PromptBank

logger = logging.getLogger(__name__)


class FreeformEnvironment(Environment):
    """
    Freeform conversation with no task constraints
    Agents just talk to each other
    """

    def __init__(self, config: dict):
        super().__init__(config)

        self.warmup_steps = config.get('warmup_steps', 5)
        self.max_turns = config.get('max_turns', 20)

        # Prompt bank
        prompt_file = config.get('prompt_file')
        if prompt_file:
            self.prompt_bank = PromptBank.load_from_file(prompt_file)
        else:
            self.prompt_bank = PromptBank()

        # Episode state
        self.current_episode = None
        self.current_prompt = None
        self.step_count = 0

    def reset(self) -> str:
        """Start new episode"""
        self.current_prompt = self.prompt_bank.sample()
        self.step_count = 0
        logger.debug(f"Environment reset with prompt: {self.current_prompt[:50]}...")
        return self.current_prompt

    def step(self, agent_output: AgentOutput) -> Tuple[str, bool, Dict]:
        """
        Execute one step

        Returns:
            next_prompt: Text for next agent
            done: Whether episode is complete
            info: Additional information
        """
        self.step_count += 1

        # Next prompt is current agent's output
        self.current_prompt = agent_output.text

        # Check termination
        done = self.step_count >= self.max_turns

        info = {
            'step': self.step_count,
            'in_warmup': self.step_count <= self.warmup_steps,
        }

        return self.current_prompt, done, info

    def run_episode(self, agents: Dict[str, Agent]) -> Episode:
        """
        Run complete episode with agent alternation

        Args:
            agents: Dict mapping agent_name -> Agent instance
        """
        # Initialize
        episode_id = str(uuid.uuid4())[:8]
        initial_prompt = self.reset()

        agent_names = list(agents.keys())
        turns = []

        current_agent_idx = 0
        current_prompt = initial_prompt

        logger.info(f"Starting episode {episode_id}")

        for step in range(self.max_turns):
            # Get current agent
            agent_name = agent_names[current_agent_idx]
            agent = agents[agent_name]

            # Generate
            output = agent.generate(current_prompt)

            # Store turn
            turn = Turn(
                agent_name=agent_name,
                output=output,
                step=step,
            )
            turns.append(turn)

            # Update for next turn
            current_prompt, done, info = self.step(output)

            # Alternate agents
            current_agent_idx = (current_agent_idx + 1) % len(agent_names)

            if done:
                break

        episode = Episode(
            id=episode_id,
            turns=turns,
            initial_prompt=initial_prompt,
            metadata={
                'num_turns': len(turns),
                'agents': agent_names,
            }
        )

        logger.info(f"Episode {episode_id} complete: {len(turns)} turns")

        return episode
