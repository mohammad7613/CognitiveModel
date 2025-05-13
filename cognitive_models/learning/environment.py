from ..tasks import Task
from .base_learning import RLLearning
from typing import Any, List, Tuple
import numpy as np


class MDPEnvironment:
    def __init__(self, agent: RLLearning, task: Task):
        """
        Initialize the environment with an RL agent, MDP model, and terminal state.

        Parameters:
        - agent (Learning): The reinforcement learning agent.
        - task (Task): The Markov Decision Process model or Partially Observable MDP.
        - terminal_state (int): The index of the terminal state in the MDP/POMDP.
        """
        self.agent = agent
        self.task = task

    def simulate_interaction(self, initial_state_idx: int, max_steps: int=100)-> Tuple[RLLearning, np.ndarray]:
        """
        Simulate the interaction of the RL agent with the MDP until the terminal state is reached.

        Parameters:
        - initial_state_idx (int): The index of the initial state in the MDP.
        - max_steps (int): The maximum number of steps for the simulation.

        Returns:
        - RL: The RL agent after simulation, with updated parameters and Q-values.
        """
        state_idx = initial_state_idx
        step = 0
        trajectory = np.zeros((max_steps, 3))
        while not self.task.is_terminal(state_idx) and step < max_steps:
            # RL agent chooses an action based on the current state
            action = self.agent.choose_action(state_idx)

            # MDP determines the next state and the reward for the chosen action
            next_state_idx = self.task.next_state(state_idx, action)
            reward = self.task.get_reward(state_idx, action)

            # RL agent updates its learning based on the action and reward
            self.agent.learning_strategy(state_idx, action, reward)

            trajectory[step] = [state_idx, action, reward]

            # Move to the next state
            state_idx = next_state_idx
            step += 1

        return self.agent,trajectory