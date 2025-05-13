from abc import ABC, abstractmethod
from typing import Any, List, Tuple
import numpy as np
from numpy.typing import NDArray
import json

def parse_mdp_config(file_path):
    CONTEXTS = {}
    TRANSITIONS = {}

    with open(file_path, 'r') as f:
        data = json.load(f)

    # Parse contexts and reward probabilities
    for context, states in data["contexts"].items():
        CONTEXTS[context] = {}
        for state, reward_probs in states.items():
            CONTEXTS[context][state] = {float(k): float(v) for k, v in reward_probs.items()}  # Convert to float

    # Parse transition probabilities
    for context, states in data["transitions"].items():
        TRANSITIONS[context] = {}
        for state, transitions in states.items():
            TRANSITIONS[context][state] = {ctx: float(prob) for ctx, prob in transitions.items()}  # Ensure float

    return CONTEXTS, TRANSITIONS



class Action:
    def __init__(self, name: str):
        self.name = name


class State:
    def __init__(self, name: str,sub_task: 'Task' = None):
        self.name = name
        self.sub_task = sub_task


class Task(ABC):
    def __init__(self, states: NDArray, terminal_states: List[int]=None):
        self.states = states
        self.terminal_states = terminal_states

    @abstractmethod
    def next_state(self, state: State, action: Action) -> State:
        pass

    @abstractmethod
    def transition_probability(self, state: State, action: Action, next_state: State) -> float:
        pass

    def is_terminal(self, state: State) -> bool:
        return state in self.terminal_states

class MDP(Task):
    """Abstract Base Class for a Markov Decision Process (MDP)."""

    def __init__(self, states: NDArray):
        """
        Initialize the MDP.

        Args:
            states (List[Any]): A list of all possible states.
            actions (List[Any]): A list of all possible actions.
        """
        super().__init__(states)
        

    @abstractmethod
    def get_reward(self, state: Any, action: Any) -> float:
        """
        Return the reward for a given state-action pair.

        Args:
            state (Any): The current state.
            action (Any): The action taken.

        Returns:
            float: The reward for taking `action` in `state`.
        """
        pass

    @abstractmethod
    def reward_probability(self, state: Any, action: Any, reward: float) -> float:
        """
        Compute the probability of receiving `reward` given `state` and `action`.

        Args:
            state (Any): The current state.
            action (Any): The action taken.
            reward (float): The reward received.

        Returns:
            float: The probability of receiving `reward`.
        """
        pass

    def is_terminal(self, state: Any) -> bool:
        """
        Check if a given state is a terminal state.

        Args:
            state (Any): The state to check.

        Returns:
            bool: True if `state` is terminal, else False.
        """
        return state not in self.states  # Default: terminal if not in states

class POMDP(Task):
    """Abstract Base Class for a Partially Observable Markov Decision Process (POMDP)."""

    def __init__(self, states: List[Any], actions: List[Any], observations: List[Any],
                 A: np.ndarray, B: np.ndarray, prior_states: np.ndarray):
        """
        Initialize the POMDP.

        Args:
            states (List[Any]): List of all possible states.
            actions (List[Any]): List of all possible actions.
            observations (List[Any]): List of all possible observations.
            A (np.ndarray): Observation model P(o | s) with shape (observations, states).
            B (np.ndarray): Transition model P(s' | s, a) with shape (actions, states, states).
            prior_states (np.ndarray): Prior probability distribution over initial states.
        """
        self.states = states
        self.actions = actions
        self.observations = observations
        self.A = A  # Observation model: P(o | s)
        self.B = B  # Transition model: P(s' | s, a)
        self.prior_states = prior_states  # Prior over initial states

    def transition_probability(self, state: Any, action: Any, next_state: Any) -> float:
        """
        Return the probability of transitioning to `next_state` given `state` and `action`.
        """
        state_idx = self.states.index(state)
        action_idx = self.actions.index(action)
        next_state_idx = self.states.index(next_state)
        return self.B[action_idx, state_idx, next_state_idx]

    def observation_probability(self, state: Any, observation: Any) -> float:
        """
        Return the probability of observing `observation` given `state`.
        """
        state_idx = self.states.index(state)
        obs_idx = self.observations.index(observation)
        return self.A[obs_idx, state_idx]
    
    def is_terminal(self, state: Any) -> bool:
        """
        Check if a given state is a terminal state.
        """
        return state not in self.states  # Default: terminal if not in states


