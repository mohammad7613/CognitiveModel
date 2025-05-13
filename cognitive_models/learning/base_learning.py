from ..tasks import Task
from typing import Any, List, Tuple
from abc import ABC, abstractmethod

from numpy.typing import NDArray

class RLLearning(ABC):
    """Asbstract Base Class for Reinforcement Learning (RL) algorithms."""

    @abstractmethod
    def __init__(self, task: Task, **kwargs):
        """
        Initialize the RL algorithm.

        Args:
            task (MDP): The task to learn.
            kwargs: Additional keyword arguments for the RL algorithm.
        """
        pass

    @abstractmethod
    def learning_strategy(self, state: Any, action: Any, reward: float, next_state: Any) -> None:
        """
        Update the RL algorithm based on the observed transition.

        Args:
            state (Any): The current state.
            action (Any): The action taken.
            reward (float): The reward received.
            next_state (Any): The next state.
        """
        pass

    @abstractmethod
    def choose_action(self, state: Any) -> Any:
        """
        Select an action based on the current state.

        Args:
            state (Any): The current state.

        Returns:
            Any: The selected action.
        """
        pass
    @abstractmethod
    def action_probabilities_sequence(self, state_sequence: NDArray, action_sequence: NDArray, rewards_sequence: NDArray) -> NDArray:
        """
        Return the action probabilities for each action in the sequence.

        Args:
            state_sequence (List[Any]): The sequence of states.
            action_sequence (List[Any]): The sequence of actions.
            rewards_sequence (List[float]): The sequence of rewards.

        Returns:
            List[float]: The action probabilities for each action in the sequence.
        """
        pass
    @abstractmethod
    def action_probability(self, state: Any, action: Any) -> float:
        """
        Return the probability of selecting an action in a given state.

        Args:
            state (Any): The current state.
            action (Any): The action to evaluate.

        Returns:
            float: The probability of selecting `action` in `state`.
        """
        pass
    @abstractmethod
    def set_parameters(self,parameters):
        pass