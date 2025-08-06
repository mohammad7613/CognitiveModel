from abc import ABC, abstractmethod
from cognitive_models.learning.base_learning import RLLearning
from cognitive_models.tasks.base_tasks import MDP, State, Action
import numpy as np
from numpy.typing import NDArray


class AbstractQL(RLLearning, ABC):
    def __init__(self, mdp: MDP,states: NDArray):
        super().__init__(mdp)
        self.states = states

    @abstractmethod
    def choose_action(self, state_idx: int) -> int:
        """Selects an action using a softmax-based policy with noise handling."""
        pass

    @abstractmethod
    def update_value(self, state_idx: int, action: int, reward: float) -> NDArray:
        """Abstract method for updating Q-values. Implement in subclasses."""
        """ you need to return the currne and the updated and delta value """
        pass
    @abstractmethod
    def action_probability(self, state_idx: int, action: int) -> float:
        """Computes the probability of selecting a given action."""
        pass
  
    def action_probabilities_sequence(
        self, state_sequence: NDArray, action_sequence: NDArray, rewards_sequence: NDArray
    ) -> NDArray:
        """Computes action probabilities over a sequence and updates Q-values."""
        action_probabilities = []
        for state_idx, action, reward in zip(state_sequence, action_sequence, rewards_sequence):
            action_probabilities.append(self.action_probability(state_idx, action))
            self.update_value(state_idx, action, reward)
        return np.array(action_probabilities)
    
    def value_information_sequence(self,state_sequence: NDArray, action_sequence: NDArray, rewards_sequence: NDArray) -> NDArray:
        value_information = []
        for state_idx, action, reward in zip(state_sequence, action_sequence, rewards_sequence):
            values = self.update_value(state_idx, action, reward)
            value_information.append(values)
        return np.array(value_information)

    @staticmethod
    def logistic_transform(x: float) -> float:
        """Applies the logistic function from (0,1) to (-inf, inf)."""
        return np.log(x / (1 - x))

    @staticmethod
    def inv_logistic_transform(x: float) -> float:
        """Applies the inverse logistic function (-inf, inf) to (0,1)."""
        return 1 / (1 + np.exp(-x))
    @abstractmethod
    def update_parameters(self, h: np.ndarray) -> None:
        """Updates model parameters using transformed values."""
        pass # Implement in subclasses


### **QL_RL (inherits from AbstractQL)**
class QL_RL(AbstractQL):
    def __init__(self, mdp: MDP, states: NDArray,rho: float = 1.0, epsilon: float = 0.1, irreducible_noise: float = 0.1, bias: float = 0):
        super().__init__(mdp,states)
        self.irreducible_noise: float = irreducible_noise
        self.bias: float = bias
        self.rho: float = rho
        self.epsilon: float = epsilon


    def update_value(self, state_idx: int, action: int, reward: float) -> None:
        """Updates Q-values using a simple Q-learning rule."""
        state = self.states[state_idx]
        delta = (reward - state.Q[action])
        state.Q[action] += self.rho * (reward - state.Q[action])
        return delta, state.Q
    def choose_action(self, state_idx: int) -> int:
        """Selects an action using a softmax-based policy with noise handling."""
        state = self.states[state_idx]
        weights = state.Q + self.bias
        exp_weights = np.exp(weights - np.max(weights))  # Numerical stability
        probabilities = (1 - self.irreducible_noise) * (exp_weights / np.sum(exp_weights))
        probabilities += self.irreducible_noise / len(state.Q)
        return np.random.choice(len(state.Q), p=probabilities)
    def action_probability(self, state_idx: int, action: int) -> float:
        """Computes the probability of selecting a given action."""
        state = self.states[state_idx]
        weights = state.Q + self.bias
        exp_weights = np.exp(weights - np.max(weights))  # Numerical stability
        probabilities = (1 - self.irreducible_noise) * (exp_weights / np.sum(exp_weights))
        probabilities += self.irreducible_noise / len(state.Q)
        return probabilities[action]
    def learning_strategy(self, state, action, reward, next_state=None):
        return self.update_value(state, action, reward)
    
    def update_parameters(self, h):
        self.rho = self.inv_logistic_transform(h[0])
        self.epsilon = self.inv_logistic_transform(h[1])
        self.irreducible_noise = self.inv_logistic_transform(h[2])
        self.bias = np.log(h[3])
    
    def set_parameters(self, parameters):
        self.update_parameters(parameters)


### **QL1_RL (inherits from AbstractQL)**
class QL1_RL(AbstractQL):
    def __init__(self, mdp: MDP, states: NDArray,alphaP: float = 0.1, alphaN: float = 0.1):
        super().__init__(mdp,states)
        self.alphaP: float = alphaP
        self.alphaN: float = alphaN

    def update_value(self, state_idx: int, action: int, reward: float) -> None:
        """Updates Q-values with separate learning rates for positive and negative rewards."""
        state = self.states[state_idx]
        delta = reward - state.Q[action]
        if  delta> 0:
            state.Q[action] += self.alphaP * delta
        else:
            state.Q[action] += self.alphaN * delta
        return delta,state.Q[action]
    def choose_action(self, state_idx: int) -> int:
        """Selects an action using a softmax-based policy with noise handling."""
        state = self.states[state_idx]
        weights = state.Q
        exp_weights = np.exp(weights - np.max(weights))  # Numerical stability
        probabilities = (exp_weights / np.sum(exp_weights))
        return np.random.choice(len(state.Q), p=probabilities)
    def action_probability(self, state_idx: int, action: int) -> float:
        """Computes the probability of selecting a given action."""
        state = self.states[state_idx]
        weights = state.Q
        exp_weights = np.exp(weights - np.max(weights))  # Numerical stability
        probabilities = (exp_weights / np.sum(exp_weights))
        return probabilities[action]
    def learning_strategy(self, state, action, reward, next_state=None):
        return self.update_value(state, action, reward)
    
    def update_parameters(self, h):
        self.alphaP = self.inv_logistic_transform(h[0])
        self.alphaN = self.inv_logistic_transform(h[1])

    def set_parameters(self, parameters):
        self.update_parameters(parameters)
