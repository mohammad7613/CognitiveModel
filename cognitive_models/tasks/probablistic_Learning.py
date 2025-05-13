import numpy as np
import random
from .base_tasks import MDP, State, Action
from typing import Tuple, List
from numpy.typing import NDArray

class PL(MDP):
    """Probablistic Learning Task in the Multiple Dissociations 
    Between Comorbid Depression and Anxiety on 
    Reward and Punishment Processing: Evidence From Computationally Informed EEG
    James F Cavanagh , Andrew W Bismark, Michael J Frank, John J B Allen"""

    TERMINAL_STATES = set()  # Optional future terminal states
    


    def __init__(self,task_states: NDArray, CONTEXTS: dict,TRANSITIONS:dict):
        """Initialize the TaskMDP with predefined states and actions."""
        super().__init__(task_states)
        self.CONTEXTS = CONTEXTS
        self.TRANSITIONS = TRANSITIONS
        # Create corresponding mapping to translate between action and state and their indexes. It is good approach for descrete state
        self.create_state_index_map()
    def create_state_index_map(self):
        self.indexes_to_state_action_map = {}
        self.state_action_to_indexes_map = {}
        self.states_to_indexes_map = {}
        self.indexes_to_state_map = {}
        context_keys = list(self.CONTEXTS.keys())
        for state_idx, state_name in enumerate(context_keys):
            action_keys = list(self.CONTEXTS[state_name].keys())
            self.states_to_indexes_map[state_name] = state_idx
            self.indexes_to_state_map[state_idx] = state_name
            self.indexes_to_state_action_map[state_idx] = state_name
            for action_idx, action_name in enumerate(action_keys):
                self.indexes_to_state_action_map[(state_idx, action_idx)] = (state_name, action_name)
                self.state_action_to_indexes_map[(state_name, action_name)] = (state_idx, action_idx)
    # This is a translator which convert the index of state and action in descrete task to the corresponding state_name and action_name
    # in the task. Ginen the name, we can access to the corresponding reward, transition and reward probability 
    # specified in CONTEXTS and TRANSITIONS.
    def translator(self, state: int, action: int)-> Tuple[str,str]:
        """Translate the state and action to the corresponding state and action in the task"""
        return self.indexes_to_state_action_map[(state, action)]
    

    def get_reward(self, state: int, action: int) -> float:
        """Return the reward for a given state-action pair."""
        state_name,action_name = self.translator(state,action)
        reward_values = list(self.CONTEXTS[state_name][action_name].keys())
        probabilities = list(self.CONTEXTS[state_name][action_name].values())
        reward_value = np.random.choice(reward_values, size=1, p=probabilities)
        return reward_value[0]


    def next_state(self, state: int, action: int) -> int:
        """Uniform transition to a new state."""
        state_name,action_name  = self.translator(state,action)
        TransitionProbabilities = list(self.TRANSITIONS[state_name][action_name].values())
        states = list(self.TRANSITIONS[state_name][action_name].keys())
        next_state_task = np.random.choice(states,size=1,p=TransitionProbabilities)[0]
        return self.states_to_indexes_map[next_state_task]  

    def transition_probability(self, state: int, action: int, next_state: int) -> float:
        """Return transition probability."""
        state_name,action_name  = self.translator(state,action)
        return self.TRANSITIONS[state_name][action_name][self.indexes_to_state_map[next_state]]

    def reward_probability(self, state: int, action: int, reward: float) -> float:
        """Calculate probability of receiving a specific reward."""
        state_name,action_name  = self.translator(state,action)
        reward_prob = self.CONTEXTS[state_name][action_name][reward]
        return reward_prob

    def is_terminal(self, state: int) -> bool:
        """Check if the state is terminal."""
        state_name = self.indexes_to_state_action_map[state]
        return state_name in self.TERMINAL_STATES
