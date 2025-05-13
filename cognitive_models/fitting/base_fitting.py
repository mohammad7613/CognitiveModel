from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple, List
import pandas as pd
from pandas import DataFrame
from numpy.typing import NDArray
from cognitive_models import MDPEnvironment, State, Action


class BaseFittingStrategy(ABC):
    def __init__(self, environment: MDPEnvironment,num_params: int):
        """
        Initialize the fitting strategy with an environment.
        
        Parameters:
        - environment (Any): The environment in which the agent interacts with the task.
        """
        self.environment = environment
        self.num_params = num_params

    @abstractmethod
    def loadData(self, data: Any) -> Tuple[NDArray, NDArray, NDArray]:
        """
        Load the data from any data structure to three arrays of Size N*T, N*T, N*T where N is the number of subjects and T is the number of trials.
        These arrays are States and Actions and Rewards/Observations.
        
        Parameters:
        - data (List[Dict[str, Any]]): A list of dictionaries, where each dictionary contains the data of one subject.
        
        Returns:
        - DataFrame: The data as a DataFrame.
        """
        pass

    @abstractmethod
    def fit(self, behavioral_data: Tuple[NDArray, NDArray, NDArray], **kwargs) -> Dict:
        """
        Fit the model to behavioral data and return estimated parameters.
        
        Parameters:
        - behavioral_data (List[Dict[str, Any]]): A list where each entry corresponds to a subject's data, 
          containing states, actions, and rewards or observations.

        Returns:
        - Dict[paramater_means, variance_means, NumpyArray of N*H]: A dictionary of estimated parameters for each subject
        H is the dimention of parameter.
        """
        pass

    @abstractmethod
    def log_likelihood(self, behavioral_data: Tuple[NDArray, NDArray, NDArray], parameters: NDArray) -> float:
        """
        Compute the log-likelihood of the behavioral data given the model parameters.

        Parameters:
        - behavioral_data (List[Dict[str, Any]]): A list of sequences of states, actions, and rewards.

        Returns:
        - float: The log-likelihood value.
        """
        pass
