from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple, Callable
import numpy as np
from numpy.typing import NDArray
import torch
import torch.optim as optim
from torch.distributions import MultivariateNormal
from joblib import Parallel, delayed
from tqdm import tqdm
from .base_fitting import BaseFittingStrategy
from cognitive_models import MDPEnvironment, RLLearning

class EMAbstract(BaseFittingStrategy, ABC):
    def __init__(self, environment: MDPEnvironment, num_params: int, num_iteration: int):
        super().__init__(environment, num_params)
        self.num_iteration = num_iteration

    @abstractmethod
    def InitializeEstep(self, Num) -> NDArray:
        pass

    @abstractmethod
    def ParameterEstimateQ(self, Q_parameters: NDArray) -> NDArray:
        pass

    @abstractmethod
    def E_step(self, behavioral_data: NDArray) -> Tuple[NDArray, NDArray]:
        pass

    @abstractmethod
    def OptimizeQ(self, behavioral_data: NDArray, initial_guess: NDArray) -> NDArray:
        pass

    @abstractmethod
    def M_step(self, QParameters: NDArray) -> NDArray:
        pass

    @abstractmethod
    def loadData(self, data, state_converter: Callable[[Any], Any] = None):
        pass

class EMGuassian(EMAbstract):
    def __init__(self, environment: MDPEnvironment, num_params: int, num_iteration_em: int = 10,
                 num_iteration_garadian_decent: int = 10000, learning_rate: float = 0.01,
                 tol: float = 1e-6):
        super().__init__(environment, num_params, num_iteration_em)
        self.prior_variance = np.ones((self.num_params,))
        self.prior_mean = np.zeros((self.num_params,))
        self.num_iters_garadian_decent = num_iteration_garadian_decent
        self.tol = tol
        self.prior_variance_tensor = torch.tensor(self.prior_variance, dtype=torch.float32)
        self.prior_mean_tensor = torch.tensor(self.prior_mean, dtype=torch.float32)
        self.learning_rate = learning_rate
        self.state_mapping = {}  # Store original states for log_likelihood

    def InitializeEstep(self, Num) -> NDArray:
        Initial = (2 * np.ones((Num, 2, self.num_params))) ** 0.5
        Initial[:, 0, :] = 0
        return Initial

    def ParameterEstimateQ(self, Q_parameters: NDArray) -> NDArray:
        return Q_parameters[0]

    def loadData(self, data, state_converter: Callable[[Any], Any] = None):
        """
        Preprocess behavioral data to ensure picklability.
        
        Parameters:
        - data: List or array of [states, actions, rewards] per subject.
        - state_converter: Function to convert custom state objects to picklable types.
                          If None, assumes states are already picklable scalars.
        """
        processed_data = np.zeros((len(data), len(data[0]), 3), dtype=np.float64)
        self.state_mapping = {}  # Reset state mapping
        for i, subject_data in enumerate(data):
            for j, (state, action, reward) in enumerate(subject_data):
                if state_converter is not None:
                    state_value = state_converter(state)
                    self.state_mapping[state_value] = state  # Store for log_likelihood
                    processed_data[i, j, 0] = state_value
                else:
                    processed_data[i, j, 0] = state
                processed_data[i, j, 1] = action.value if hasattr(action, 'value') else action
                processed_data[i, j, 2] = reward
        self.behavioral_data = processed_data

    def OptimizeQ(self, behavioral_data: NDArray, initial_guess: NDArray) -> NDArray:
        """
        Optimize Q distribution over latent variables to obtain the posterior in E Step.
        
        Parameters:
        - behavioral_data (NDArray): Array containing states, actions, and rewards/observations.
        - initial_guess (NDArray): The initial parameter estimates.
        
        Returns:
        - NDArray: The optimized parameter estimates.
        """
        mu = torch.tensor(initial_guess[0], dtype=torch.float32, requires_grad=True)
        sigma = torch.tensor(initial_guess[1], dtype=torch.float32, requires_grad=False)
        optimizer = torch.optim.Adam([mu], lr=self.learning_rate)

        for i in range(self.num_iters_garadian_decent):
            optimizer.zero_grad()
            dist = MultivariateNormal(mu, torch.diag(sigma))
            h_i = dist.sample((1,))
            logP = torch.tensor(
                self.log_likelihood(behavioral_data, h_i.detach().numpy()),
                dtype=torch.float32
            )
            grad1_mu = ((h_i[0] - mu) / (sigma ** 2)) * (logP + 120)
            grad_mu = grad1_mu - ((mu - self.prior_mean_tensor) / self.prior_variance_tensor)
            mu.grad = -grad_mu
            optimizer.step()
            if (i + 1) % 1000 == 0:
                print(f"Iteration {i + 1}: sigma = {sigma.detach().numpy()}, mu = {mu.detach().numpy()}")

        return np.array([mu.detach().numpy(), sigma.detach().numpy()])

    def log_likelihood(self, data: NDArray, parameters: NDArray) -> float:
        States = data[:, 0]
        Actions = data[:, 1]
        Rewards = data[:, 2]
        if self.state_mapping:
            States = [self.state_mapping.get(s, s) for s in States]
        self.environment.agent.set_parameters(parameters=parameters[0])
        Probabilities = self.environment.agent.action_probabilities_sequence(States, Actions, Rewards)
        return np.sum(np.log(Probabilities))

    def E_step(self, behavioral_data: NDArray) -> Tuple[NDArray, NDArray]:
        """
        Perform the E-step of the EM algorithm.
        
        Parameters:
        - behavioral_data (NDArray): Array containing states, actions, and rewards/observations.
        
        Returns:
        - Tuple[NDArray, NDArray]: The expected sufficient statistics for each subject.
        """
        States = behavioral_data[:, :, 0]
        Actions = behavioral_data[:, :, 1]
        Rewards = behavioral_data[:, :, 2]
        assert States.shape[0] == Actions.shape[0] == Rewards.shape[0]
        initial_guess = self.InitializeEstep(behavioral_data.shape[0])
        Num_subjects = States.shape[0]
        Q_parameters = np.zeros_like(initial_guess)
        Parameters = np.zeros((Num_subjects, self.num_params))

        def process_subject(i: int, subject_data: NDArray, init_guess: NDArray,
                           em_instance: 'EMGuassian') -> Tuple[NDArray, NDArray]:
            q_param = em_instance.OptimizeQ(subject_data, init_guess)
            param = em_instance.ParameterEstimateQ(q_param)
            return q_param, param

        try:
            results = Parallel(n_jobs=-1, backend='loky')(
                delayed(process_subject)(
                    i, behavioral_data[i, :, :], initial_guess[i], self
                ) for i in range(Num_subjects)
            )
        except Exception as e:
            print(f"Parallel execution failed: {e}")
            raise

        for i, (q_param, param) in enumerate(results):
            Q_parameters[i] = q_param
            Parameters[i] = param

        return Q_parameters, Parameters

    def M_step(self, QParameters: NDArray) -> NDArray:
        self.prior_mean = np.mean(QParameters[:, 0, 0:self.num_params], axis=0)
        TempArray = QParameters[:, 1, 0:self.num_params] ** 2 + (QParameters[:, 0, 0:self.num_params] - self.prior_mean) ** 2
        self.prior_variance = np.mean(TempArray, axis=0)
        self.prior_variance_tensor = torch.tensor(self.prior_variance, dtype=torch.float32)
        self.prior_mean_tensor = torch.tensor(self.prior_mean, dtype=torch.float32)

    def fit(self, behavioral_data: NDArray) -> Dict:
        prev_prior_mean = np.copy(self.prior_mean)
        print("Hi")
        for iteration in tqdm(range(self.num_iteration), desc="EM Iterations", unit="iter"):
            EstimatedParameters, QP = self.E_step(behavioral_data=behavioral_data)
            self.M_step(QP)
            mean_change = np.linalg.norm(self.prior_mean - prev_prior_mean)
            prev_prior_mean = np.copy(self.prior_mean)
            print(f"\nIteration {iteration + 1} completed, mean change: {mean_change:.6f}, prior mean: {self.prior_mean}")
            if mean_change < self.tol:
                print(f"Convergence achieved after {iteration + 1} iterations.")
                break
        return {
            'prior_mean': self.prior_mean,
            'prior_variance': self.prior_variance,
            'subject_means': EstimatedParameters
        }