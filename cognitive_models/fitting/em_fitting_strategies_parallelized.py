from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple, List
import pandas as pd
from pandas import DataFrame
from ..tasks import State, Action
from numpy.typing import NDArray
from .base_fitting import BaseFittingStrategy
from cognitive_models import MDPEnvironment, RLLearning
import numpy as np
import torch
import torch.optim as optim
from torch.distributions import MultivariateNormal
from joblib import Parallel, delayed
from tqdm import tqdm
import time
from IPython.display import clear_output

# Check if cloudpickle is available
try:
    import cloudpickle
    CLOUDPICKLE_ENABLED = True
except ImportError:
    import warnings
    warnings.warn(
        "cloudpickle is not installed. Parallel execution requires cloudpickle for user-defined classes. "
        "Install it with `pip install cloudpickle`.",
        UserWarning
    )
    CLOUDPICKLE_ENABLED = False

def _process_subject(subject_idx: int, behavioral_data: NDArray, initial_guess: NDArray,
                    environment: MDPEnvironment, num_params: int, num_iters_gd: int,
                    learning_rate: float, prior_variance: NDArray, prior_mean: NDArray) -> Tuple[NDArray, NDArray]:
    """
    Process a single subject's data for the E-step, independent of EMGuassian instance.
    
    Parameters:
    - subject_idx (int): Index of the subject.
    - behavioral_data (NDArray): Behavioral data for the subject.
    - initial_guess (NDArray): Initial guess for Q parameters.
    - environment (MDPEnvironment): The environment object.
    - num_params (int): Number of parameters.
    - num_iters_gd (int): Number of gradient descent iterations.
    - learning_rate (float): Learning rate for optimization.
    - prior_variance (NDArray): Prior variance for parameters.
    - prior_mean (NDArray): Prior mean for parameters.
    
    Returns:
    - Tuple[NDArray, NDArray]: Q parameters and estimated parameters for the subject.
    """
    mu = torch.tensor(initial_guess[0], dtype=torch.float32, requires_grad=True)
    sigma = torch.tensor(initial_guess[1], dtype=torch.float32, requires_grad=False)
    prior_variance_tensor = torch.tensor(prior_variance, dtype=torch.float32)
    prior_mean_tensor = torch.tensor(prior_mean, dtype=torch.float32)
    
    optimizer = torch.optim.Adam([mu], lr=learning_rate)
    for i in range(num_iters_gd):
        optimizer.zero_grad()
        dist = MultivariateNormal(mu, torch.diag(sigma))
        h_i = dist.sample((1,))
        logP = torch.tensor(
            _log_likelihood(behavioral_data, h_i.detach().numpy(), environment),
            dtype=torch.float32
        )
        grad1_mu = ((h_i[0] - mu) / (sigma ** 2)) * (logP + 120)
        grad_mu = grad1_mu - ((mu - prior_mean_tensor) / prior_variance_tensor)
        mu.grad = -grad_mu
        optimizer.step()
        if (i + 1) % 1000 == 0:
            print(f"Subject {subject_idx}, Iteration {i + 1}: sigma = {sigma.detach().numpy()}, mu = {mu.detach().numpy()}")

    q_params = np.array([mu.detach().numpy(), sigma.detach().numpy()])
    params = q_params[0]  # ParameterEstimateQ: return mu
    return q_params, params

def _log_likelihood(behavioral_data: NDArray, parameters: NDArray, environment: MDPEnvironment) -> float:
    """
    Compute log likelihood for a subject's behavioral data.
    
    Parameters:
    - behavioral_data (NDArray): Array containing states, actions, and rewards.
    - parameters (NDArray): Parameters to set in the agent.
    - environment (MDPEnvironment): The environment object.
    
    Returns:
    - float: Log likelihood of the data.
    """
    States = behavioral_data[:, 0]
    Actions = behavioral_data[:, 1]
    Rewards = behavioral_data[:, 2]
    environment.agent.set_parameters(parameters=parameters[0])
    Probabilities = environment.agent.action_probabilities_sequence(States, Actions, Rewards)
    return np.sum(np.log(Probabilities))

class EMAbstract(BaseFittingStrategy, ABC):
    def __init__(self, environment: MDPEnvironment, num_params: int, num_iteration: int):
        """
        Initialize the fitting strategy with an environment.
        
        Parameters:
        - environment (Any): The environment in which the agent interacts with the task.
        - num_params (int): Number of parameters.
        - num_iteration (int): Number of EM iterations.
        """
        super().__init__(environment, num_params)
        self.num_iteration = num_iteration
    
    @abstractmethod
    def InitializeEstep(self) -> NDArray:
        """Initialize the parameters of Q-distribution over Latent Variables during E-step for each subject."""
        pass
    
    @abstractmethod
    def ParameterEstimateQ(self, Q_parameters: NDArray) -> NDArray:
        """Return Estimate Latent Variables given the optimized parameter of Q using MAP or Expectation or other strategies."""
        pass
    
    @abstractmethod  
    def E_step(self, behavioral_data: Tuple[NDArray, NDArray, NDArray]) -> NDArray:
        """Perform the E-step of the EM algorithm."""
        pass
    
    @abstractmethod
    def OptimizeQ(self, behavioral_data: NDArray, initial_guess: NDArray) -> NDArray:
        """Optimize Q distribution over Latent variables to obtain the posterior in E Step."""
        pass
    
    @abstractmethod
    def M_step(self, QParameters: NDArray) -> NDArray:
        """Perform the M-step of the EM algorithm."""
        pass

class EMGuassian(EMAbstract):
    def __init__(self, environment: MDPEnvironment, num_params: int, num_iteration_em: int=10,
                 num_iteration_garadian_decent: int=10000,
                 learning_rate: float = 0.01,
                 tol: float =1e-6):
        """
        Initialize the fitting strategy with an environment.
        
        Parameters:
        - environment (MDPEnvironment): The environment in which the agent interacts with the task.
        - num_params (int): Number of parameters.
        - num_iteration_em (int): Number of EM iterations.
        - num_iteration_garadian_decent (int): Number of gradient descent iterations.
        - learning_rate (float): Learning rate for optimization.
        - tol (float): Convergence tolerance.
        """
        super().__init__(environment, num_params, num_iteration_em)
        self.prior_variance = np.ones((self.num_params,))
        self.prior_mean = np.zeros((self.num_params,))
        self.num_iters_garadian_decent = num_iteration_garadian_decent
        self.tol = tol
        self.prior_variance_tensor = torch.tensor(self.prior_variance, dtype=torch.float32)
        self.prior_mean_tensor = torch.tensor(self.prior_mean, dtype=torch.float32)
        self.learning_rate = learning_rate
    
    def InitializeEstep(self, Num) -> NDArray:
        """Initialize the parameters of Q-distribution over Latent Variables during E-step for each subject."""
        Initial = (2 * np.ones((Num, 2, self.num_params))) ** 0.5
        Initial[:, 0, :] = 0
        return Initial

    def ParameterEstimateQ(self, Q_parameters: NDArray) -> NDArray:
        """Return Estimate Latent Variables given the optimized parameter of Q using MAP or Expectation or other strategies."""
        return Q_parameters[0]
    
    def E_step(self, behavioral_data: NDArray) -> Tuple[NDArray, NDArray]:
        """
        Perform the E-step of the EM algorithm with parallel processing.
        
        Parameters:
        - behavioral_data (NDArray): Array containing states, actions, and rewards/observations.
        
        Returns:
        - Tuple[NDArray, NDArray]: The expected sufficient statistics and Q parameters for each subject.
        """
        if not CLOUDPICKLE_ENABLED:
            raise ImportError(
                "cloudpickle is required for parallel execution in E_step. "
                "Install it with `pip install cloudpickle`."
            )

        States = behavioral_data[:, :, 0]
        Actions = behavioral_data[:, :, 1]
        Rewards = behavioral_data[:, :, 2]
        assert States.shape[0] == Actions.shape[0] == Rewards.shape[0]
        initial_guess = self.InitializeEstep(behavioral_data.shape[0])
        Num_subjects = States.shape[0]
        Q_parameters = np.zeros_like(initial_guess)
        Parameters = np.zeros((Num_subjects, self.num_params))

        # Parallel execution with cloudpickle
        start_time_par = time.time()
        try:
            def cloudpickle_delayed(i, data, guess, env, n_params, n_iters, lr, p_var, p_mean):
                # Deserialize arguments with cloudpickle
                data = cloudpickle.loads(data)
                guess = cloudpickle.loads(guess)
                env = cloudpickle.loads(env)
                p_var = cloudpickle.loads(p_var)
                p_mean = cloudpickle.loads(p_mean)
                return _process_subject(i, data, guess, env, n_params, n_iters, lr, p_var, p_mean)

            results = Parallel(n_jobs=-1, backend='loky')(
                delayed(cloudpickle_delayed)(
                    i,
                    cloudpickle.dumps(behavioral_data[i, :, :]),
                    cloudpickle.dumps(initial_guess[i]),
                    cloudpickle.dumps(self.environment),
                    self.num_params,
                    self.num_iters_garadian_decent,
                    self.learning_rate,
                    cloudpickle.dumps(self.prior_variance),
                    cloudpickle.dumps(self.prior_mean)
                )
                for i in range(Num_subjects)
            )
            for i, (q_params, params) in enumerate(results):
                Q_parameters[i] = q_params
                Parameters[i] = params
            par_time = time.time() - start_time_par
            print(f"Parallel E-step time: {par_time:.2f} seconds")
        except Exception as e:
            print(f"Parallel execution failed: {e}")
            raise

        return Parameters, Q_parameters
    
    def OptimizeQ(self, behavioral_data: NDArray, initial_guess: NDArray) -> NDArray:
        """Optimize Q distribution over Latent variables to obtain the posterior in E Step."""
        mu = torch.tensor(initial_guess[0], dtype=torch.float32, requires_grad=True)
        sigma = torch.tensor(initial_guess[1], dtype=torch.float32, requires_grad=False)
        optimizer = torch.optim.Adam([mu], lr=self.learning_rate)
        for i in range(self.num_iters_garadian_decent):
            optimizer.zero_grad()
            grad_mu = self.stochastic_gradient(sigma, mu, behavioral_data)
            mu.grad = -grad_mu
            optimizer.step()
            if (i + 1) % 1000 == 0:
                print(f"Iteration {i + 1}: sigma = {sigma.detach().numpy()}, mu = {mu.detach().numpy()}")
        QParameters = np.array([mu.detach().numpy(), sigma.detach().numpy()])
        return QParameters
    
    def stochastic_gradient(self, sigma, mu, behavioral_data):
        dist = MultivariateNormal(mu, torch.diag(sigma))
        h_i = dist.sample((1,))
        logP = torch.tensor(self.log_likelihood(behavioral_data=behavioral_data, parameters=h_i.detach().numpy()), dtype=torch.float32)
        grad1_mu = ((h_i[0] - mu) / (sigma ** 2)) * (logP + 120)
        grad_mu = grad1_mu - ((mu - self.prior_mean_tensor) / self.prior_variance_tensor)
        return grad_mu

    def log_likelihood(self, behavioral_data: NDArray, parameters: NDArray) -> float:
        return _log_likelihood(behavioral_data, parameters, self.environment)
    
    def M_step(self, QParameters: NDArray) -> NDArray:
        """Perform the M-step of the EM algorithm."""
        self.prior_mean = np.mean(QParameters[:, 0, 0:self.num_params], axis=0)
        TempArray = QParameters[:, 1, 0:self.num_params] ** 2 + (QParameters[:, 0, 0:self.num_params] - self.prior_mean) ** 2
        self.prior_variance = np.mean(TempArray, axis=0)
        self.prior_variance_tensor = torch.tensor(self.prior_variance, dtype=torch.float32)
        self.prior_mean_tensor = torch.tensor(self.prior_mean, dtype=torch.float32)

    def loadData(self, data):
        pass

    def fit(self, behavioral_data: NDArray) -> Dict:
        """
        Fit the model parameters using the EM algorithm until convergence.
        
        Parameters:
        - behavioral_data (NDArray): Array containing states, actions, and rewards/observations.
        
        Returns:
        - Dict: Contains the fitted prior mean, prior variance, and subject means.
        """
        prev_prior_mean = np.copy(self.prior_mean)
        print("Hi")
        for iteration in tqdm(range(self.num_iteration), desc="EM Iterations", unit="iter"):
            EstimatedParameters, QP = self.E_step(behavioral_data=behavioral_data)
            self.M_step(QP)
            mean_change = np.linalg.norm(self.prior_mean - prev_prior_mean)
            prev_prior_mean = np.copy(self.prior_mean)
            clear_output(wait=True)
            print(f"Iteration {iteration + 1} completed, mean change: {mean_change:.6f}, prior mean: {self.prior_mean}")
            if mean_change < self.tol:
                print(f"Convergence achieved after {iteration + 1} iterations.")
                break
        return {
            'prior_mean': self.prior_mean,
            'prior_variance': self.prior_variance,
            'subject_means': EstimatedParameters
        }