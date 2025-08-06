from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple, List, Union, Callable
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
import os

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
    - behavioral_data (NDArray): Behavioral data for the subject, shape (n_trials, 3).
    - initial_guess (NDArray): Initial guess for Q parameters, shape (2, num_params).
    - environment (MDPEnvironment): The environment object.
    - num_params (int): Number of parameters.
    - num_iters_gd (int): Number of gradient descent iterations.
    - learning_rate (float): Learning rate for optimization.
    - prior_variance (NDArray): Prior variance for parameters, shape (num_params,).
    - prior_mean (NDArray): Prior mean for parameters, shape (num_params,).
    
    Returns:
    - Tuple[NDArray, NDArray]: Q parameters (shape (2, num_params)) and estimated parameters (shape (num_params,)).
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
    # params = q_params[0]  # ParameterEstimateQ: return mu
    return q_params

def _log_likelihood(behavioral_data: NDArray, parameters: NDArray, environment: MDPEnvironment) -> float:
    """
    Compute log likelihood for a subject's behavioral data.
    
    Parameters:
    - behavioral_data (NDArray): Array containing states, actions, and rewards, shape (n_trials, 3).
    - parameters (NDArray): Parameters to set in the agent, shape (num_params,).
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
        - environment (MDPEnvironment): The environment in which the agent interacts with the task.
        - num_params (int): Number of parameters.
        - num_iteration (int): Number of EM iterations.
        """
        super().__init__(environment, num_params)
        self.num_iteration = num_iteration
    
    @abstractmethod
    def InitializeEstep(self, num_subjects: int) -> NDArray:
        """Initialize the parameters of Q-distribution over Latent Variables during E-step for each subject."""
        pass
    
    @abstractmethod
    def ParameterEstimateQ(self, Q_parameters: NDArray) -> NDArray:
        """Return Estimate Latent Variables given the optimized parameter of Q using MAP or Expectation or other strategies."""
        pass
    
    @abstractmethod
    def E_step(self, behavioral_data: List[NDArray]) -> Tuple[NDArray, NDArray]:
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
                 num_iteration_gradient_descent: int=10000, learning_rate: float = 0.01,
                 tol: float =1e-6, data_loader: Callable[[str, Any], List[NDArray]] = None):
        """
        Initialize the fitting strategy with an environment.
        
        Parameters:
        - environment (MDPEnvironment): The environment in which the agent interacts with the task.
        - num_params (int): Number of parameters.
        - num_iteration_em (int): Number of EM iterations.
        - num_iteration_gradient_descent (int): Number of gradient descent iterations.
        - learning_rate (float): Learning rate for optimization.
        - tol (float): Convergence tolerance.
        - data_loader (Callable, optional): Custom function to load behavioral data from a file.
          Must return a List[NDArray] where each NDArray is shape (n_trials, 3).
        """
        super().__init__(environment, num_params, num_iteration_em)
        self.prior_variance = np.ones((self.num_params,))
        self.prior_mean = np.zeros((self.num_params,))
        self.num_iters_gradient_descent = num_iteration_gradient_descent
        self.tol = tol
        self.prior_variance_tensor = torch.tensor(self.prior_variance, dtype=torch.float32)
        self.prior_mean_tensor = torch.tensor(self.prior_mean, dtype=torch.float32)
        self.learning_rate = learning_rate
        self._data_loader = data_loader if data_loader is not None else self.load_behavioral_data
    
    def loadData(self, file_path: str, subject_col: str = 'subject_id',
                            state_col: str = 'state', action_col: str = 'action',
                            reward_col: str = 'reward', **kwargs) -> List[NDArray]:
        """
        Default loader for behavioral data from a CSV or Excel file.
        
        Parameters:
        - file_path (str): Path to the CSV or Excel file.
        - subject_col (str): Column name for subject IDs (used for grouping).
        - state_col (str): Column name for states.
        - action_col (str): Column name for actions.
        - reward_col (str): Column name for rewards.
        - **kwargs: Additional arguments for pandas.read_csv or read_excel.
        
        Returns:
        - List[NDArray]: List of per-subject data arrays (shape (n_trials, 3)).
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path, **kwargs)
        elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
            df = pd.read_excel(file_path, **kwargs)
        else:
            raise ValueError("File must be CSV or Excel (.csv, .xlsx, .xls)")

        required_cols = [subject_col, state_col, action_col, reward_col]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        df = df.dropna(subset=required_cols)
        if df.empty:
            raise ValueError("No valid data after removing missing values")

        subject_groups = df.groupby(subject_col)
        behavioral_data = []

        for _, group in subject_groups:
            data = group[[state_col, action_col, reward_col]].to_numpy()
            if data.shape[0] == 0:
                continue
            behavioral_data.append(data)

        if not behavioral_data:
            raise ValueError("No valid subject data found")

        return behavioral_data
    
    def InitializeEstep(self, num_subjects: int) -> NDArray:
        """Initialize the parameters of Q-distribution over Latent Variables during E-step for each subject."""
        Initial = (2 * np.ones((num_subjects, 2, self.num_params))) ** 0.5
        Initial[:, 0, :] = 0
        return Initial

    def ParameterEstimateQ(self, Q_parameters: NDArray) -> NDArray:
        """Return Estimate Latent Variables given the optimized parameter of Q using MAP or Expectation or other strategies."""
        return Q_parameters[0]
    
    def E_step(self, behavioral_data: List[NDArray]) -> Tuple[NDArray, NDArray]:
        """
        Perform the E-step of the EM algorithm with parallel processing.
        
        Parameters:
        - behavioral_data (List[NDArray]): List of per-subject data arrays, each shape (n_trials, 3).
        
        Returns:
        - Tuple[NDArray, NDArray]: The expected sufficient statistics (shape (n_subjects, num_params))
          and Q parameters (shape (n_subjects, 2, num_params)) for each subject.
        """
        if not CLOUDPICKLE_ENABLED:
            raise ImportError(
                "cloudpickle is required for parallel execution in E_step. "
                "Install it with `pip install cloudpickle`."
            )

        Num_subjects = len(behavioral_data)
        if Num_subjects == 0:
            raise ValueError("Behavioral data list is empty")

        for i, data in enumerate(behavioral_data):
            if data.shape[1] != 3:
                raise ValueError(f"Subject {i} data must have 3 columns (state, action, reward), got shape {data.shape}")

        initial_guess = self.InitializeEstep(Num_subjects)
        Q_parameters = np.zeros_like(initial_guess)
        # Parameters = np.zeros((Num_subjects, self.num_params))

        start_time_par = time.time()
        try:
            def cloudpickle_delayed(i, data, guess, env, n_params, n_iters, lr, p_var, p_mean):
                data = cloudpickle.loads(data)
                guess = cloudpickle.loads(guess)
                env = cloudpickle.loads(env)
                p_var = cloudpickle.loads(p_var)
                p_mean = cloudpickle.loads(p_mean)
                return i, _process_subject(i, data, guess, env, n_params, n_iters, lr, p_var, p_mean)

            results = Parallel(n_jobs=-1, backend='loky')(
                delayed(cloudpickle_delayed)(
                    i,
                    cloudpickle.dumps(behavioral_data[i]),
                    cloudpickle.dumps(initial_guess[i]),
                    cloudpickle.dumps(self.environment),
                    self.num_params,
                    self.num_iters_gradient_descent,
                    self.learning_rate,
                    cloudpickle.dumps(self.prior_variance),
                    cloudpickle.dumps(self.prior_mean)
                )
                for i in range(Num_subjects)
            )
            for idx, (q_params) in results:
                Q_parameters[idx] = q_params
                # Parameters[idx] = params
            par_time = time.time() - start_time_par
            print(f"Parallel E-step time: {par_time:.2f} seconds")
        except Exception as e:
            print(f"Parallel execution failed: {e}")
            raise

        return Q_parameters
    
    def OptimizeQ(self, behavioral_data: NDArray, initial_guess: NDArray) -> NDArray:
        """Optimize Q distribution over Latent variables to obtain the posterior in E Step."""
        mu = torch.tensor(initial_guess[0], dtype=torch.float32, requires_grad=True)
        sigma = torch.tensor(initial_guess[1], dtype=torch.float32, requires_grad=False)
        optimizer = torch.optim.Adam([mu], lr=self.learning_rate)
        for i in range(self.num_iters_gradient_descent):
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

    def fit(self, behavioral_data: Union[str, List[NDArray]],
            data_loader: Callable[[str, Any], List[NDArray]] = None,
            **load_kwargs) -> Dict:
        """
        Fit the model parameters using the EM algorithm until convergence.
        
        Parameters:
        - behavioral_data (Union[str, List[NDArray]]): Either a file path to a CSV/Excel file or
          a list of per-subject data arrays (each shape (n_trials, 3)).
        - data_loader (Callable, optional): Custom function to load behavioral data from a file.
          Must return a List[NDArray] where each NDArray is shape (n_trials, 3). Overrides
          self._data_loader if provided.
        - **load_kwargs: Additional arguments for the data loader.
        
        Returns:
        - Dict: Contains the fitted prior mean, prior variance, and subject means. The order of
          subject_means corresponds to the order of behavioral_data.
        """
        if isinstance(behavioral_data, str):
            loader = data_loader if data_loader is not None else self._data_loader
            if not callable(loader):
                raise TypeError("data_loader must be a callable")
            behavioral_data = loader(behavioral_data, **load_kwargs)
        else:
            if not isinstance(behavioral_data, list):
                raise TypeError("behavioral_data must be a list of NDArrays when not a file path")

        for i, data in enumerate(behavioral_data):
            if not isinstance(data, np.ndarray):
                raise TypeError(f"Subject {i} data must be a NumPy array, got {type(data)}")
            if data.shape[1] != 3:
                raise ValueError(f"Subject {i} data must have 3 columns (state, action, reward), got shape {data.shape}")

        prev_prior_mean = np.copy(self.prior_mean)
        print(f"Number of Subject:{len(behavioral_data)}")
        for iteration in tqdm(range(self.num_iteration), desc="EM Iterations", unit="iter"):
            QP = self.E_step(behavioral_data=behavioral_data)
            self.M_step(QP)
            EstimatedParameters = QP[:,0,:]
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