from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple, List
import pandas as pd
from pandas import DataFrame
from ..tasks import State,Action
from numpy.typing import NDArray
from .base_fitting import BaseFittingStrategy
from cognitive_models import MDPEnvironment, RLLearning
import numpy as np
import torch
import torch.optim as optim
from torch.distributions import MultivariateNormal
from joblib import Parallel, delayed
from tqdm import tqdm

class EMAbstract(BaseFittingStrategy,ABC):
      def __init__(self, environment: MDPEnvironment, num_params: int, num_iteration: int):
          """
          Initialize the fitting strategy with an environment.
          
          Parameters:
          - environment (Any): The environment in which the agent interacts with the task.
          """
          super().__init__(environment,num_params)
          self.num_iteration = num_iteration
      

      @abstractmethod
      def InitializeEstep(self) -> NDArray:
           """
           Initialize the parameters of Q-distribution over Latent Variables during E-step for each subject. 
           """
           pass
      @abstractmethod
      def ParameterEstimateQ(self, Q_parameters: NDArray) -> NDArray:
           """
           Return Estimate Latent Variables given the optimized parameter of Q using MAP or Expecation or other startegies.
           """
           pass
      @abstractmethod  
      def E_step(self, behavioral_data: Tuple[NDArray, NDArray, NDArray]) -> NDArray:
            """
            Perform the E-step of the EM algorithm.
            
            Parameters:
            - behavioral_data (Tuple[NDArray, NDArray, NDArray]): A tuple of arrays containing states, actions, and rewards/observations.
            - parameters (NDArray): The current parameter estimates.
            
            Returns:
            - Tuple[NDArray, NDArray]: The expected sufficient statistics for each subject.
            """
 
      
      @abstractmethod
      def OptimizeQ(self,behavioral_data: Tuple[NDArray,NDArray,NDArray], initial_guess: NDArray) -> NDArray:
            """
            Optimize Q distribution over Latant variabels to obtain the posterior in E Step.
            
            Parameters:
            - behavioral_data (Tuple[NDArray, NDArray, NDArray]): A tuple of arrays containing states, actions, and rewards/observations.
            - initial_guess (NDArray): The initial parameter estimates.
            
            Returns:
            - NDArray: The optimized parameter estimates.
            """
            pass
      
      @abstractmethod
      def M_step(self, QParmaters: NDArray) -> NDArray:
            """
            Perform the M-step of the EM algorithm.
            
            Parameters:
            - ParametersQ (NDArray): The expected sufficient statistics for each subject.
            
            Returns:
            - NDArray: The return the parameter mean over population and variance given the specified prior distribution.
            """
            pass


class EMGuassian(EMAbstract):
      def __init__(self, environment: MDPEnvironment,num_params: int, num_iteration_em: int=10,
                   num_iteration_garadian_decent:int=10000,
                   learning_rate: float = 0.01,
                   tol: float =1e-6):
          """
          Initialize the fitting strategy with an environment.
          
          Parameters:
          - environment (Any): The environment in which the agent interacts with the task.
          """
          super().__init__(environment,num_params,num_iteration_em)
          self.prior_variance = np.ones((self.num_params,))
          self.prior_mean = np.zeros((self.num_params,))
          self.num_iters_garadian_decent = num_iteration_garadian_decent
          self.tol = tol

          # Ensure prior_variance and prior_mean are tensors
          self.prior_variance_tensor = torch.tensor(self.prior_variance, dtype=torch.float32)
          self.prior_mean_tensor = torch.tensor(self.prior_mean, dtype=torch.float32)
          self.learning_rate = learning_rate
      
      def InitializeEstep(self,Num) -> NDArray:
           """
           Initialize the parameters of Q-distribution over Latent Variables during E-step for each subject. 
           """
           Initial = (2*np.ones((Num,2,self.num_params)))**0.5
           Initial[:,0,:] = 0


           return Initial

      def ParameterEstimateQ(self, Q_parameters: NDArray) -> NDArray:
           """
           Return Estimate Latent Variables given the optimized parameter of Q using MAP or Expecation or other startegies.
           """
           return Q_parameters[0]
      

      def E_step(self, behavioral_data: NDArray) -> Tuple[NDArray,NDArray]:

            """
            Perform the E-step of the EM algorithm.
            
            Parameters:
            - behavioral_data (Tuple[NDArray, NDArray, NDArray]): A tuple of arrays containing states, actions, and rewards/observations.
            - parameters (NDArray): The current parameter estimates.
            
            Returns:
            - Tuple[NDArray, NDArray]: The expected sufficient statistics for each subject.
            """
            
            States = behavioral_data[:,:,0]
            Actions = behavioral_data[:,:,1]
            Rewards = behavioral_data[:,:,2]
            assert States.shape[0] == Actions.shape[0] == Rewards.shape[0]
            initial_guess = self.InitializeEstep(behavioral_data.shape[0])
            Num_subjects = States.shape[0]
            Q_parameters = np.zeros_like(initial_guess)
            Parameters = np.zeros((Num_subjects,self.num_params))
            for i in range(Num_subjects):
                Q_parameters[i] = self.OptimizeQ(behavioral_data=behavioral_data[i,:,:],
                                    initial_guess=initial_guess[i])
                Parameters[i] = self.ParameterEstimateQ(Q_parameters[i])
                
            
            return Parameters,Q_parameters
        
      def OptimizeQ(self,behavioral_data: NDArray, initial_guess: NDArray) -> NDArray:
            """
            Optimize Q distribution over Latant variabels to obtain the posterior in E Step.
            
            Parameters:
            - behavioral_data (Tuple[NDArray, NDArray, NDArray]): A tuple of arrays containing states, actions, and rewards/observations.
            - initial_guess (NDArray): The initial parameter estimates.
            
            Returns:
            - NDArray: The optimized parameter estimates.
            """
            mu = torch.tensor(initial_guess[0], dtype=torch.float32, requires_grad=True)
            # sigma = torch.tensor(initial_guess[1], dtype=torch.float32, requires_grad=True)
            sigma = torch.tensor(initial_guess[1], dtype=torch.float32, requires_grad=False)
            
            
            # optimizer = torch.optim.Adam([sigma, mu], lr=learning_rate)
            optimizer = torch.optim.Adam([mu], lr=self.learning_rate)
            
            
            for i in range(self.num_iters_garadian_decent):
                  optimizer.zero_grad()
                  # grad_sigma, grad_mu = self.stochastic_gradient(sigma, mu, behavioral_data)
                  
                  grad_mu = self.stochastic_gradient(sigma, mu, behavioral_data)

                  # sigma.grad = -grad_sigma
                  mu.grad = -grad_mu
                  
                  optimizer.step()
                  
                  if (i+1) % 1000 == 0:
                        print(f"Iteration {i+1}: sigma = {sigma.detach().numpy()}, mu = {mu.detach().numpy()}")
            
            QParameters = np.array([mu.detach().numpy(),sigma.detach().numpy()])
            return QParameters
      

      def stochastic_gradient(self, sigma, mu, behavioral_data):
            # sigma, mu = torch.tensor(sigma,dtype=torch.float32), torch.tensor(mu,dtype=torch.float32)  # Convert inputs to tensors
            
            # h_i = torch.tensor(torch.multivariate_normal(mu, torch.diag(sigma), size=(10,)), requires_grad=False)  # Ensure h_i is a tensor but not part of computation graph
            # Sample from multivariate normal distribution
            dist = MultivariateNormal(mu, torch.diag(sigma))
            h_i = dist.sample((1,))  # Generate 1 samples

            logP = torch.tensor(self.log_likelihood(behavioral_data=behavioral_data, parameters=h_i.detach().numpy()),dtype=torch.float32)  # Convert h_i to NumPy before passing to log_likelihood
            # For test
            # grad1_sigma = (-1 / sigma + ((h_i[0] - mu)**2 / sigma**3)) * logP
            # grad_sigma = 1 / sigma - sigma / self.prior_variance_tensor + grad1_sigma
            
            grad1_mu = ((h_i[0] - mu) / (sigma**2)) * (logP+120)
            grad_mu = grad1_mu-((mu - self.prior_mean_tensor) / self.prior_variance_tensor) 
            
            return grad_mu
            # return grad_sigma, grad_mu

      def log_likelihood(self, behavioral_data:NDArray, parameters: NDArray) -> float:
            States = behavioral_data[:,0]
            Actions = behavioral_data[:,1]
            Rewards = behavioral_data[:,2]
            self.environment.agent.set_parameters(parameters=parameters[0])
            Probablities = self.environment.agent.action_probabilities_sequence(States,Actions,Rewards)
            return np.sum(np.log(Probablities))

            
      
      def M_step(self,QParmaters: NDArray) -> NDArray:
            """
            Perform the M-step of the EM algorithm.
            
            Parameters:(self.Trajectories[:,:,0], self.Trajectories[:,:,1], self.Trajectories[:,:,2])
            - ParametersQ (NDArray): The expected sufficient statistics for each subject.
            (self.Trajectories[:,:,0], self.Trajectories[:,:,1], self.Trajectories[:,:,2])
            Returns:
            - NDArray: The return (self.Trajectories[:,:,0], self.Trajectories[:,:,1], self.Trajectories[:,:,2])the parameter mean over population and variance given the specified prior distribution.
            """
            self.prior_mean = np.mean(QParmaters[:,0,0:self.num_params],axis=0)
            TempArray = QParmaters[:,1,0:self.num_params] ** 2 + (QParmaters[:,0,0:self.num_params] - self.prior_mean) ** 2
            self.prior_variance = np.mean(TempArray,axis=0) 

            
            # Ensure prior_variance and prior_mean are tensors
            self.prior_variance_tensor = torch.tensor(self.prior_variance, dtype=torch.float32)
            self.prior_mean_tensor = torch.tensor(self.prior_mean, dtype=torch.float32)
            

      def loadData(self, data):
            pass

      def fit(self,behavioral_data: NDArray) -> Dict:
            """
            Fit the model parameters using the EM algorithm until convergence.

            Parameters:
            - states (array): States for each subject.
            - actions (array): Actions for each subject.
            - rewards (array): Rewards for each subject.
            - tol (float): Convergence tolerance for parameter change.
            - max_iterations (int): Maximum number of iterations to prevent infinite loop.
            
            Returns:
            - dict: Contains the fitted prior mean, prior variance, subject means, and subject variances.
            """
            prev_prior_mean = np.copy(self.prior_mean)

            
            # Use tqdm to create a progress bar for the iterations
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
            # for iteration in tqdm(range(self.num_iteration), desc="EM Iterations", unit="iter"):
            # # for iteration in range(self.num_iterations):
            #       # E-step
            #       EstimatedParameters, QP = self.E_step(behavioral_data=behavioral_data)

            #       # M-step
            #       self.M_step(QP)

            #       # Check for convergence based on changes in prior_mean and prior_variance
            #       mean_change = np.linalg.norm(self.prior_mean - prev_prior_mean)

            #       # Update previous mean and variance for next iteration comparison
            #       prev_prior_mean = np.copy(self.prior_mean)
            #       print(f"\nIteration {iteration} has completed and")
            #       print(f"mean change is{self.prior_mean}")

            #       if mean_change < self.tol:
            #             print(f"Convergence achieved after {iteration + 1} iterations.")
            #             break
            
            # return {
            #       'prior_mean': self.prior_mean,
            #       'prior_variance': self.prior_variance,
            #       'subject_means': EstimatedParameters
            # } 