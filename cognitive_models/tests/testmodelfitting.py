from cognitive_models import *
import unittest
import numpy as np
from numpy.typing import NDArray
import copy
import matplotlib.pyplot as plt

class TestPL(unittest.TestCase):
      
    def setUp(self):
        # CONTEXTS = {
        #     'A_B': {'A': {1: 0.8, -1: 0.2}, 'B': {1: 0.2, -1: 0.8}},
        #     'C_D': {'C': {1: 0.7, -1: 0.3}, 'D': {1: 0.3, -1: 0.7}},
        #     'E_F': {'E': {1: 0.6, -1: 0.4}, 'F': {1: 0.4, -1: 0.6}},
        # }
        # TRANSITIONS = {
        #     'A_B': {'A': {'A_B':1/3,'C_D':1/3,'E_F':1/3},'B': {'A_B':1/3,'C_D':1/3,'E_F':1/3}},
        #     'C_D': {'C': {'A_B':1/3,'C_D':1/3,'E_F':1/3},'D': {'A_B':1/3,'C_D':1/3,'E_F':1/3}},
        #     'E_F': {'E': {'A_B':1/3,'C_D':1/3,'E_F':1/3},'F': {'A_B':1/3,'C_D':1/3,'E_F':1/3}},
        # }
        CONTEXTS,TRANSITIONS = parse_mdp_config("cognitive_models/tasks/mdp_pl_config.json")

        class StateTask(State):
            def __init__(self, name: str=None):
                super().__init__(name)
        class StateQ(State):
            def __init__(self, name: str=None, Q: NDArray[np.float64]=None):
                super().__init__(name)
                self.Q = Q



       



        StatesTask = np.empty((3,),dtype=object)
        for i,k in enumerate(list(CONTEXTS.keys())):
            StatesTask[i] = StateTask(k)           
        self.task_mdp = PL(StatesTask,CONTEXTS,TRANSITIONS)



        StatesAgent = np.empty((3,),dtype=object)
        for i,k in enumerate(list(CONTEXTS.keys())):
            StatesAgent[i] = StateQ(k,np.zeros((2,)))


            
              




        ## Create a simulated data
        ### Define the range parameters using uniform strategy
        self.NumberSubjects = 75

        #### Define ranges for each parameter
        alphaPrange = (0.1, 1)
        alphaNrange = (0.1, 1)
    
        #### Generate uniformly distributed samples
        self.alphaPsamples = np.random.uniform(*alphaPrange, self.NumberSubjects)
        self.alphaNsamples = np.random.uniform(*alphaNrange, self.NumberSubjects)

        #### Generate Guassian Range

        self.alphaNsamplesG = AbstractQL.logistic_transform(self.alphaNsamples)
        self.alphaPsamplesG = AbstractQL.logistic_transform(self.alphaPsamples)



        ### Define the range parameters using guassian(I am going to implement it) 





        #### Combine into a single array
        self.samples = np.column_stack((self.alphaPsamples , self.alphaNsamples ))



        ### Generate the simulate actions 
        Trials = 200
        self.Agents = np.empty(shape=(self.NumberSubjects,1),dtype=object)
        self.Trajectories = np.zeros((self.NumberSubjects,Trials,3)).astype(int)
        for i in np.arange(self.NumberSubjects).astype(int):
            # you need to take copy to sure that in each simulation the initial Q value are the same for all simulated subjects
            self.Agents[i,0] = QL1_RL(self.task_mdp,copy.deepcopy(StatesAgent),self.samples[i,0],self.samples[i,1])
            self.environment = MDPEnvironment(self.Agents[i,0], self.task_mdp)
            self.Agents[i],self.Trajectories[i] = self.environment.simulate_interaction(initial_state_idx=0,max_steps=Trials)
        
        DummyAgent = QL1_RL(self.task_mdp,copy.deepcopy(StatesAgent),0.1,0.1)
        EnvFit = MDPEnvironment(DummyAgent, self.task_mdp)
        self.EMFit = EMGuassian(EnvFit,2)

        self.behavioraldata  = self.Trajectories


    

    def testSimulated(self):
        i = np.random.randint(0,74,1)
        i = i[0]
        print(self.Agents[i,0].states[0].name)
        print(self.Agents[i,0].states[0].Q)
        print("#####################################3")
        print(self.Agents[i,0].states[1].name)
        print(self.Agents[i,0].states[1].Q)
        print("#####################################3")
        print(self.Agents[i,0].states[2].name)
        print(self.Agents[i,0].states[2].Q)



        


    def testModelfitting(self):

        ##### Apply Model Fitting
        EstimatedData = self.EMFit.fit(self.behavioraldata,tol=0.01,num_iterations=10)
        
        estimated_params = EstimatedData['subject_means']

        estimated_params = AbstractQL.inv_logistic_transform(estimated_params)
        
        real_params = self.samples 

        parameter_names=["alphaP", "alphaN"]
        ##### Plot the real and estimated data
        num_parameters = real_params.shape[1]
        fig, axes = plt.subplots(1, num_parameters, figsize=(5 * num_parameters, 5), sharey=False)

        # if parameter_names is None:
        #     parameter_names = [f"Parameter {i+1}" for i in range(num_parameters)]

        for i in range(num_parameters):
            ax = axes[i] if num_parameters > 1 else axes  # Adjust if only one subplot

            # Scatter plot for each parameter
            ax.scatter(real_params[:, i], estimated_params[:, i], color='blue', alpha=0.7)
            
            # Plot y=x line for reference
            min_val = min(np.min(real_params[:, i]), np.min(estimated_params[:, i]))
            max_val = max(np.max(real_params[:, i]), np.max(estimated_params[:, i]))
            ax.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', linewidth=1)

            # Label axes and add title for each subplot
            ax.set_xlabel("Real Values")
            ax.set_ylabel("Estimated Values")
            ax.set_title(parameter_names[i])
            ax.set_xlim(min_val, max_val)
            ax.set_ylim(min_val, max_val)

        plt.tight_layout()
        plt.show()



        ##### Calculate the correlation


if __name__ == "__main__":
    unittest.main()







