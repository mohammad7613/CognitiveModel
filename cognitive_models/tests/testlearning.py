from cognitive_models import *
import unittest
import numpy as np
from numpy.typing import NDArray



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
            
              

        self.agent = QL1_RL(self.task_mdp,StatesAgent,0.1,0.3)
        self.environment = MDPEnvironment(self.agent, self.task_mdp)
    def testLearning(self):
        self.environment.simulate_interaction(initial_state_idx=0,max_steps=100)
        print(self.agent.states[0].name)
        print(self.agent.states[0].Q)
        print("#####################################3")
        print(self.agent.states[1].name)
        print(self.agent.states[1].Q)
        print("#####################################3")
        print(self.agent.states[2].name)
        print(self.agent.states[2].Q)
        # self.assertTrue(np.allclose(self.agent.Q, [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], atol=1e-8, rtol=1e-5))


if __name__ == "__main__":
    unittest.main()







