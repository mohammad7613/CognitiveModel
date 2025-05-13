# cognitive_models/tasks/test_go_nogo_task.py

import unittest
from cognitive_models import *
import numpy as np
from numpy.typing import NDArray

class TestTaskMDPPL(unittest.TestCase):

    def setUp(self):
        CONTEXTS,TRANSITIONS = parse_mdp_config("cognitive_models/tasks/mdp_pl_config.json")
        class StateTask(State):
            def __init__(self, name: str=None):
                super().__init__(name)

        StatesTask = np.empty((3,),dtype=object)
        for i,k in enumerate(list(CONTEXTS.keys())):
            StatesTask[i] = StateTask(k)           
        self.task_mdp = PL(StatesTask,CONTEXTS,TRANSITIONS)

    def testTranslator(self):
        A, B = self.task_mdp.translator(0,0)
        assert A == 'A_B'
        assert B == 'A'
        A, B = self.task_mdp.translator(1,0)
        assert A == 'C_D'
        assert B == 'C'

        A, B = self.task_mdp.translator(2,0)
        assert A == 'E_F'
        assert B == 'E'

        A, B = self.task_mdp.translator(0,1)
        assert A == 'A_B'
        assert B == 'B'

        A, B = self.task_mdp.translator(1,1)
        assert A == 'C_D'
        assert B == 'D'

        A, B = self.task_mdp.translator(2,1)
        assert A == 'E_F'
        assert B == 'F'
    def testGetReward(self):
        reward = self.task_mdp.get_reward(0,0)
        assert reward in [1,-1]
        reward = self.task_mdp.get_reward(1,0)
        assert reward in [1,-1]
        reward = self.task_mdp.get_reward(2,0)
        assert reward in [1,-1]
        reward = self.task_mdp.get_reward(0,1)
        assert reward in [1,-1]
        reward = self.task_mdp.get_reward(1,1)
        assert reward in [1,-1]
        reward = self.task_mdp.get_reward(2,1)
        assert reward in [1,-1]
    def testNextState(self):
        state = self.task_mdp.next_state(0,0)
        assert state in [0,1,2,3]
        state = self.task_mdp.next_state(1,0)
        assert state in [0,1,2,3]
        state = self.task_mdp.next_state(2,0)
        assert state in [0,1,2,3]
        state = self.task_mdp.next_state(0,1)
        assert state in [0,1,2,3]
        state = self.task_mdp.next_state(1,1)
        assert state in [0,1,2,3]
        state = self.task_mdp.next_state(2,1)
        assert state in [0,1,2,3]
    def testRewardProbability(self):
        assert 0.8 == self.task_mdp.reward_probability(0,0,1)
        assert 0.2 == self.task_mdp.reward_probability(0,0,-1)
        assert 0.2 == self.task_mdp.reward_probability(0,1,1)
        assert 0.8 == self.task_mdp.reward_probability(0,1,-1)
        assert 0.7 == self.task_mdp.reward_probability(1,0,1)
        assert 0.3 == self.task_mdp.reward_probability(1,0,-1)
        assert 0.3 == self.task_mdp.reward_probability(1,1,1)
        assert 0.7 == self.task_mdp.reward_probability(1,1,-1)
        assert 0.6 == self.task_mdp.reward_probability(2,0,1)
        assert 0.4 == self.task_mdp.reward_probability(2,0,-1)
        assert 0.4 == self.task_mdp.reward_probability(2,1,1)
        assert 0.6 == self.task_mdp.reward_probability(2,1,-1)
    def testTransitionProbability(self):
        assert 0.333 == self.task_mdp.transition_probability(0,0,0)
        assert 0.333 == self.task_mdp.transition_probability(0,0,1)
        assert 0.334 == self.task_mdp.transition_probability(0,0,2)
        assert 0.333 == self.task_mdp.transition_probability(0,1,0)
        assert 0.333 == self.task_mdp.transition_probability(0,1,1)
        assert 0.334 == self.task_mdp.transition_probability(0,1,2)


class TestTaskMDPGoNoGo(unittest.TestCase):

    def setUp(self):
        CONTEXTS,TRANSITIONS = parse_mdp_config("cognitive_models/tasks/mdp_no_go_no_config.json")
        print(CONTEXTS)
        print(TRANSITIONS)
        class StateTask(State):
            def __init__(self, name: str=None):
                super().__init__(name)

        StatesTask = np.empty((4,),dtype=object)
        for i,k in enumerate(list(CONTEXTS.keys())):
            StatesTask[i] = StateTask(k)           
        self.task_mdp = GoNoGo(StatesTask,CONTEXTS,TRANSITIONS)

    def testTranslator(self):
        A, B = self.task_mdp.translator(0,0)
        assert A == 'go_to_win'
        assert B == 'go'
        A, B = self.task_mdp.translator(1,0)
        assert A == 'no_go_to_win'
        assert B == 'go'

        A, B = self.task_mdp.translator(2,0)
        assert A == 'go_to_avoid_loss'
        assert B == 'go'

        A, B = self.task_mdp.translator(0,1)
        assert A == 'go_to_win'
        assert B == 'no-go'

        A, B = self.task_mdp.translator(1,1)
        assert A == 'no_go_to_win'
        assert B == 'no-go'

        A, B = self.task_mdp.translator(2,1)
        assert A == 'go_to_avoid_loss'
        assert B == 'no-go'
    def testGetReward(self):
        reward = self.task_mdp.get_reward(0,0)
        assert reward in [1,0]
        reward = self.task_mdp.get_reward(1,0)
        assert reward in [1,0]
        reward = self.task_mdp.get_reward(2,0)
        assert reward in [0,-1]
        reward = self.task_mdp.get_reward(3,0)
        assert reward in [0,-1]
        reward = self.task_mdp.get_reward(0,1)
        assert reward in [1,0]
        reward = self.task_mdp.get_reward(1,1)
        assert reward in [1,0]
        reward = self.task_mdp.get_reward(2,1)
        assert reward in [0,-1]
        reward = self.task_mdp.get_reward(3,1)
        assert reward in [0,-1]
    def testNextState(self):
        state = self.task_mdp.next_state(0,0)
        assert state in [0,1,2,3]
        state = self.task_mdp.next_state(1,0)
        assert state in [0,1,2,3]
        state = self.task_mdp.next_state(2,0)
        assert state in [0,1,2,3]
        state = self.task_mdp.next_state(0,1)
        assert state in [0,1,2,3]
        state = self.task_mdp.next_state(1,1)
        assert state in [0,1,2,3]
        state = self.task_mdp.next_state(2,1)
        assert state in [0,1,2,3]
    def testRewardProbability(self):
        assert 0.8 == self.task_mdp.reward_probability(0,0,1)
        assert 0.2 == self.task_mdp.reward_probability(0,0,0)
        assert 0.2 == self.task_mdp.reward_probability(0,1,1)
        assert 0.8 == self.task_mdp.reward_probability(0,1,0)
        assert 0.2 == self.task_mdp.reward_probability(1,0,1)
        assert 0.8 == self.task_mdp.reward_probability(1,0,0)
        assert 0.8 == self.task_mdp.reward_probability(1,1,1)
        assert 0.2 == self.task_mdp.reward_probability(1,1,0)
        assert 0.8 == self.task_mdp.reward_probability(2,0,0)
        assert 0.2 == self.task_mdp.reward_probability(2,0,-1)
        assert 0.2 == self.task_mdp.reward_probability(2,1,0)
        assert 0.8 == self.task_mdp.reward_probability(2,1,-1)
    def testTransitionProbability(self):
        assert 1/4 == self.task_mdp.transition_probability(0,0,0)
        assert 1/4 == self.task_mdp.transition_probability(0,0,1)
        assert 1/4 == self.task_mdp.transition_probability(0,0,2)
        assert 1/4 == self.task_mdp.transition_probability(0,0,3)
        assert 1/4 == self.task_mdp.transition_probability(0,1,0)
        assert 1/4 == self.task_mdp.transition_probability(0,1,1)
        assert 1/4 == self.task_mdp.transition_probability(0,1,2)
        assert 1/4 == self.task_mdp.transition_probability(0,0,0)
        assert 1/4 == self.task_mdp.transition_probability(0,0,1)
        assert 1/4 == self.task_mdp.transition_probability(0,0,2)
        assert 1/4 == self.task_mdp.transition_probability(0,1,0)
        assert 1/4 == self.task_mdp.transition_probability(0,1,1)
        assert 1/4 == self.task_mdp.transition_probability(0,1,2)
        

if __name__ == "__main__":
    unittest.main()
