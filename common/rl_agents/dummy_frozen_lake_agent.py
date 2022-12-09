
import os
import shutil
import mlflow
import json
from common.rl_agents.agent import Agent

class DummyFrozenLakeAgent(Agent):

    def __init__(self, observation_space, number_of_actions, always_action):
        self.observation_space = observation_space
        self.number_of_actions = number_of_actions
        self.always_action = always_action
        

    def select_action(self, state, deploy=False):
        if state.tolist() == [0, 0, 3]:
            return 0
        elif state.tolist() == [0, 0, 2]:
            return 0
        elif state.tolist() == [0, 0, 1]:
            return 2
        elif state.tolist() == [0, 1, 1]:
            return 2
        elif state.tolist() == [0, 2, 1]:
            return 0
        elif state.tolist() == [0, 2, 0]:
            return 2
        elif state.tolist() == [0, 3, 0]:
            return 0
        elif state.tolist() == [1, 3, 0]:
            return 0
        return 0

    def store_experience(self, state, action, reward, next_state, terminal):
        pass

    def step_learn(self):
        pass

    def episodic_learn(self):
        pass
    
    def get_hyperparameters(self):
        pass

    def save(self):
        os.mkdir('tmp_model')
        weights = {"always_action": self.always_action}
        with open("tmp_model/weights.json", 'w', encoding='utf-8') as f:
            json.dump(weights, f, indent=2)
        mlflow.log_artifacts("tmp_model", artifact_path="model")
        shutil.rmtree('tmp_model')

    def load(self, root_folder):
        with open(os.path.join(root_folder, 'weights.json')) as json_file:
            self.always_action = json.load(json_file)['always_action']