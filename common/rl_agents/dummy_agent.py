
import os
import shutil
import mlflow
import json
from common.rl_agents.agent import Agent
class DummyAgent(Agent):

    def __init__(self, observation_space, number_of_actions, always_action):
        self.observation_space = observation_space
        self.number_of_actions = number_of_actions
        self.always_action = always_action
        

    def select_action(self, time_step, deploy=False):
        return self.always_action

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
            print("Loaded", self.always_action)
