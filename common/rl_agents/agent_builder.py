import os
from common.rl_agents.dummy_agent import DummyAgent
from common.rl_agents.dummy_frozen_lake_agent import DummyFrozenLakeAgent
from common.rl_agents.sarsa_max_agent import SarsaMaxAgent
from common.rl_agents.deep_q_agent import DQNAgent
from common.rl_agents.hillclimbing_agent import HillClimbingAgent
from common.rl_agents.reinforce_agent import ReinforceAgent
'''
HOW TO ADD MORE AGENTS?
1) Create a new AGENTNAME.py with an AGENTNAME class
2) Inherit the agent-class
3) Override the methods
4) Import this py-script into this script
5) Add additional agent hyperparameters to the argparser
6) Add to build_agent the building procedure of your agent
'''
class AgentBuilder():

    @staticmethod
    def layers_neurons_to_number_of_neurons(layers, neurons):
        number_of_neurons = []
        for i in range(layers):
            number_of_neurons.append(neurons)
        return number_of_neurons

    @staticmethod
    def build_agent(model_root_folder_path, command_line_arguments, observation_space, action_space):
        #print('Build model with', model_root_folder_path, command_line_arguments)
        #print('Environment', observation_space.shape, action_space.n)
        try:
            state_dimension = observation_space.shape[0]
        except:
            state_dimension = 1
        agent = None
        if command_line_arguments['rl_algorithm'] == 'dummy_agent':
            #print("Build Dummy Agent.")
            agent = DummyAgent(state_dimension, action_space.n, command_line_arguments['always_action'])
            if model_root_folder_path!= None:
                agent.load(model_root_folder_path)
        elif command_line_arguments['rl_algorithm'] == 'dummy_frozen_lake_agent':
            #print("Build Dummy Frozen Lake Agent.")
            agent = DummyFrozenLakeAgent(state_dimension, action_space.n, command_line_arguments['always_action'])
            if model_root_folder_path!= None:
                agent.load(model_root_folder_path)
        elif command_line_arguments['rl_algorithm'] == 'sarsamax':
            #print("Build SARSAMAX Agent.")
            agent = SarsaMaxAgent(action_space.n, epsilon=command_line_arguments['epsilon'], epsilon_dec=command_line_arguments['epsilon_dec'], epsilon_min=command_line_arguments['epsilon_min'], alpha=command_line_arguments['alpha'], gamma=command_line_arguments['gamma'])
            if model_root_folder_path!= None:
                agent.load(model_root_folder_path)
        elif command_line_arguments['rl_algorithm'] == 'hillclimbing':
            #print("Build SARSAMAX Agent.")
            agent = HillClimbingAgent(state_dimension, action_space.n, gamma=command_line_arguments['gamma'], noise_scale= command_line_arguments['noise_scale'])
            if model_root_folder_path!= None:
                agent.load(model_root_folder_path)
        elif command_line_arguments['rl_algorithm'] == 'reinforce':
            #print("Build SARSAMAX Agent.")
            agent = ReinforceAgent(state_dimension=state_dimension, number_of_actions=action_space.n, gamma=command_line_arguments['gamma'], hidden_layer_size= command_line_arguments['neurons'],lr=command_line_arguments['lr'])
            if model_root_folder_path!= None:
                agent.load(model_root_folder_path)
        elif command_line_arguments['rl_algorithm'] == 'dqn_agent':
            #print("Build DQN Agent.", state_dimension, action_space.n)
            number_of_neurons = AgentBuilder.layers_neurons_to_number_of_neurons(command_line_arguments['layers'],command_line_arguments['neurons'])
            agent = DQNAgent(state_dimension, number_of_neurons, action_space.n, epsilon=command_line_arguments['epsilon'], epsilon_dec=command_line_arguments['epsilon_dec'], epsilon_min=command_line_arguments['epsilon_min'], gamma=command_line_arguments['gamma'], learning_rate=command_line_arguments['lr'], replace=command_line_arguments['replace'], batch_size=command_line_arguments['batch_size'], replay_buffer_size=command_line_arguments['replay_buffer_size'])
            #print(model_root_folder_path, model_root_folder_path.find("mlruns"))
            #import os
            #first_part_of_path = str(os.getcwd()).replace("safe_gym_training","")
            #second_part = model_root_folder_path[model_root_folder_path.find("mlruns"):]
            #model_root_folder_path = os.path.join(first_part_of_path, second_part)
            #print(model_root_folder_path)
            agent.load(model_root_folder_path)
        return agent