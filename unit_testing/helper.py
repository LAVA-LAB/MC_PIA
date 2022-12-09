import argparse
import sys
import argparse
import sys
sys.path.insert(0, '../')
from common.tasks.helper import *

def parse_prop_type(prop):
    if prop.find("min") < prop.find("=") and prop.find("min") != -1:
        return "min_prop"
    elif prop.find("max") < prop.find("=") and prop.find("max") != -1:
        return "max_prop"
    else:
        return "reward"


def get_frozen_lake_arguments():
    arg_parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    args = argparse.Namespace()
    unknown_args = list()
    arg_parser.add_argument('--project_dir', help='In which folder should we save your projects?', type=str,
                            default='projects')
    arg_parser.add_argument('--project_name', help='What is the name of your project?', type=str,
                            default='Frozen Lake')
    arg_parser.add_argument('--parent_run_id', help='Do you want to continue training of a RL agent? Name the run_id of the last training unit (see mlflow ui).', type=str,
                            default='')
    arg_parser.add_argument('--env', help='In which environment do you want to train your RL agent?', type=str,
                            default='')
    arg_parser.add_argument('--sliding_window_size', help='What is the sliding window size for the reward averaging?', type=int,
                            default=100)
    arg_parser.add_argument('--num_episodes', help='What is the number of training episodes?', type=int,
                            default=1000)
    arg_parser.add_argument('--rl_algorithm', help='What is the  used RL algorithm?', type=str,
                            default='dummy_agent')
    # Model Checking
    arg_parser.add_argument('--prism_dir', help='In which folder should we save your projects?', type=str,
                            default='../prism_files')
    arg_parser.add_argument('--prism_file_path', help='In which folder should we save your projects?', type=str,
                            default='frozen_lake_4x4.prism')
    arg_parser.add_argument('--constant_definitions', help='Constant definitions of the formal model (PRISM model)', type=str,
                            default='slippery=0.0')
    arg_parser.add_argument('--prop', help='Property Specification', type=str,
                            default='Pmin=? [F "water"]')
    arg_parser.add_argument('--prop_type', help='Maximal Reward (max_reward), Minimal Reward (min_reward), Property minimization (min_prop) or maximization (max_prop).', type=str,
                            default='min_reward')
    arg_parser.add_argument('--max_steps', help='Maximal steps in environment', type=int,
                            default=20)
    arg_parser.add_argument('--disabled_features', help='Features which should not be used by the RL agent: FEATURE1,FEATURE2,...', type=str,
                            default='')
    arg_parser.add_argument('--wrong_action_penalty', help='Wrong action penalty', type=int,
                            default=1000)
    arg_parser.add_argument('--reward_flag', help='Reward Flag', type=bool,
                            default=False)
    # Dummy Agent
    arg_parser.add_argument('--always_action', help='DummyAgent-Parameter: Which action should the dummy agent choose?', type=int,
                            default=0)
    # Double DQN Agent
    arg_parser.add_argument('--layers', help='DummyAgent-Parameter: Which action should the dummy agent choose?', type=int,
                            default=0)
    arg_parser.add_argument('--neurons', help='DummyAgent-Parameter: Which action should the dummy agent choose?', type=int,
                            default=0)
    arg_parser.add_argument('--replay_buffer_size', help='DummyAgent-Parameter: Which action should the dummy agent choose?', type=int,
                            default=300000)
    arg_parser.add_argument('--epsilon', help='Epsilon Starting Rate', type=float,
                            default=1)
    arg_parser.add_argument('--epsilon_dec', help='Epsilon Decreasing Rate', type=float,
                            default=0.9994)
    arg_parser.add_argument('--epsilon_min', help='Minimal Epsilon Value', type=float,
                            default=0.1)
    arg_parser.add_argument('--gamma', help='Gamma', type=float,
                            default=0.99)
    arg_parser.add_argument('--replace', help='Replace Target Network Intervals', type=int,
                            default=304)
    arg_parser.add_argument('--lr', help='Learning Rate', type=float,
                            default=0.004)
    arg_parser.add_argument('--batch_size', help='Batch Size', type=int,
                            default=32)
    arg_parser.add_argument('--seed', help='Random Seed for numpy, random, storm, pytorch', type=int,
                            default=-1)
    args, unknown_args = arg_parser.parse_known_args(sys.argv)
    return vars(args)





def get_frozen_lake_v1_arguments():
    arg_parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    args = argparse.Namespace()
    unknown_args = list()
    arg_parser.add_argument('--project_dir', help='In which folder should we save your projects?', type=str,
                            default='projects')
    arg_parser.add_argument('--project_name', help='What is the name of your project?', type=str,
                            default='Frozen Lake')
    arg_parser.add_argument('--parent_run_id', help='Do you want to continue training of a RL agent? Name the run_id of the last training unit (see mlflow ui).', type=str,
                            default='')
    arg_parser.add_argument('--env', help='In which environment do you want to train your RL agent?', type=str,
                            default='')
    arg_parser.add_argument('--sliding_window_size', help='What is the sliding window size for the reward averaging?', type=int,
                            default=100)
    arg_parser.add_argument('--num_episodes', help='What is the number of training episodes?', type=int,
                            default=1000)
    arg_parser.add_argument('--rl_algorithm', help='What is the  used RL algorithm?', type=str,
                            default='dummy_agent')
    # Model Checking
    arg_parser.add_argument('--prism_dir', help='In which folder should we save your projects?', type=str,
                            default='../prism_files')
    arg_parser.add_argument('--prism_file_path', help='In which folder should we save your projects?', type=str,
                            default='frozen_lake-v1.prism')
    arg_parser.add_argument('--constant_definitions', help='Constant definitions of the formal model (PRISM model)', type=str,
                            default='slippery=0.0')
    arg_parser.add_argument('--prop', help='Property Specification', type=str,
                            default='Pmin=? [F "water"]')
    arg_parser.add_argument('--prop_type', help='Maximal Reward (max_reward), Minimal Reward (min_reward), Property minimization (min_prop) or maximization (max_prop).', type=str,
                            default='min_reward')
    arg_parser.add_argument('--max_steps', help='Maximal steps in environment', type=int,
                            default=20)
    arg_parser.add_argument('--disabled_features', help='Features which should not be used by the RL agent: FEATURE1,FEATURE2,...', type=str,
                            default='')
    arg_parser.add_argument('--wrong_action_penalty', help='Wrong action penalty', type=int,
                            default=1000)
    arg_parser.add_argument('--reward_flag', help='Reward Flag', type=bool,
                            default=False)
    # Dummy Agent
    arg_parser.add_argument('--always_action', help='DummyAgent-Parameter: Which action should the dummy agent choose?', type=int,
                            default=0)
    # Double DQN Agent
    arg_parser.add_argument('--layers', help='DummyAgent-Parameter: Which action should the dummy agent choose?', type=int,
                            default=0)
    arg_parser.add_argument('--neurons', help='DummyAgent-Parameter: Which action should the dummy agent choose?', type=int,
                            default=0)
    arg_parser.add_argument('--replay_buffer_size', help='DummyAgent-Parameter: Which action should the dummy agent choose?', type=int,
                            default=300000)
    arg_parser.add_argument('--epsilon', help='Epsilon Starting Rate', type=float,
                            default=1)
    arg_parser.add_argument('--epsilon_dec', help='Epsilon Decreasing Rate', type=float,
                            default=0.9994)
    arg_parser.add_argument('--epsilon_min', help='Minimal Epsilon Value', type=float,
                            default=0.1)
    arg_parser.add_argument('--gamma', help='Gamma', type=float,
                            default=0.99)
    arg_parser.add_argument('--replace', help='Replace Target Network Intervals', type=int,
                            default=304)
    arg_parser.add_argument('--lr', help='Learning Rate', type=float,
                            default=0.004)
    arg_parser.add_argument('--batch_size', help='Batch Size', type=int,
                            default=32)
    arg_parser.add_argument('--seed', help='Random Seed for numpy, random, storm, pytorch', type=int,
                            default=-1)
    args, unknown_args = arg_parser.parse_known_args(sys.argv)
    return vars(args)


def get_transporter_with_fuel_arguments():
    arg_parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    args = argparse.Namespace()
    unknown_args = list()
    arg_parser.add_argument('--project_dir', help='In which folder should we save your projects?', type=str,
                            default='projects')
    arg_parser.add_argument('--project_name', help='What is the name of your project?', type=str,
                            default='Frozen Lake')
    arg_parser.add_argument('--parent_run_id', help='Do you want to continue training of a RL agent? Name the run_id of the last training unit (see mlflow ui).', type=str,
                            default='')
    arg_parser.add_argument('--env', help='In which environment do you want to train your RL agent?', type=str,
                            default='')
    arg_parser.add_argument('--sliding_window_size', help='What is the sliding window size for the reward averaging?', type=int,
                            default=100)
    arg_parser.add_argument('--num_episodes', help='What is the number of training episodes?', type=int,
                            default=1000)
    arg_parser.add_argument('--rl_algorithm', help='What is the  used RL algorithm?', type=str,
                            default='dummy_agent')
    # Model Checking
    arg_parser.add_argument('--prism_dir', help='In which folder should we save your projects?', type=str,
                            default='../prism_files')
    arg_parser.add_argument('--prism_file_path', help='In which folder should we save your projects?', type=str,
                            default='transporter.prism')
    arg_parser.add_argument('--constant_definitions', help='Constant definitions of the formal model (PRISM model)', type=str,
                            default='MAX_JOBS=2,MAX_FUEL=21')
    arg_parser.add_argument('--prop', help='Property Specification', type=str,
                            default='Pmin=? [F jobs_done=2]')
    arg_parser.add_argument('--max_steps', help='Maximal steps in environment', type=int,
                            default=100)
    arg_parser.add_argument('--disabled_features', help='Features which should not be used by the RL agent: FEATURE1,FEATURE2,...', type=str,
                            default='')
    arg_parser.add_argument('--wrong_action_penalty', help='Wrong action penalty', type=int,
                            default=100)
    # Dummy Agent
    arg_parser.add_argument('--always_action', help='DummyAgent-Parameter: Which action should the dummy agent choose?', type=int,
                            default=0)
    # Double DQN Agent
    arg_parser.add_argument('--layers', help='DummyAgent-Parameter: Which action should the dummy agent choose?', type=int,
                            default=0)
    arg_parser.add_argument('--neurons', help='DummyAgent-Parameter: Which action should the dummy agent choose?', type=int,
                            default=0)
    arg_parser.add_argument('--replay_buffer_size', help='DummyAgent-Parameter: Which action should the dummy agent choose?', type=int,
                            default=300000)
    arg_parser.add_argument('--epsilon', help='Epsilon Starting Rate', type=float,
                            default=1)
    arg_parser.add_argument('--epsilon_dec', help='Epsilon Decreasing Rate', type=float,
                            default=0.9994)
    arg_parser.add_argument('--epsilon_min', help='Minimal Epsilon Value', type=float,
                            default=0.1)
    arg_parser.add_argument('--gamma', help='Gamma', type=float,
                            default=0.99)
    arg_parser.add_argument('--replace', help='Replace Target Network Intervals', type=int,
                            default=304)
    arg_parser.add_argument('--lr', help='Learning Rate', type=float,
                            default=0.004)
    arg_parser.add_argument('--batch_size', help='Batch Size', type=int,
                            default=32)
    arg_parser.add_argument('--seed', help='Random Seed for numpy, random, storm, pytorch', type=int,
                            default=-1)
    args, unknown_args = arg_parser.parse_known_args(sys.argv)
    return vars(args)
