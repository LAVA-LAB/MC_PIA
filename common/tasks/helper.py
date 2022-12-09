"""This module provides helper functions for COOL-MC."""
import argparse
import sys
import random
from typing import Any, Dict
import numpy as np
import torch


def get_arguments() -> Dict[str, Any]:
    """Parses all the COOL-MC arguments

    Returns:
        Dict[str, Any]: dictionary with the command line arguments as key and their assignment as value
    """
    arg_parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    args = argparse.Namespace()
    arg_parser.add_argument('--task', help='What type of task do you want to perform(safe_training, openai_training, rl_model_checking)?', type=str,
                            default='safe_training')
    arg_parser.add_argument('--project_name', help='What is the name of your project?', type=str,
                            default='defaultproject')
    arg_parser.add_argument('--parent_run_id', help='Do you want to continue training of a RL agent? Name the run_id of the last training unit (see mlflow ui).', type=str,
                            default='')
    arg_parser.add_argument('--num_episodes', help='What is the number of training episodes?', type=int,
                            default=1000)
    arg_parser.add_argument('--eval_interval', help='What is the number of training episodes?', type=int,
                            default=9)
    arg_parser.add_argument('--sliding_window_size', help='What is the sliding window size for the reward averaging?', type=int,
                            default=100)
    arg_parser.add_argument('--rl_algorithm', help='What is the  used RL algorithm?', type=str,
                            default='dqn_agent')
    # OpenAI Gym
    arg_parser.add_argument('--env', help='In which environment do you want to train your RL agent?', type=str,
                            default='')
    # Model Checking
    arg_parser.add_argument('--prism_dir', help='In which folder should we save your projects?', type=str,
                            default='../prism_files')
    arg_parser.add_argument('--prism_file_path', help='In which folder should we save your projects?', type=str,
                            default='')
    arg_parser.add_argument('--constant_definitions', help='Constant definitions of the formal model (PRISM model)', type=str,
                            default='')
    arg_parser.add_argument('--prop', help='Property Specification', type=str,
                            default='   ')
    arg_parser.add_argument('--max_steps', help='Maximal steps in environment', type=int,
                            default=100)
    arg_parser.add_argument('--disabled_features', help='Features which should not be used by the RL agent: FEATURE1,FEATURE2,...', type=str,
                            default='')
    arg_parser.add_argument('--permissive_input', help='Constant definitions of the formal model (e.g. pos=[0,3];...)', type=str,
                            default='')
    arg_parser.add_argument('--abstract_features', help='', type=str,
                            default='')
    arg_parser.add_argument('--wrong_action_penalty', help='Wrong action penalty', type=int,
                            default=1000)
    arg_parser.add_argument('--reward_flag', help='Reward Flag (', type=int,
                            default=0)
    arg_parser.add_argument('--deploy', help='Deploy Flag (', type=int,
                            default=0)
    arg_parser.add_argument('--range_plotting', help='Range Plotting Flag', type=int,
                            default=1)
    # Dummy Agent
    arg_parser.add_argument('--always_action', help='DummyAgent-Parameter: Which action should the dummy agent choose?', type=int,
                            default=0)
    # Sarsamax
    arg_parser.add_argument('--alpha', help='Gamma', type=float,
                            default=0.99)
    # Hillclimbing
    arg_parser.add_argument('--noise_scale', help='Noise Scale for Hillclimbing', type=float,
                            default=1e-2)
    # DQN Agent
    arg_parser.add_argument('--layers', help='DummyAgent-Parameter: Which action should the dummy agent choose?', type=int,
                            default=2)
    arg_parser.add_argument('--neurons', help='DummyAgent-Parameter: Which action should the dummy agent choose?', type=int,
                            default=64)
    arg_parser.add_argument('--replay_buffer_size', help='DummyAgent-Parameter: Which action should the dummy agent choose?', type=int,
                            default=300000)
    arg_parser.add_argument('--epsilon', help='Epsilon Starting Rate', type=float,
                            default=1)
    arg_parser.add_argument('--epsilon_dec', help='Epsilon Decreasing Rate', type=float,
                            default=0.9999)
    arg_parser.add_argument('--epsilon_min', help='Minimal Epsilon Value', type=float,
                            default=0.1)
    arg_parser.add_argument('--gamma', help='Gamma', type=float,
                            default=0.99)
    arg_parser.add_argument('--replace', help='Replace Target Network Intervals', type=int,
                            default=304)
    arg_parser.add_argument('--lr', help='Learning Rate', type=float,
                            default=0.0001)
    arg_parser.add_argument('--batch_size', help='Batch Size', type=int,
                            default=32)
    arg_parser.add_argument('--seed', help='Random Seed for numpy, random, storm, pytorch', type=int,
                            default=-1)
    arg_parser.add_argument('--attack_config', help='Attack config in csv format', type=str, default="")

    args, _ = arg_parser.parse_known_args(sys.argv)
    return vars(args)


def parse_prop_type(prop: str) -> str:
    """Guides in Safe training, if we want to maximize or minimize the property result
    or do normal reward maximization.

    Args:
        prop (str): Property Query

    Returns:
        str: type of optimization
    """
    assert isinstance(prop, str)
    if prop.find("min") < prop.find("=") and prop.find("min") != -1:
        return "min_prop"
    if prop.find("max") < prop.find("=") and prop.find("max") != -1:
        return "max_prop"
    return "reward"


def set_random_seed(seed: int):
    """Set global seed to all used libraries. If you use other libraries too,
    add them here.

    Args:
        seed (int): Random Seed
    """
    assert isinstance(seed, int)
    if seed != -1:
        print("Set Seed to", seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
