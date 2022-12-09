"""This module manages the openai gym training."""
from common.utilities.project import Project
from common.utilities.training import train
import sys
from typing import Dict, Any
import gym
sys.path.insert(0, '../')


def run_openai_gym_training(command_line_arguments: Dict[str, Any]) -> int:
    """Run the openai gym training with the command line arguments.

    Args:
        command_line_arguments (Dict[str,Any]): Command Line Arguments.

    Returns:
        int: Experiment ID
    """
    command_line_arguments['task'] = 'openai_gym_training'
    command_line_arguments['eval_interval'] = 1
    command_line_arguments['max_steps'] = 20
    command_line_arguments['deploy'] = (1 == command_line_arguments['deploy'])
    env = gym.make(command_line_arguments['env'])
    m_project = Project(command_line_arguments)
    m_project.init_mlflow_bridge(
        command_line_arguments['project_name'],
        command_line_arguments['task'], command_line_arguments['parent_run_id'])
    m_project.load_saved_command_line_arguments()
    m_project.create_agent(command_line_arguments,
                           env.observation_space, env.action_space)
    train(m_project, env, prop_type='reward')
    experiment_id = m_project.mlflow_bridge.get_run_id()
    m_project.close()
    return experiment_id
