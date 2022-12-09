"""This module manages the safe gym training task."""
import os
from typing import Any, Dict
from common.utilities.project import Project
from common.utilities.training import train
from common.safe_gym.safe_gym import SafeGym
from common.tasks.helper import *


def run_safe_gym_training(command_line_arguments: Dict[str, Any]) -> int:
    """Runs the safe gym training with the passed command line arguments

    Args:
        command_line_arguments (Dict[str, Any]): Command Line Arguments.

    Returns:
        int: Experiment ID
    """
    command_line_arguments['task'] = 'safe_training'
    command_line_arguments['prop_type'] = parse_prop_type(
        command_line_arguments['prop'])
    command_line_arguments['reward_flag'] = command_line_arguments['reward_flag'] == 1
    command_line_arguments['deploy'] = (1 == command_line_arguments['deploy'])
    prism_file_path = os.path.join(
        command_line_arguments['prism_dir'], command_line_arguments['prism_file_path'])
    # Environment
    env = SafeGym(prism_file_path, command_line_arguments['constant_definitions'], 
                command_line_arguments['max_steps'], command_line_arguments['wrong_action_penalty'],
                  command_line_arguments['reward_flag'], 
                  command_line_arguments['seed'], command_line_arguments['permissive_input'],
                  command_line_arguments['disabled_features'], attack_config=command_line_arguments['attack_config'])
    # Project
    m_project = Project(command_line_arguments)
    m_project.init_mlflow_bridge(
        command_line_arguments['project_name'], command_line_arguments['task'],
        command_line_arguments['parent_run_id'])
    m_project.load_saved_command_line_arguments()
    print(m_project.command_line_arguments)
    m_project.create_agent(command_line_arguments,
                           env.observation_space, env.action_space)
    m_project.mlflow_bridge.set_property_query_as_run_name(
        command_line_arguments['prop'] + " for " + command_line_arguments['constant_definitions'])
    train(m_project, env, prop_type=command_line_arguments['prop_type'])
    run_id = m_project.mlflow_bridge.get_run_id()
    m_project.close()
    return run_id
