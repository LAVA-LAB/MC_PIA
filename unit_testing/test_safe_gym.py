import pytest

from helper import *
import os
sys.path.insert(0, '../')
from common.utilities.project import Project
from common.utilities.training import train
from common.safe_gym.safe_gym import SafeGym


@pytest.fixture(scope='session')
def dummy_agent_safe_frozen_lake():
    command_line_arguments = get_frozen_lake_arguments()
    command_line_arguments['task'] = 'safe_training'
    command_line_arguments['deploy'] = 0
    prism_file_path = os.path.join(command_line_arguments['prism_dir'], command_line_arguments['prism_file_path'])
    env = SafeGym(prism_file_path,command_line_arguments['constant_definitions'], command_line_arguments['max_steps'], command_line_arguments['wrong_action_penalty'], command_line_arguments['reward_flag'], command_line_arguments['seed'], "", command_line_arguments['disabled_features'],attack_config="")
    m_project = Project(command_line_arguments)
    m_project.init_mlflow_bridge(command_line_arguments['project_name'],command_line_arguments['task'],command_line_arguments['parent_run_id'])
    m_project.load_saved_command_line_arguments()
    m_project.create_agent(command_line_arguments, env.observation_space, env.action_space)
    return m_project, env

@pytest.fixture(scope='session')
def dummy_agent_safe_frozen_lake_dqn_agent():
    command_line_arguments = get_frozen_lake_arguments()
    command_line_arguments['task'] = 'safe_training'
    command_line_arguments['deploy'] = 0
    command_line_arguments['rl_algorithm'] = 'dqn_agent'
    prism_file_path = os.path.join(command_line_arguments['prism_dir'], command_line_arguments['prism_file_path'])
    env = SafeGym(prism_file_path,command_line_arguments['constant_definitions'], command_line_arguments['max_steps'], command_line_arguments['wrong_action_penalty'], command_line_arguments['reward_flag'], command_line_arguments['seed'], "", command_line_arguments['disabled_features'],attack_config="")
    m_project = Project(command_line_arguments)
    m_project.init_mlflow_bridge(command_line_arguments['project_name'],command_line_arguments['task'],command_line_arguments['parent_run_id'])
    m_project.load_saved_command_line_arguments()
    m_project.create_agent(command_line_arguments, env.observation_space, env.action_space)
    return m_project, env
    

@pytest.fixture(scope='session')
def dummy_agent_safe_frozen_lake_dummy_frozen_lake_agent():
    command_line_arguments = get_frozen_lake_arguments()
    command_line_arguments['task'] = 'safe_training'
    command_line_arguments['deploy'] = 0
    command_line_arguments['rl_algorithm'] = 'dummy_frozen_lake_agent'
    prism_file_path = os.path.join(command_line_arguments['prism_dir'], command_line_arguments['prism_file_path'])
    env = SafeGym(prism_file_path,command_line_arguments['constant_definitions'], command_line_arguments['max_steps'], command_line_arguments['wrong_action_penalty'], command_line_arguments['reward_flag'], command_line_arguments['seed'], "", command_line_arguments['disabled_features'],attack_config="")
    m_project = Project(command_line_arguments)
    m_project.init_mlflow_bridge(command_line_arguments['project_name'],command_line_arguments['task'],command_line_arguments['parent_run_id'])
    m_project.load_saved_command_line_arguments()
    m_project.create_agent(command_line_arguments, env.observation_space, env.action_space)
    return m_project, env

def test_frozen_lake_3_steps_fall_in_water_but_penalty_state_after_4(dummy_agent_safe_frozen_lake):
    m_project, env = dummy_agent_safe_frozen_lake
    m_project.command_line_arguments['num_episodes'] = 1
    m_project.command_line_arguments['eval_interval'] = 1
    m_project.command_line_arguments['always_action'] = 0
    env.max_steps = 3
    all_states, all_actions, all_rewards, all_terminals = train(m_project, env)
    m_project.mlflow_bridge.close()
    print(all_states)
    print(all_rewards)
    assert sum(all_rewards) == -300

def test_frozen_lake_Falling_into_water(dummy_agent_safe_frozen_lake):
    m_project, env = dummy_agent_safe_frozen_lake
    m_project.command_line_arguments['num_episodes'] = 1
    m_project.command_line_arguments['eval_interval'] = 1
    m_project.command_line_arguments['always_action'] = 0
    env.max_steps = 4
    all_states, all_actions, all_rewards, all_terminals = train(m_project, env)
    m_project.mlflow_bridge.close()
    print(all_states)
    assert sum(all_rewards) == -1100
    assert len(all_states) == 5 # 4 states + 1 terminal state


def test_frozen_lake_dqn_agent_different_actions(dummy_agent_safe_frozen_lake_dqn_agent):
    m_project, env = dummy_agent_safe_frozen_lake_dqn_agent
    m_project.command_line_arguments['num_episodes'] = 1
    m_project.command_line_arguments['eval_interval'] = 1
    env.max_steps = 1000
    all_states, all_actions, all_rewards, all_terminals = train(m_project, env)
    m_project.mlflow_bridge.close()
    assert len(list(set(all_actions))) > 1

def test_dummy_agent_safe_frozen_lake_dummy_frozen_lake_agent(dummy_agent_safe_frozen_lake_dummy_frozen_lake_agent):
    m_project, env = dummy_agent_safe_frozen_lake_dummy_frozen_lake_agent
    m_project.command_line_arguments['num_episodes'] = 1
    m_project.command_line_arguments['eval_interval'] = 1
    env.max_steps = 1000
    all_states, all_actions, all_rewards, all_terminals = train(m_project, env)
    m_project.mlflow_bridge.close()
    assert sum(all_rewards) == -600
    assert True in all_terminals

