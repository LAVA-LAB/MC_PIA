import pytest
from helper import *
import os
import argparse
import sys
sys.path.insert(0, '../')
from common.utilities.project import Project
from common.utilities.training import train
from common.safe_gym.safe_gym import SafeGym
from common.utilities.front_end_printer import *
from common.tasks.safe_gym_training import *
from common.tasks.openai_gym_training import *
from common.tasks.verify_rl_agent import *
from common.tasks.helper import *

@pytest.mark.skip(reason="no way of currently testing this")
def test_frozen_lake_agent_a():
    arg_parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    args = argparse.Namespace()
    unknown_args = list()
    arg_parser.add_argument('--task', help='What is the name of your project?', type=str,
                            default='openai_training')
    arg_parser.add_argument('--project_name', help='What is the name of your project?', type=str,
                            default='experiments')
    arg_parser.add_argument('--parent_run_id', help='Do you want to continue training of a RL agent? Name the run_id of the last training unit (see mlflow ui).', type=str,
                            default='')
    arg_parser.add_argument('--sliding_window_size', help='What is the sliding window size for the reward averaging?', type=int,
                            default=100)
    arg_parser.add_argument('--num_episodes', help='What is the number of training episodes?', type=int,
                            default=100000)
    arg_parser.add_argument('--eval_interval', help='What is the number of training episodes?', type=int,
                            default=10)
    arg_parser.add_argument('--rl_algorithm', help='What is the  used RL algorithm?', type=str,
                            default='dqn_agent')
    # OpenAI Gym
    arg_parser.add_argument('--env', help='In which environment do you want to train your RL agent?', type=str,
                            default='FrozenLake-v0')
    # Model Checking
    arg_parser.add_argument('--prism_dir', help='In which folder should we save your projects?', type=str,
                            default='../prism_files')
    arg_parser.add_argument('--prism_file_path', help='In which folder should we save your projects?', type=str,
                            default='')
    arg_parser.add_argument('--constant_definitions', help='Constant definitions of the formal model (PRISM model)', type=str,
                            default='')
    arg_parser.add_argument('--prop', help='Property Specification', type=str,
                            default='')
    arg_parser.add_argument('--max_steps', help='Maximal steps in environment', type=int,
                            default=20)
    arg_parser.add_argument('--disabled_features', help='Features which should not be used by the RL agent: FEATURE1,FEATURE2,...', type=str,
                            default='')
    arg_parser.add_argument('--permissive_input', help='Constant definitions of the formal model (e.g. pos=[0,3];...)', type=str,
                            default='')
    arg_parser.add_argument('--abstract_features', help='', type=str,
                            default='')
    arg_parser.add_argument('--wrong_action_penalty', help='Wrong action penalty', type=int,
                            default=1000)
    arg_parser.add_argument('--reward_flag', help='Reward Flag (', type=int,
                            default=1)
    arg_parser.add_argument('--deploy', help='Deploy Flag (', type=int,
                            default=0)        
    # Double DQN Agent
    arg_parser.add_argument('--layers', help='DummyAgent-Parameter: Which action should the dummy agent choose?', type=int,
                            default=4)
    arg_parser.add_argument('--neurons', help='DummyAgent-Parameter: Which action should the dummy agent choose?', type=int,
                            default=128)
    arg_parser.add_argument('--replay_buffer_size', help='DummyAgent-Parameter: Which action should the dummy agent choose?', type=int,
                            default=100000)
    arg_parser.add_argument('--epsilon', help='Epsilon Starting Rate', type=float,
                            default=1)
    arg_parser.add_argument('--epsilon_dec', help='Epsilon Decreasing Rate', type=float,
                            default=0.99999)
    arg_parser.add_argument('--epsilon_min', help='Minimal Epsilon Value', type=float,
                            default=0.1)
    arg_parser.add_argument('--gamma', help='Gamma', type=float,
                            default=0.99)
    arg_parser.add_argument('--replace', help='Replace Target Network Intervals', type=int,
                            default=100)
    arg_parser.add_argument('--lr', help='Learning Rate', type=float,
                            default=0.001)
    arg_parser.add_argument('--batch_size', help='Batch Size', type=int,
                            default=128)
    arg_parser.add_argument('--seed', help='Batch Size', type=int,
                            default=128)
    arg_parser.add_argument('--attack_config', help='Attack config in csv format', type=str, default="")

    args, unknown_args = arg_parser.parse_known_args(sys.argv)
    command_line_arguments = vars(args)
    set_random_seed(command_line_arguments['seed'])
    parent_run_id = run_openai_gym_training(command_line_arguments)
    # Verify
    command_line_arguments['parent_run_id'] = parent_run_id
    command_line_arguments['task'] = 'rl_model_checking'
    command_line_arguments['prism_dir'] = '../prism_files'
    command_line_arguments['prism_file_path'] = 'frozen_lake3-v1.prism'
    command_line_arguments['constant_definitions'] = 'start_position=0,control=0.333'
    command_line_arguments['prop'] = 'P=? [F AT_FRISBEE=true]'
    result = run_verify_rl_agent(command_line_arguments)
    # Verify
    command_line_arguments['parent_run_id'] = parent_run_id
    command_line_arguments['task'] = 'rl_model_checking'
    command_line_arguments['range_plotting'] = 0
    command_line_arguments['prism_dir'] = '../prism_files'
    command_line_arguments['prism_file_path'] = 'frozen_lake3-v1.prism'
    command_line_arguments['constant_definitions'] = 'start_position=[0;1;16],control=0.333'
    command_line_arguments['prop'] = 'P=? [F AT_FRISBEE=true]'
    result = run_verify_rl_agent(command_line_arguments)
    # Verify
    command_line_arguments['parent_run_id'] = parent_run_id
    command_line_arguments['task'] = 'rl_model_checking'
    command_line_arguments['prism_dir'] = '../prism_files'
    command_line_arguments['prism_file_path'] = 'frozen_lake3-v1.prism'
    command_line_arguments['constant_definitions'] = 'start_position=[0;1;16],control=0.333'
    command_line_arguments['prop'] = 'P=? [F WATER=true]'
    result = run_verify_rl_agent(command_line_arguments)
    assert 1 == 1

@pytest.mark.skip(reason="no way of currently testing this")
def test_taxi_experiment_section_agent_b():
    import time
    start_time = time.time()
    arg_parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    args = argparse.Namespace()
    unknown_args = list()
    arg_parser.add_argument('--task', help='What is the name of your project?', type=str,
                            default='safe_training')
    arg_parser.add_argument('--project_name', help='What is the name of your project?', type=str,
                            default='experiments')
    arg_parser.add_argument('--parent_run_id', help='Do you want to continue training of a RL agent? Name the run_id of the last training unit (see mlflow ui).', type=str,
                            default='')
    arg_parser.add_argument('--sliding_window_size', help='What is the sliding window size for the reward averaging?', type=int,
                            default=100)
    arg_parser.add_argument('--num_episodes', help='What is the number of training episodes?', type=int,
                            default=37000)
    arg_parser.add_argument('--eval_interval', help='What is the number of training episodes?', type=int,
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
                            default='transporter.prism')
    arg_parser.add_argument('--constant_definitions', help='Constant definitions of the formal model (PRISM model)', type=str,
                            default='MAX_FUEL=10,MAX_JOBS=2')
    arg_parser.add_argument('--prop', help='Property Specification', type=str,
                            default='Pmax=? [F jobs_done=2]')
    arg_parser.add_argument('--max_steps', help='Maximal steps in environment', type=int,
                            default=30)
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
                             
    # Double DQN Agent
    arg_parser.add_argument('--layers', help='DummyAgent-Parameter: Which action should the dummy agent choose?', type=int,
                            default=4)
    arg_parser.add_argument('--neurons', help='DummyAgent-Parameter: Which action should the dummy agent choose?', type=int,
                            default=512)
    arg_parser.add_argument('--replay_buffer_size', help='DummyAgent-Parameter: Which action should the dummy agent choose?', type=int,
                            default=100000)
    arg_parser.add_argument('--epsilon', help='Epsilon Starting Rate', type=float,
                            default=1)
    arg_parser.add_argument('--epsilon_dec', help='Epsilon Decreasing Rate', type=float,
                            default=0.99999)
    arg_parser.add_argument('--epsilon_min', help='Minimal Epsilon Value', type=float,
                            default=0.1)
    arg_parser.add_argument('--gamma', help='Gamma', type=float,
                            default=0.99)
    arg_parser.add_argument('--replace', help='Replace Target Network Intervals', type=int,
                            default=100)
    arg_parser.add_argument('--lr', help='Learning Rate', type=float,
                            default=0.0001)
    arg_parser.add_argument('--batch_size', help='Batch Size', type=int,
                            default=100)
    arg_parser.add_argument('--seed', help='Batch Size', type=int,
                            default=128)
    arg_parser.add_argument('--attack_config', help='Attack config in csv format', type=str, default="")
    args, unknown_args = arg_parser.parse_known_args(sys.argv)
    command_line_arguments = vars(args)
    set_random_seed(command_line_arguments['seed'])
    parent_run_id = run_safe_gym_training(command_line_arguments)
    # Verify
    command_line_arguments['parent_run_id'] = parent_run_id
    command_line_arguments['task'] = 'rl_model_checking'
    command_line_arguments['prop'] = 'P=? [F jobs_done=2]'
    result = run_verify_rl_agent(command_line_arguments)
    end_time = time.time()
    print("Total Time", end_time-start_time)
    assert 1 == 1

@pytest.mark.skip(reason="no way of currently testing this")
def test_taxi_experiment_section_agent_c():
    arg_parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    args = argparse.Namespace()
    unknown_args = list()
    arg_parser.add_argument('--task', help='What is the name of your project?', type=str,
                            default='safe_training')
    arg_parser.add_argument('--project_name', help='What is the name of your project?', type=str,
                            default='experiments')
    arg_parser.add_argument('--parent_run_id', help='Do you want to continue training of a RL agent? Name the run_id of the last training unit (see mlflow ui).', type=str,
                            default='')
    arg_parser.add_argument('--sliding_window_size', help='What is the sliding window size for the reward averaging?', type=int,
                            default=100)
    arg_parser.add_argument('--num_episodes', help='What is the number of training episodes?', type=int,
                            default=60000)
    arg_parser.add_argument('--eval_interval', help='What is the number of training episodes?', type=int,
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
                            default='transporter.prism')
    arg_parser.add_argument('--constant_definitions', help='Constant definitions of the formal model (PRISM model)', type=str,
                            default='MAX_FUEL=10,MAX_JOBS=2')
    arg_parser.add_argument('--prop', help='Property Specification', type=str,
                            default='Pmax=? [F jobs_done=2]')
    arg_parser.add_argument('--max_steps', help='Maximal steps in environment', type=int,
                            default=30)
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
                             

    arg_parser.add_argument('--layers', help='DummyAgent-Parameter: Which action should the dummy agent choose?', type=int,
                            default=4)
    arg_parser.add_argument('--neurons', help='DummyAgent-Parameter: Which action should the dummy agent choose?', type=int,
                            default=512)
    arg_parser.add_argument('--replay_buffer_size', help='DummyAgent-Parameter: Which action should the dummy agent choose?', type=int,
                            default=100000)
    arg_parser.add_argument('--epsilon', help='Epsilon Starting Rate', type=float,
                            default=1)
    arg_parser.add_argument('--epsilon_dec', help='Epsilon Decreasing Rate', type=float,
                            default=0.99999)
    arg_parser.add_argument('--epsilon_min', help='Minimal Epsilon Value', type=float,
                            default=0.1)
    arg_parser.add_argument('--gamma', help='Gamma', type=float,
                            default=0.99)
    arg_parser.add_argument('--replace', help='Replace Target Network Intervals', type=int,
                            default=100)
    arg_parser.add_argument('--lr', help='Learning Rate', type=float,
                            default=0.0001)
    arg_parser.add_argument('--batch_size', help='Batch Size', type=int,
                            default=100)
    arg_parser.add_argument('--seed', help='Batch Size', type=int,
                            default=128)
    arg_parser.add_argument('--attack_config', help='Attack config in csv format', type=str, default="")
    args, unknown_args = arg_parser.parse_known_args(sys.argv)
    command_line_arguments = vars(args)
    set_random_seed(command_line_arguments['seed'])
    parent_run_id = run_safe_gym_training(command_line_arguments)
    # Verify
    command_line_arguments['parent_run_id'] = parent_run_id
    command_line_arguments['task'] = 'rl_model_checking'
    command_line_arguments['prop'] = 'P=? [F jobs_done=2]'
    result = run_verify_rl_agent(command_line_arguments)
    assert 1 == 1

@pytest.mark.skip(reason="no way of currently testing this")
def test_collision_avoidance_experiment_agent_d():
    import time
    start_time = time.time()
    arg_parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    args = argparse.Namespace()
    unknown_args = list()
    arg_parser.add_argument('--task', help='What is the name of your project?', type=str,
                            default='safe_training')
    arg_parser.add_argument('--project_name', help='What is the name of your project?', type=str,
                            default='experiments')
    arg_parser.add_argument('--parent_run_id', help='Do you want to continue training of a RL agent? Name the run_id of the last training unit (see mlflow ui).', type=str,
                            default='')
    arg_parser.add_argument('--sliding_window_size', help='What is the sliding window size for the reward averaging?', type=int,
                            default=100)
    arg_parser.add_argument('--num_episodes', help='What is the number of training episodes?', type=int,
                            default=49700)
    arg_parser.add_argument('--eval_interval', help='What is the number of training episodes?', type=int,
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
                            default='avoid.prism')
    arg_parser.add_argument('--constant_definitions', help='Constant definitions of the formal model (PRISM model)', type=str,
                            default='xMax=4,yMax=4,slickness=0')
    arg_parser.add_argument('--prop', help='Property Specification', type=str,
                            default='')
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
                            default=1)
    arg_parser.add_argument('--deploy', help='Deploy Flag (', type=int,
                            default=0)       
                             
    # Double DQN Agent
    arg_parser.add_argument('--layers', help='DummyAgent-Parameter: Which action should the dummy agent choose?', type=int,
                            default=4)
    arg_parser.add_argument('--neurons', help='DummyAgent-Parameter: Which action should the dummy agent choose?', type=int,
                            default=512)
    arg_parser.add_argument('--replay_buffer_size', help='DummyAgent-Parameter: Which action should the dummy agent choose?', type=int,
                            default=100000)
    arg_parser.add_argument('--epsilon', help='Epsilon Starting Rate', type=float,
                            default=1)
    arg_parser.add_argument('--epsilon_dec', help='Epsilon Decreasing Rate', type=float,
                            default=0.99999)
    arg_parser.add_argument('--epsilon_min', help='Minimal Epsilon Value', type=float,
                            default=0.1)
    arg_parser.add_argument('--gamma', help='Gamma', type=float,
                            default=0.99)
    arg_parser.add_argument('--replace', help='Replace Target Network Intervals', type=int,
                            default=100)
    arg_parser.add_argument('--lr', help='Learning Rate', type=float,
                            default=0.0001)
    arg_parser.add_argument('--batch_size', help='Batch Size', type=int,
                            default=100)
    arg_parser.add_argument('--seed', help='Batch Size', type=int,
                            default=128)
    arg_parser.add_argument('--attack_config', help='Attack config in csv format', type=str, default="")
    args, unknown_args = arg_parser.parse_known_args(sys.argv)
    command_line_arguments = vars(args)
    set_random_seed(command_line_arguments['seed'])
    parent_run_id = run_safe_gym_training(command_line_arguments)
    end_time = time.time()
    print("Total Time:", (end_time-start_time))
    assert 1 == 1


@pytest.mark.skip(reason="no way of currently testing this")
def test_smart_grid_agent_e():
    import time
    start_time = time.time()
    arg_parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    args = argparse.Namespace()
    unknown_args = list()
    arg_parser.add_argument('--task', help='What is the name of your project?', type=str,
                            default='safe_training')
    arg_parser.add_argument('--project_name', help='What is the name of your project?', type=str,
                            default='experiments')
    arg_parser.add_argument('--parent_run_id', help='Do you want to continue training of a RL agent? Name the run_id of the last training unit (see mlflow ui).', type=str,
                            default='')
    arg_parser.add_argument('--sliding_window_size', help='What is the sliding window size for the reward averaging?', type=int,
                            default=100)
    arg_parser.add_argument('--num_episodes', help='What is the number of training episodes?', type=int,
                            default=5000)
    arg_parser.add_argument('--eval_interval', help='What is the number of training episodes?', type=int,
                            default=50)
    arg_parser.add_argument('--rl_algorithm', help='What is the  used RL algorithm?', type=str,
                            default='dqn_agent')
    # OpenAI Gym
    arg_parser.add_argument('--env', help='In which environment do you want to train your RL agent?', type=str,
                            default='')
    # Model Checking
    arg_parser.add_argument('--prism_dir', help='In which folder should we save your projects?', type=str,
                            default='../prism_files')
    arg_parser.add_argument('--prism_file_path', help='In which folder should we save your projects?', type=str,
                            default='smart_grid.prism')
    arg_parser.add_argument('--constant_definitions', help='Constant definitions of the formal model (PRISM model)', type=str,
                            default='max_consumption=4,renewable_limit=3,non_renewable_limit=3,grid_upper_bound=2')
    arg_parser.add_argument('--prop', help='Property Specification', type=str,
                            default='Pmin=? [F<=10000 IS_BLACKOUT=true]')
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
                             
    # Double DQN Agent
    arg_parser.add_argument('--layers', help='DummyAgent-Parameter: Which action should the dummy agent choose?', type=int,
                            default=4)
    arg_parser.add_argument('--neurons', help='DummyAgent-Parameter: Which action should the dummy agent choose?', type=int,
                            default=512)
    arg_parser.add_argument('--replay_buffer_size', help='DummyAgent-Parameter: Which action should the dummy agent choose?', type=int,
                            default=100000)
    arg_parser.add_argument('--epsilon', help='Epsilon Starting Rate', type=float,
                            default=1)
    arg_parser.add_argument('--epsilon_dec', help='Epsilon Decreasing Rate', type=float,
                            default=0.99999)
    arg_parser.add_argument('--epsilon_min', help='Minimal Epsilon Value', type=float,
                            default=0.1)
    arg_parser.add_argument('--gamma', help='Gamma', type=float,
                            default=0.99)
    arg_parser.add_argument('--replace', help='Replace Target Network Intervals', type=int,
                            default=100)
    arg_parser.add_argument('--lr', help='Learning Rate', type=float,
                            default=0.0001)
    arg_parser.add_argument('--batch_size', help='Batch Size', type=int,
                            default=100)
    arg_parser.add_argument('--seed', help='Batch Size', type=int,
                            default=128)
    arg_parser.add_argument('--attack_config', help='Attack config in csv format', type=str, default="")
    args, unknown_args = arg_parser.parse_known_args(sys.argv)
    command_line_arguments = vars(args)
    set_random_seed(command_line_arguments['seed'])
    parent_run_id = run_safe_gym_training(command_line_arguments)
    end_time = time.time()
    print("Total Time:", (end_time-start_time))
    assert 1 == 1

@pytest.mark.skip(reason="no way of currently testing this")
def test_stock_market_agent_f():
    import time
    start_time = time.time()
    arg_parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    args = argparse.Namespace()
    unknown_args = list()
    arg_parser.add_argument('--task', help='What is the name of your project?', type=str,
                            default='safe_training')
    arg_parser.add_argument('--project_name', help='What is the name of your project?', type=str,
                            default='experiments')
    arg_parser.add_argument('--parent_run_id', help='Do you want to continue training of a RL agent? Name the run_id of the last training unit (see mlflow ui).', type=str,
                            default='')
    arg_parser.add_argument('--sliding_window_size', help='What is the sliding window size for the reward averaging?', type=int,
                            default=100)
    arg_parser.add_argument('--num_episodes', help='What is the number of training episodes?', type=int,
                            default=10000)
    arg_parser.add_argument('--eval_interval', help='What is the number of training episodes?', type=int,
                            default=10)
    arg_parser.add_argument('--rl_algorithm', help='What is the  used RL algorithm?', type=str,
                            default='dqn_agent')
    # OpenAI Gym
    arg_parser.add_argument('--env', help='In which environment do you want to train your RL agent?', type=str,
                            default='')
    # Model Checking
    arg_parser.add_argument('--prism_dir', help='In which folder should we save your projects?', type=str,
                            default='../prism_files')
    arg_parser.add_argument('--prism_file_path', help='In which folder should we save your projects?', type=str,
                            default='stock_market.prism')
    arg_parser.add_argument('--constant_definitions', help='Constant definitions of the formal model (PRISM model)', type=str,
                            default='')
    arg_parser.add_argument('--prop', help='Property Specification', type=str,
                            default='Pmax=? [F<=100 "success"]')
    arg_parser.add_argument('--max_steps', help='Maximal steps in environment', type=int,
                            default=100)
    arg_parser.add_argument('--disabled_features', help='Features which should not be used by the RL agent: FEATURE1,FEATURE2,...', type=str,
                            default='')
    arg_parser.add_argument('--permissive_input', help='Constant definitions of the formal model (e.g. pos=[0,3];...)', type=str,
                            default='')
    arg_parser.add_argument('--abstract_features', help='', type=str,
                            default='')
    arg_parser.add_argument('--wrong_action_penalty', help='Wrong action penalty', type=int,
                            default=10)
    arg_parser.add_argument('--reward_flag', help='Reward Flag (', type=int,
                            default=1)
    arg_parser.add_argument('--deploy', help='Deploy Flag (', type=int,
                            default=0)       
                             
    # Double DQN Agent
    arg_parser.add_argument('--layers', help='DummyAgent-Parameter: Which action should the dummy agent choose?', type=int,
                            default=4)
    arg_parser.add_argument('--neurons', help='DummyAgent-Parameter: Which action should the dummy agent choose?', type=int,
                            default=512)
    arg_parser.add_argument('--replay_buffer_size', help='DummyAgent-Parameter: Which action should the dummy agent choose?', type=int,
                            default=100000)
    arg_parser.add_argument('--epsilon', help='Epsilon Starting Rate', type=float,
                            default=1)
    arg_parser.add_argument('--epsilon_dec', help='Epsilon Decreasing Rate', type=float,
                            default=0.99999)
    arg_parser.add_argument('--epsilon_min', help='Minimal Epsilon Value', type=float,
                            default=0.1)
    arg_parser.add_argument('--gamma', help='Gamma', type=float,
                            default=0.99)
    arg_parser.add_argument('--replace', help='Replace Target Network Intervals', type=int,
                            default=100)
    arg_parser.add_argument('--lr', help='Learning Rate', type=float,
                            default=0.0001)
    arg_parser.add_argument('--batch_size', help='Batch Size', type=int,
                            default=100)
    arg_parser.add_argument('--seed', help='Batch Size', type=int,
                            default=128)
    arg_parser.add_argument('--attack_config', help='Attack config in csv format', type=str, default="")
    args, unknown_args = arg_parser.parse_known_args(sys.argv)
    command_line_arguments = vars(args)
    set_random_seed(command_line_arguments['seed'])
    parent_run_id = run_safe_gym_training(command_line_arguments)
    end_time = time.time()
    print("Total Time:", (end_time-start_time))
    assert 1 == 1

#@pytest.mark.skip(reason="no way of currently testing this")
def test_james_bond_g():
    import time
    start_time = time.time()
    arg_parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    args = argparse.Namespace()
    unknown_args = list()
    arg_parser.add_argument('--task', help='What is the name of your project?', type=str,
                            default='safe_training')
    arg_parser.add_argument('--project_name', help='What is the name of your project?', type=str,
                            default='experiments')
    arg_parser.add_argument('--parent_run_id', help='Do you want to continue training of a RL agent? Name the run_id of the last training unit (see mlflow ui).', type=str,
                            default='')
    arg_parser.add_argument('--sliding_window_size', help='What is the sliding window size for the reward averaging?', type=int,
                            default=100)
    arg_parser.add_argument('--num_episodes', help='What is the number of training episodes?', type=int,
                            default=5000)
    arg_parser.add_argument('--eval_interval', help='What is the number of training episodes?', type=int,
                            default=10)
    arg_parser.add_argument('--rl_algorithm', help='What is the  used RL algorithm?', type=str,
                            default='dqn_agent')
    # OpenAI Gym
    arg_parser.add_argument('--env', help='In which environment do you want to train your RL agent?', type=str,
                            default='')
    # Model Checking
    arg_parser.add_argument('--prism_dir', help='In which folder should we save your projects?', type=str,
                            default='../prism_files')
    arg_parser.add_argument('--prism_file_path', help='In which folder should we save your projects?', type=str,
                            default='james_bond007.prism')
    arg_parser.add_argument('--constant_definitions', help='Constant definitions of the formal model (PRISM model)', type=str,
                            default='')
    arg_parser.add_argument('--prop', help='Property Specification', type=str,
                            default='')
    arg_parser.add_argument('--max_steps', help='Maximal steps in environment', type=int,
                            default=100)
    arg_parser.add_argument('--disabled_features', help='Features which should not be used by the RL agent: FEATURE1,FEATURE2,...', type=str,
                            default='')
    arg_parser.add_argument('--permissive_input', help='Constant definitions of the formal model (e.g. pos=[0,3];...)', type=str,
                            default='')
    arg_parser.add_argument('--abstract_features', help='', type=str,
                            default='')
    arg_parser.add_argument('--wrong_action_penalty', help='Wrong action penalty', type=int,
                            default=10)
    arg_parser.add_argument('--reward_flag', help='Reward Flag (', type=int,
                            default=1)
    arg_parser.add_argument('--deploy', help='Deploy Flag (', type=int,
                            default=0)       
                             
    # Double DQN Agent
    arg_parser.add_argument('--layers', help='DummyAgent-Parameter: Which action should the dummy agent choose?', type=int,
                            default=4)
    arg_parser.add_argument('--neurons', help='DummyAgent-Parameter: Which action should the dummy agent choose?', type=int,
                            default=512)
    arg_parser.add_argument('--replay_buffer_size', help='DummyAgent-Parameter: Which action should the dummy agent choose?', type=int,
                            default=100000)
    arg_parser.add_argument('--epsilon', help='Epsilon Starting Rate', type=float,
                            default=1)
    arg_parser.add_argument('--epsilon_dec', help='Epsilon Decreasing Rate', type=float,
                            default=0.99999)
    arg_parser.add_argument('--epsilon_min', help='Minimal Epsilon Value', type=float,
                            default=0.1)
    arg_parser.add_argument('--gamma', help='Gamma', type=float,
                            default=0.99)
    arg_parser.add_argument('--replace', help='Replace Target Network Intervals', type=int,
                            default=100)
    arg_parser.add_argument('--lr', help='Learning Rate', type=float,
                            default=0.0001)
    arg_parser.add_argument('--batch_size', help='Batch Size', type=int,
                            default=100)
    arg_parser.add_argument('--seed', help='Batch Size', type=int,
                            default=128)
    arg_parser.add_argument('--attack_config', help='Attack config in csv format', type=str, default="")
    args, unknown_args = arg_parser.parse_known_args(sys.argv)
    command_line_arguments = vars(args)
    set_random_seed(command_line_arguments['seed'])
    parent_run_id = run_safe_gym_training(command_line_arguments)
    end_time = time.time()
    print("Total Time:", (end_time-start_time))
    assert 1 == 1

#@pytest.mark.skip(reason="no way of currently testing this")
def test_crazy_climber_h():
    import time
    start_time = time.time()
    arg_parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    args = argparse.Namespace()
    unknown_args = list()
    arg_parser.add_argument('--task', help='What is the name of your project?', type=str,
                            default='safe_training')
    arg_parser.add_argument('--project_name', help='What is the name of your project?', type=str,
                            default='experiments')
    arg_parser.add_argument('--parent_run_id', help='Do you want to continue training of a RL agent? Name the run_id of the last training unit (see mlflow ui).', type=str,
                            default='')
    arg_parser.add_argument('--sliding_window_size', help='What is the sliding window size for the reward averaging?', type=int,
                            default=100)
    arg_parser.add_argument('--num_episodes', help='What is the number of training episodes?', type=int,
                            default=3000)
    arg_parser.add_argument('--eval_interval', help='What is the number of training episodes?', type=int,
                            default=10)
    arg_parser.add_argument('--rl_algorithm', help='What is the  used RL algorithm?', type=str,
                            default='dqn_agent')
    # OpenAI Gym
    arg_parser.add_argument('--env', help='In which environment do you want to train your RL agent?', type=str,
                            default='')
    # Model Checking
    arg_parser.add_argument('--prism_dir', help='In which folder should we save your projects?', type=str,
                            default='../prism_files')
    arg_parser.add_argument('--prism_file_path', help='In which folder should we save your projects?', type=str,
                            default='crazy_climber.prism')
    arg_parser.add_argument('--constant_definitions', help='Constant definitions of the formal model (PRISM model)', type=str,
                            default='')
    arg_parser.add_argument('--prop', help='Property Specification', type=str,
                            default='')
    arg_parser.add_argument('--max_steps', help='Maximal steps in environment', type=int,
                            default=100)
    arg_parser.add_argument('--disabled_features', help='Features which should not be used by the RL agent: FEATURE1,FEATURE2,...', type=str,
                            default='')
    arg_parser.add_argument('--permissive_input', help='Constant definitions of the formal model (e.g. pos=[0,3];...)', type=str,
                            default='')
    arg_parser.add_argument('--abstract_features', help='', type=str,
                            default='')
    arg_parser.add_argument('--wrong_action_penalty', help='Wrong action penalty', type=int,
                            default=10)
    arg_parser.add_argument('--reward_flag', help='Reward Flag (', type=int,
                            default=1)
    arg_parser.add_argument('--deploy', help='Deploy Flag (', type=int,
                            default=0)       
                             
    # Double DQN Agent
    arg_parser.add_argument('--layers', help='DummyAgent-Parameter: Which action should the dummy agent choose?', type=int,
                            default=4)
    arg_parser.add_argument('--neurons', help='DummyAgent-Parameter: Which action should the dummy agent choose?', type=int,
                            default=512)
    arg_parser.add_argument('--replay_buffer_size', help='DummyAgent-Parameter: Which action should the dummy agent choose?', type=int,
                            default=100000)
    arg_parser.add_argument('--epsilon', help='Epsilon Starting Rate', type=float,
                            default=1)
    arg_parser.add_argument('--epsilon_dec', help='Epsilon Decreasing Rate', type=float,
                            default=0.99999)
    arg_parser.add_argument('--epsilon_min', help='Minimal Epsilon Value', type=float,
                            default=0.1)
    arg_parser.add_argument('--gamma', help='Gamma', type=float,
                            default=0.99)
    arg_parser.add_argument('--replace', help='Replace Target Network Intervals', type=int,
                            default=100)
    arg_parser.add_argument('--lr', help='Learning Rate', type=float,
                            default=0.0001)
    arg_parser.add_argument('--batch_size', help='Batch Size', type=int,
                            default=100)
    arg_parser.add_argument('--seed', help='Batch Size', type=int,
                            default=128)
    arg_parser.add_argument('--attack_config', help='Attack config in csv format', type=str, default="")
    args, unknown_args = arg_parser.parse_known_args(sys.argv)
    command_line_arguments = vars(args)
    set_random_seed(command_line_arguments['seed'])
    parent_run_id = run_safe_gym_training(command_line_arguments)
    end_time = time.time()
    print("Total Time:", (end_time-start_time))
    assert 1 == 1

def test_freeway_experiment_section_agent_j():
    arg_parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    args = argparse.Namespace()
    unknown_args = list()
    arg_parser.add_argument('--task', help='What is the name of your project?', type=str,
                            default='safe_training')
    arg_parser.add_argument('--project_name', help='What is the name of your project?', type=str,
                            default='experiments')
    arg_parser.add_argument('--parent_run_id', help='Do you want to continue training of a RL agent? Name the run_id of the last training unit (see mlflow ui).', type=str,
                            default='')
    arg_parser.add_argument('--sliding_window_size', help='What is the sliding window size for the reward averaging?', type=int,
                            default=100)
    arg_parser.add_argument('--num_episodes', help='What is the number of training episodes?', type=int,
                            default=2900)
    arg_parser.add_argument('--eval_interval', help='What is the number of training episodes?', type=int,
                            default=700)
    arg_parser.add_argument('--rl_algorithm', help='What is the  used RL algorithm?', type=str,
                            default='dqn_agent')
    # OpenAI Gym
    arg_parser.add_argument('--env', help='In which environment do you want to train your RL agent?', type=str,
                            default='')
    # Model Checking
    arg_parser.add_argument('--prism_dir', help='In which folder should we save your projects?', type=str,
                            default='../prism_files')
    arg_parser.add_argument('--prism_file_path', help='In which folder should we save your projects?', type=str,
                            default='freeway.prism')
    arg_parser.add_argument('--constant_definitions', help='Constant definitions of the formal model (PRISM model)', type=str,
                            default='')
    arg_parser.add_argument('--prop', help='Property Specification', type=str,
                            default='Pmax=? [F won=true]')
    arg_parser.add_argument('--max_steps', help='Maximal steps in environment', type=int,
                            default=30)
    arg_parser.add_argument('--disabled_features', help='Features which should not be used by the RL agent: FEATURE1,FEATURE2,...', type=str,
                            default='')
    arg_parser.add_argument('--permissive_input', help='Constant definitions of the formal model (e.g. pos=[0,3];...)', type=str,
                            default='')
    arg_parser.add_argument('--abstract_features', help='', type=str,
                            default='')
    arg_parser.add_argument('--wrong_action_penalty', help='Wrong action penalty', type=int,
                            default=1000)
    arg_parser.add_argument('--reward_flag', help='Reward Flag (', type=int,
                            default=1)
    arg_parser.add_argument('--deploy', help='Deploy Flag (', type=int,
                            default=0)       
                             

    arg_parser.add_argument('--layers', help='DummyAgent-Parameter: Which action should the dummy agent choose?', type=int,
                            default=2)
    arg_parser.add_argument('--neurons', help='DummyAgent-Parameter: Which action should the dummy agent choose?', type=int,
                            default=512)
    arg_parser.add_argument('--replay_buffer_size', help='DummyAgent-Parameter: Which action should the dummy agent choose?', type=int,
                            default=100000)
    arg_parser.add_argument('--epsilon', help='Epsilon Starting Rate', type=float,
                            default=1)
    arg_parser.add_argument('--epsilon_dec', help='Epsilon Decreasing Rate', type=float,
                            default=0.9999)
    arg_parser.add_argument('--epsilon_min', help='Minimal Epsilon Value', type=float,
                            default=0.1)
    arg_parser.add_argument('--gamma', help='Gamma', type=float,
                            default=0.99)
    arg_parser.add_argument('--replace', help='Replace Target Network Intervals', type=int,
                            default=100)
    arg_parser.add_argument('--lr', help='Learning Rate', type=float,
                            default=0.0001)
    arg_parser.add_argument('--batch_size', help='Batch Size', type=int,
                            default=100)
    arg_parser.add_argument('--seed', help='Batch Size', type=int,
                            default=128)
    arg_parser.add_argument('--attack_config', help='Attack config in csv format', type=str, default="")
    
    args, unknown_args = arg_parser.parse_known_args(sys.argv)
    command_line_arguments = vars(args)
    set_random_seed(command_line_arguments['seed'])
    parent_run_id = run_safe_gym_training(command_line_arguments)
    assert 1 == 1