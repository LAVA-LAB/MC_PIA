import pytest
from helper import *
import os
import numpy as np
sys.path.insert(0, '../')
from common.utilities.project import Project
from common.utilities.training import train
from common.safe_gym.safe_gym import SafeGym
from common.safe_gym.permissive_manager import PStateVariable

def test_all_states_1d_0_2():
    command_line_arguments = get_frozen_lake_v1_arguments()
    command_line_arguments['task'] = 'safe_training'
    command_line_arguments['deploy'] = 0
    command_line_arguments['rl_algorithm'] = 'dummy_frozen_lake_agent'
    command_line_arguments['permissive_input'] = 'pos=[0;2]'
    pstate_variable = PStateVariable('pos', 0, 2)
    prism_file_path = os.path.join(command_line_arguments['prism_dir'], command_line_arguments['prism_file_path'])
    env = SafeGym(prism_file_path,command_line_arguments['constant_definitions'], command_line_arguments['max_steps'], command_line_arguments['wrong_action_penalty'], command_line_arguments['reward_flag'], command_line_arguments['seed'], command_line_arguments['permissive_input'], command_line_arguments['disabled_features'],attack_config="")
    all_states = PStateVariable.generate_all_states(env.storm_bridge.state_mapper.mapper, np.array([[0]],dtype=np.int32), [pstate_variable])

    assert len(all_states) == 3

def test_all_states_3d_1var():
    command_line_arguments = get_frozen_lake_arguments()
    command_line_arguments['task'] = 'safe_training'
    command_line_arguments['deploy'] = 0
    command_line_arguments['rl_algorithm'] = 'dummy_frozen_lake_agent'
    command_line_arguments['permissive_input'] = 'x=[0;2]'
    fix_state = np.array([0, 0, 3])
    pstate_variable = PStateVariable('x', 0, 2)
    prism_file_path = os.path.join(command_line_arguments['prism_dir'], command_line_arguments['prism_file_path'])
    env = SafeGym(prism_file_path,command_line_arguments['constant_definitions'], command_line_arguments['max_steps'], command_line_arguments['wrong_action_penalty'], command_line_arguments['reward_flag'], command_line_arguments['seed'], command_line_arguments['permissive_input'], command_line_arguments['disabled_features'],attack_config="")
    all_states = PStateVariable.generate_all_states(env.storm_bridge.state_mapper.mapper,fix_state, [pstate_variable])
    assert len(all_states) == 3

def test_all_states_3d_2vars_with_out_of_range():
    command_line_arguments = get_frozen_lake_arguments()
    command_line_arguments['task'] = 'safe_training'
    command_line_arguments['deploy'] = 0
    command_line_arguments['rl_algorithm'] = 'dummy_frozen_lake_agent'
    command_line_arguments['permissive_input'] = 'x=[0;2]'
    fix_state = np.array([0, 0, 3])
    pstate_variable1 = PStateVariable('x', 0, 2)
    pstate_Variable2 = PStateVariable('y', 0, 2)
    prism_file_path = os.path.join(command_line_arguments['prism_dir'], command_line_arguments['prism_file_path'])
    env = SafeGym(prism_file_path,command_line_arguments['constant_definitions'], command_line_arguments['max_steps'], command_line_arguments['wrong_action_penalty'], command_line_arguments['reward_flag'], command_line_arguments['seed'], command_line_arguments['permissive_input'], command_line_arguments['disabled_features'],attack_config="")
    all_states = PStateVariable.generate_all_states(env.storm_bridge.state_mapper.mapper,fix_state, [pstate_variable1, pstate_Variable2])
    assert len(all_states) == 12

def test_all_states_3d_2vars():
    # Duplicate states won't be removed. this should be no problem.
    command_line_arguments = get_frozen_lake_arguments()
    command_line_arguments['task'] = 'safe_training'
    command_line_arguments['deploy'] = 0
    command_line_arguments['rl_algorithm'] = 'dummy_frozen_lake_agent'
    command_line_arguments['permissive_input'] = 'x=[0;2]'
    fix_state = np.array([0, 0, 2])
    pstate_variable1 = PStateVariable('x', 0, 2)
    pstate_Variable2 = PStateVariable('y', 0, 2)
    prism_file_path = os.path.join(command_line_arguments['prism_dir'], command_line_arguments['prism_file_path'])
    env = SafeGym(prism_file_path,command_line_arguments['constant_definitions'], command_line_arguments['max_steps'], command_line_arguments['wrong_action_penalty'], command_line_arguments['reward_flag'], command_line_arguments['seed'], command_line_arguments['permissive_input'], command_line_arguments['disabled_features'],attack_config="")
    all_states = PStateVariable.generate_all_states(env.storm_bridge.state_mapper.mapper,fix_state, [pstate_variable1, pstate_Variable2])
    assert len(all_states) == 12

def test_all_states_3d_2vars2():
    # Duplicate states won't be removed. this should be no problem.
    command_line_arguments = get_frozen_lake_arguments()
    command_line_arguments['task'] = 'safe_training'
    command_line_arguments['deploy'] = 0
    command_line_arguments['rl_algorithm'] = 'dummy_frozen_lake_agent'
    command_line_arguments['permissive_input'] = 'x=[0;2]'
    fix_state = np.array([0, 0, 2])
    pstate_variable1 = PStateVariable('x', 2, 4)
    #pstate_Variable2 = PStateVariable('y', 0, 2)
    prism_file_path = os.path.join(command_line_arguments['prism_dir'], command_line_arguments['prism_file_path'])
    env = SafeGym(prism_file_path,command_line_arguments['constant_definitions'], command_line_arguments['max_steps'], command_line_arguments['wrong_action_penalty'], command_line_arguments['reward_flag'], command_line_arguments['seed'], command_line_arguments['permissive_input'], command_line_arguments['disabled_features'],attack_config="")
    all_states = PStateVariable.generate_all_states(env.storm_bridge.state_mapper.mapper,fix_state, [pstate_variable1])
    print(all_states)
    # IMPORTANT: The initial state wont be taken into account, if it is not in the interval
    assert len(all_states) == 3

def test_transporter_with_fuel():
    # Duplicate states won't be removed. this should be no problem.
    command_line_arguments = get_transporter_with_fuel_arguments()
    command_line_arguments['task'] = 'safe_training'
    command_line_arguments['reward_flag'] = 0
    command_line_arguments['deploy'] = 0
    command_line_arguments['rl_algorithm'] = 'dummy_frozen_lake_agent'
    command_line_arguments['permissive_input'] = 'fuel=[0;2]'
    fix_state = np.array([0, 0, 0, 0, 0, 0, 0, 0])
    pstate_variable1 = PStateVariable('fuel', 0, 23)
    #pstate_Variable2 = PStateVariable('y', 0, 2)
    prism_file_path = os.path.join(command_line_arguments['prism_dir'], command_line_arguments['prism_file_path'])
    env = SafeGym(prism_file_path,command_line_arguments['constant_definitions'], command_line_arguments['max_steps'], command_line_arguments['wrong_action_penalty'], command_line_arguments['reward_flag'], command_line_arguments['seed'], command_line_arguments['permissive_input'], command_line_arguments['disabled_features'],attack_config="")
    all_states = PStateVariable.generate_all_states(env.storm_bridge.state_mapper.mapper,fix_state, [pstate_variable1])
    print(all_states)
    # IMPORTANT: The initial state wont be taken into account, if it is not in the interval
    assert len(all_states) == 24

