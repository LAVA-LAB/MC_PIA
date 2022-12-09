import pytest
from helper import *
import os
import numpy as np
sys.path.insert(0, '../')
from common.utilities.project import Project
from common.utilities.training import train
from common.safe_gym.safe_gym import SafeGym
from common.safe_gym.permissive_manager import PStateVariable

#@pytest.mark.skip(reason="no way of currently testing this")
def test_state_space():
    env = SafeGym('../prism_files/dummy_abc.prism', constant_definitions='', max_steps=100, wrong_action_penalty=0, reward_flag=True,seed= -1, permissive_input='', disabled_features='', abstraction_input='',attack_config="")
    original_state = env.reset()
    assert original_state.shape[0] == 3

#@pytest.mark.skip(reason="no way of currently testing this")
def test_disabled_state_space1():
    env = SafeGym('../prism_files/dummy_abc.prism', constant_definitions='', max_steps=100, wrong_action_penalty=0, reward_flag=True, seed= -1, permissive_input='', disabled_features='a', abstraction_input='',attack_config="")
    disabled_state = env.reset()
    assert disabled_state.shape[0] == 2
    assert disabled_state[0] == 0

#@pytest.mark.skip(reason="no way of currently testing this")
def test_disabled_state_space2():
    env = SafeGym('../prism_files/dummy_abc.prism', constant_definitions='', max_steps=100, wrong_action_penalty=0, reward_flag=True, seed= -1, permissive_input='', disabled_features='a,b', abstraction_input='',attack_config="")
    disabled_state = env.reset()
    assert disabled_state.shape[0] == 1
    assert disabled_state[0] == 0

#@pytest.mark.skip(reason="no way of currently testing this")
def test_disabled_state_space3():
    env = SafeGym('../prism_files/dummy_abc.prism', constant_definitions='', max_steps=100, wrong_action_penalty=0, reward_flag=True, seed= -1, permissive_input='', disabled_features='a,c', abstraction_input='',attack_config="")
    disabled_state = env.reset()
    assert disabled_state.shape[0] == 1
    assert disabled_state[0] == 0


#@pytest.mark.skip(reason="no way of currently testing this")
def test_disabled_state_space4():
    env = SafeGym('../prism_files/dummy_abc.prism', constant_definitions='', max_steps=100, wrong_action_penalty=0, reward_flag=True, seed= -1, permissive_input='', disabled_features='b,c', abstraction_input='',attack_config="")
    disabled_state = env.reset()
    assert disabled_state.shape[0] == 1
    assert disabled_state[0] == 1

#@pytest.mark.skip(reason="no way of currently testing this")
def test_mapping_state_space():
    env = SafeGym('../prism_files/dummy_abc2.prism', constant_definitions='', max_steps=100, wrong_action_penalty=0, reward_flag=True, seed= -1, permissive_input='', disabled_features='', abstraction_input='',attack_config="")
    original_state = env.reset()
    assert original_state.shape[0] == 3
    assert original_state[2] == 1
    assert original_state[1] == 0
    assert original_state[0] == 0

#@pytest.mark.skip(reason="no way of currently testing this")
def test_mapping_disabled_state_space1():
    env = SafeGym('../prism_files/dummy_abc2.prism', constant_definitions='', max_steps=100, wrong_action_penalty=0, reward_flag=True, seed= -1, permissive_input='', disabled_features='a', abstraction_input='',attack_config="")
    disabled_state = env.reset()
    assert disabled_state.shape[0] == 2
    assert disabled_state[0] == 0
    assert disabled_state[1] == 0

#@pytest.mark.skip(reason="no way of currently testing this")
def test_mapping_disabled_state_space2():
    env = SafeGym('../prism_files/dummy_abc2.prism', constant_definitions='', max_steps=100, wrong_action_penalty=0, reward_flag=True, seed= -1, permissive_input='', disabled_features='a,b', abstraction_input='',attack_config="")
    disabled_state = env.reset()
    assert disabled_state.shape[0] == 1
    assert disabled_state[0] == 0

#@pytest.mark.skip(reason="no way of currently testing this")
def test_mapping_disabled_state_space3():
    env = SafeGym('../prism_files/dummy_abc2.prism', constant_definitions='', max_steps=100, wrong_action_penalty=0, reward_flag=True, seed= -1, permissive_input='', disabled_features='c,b', abstraction_input='',attack_config="")
    disabled_state = env.reset()
    assert disabled_state.shape[0] == 1
    assert disabled_state[0] == 1

#@pytest.mark.skip(reason="no way of currently testing this")
def test_mapping_disabled_state_space4():
    env = SafeGym('../prism_files/dummy_abc2.prism', constant_definitions='', max_steps=100, wrong_action_penalty=0, reward_flag=True, seed= -1, permissive_input='', disabled_features='b,c', abstraction_input='',attack_config="")
    disabled_state = env.reset()
    assert disabled_state.shape[0] == 1
    assert disabled_state[0] == 1

def test_mapping_disabled_state_space4():
    env = SafeGym('../prism_files/dummy_abcd.prism', constant_definitions='', max_steps=100, wrong_action_penalty=0, reward_flag=True, seed= -1, permissive_input='', disabled_features='b,c', abstraction_input='',attack_config="")
    disabled_state = env.reset()
    assert disabled_state.shape[0] == 2
    assert disabled_state[0] == 4


