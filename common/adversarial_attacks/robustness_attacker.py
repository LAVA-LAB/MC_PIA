from common.adversarial_attacks.adversarial_attack import AdversarialAttack
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torch
import math
from numpy import linalg as LA
import itertools

NORM = np.inf
def cartesian_coord2(*arrays, epsilon, feature_map=None):
    grid = np.meshgrid(*arrays)      
    coord_list = [entry.ravel() for entry in grid]
    points = np.vstack(coord_list).T
    if feature_map!=None:
        for col in range(points.shape[1]):
            target_feature = col in feature_map.values()
            if target_feature==False:
                points=points[~(points[:,col] != 0),:]
    if NORM==1:
        return list(points[np.absolute(points).sum(axis=1) <= epsilon,:])
    else:
        return list(points[np.absolute(points).max(axis=1) <= epsilon,:])
    

def cartesian_coord(*arrays, epsilon, feature_map=None, state=None, rl_agent=None, original_action_idx=None):
    
    length = len(arrays)
    values = arrays[0]
    all_action_idizes = [original_action_idx]
    already_existed = []
    counter = 0
    #print(state)
    if feature_map != None:
        for combination in itertools.product(values, repeat=len(feature_map.values())):
            # Create list with only zeroes
            zero_vector = [0] * length
            counter+=1
            for i, idx in enumerate(feature_map.values()):
                zero_vector[idx] = combination[i]
            if LA.norm(zero_vector,ord=NORM) <= epsilon:
                action_idx = rl_agent.select_action(state+np.array(zero_vector),deploy=True)
                if action_idx != original_action_idx:
                    if action_idx not in already_existed:
                        already_existed.append(action_idx)
                        all_action_idizes.append(action_idx)
                        if len(all_action_idizes) == rl_agent.number_of_actions:
                            #print(rl_agent.number_of_actions, len(all_action_idizes),"Counter",counter)
                            #print("Early Ending")
                            return all_action_idizes

    else:
        for combination in itertools.product(values, repeat=length):
            counter+=1
            if LA.norm(combination,ord=NORM) <= epsilon:
                action_idx = rl_agent.select_action(state+np.array(combination),deploy=True)
                if action_idx != original_action_idx:
                    if action_idx not in already_existed:
                        already_existed.append(action_idx)
                        all_action_idizes.append(action_idx)
                        if len(all_action_idizes) == rl_agent.number_of_actions:
                            #print(rl_agent.number_of_actions, len(all_action_idizes),"Counter",counter)
                            #print("Early Ending")
                            return all_action_idizes
    #print(rl_agent.number_of_actions, len(all_action_idizes),"Counter",counter)
    return all_action_idizes
    
        
    


class RobustnessAttacker(AdversarialAttack):

    def __init__(self, state_mapper, attack_config_str: str) -> None:
        super().__init__(state_mapper, attack_config_str)
        self.parse_attack_config(attack_config_str)
        self.max_actions = -1
        self.max_counter = math.inf
        self.valid_attacks = None
        self.all_actions_found = 0
        self.not_all_actions_found = 0
        self.totally_robust = 0
        self.failed_once = False

    def get_feature_mapping(self, features):
        feature_map = {}
        for feature in features:
            feature_map[feature] = self.state_mapper.mapper[feature]
        return feature_map

    def zeroize_attack_for_none_features(self, attack):
        for i in range(attack.shape[0]):
            if i not in self.feature_map.values():
                attack[i] = 0
        return attack

    def parse_attack_config(self, attack_config_str: str) -> None:
        try:
            attack_name, epsilon = attack_config_str.split(',')
            self.attack_name = attack_name
            self.epsilon = int(epsilon)
            self.current_state = None
            self.feature_map = None
        except:
            attack_name, epsilon, features = attack_config_str.split(',')
            self.attack_name = attack_name
            self.epsilon = int(epsilon)
            self.current_state = None
            self.feature_map = self.get_feature_mapping(features.split("+"))




    def action_dict_full(self, all_actions_idizes):
        # Check if all actions are present
        return len(list(all_actions_idizes.keys())) == self.max_actions

    def only_original_actions(self, all_actions_idizes, original_action_idx):
        if str(original_action_idx) in all_actions_idizes.keys() and len(all_actions_idizes.keys()) == 1 or len(all_actions_idizes.keys()) == 0:
            return True
        else:
            return False

    def fast_attack(self, action_mapper, rl_agent, state:np.ndarray, current_model_checking_action_idx: int = -1) -> np.ndarray:
        if (self.current_state is None) or (np.array_equal(self.current_state, state) == False):
            #print("New State", state)
            # Original Action
            original_action_idx = rl_agent.select_action(state,deploy=True)
            # Reset Actions
            self.current_state = state
            self.permissive_actions = []
            all_actions_idizes = {}
            tmp_valid_attacks = []
            tries = 0
            counter=0
            if self.max_counter == math.inf:
                all_possible_attacks = cartesian_coord2(*len(state)*[np.arange(-self.epsilon, self.epsilon+1)],epsilon=self.epsilon,feature_map=self.feature_map)
            else:
                all_possible_attacks = self.valid_attacks
            for idx, attack in enumerate(all_possible_attacks):
                attack = np.array(attack)
                tries+=1
                if True:
                    #print(attack)
                    counter+=1
                    if self.max_counter == math.inf:
                        # Only collect all valid actions until the max_counter is defined
                        tmp_valid_attacks.append(attack.tolist())
                    # Add the autoencoder here
                    action_idx = rl_agent.select_action(state+attack,deploy=True)
                    # If attack was successful and the action_idx is not yet in all_action_idizes, we add the successful attack in front
                    
                    if action_idx != original_action_idx and action_idx not in all_actions_idizes:
                        tmp = all_possible_attacks[-idx]
                        del all_possible_attacks[-idx]
                        all_possible_attacks.insert(0,tmp)
                    
                    
                    all_actions_idizes[str(action_idx)] = state+attack
                    if self.action_dict_full(all_actions_idizes) or counter==self.max_counter:
                        #print(counter, self.max_counter)
                        if self.action_dict_full(all_actions_idizes) != True:
                            #print("Not all actions found")
                            self.all_actions_found +=1
                        else:
                            #print("All actions found")
                            self.not_all_actions_found +=1
                        if self.only_original_actions(all_actions_idizes, original_action_idx):
                            #print("Totally robust")
                            self.totally_robust +=1
                        break
            
            # Only the action_idizes keys are important
            for action_idx in all_actions_idizes.keys():
                action = action_mapper.action_index_to_action_name(int(action_idx))
                self.permissive_actions.append(action)
            

            if tries == len(all_possible_attacks) and self.max_counter == math.inf:
                tmp_valid_attacks.sort()
                self.valid_attacks = list(k for k,_ in itertools.groupby(tmp_valid_attacks))
                #self.valid_attacks.append([0]*len(self.valid_attacks[0]))
                self.max_counter = len(self.valid_attacks)
                print(len(self.valid_attacks))

        return self.permissive_actions


    def attack(self, action_mapper, rl_agent, state:np.ndarray, current_model_checking_action_idx: int = -1) -> np.ndarray:
        if (self.current_state is None) or (np.array_equal(self.current_state, state) == False):
            if self.failed_once == False:
                try:
                    self.permissive_actions = self.fast_attack(action_mapper, rl_agent, state, current_model_checking_action_idx)
                except:
                    self.failed_once = True
            
            if self.failed_once==True:
                # Original Action
                original_action_idx = rl_agent.select_action(state,deploy=True)
                # Reset Actions
                self.current_state = state
                self.permissive_actions = []
                all_actions_idizes = cartesian_coord(*len(state)*[np.arange(-self.epsilon, self.epsilon+1)],epsilon=self.epsilon,feature_map=self.feature_map, state=state, rl_agent=rl_agent, original_action_idx=original_action_idx)
                    
                for action_idx in all_actions_idizes:
                    action = action_mapper.action_index_to_action_name(int(action_idx))
                    self.permissive_actions.append(action)
            
        return self.permissive_actions
     

    
    def create_condition(self, available_actions, action_name):
        cond1 = False
        for selected_action in self.permissive_actions:
            if selected_action not in available_actions:
                cond1 |= (action_name == available_actions[0])
            else:
                cond1 |= (action_name == selected_action)
        return cond1