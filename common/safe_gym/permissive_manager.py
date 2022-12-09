import numpy as np
from common.safe_gym.state_mapper import StateMapper

class PStateVariable:

    def __init__(self, name: str, lower_bound: int, upper_bound: int, idx = None, current_assignment = None, step_size: int = 1, is_range: bool = False):
        self.name = name
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.idx = idx
        self.step_size = step_size
        self.is_range = is_range
        if current_assignment == None:
            self.current_assignment = lower_bound
        else:
            self.current_assignment = current_assignment

    def next(self):
        current_value = self.current_assignment
        self.current_assignment+=1
        if current_value > self.upper_bound:
            return None
        else:
            return current_value

    def set_idx(self, idx):
        self.idx = idx

    def reset(self):
        self.current_assignment = self.lower_bound

    def copy(self):
        return PStateVariable(self.name, self.lower_bound, self.upper_bound, self.idx, current_assignment=self.current_assignment, step_size=self.step_size, is_range=self.is_range)


    def __str__(self):
        return self.name + "=["+str(self.lower_bound)+','+str(self.upper_bound)+"]" + " Index: " + str(self.idx) + " Step Size: " + str(self.step_size)

    @staticmethod
    def copy_all_state_variables(variables):
        copies = []
        for variable in variables:
            copies.append(variable.copy())
        return copies

    @staticmethod
    def parse_state_variables(permissive_input):
        pstate_variables = []
        for state_variable_str in permissive_input.split(","):
            name = state_variable_str.split('=')[0]
            if name.find("<>")!=-1:
                is_range = True
                name = name.replace('<>','')
            else:
                is_range = False
            start_interval = state_variable_str.find('[')
            end_interval = state_variable_str.find(']')
            interval = state_variable_str[(start_interval+1):end_interval]
            step_size = 1
            if interval.count(';')>1:
                parts = state_variable_str[(start_interval+1):end_interval].split(';')
                lower_bound = int(parts[0])
                step_size = int(parts[1])
                upper_bound = int(parts[2])
            else:
                commata = state_variable_str[start_interval:].find(';')
                lower_bound = int(state_variable_str[start_interval+1:start_interval+commata])
                upper_bound = int(state_variable_str[start_interval+commata+1:end_interval])
            pstate_variable = PStateVariable(name, lower_bound, upper_bound, step_size = step_size, is_range=is_range)
            pstate_variables.append(pstate_variable)
        return pstate_variables

    @staticmethod
    def generate_all_states(mapper, fix_state, pstate_variables):
        fix_state = np.array(fix_state, copy=True, dtype=np.int32)
        # Assign all the variable indizes
        for i in range(len(pstate_variables)):
            pstate_variables[i].set_idx(mapper[pstate_variables[i].name])
        # Generate States
        all_states = PStateVariable.__generate_all_states(pstate_variables, fix_state, 0)
        return all_states

    @staticmethod
    def __generate_all_states(pstate_variables, state, state_var_idx):
        # Generate new state for current state var
        n_state = np.array(state, copy=True)
        all_current_states = []
        # Stop if index out of range
        if state_var_idx>=len(pstate_variables):
            return all_current_states
        
        
        if pstate_variables[state_var_idx].is_range == True and (state[pstate_variables[state_var_idx].idx] < pstate_variables[state_var_idx].lower_bound or state[pstate_variables[state_var_idx].idx] > pstate_variables[state_var_idx].upper_bound):
            all_current_states.append(n_state)
            c_state_variables = PStateVariable.copy_all_state_variables(pstate_variables)
            sub_states = PStateVariable.__generate_all_states(c_state_variables, n_state, state_var_idx+1)
            all_current_states.extend(sub_states)
            return all_current_states
        else:
            while True:
                value = pstate_variables[state_var_idx].next()
                if value == None:
                    return all_current_states
                else:
                    # Copy state
                    n_state = np.array(n_state, copy=True)
                    # Update copied state
                    n_state[pstate_variables[state_var_idx].idx] = value
                    # Add state
                    all_current_states.append(n_state)
                    # Copy state variables
                    c_state_variables = PStateVariable.copy_all_state_variables(pstate_variables)
                    sub_states = PStateVariable.__generate_all_states(c_state_variables, n_state, state_var_idx+1)
                    all_current_states.extend(sub_states)
        
        
class PermissiveManager:

    def __init__(self, permissive_input: str, state_var_mapper: StateMapper):
        self.current_state = None
        self.state_var_mapper = state_var_mapper.mapper
        self.is_permissive = (permissive_input != '')
        if self.is_permissive:
            self.pstate_variables = PStateVariable.parse_state_variables(permissive_input)
        self.permissive_actions = []
        self.action_mapper = None

    def manage_actions(self, state, agent):
        if (self.current_state is None) or (np.array_equal(self.current_state, state) == False):
            # Reset Actions
            self.permissive_actions = []
            # Reset all pstate_variables
            for pstate_variable in self.pstate_variables:
                pstate_variable.reset()
            self.current_state = state
            # Get all permissive states
            #print('New', self.current_state)
            all_states = PStateVariable.generate_all_states(self.state_var_mapper, self.current_state, self.pstate_variables)
            for state in all_states:
                action = self.action_mapper.action_index_to_action_name(agent.select_action(state, deploy=True))
                self.permissive_actions.append(action)
            # Get all actions for these permissive states
            
        return self.permissive_actions

    def manage_permissive_state_actions_pairs(self, state, agent):
        all_pairs = {}
        if (self.current_state is None) or (np.array_equal(self.current_state, state) == False):
            # Reset Actions
            self.permissive_actions = []
            # Reset all pstate_variables
            for pstate_variable in self.pstate_variables:
                pstate_variable.reset()
            #print('New', self.current_state)
            all_states = PStateVariable.generate_all_states(self.state_var_mapper, state, self.pstate_variables)
            
            for state in all_states:
                action = self.action_mapper.action_index_to_action_name(agent.select_action(state, deploy=True))
                all_pairs[str(state)] = action
        return all_pairs

    

    def create_condition(self, available_actions, action_name):
        cond1 = False
        for selected_action in self.permissive_actions:
            if selected_action not in available_actions:
                cond1 |= (action_name == available_actions[0])
            else:
                cond1 |= (action_name == selected_action)
        return cond1