"""
This module handles the state mapping process.
"""
import math
import os
import json
from typing import List
import numpy as np
from common.safe_gym.state_mapper import StateMapper


class AStateVariable:

    def __init__(self, name: str, lower_bound: int, upper_bound: int, step_size: int, mapping=None):
        """Constructor

        Args:
            name (str): Variable name for abstraction
            lower_bound (int): Lower bound of abstraction
            upper_bound (int): Upper bound of abstraction
            step_size (int): Step Size of abstractions
            mapping ([type], optional): State Variable Mapping. Defaults to None.
        """
        assert isinstance(name, str)
        assert isinstance(lower_bound, int)
        assert isinstance(upper_bound, int)
        assert isinstance(step_size, int)
        self.name = name
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.step_size = step_size
        self.all_assignments = range(
            self.lower_bound, self.upper_bound+self.step_size, self.step_size)
        self.mapping = mapping

    def round(self, value: int) -> int:
        """Rounds the value to the closes abstract state

        Args:
            value (int): Original state value

        Raises:
            ValueError: Wrong abstraction

        Returns:
            int: Closes Abstracted state value
        """
        assert isinstance(value, np.int32)
        if self.mapping is None:
            min_distance = math.inf
            min_idx = 0
            for idx in range(len(self.all_assignments)):
                distance = abs(value - self.all_assignments[idx])
                if distance < min_distance:
                    min_idx = idx
                    min_distance = distance
            assert isinstance(self.all_assignments[min_idx], int)
            return self.all_assignments[min_idx]
        else:
            for original_values in self.mapping.keys():
                for part_value in original_values.split(","):
                    if part_value == str(int(value)):
                        assert isinstance(self.mapping[original_values], int)
                        return self.mapping[original_values]
            raise ValueError("Mapping for " + str(value) + " does not exist.")

    def set_idx(self, idx: int):
        """Set index of state variable for abstraction.

        Args:
            idx (int): index
        """
        assert isinstance(idx, int)
        self.idx = idx

    def get_idx(self) -> int:
        """Get index of state variable for abstractoin

        Returns:
            int: Index
        """
        return self.idx

    def __str__(self) -> str:
        """Abstraction State Variable to string.

        Returns:
            str: Abstraction State Variable as string.
        """
        return self.name + "=["+str(self.lower_bound)+','+str(self.upper_bound)+"]" + " Index: " + str(self.idx) + " Step Size: " + str(self.step_size) + " Rounding Method: " + self.method

    @staticmethod
    def parse_abstraction_variables(abstraction_input: str) -> List:
        """Parse abstraction variables fro abstraction input string.

        Args:
            abstraction_input ([str]): Abstraction input: VAR=[LOW;STEP;MAX;METHOD], VAR=[LOW;STEP;MAX]

        Raises:
            ValueError: Abstraction Input is the wrong format

        Returns:
            List: List of AStateVariables
        """
        assert isinstance(abstraction_input, str)
        # VAR=[LOW;STEP;MAX;METHOD], VAR=[LOW;STEP;MAX]
        astate_variables = []
        for abstraction_variable_str in abstraction_input.split(","):
            name = abstraction_variable_str.split('=')[0]
            start_interval = abstraction_variable_str.find('[')
            end_interval = abstraction_variable_str.find(']')
            interval = abstraction_variable_str[(
                start_interval+1):end_interval]
            step_size = 1
            if interval.count(';') == 2:
                parts = abstraction_variable_str[(
                    start_interval+1):end_interval].split(';')
                lower_bound = int(parts[0])
                step_size = int(parts[1])
                upper_bound = int(parts[2])
                astate_variable = AStateVariable(
                    name, lower_bound, upper_bound, step_size)
                astate_variables.append(astate_variable)
            else:
                raise ValueError(
                    "Abstraction Input is the wrong format (VAR=[LOW;STEP;MAX], VAR=[LOW;STEP;MAX])")
        assert isinstance(astate_variables, list)
        return astate_variables

    @staticmethod
    def parse_abstraction_from_dict(abstraction_mapping: dict) -> List:
        """Parse abstractions from dictionary

        Args:
            abstraction_mapping (dict): Abstraction mapping

        Returns:
            List: List of abstraction state variables
        """
        assert isinstance(abstraction_mapping, dict)
        astate_variables = []
        for state_variable_name in abstraction_mapping.keys():
            astate_variable = AStateVariable(
                state_variable_name, -1, -1, -1, mapping=abstraction_mapping[state_variable_name])
            astate_variables.append(astate_variable)
        assert isinstance(astate_variables, list)
        return astate_variables


class AbstractionManager:

    def __init__(self, state_var_mapper: StateMapper, abstraction_input: str) -> None:
        """Constructor

        Args:
            state_var_mapper (dict): State variable mapper
            abstraction_input (str): Raw abstraction input.
        """
        assert isinstance(state_var_mapper, StateMapper)
        assert isinstance(abstraction_input, str)
        self.is_active = (abstraction_input != '')
        self.state_var_mapper = state_var_mapper.mapper
        if self.is_active:
            if os.path.isfile(abstraction_input):
                with open(abstraction_input) as json_file:
                    abstraction_mapping = json.load(json_file)
                    self.astate_variables = AStateVariable.parse_abstraction_from_dict(
                        abstraction_mapping)
            else:
                self.astate_variables = AStateVariable.parse_abstraction_variables(
                    abstraction_input)

            for i in range(len(self.astate_variables)):
                self.astate_variables[i].set_idx(
                    self.state_var_mapper[self.astate_variables[i].name])

    def preprocess_state(self, state: np.ndarray) -> np.ndarray:
        """Apply the abstraction on the state.

        Args:
            state (np.ndarray): raw state

        Returns:
            np.ndarray: abstracted state
        """
        assert isinstance(state, np.ndarray)
        for astate_variable in self.astate_variables:
            idx = astate_variable.get_idx()
            state[idx] = astate_variable.round(state[idx])
        assert isinstance(state, np.ndarray)
        return state
