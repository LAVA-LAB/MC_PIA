"""This module manages the verification of the RL policy."""
import os
from typing import Dict, Any, List
from common.utilities.project import Project
from common.utilities.training import train
from common.safe_gym.safe_gym import SafeGym
from common.safe_gym.constant_definition_parser import ConstantDefinitionParser
import numpy as np
import matplotlib.pyplot as plt
from common.utilities.front_end_printer import *


def run_verify_rl_agent(command_line_arguments: Dict[str, Any]) -> List[float]:
    """Runs the verification task.

    Args:
        command_line_arguments (Dict[str,Any]): Command Line Arguments.

    Raises:
        ValueError: Wrong Format.

    Returns:
        List[float]: The list of property results
    """
    command_line_arguments['task'] = 'rl_model_checking'
    m_project = Project(command_line_arguments)
    m_project.init_mlflow_bridge(
        command_line_arguments['project_name'], command_line_arguments['task'], command_line_arguments['parent_run_id'])
    m_project.load_saved_command_line_arguments()
    m_project.command_line_arguments['disabled_features'] = command_line_arguments['disabled_features']
    prism_file_path = os.path.join(
        m_project.command_line_arguments['prism_dir'], m_project.command_line_arguments['prism_file_path'])
    print(m_project.command_line_arguments)
    if command_line_arguments['constant_definitions'].count('[') == 0 and command_line_arguments['constant_definitions'].count(']') == 0:
        if command_line_arguments['permissive_input']=="":
            query = command_line_arguments['prop']
            # Insert min at second position
            operator_str = query[:1]
            min_part = "min"
            command_line_arguments['prop'] = operator_str + min_part + query[1:]
        env = SafeGym(prism_file_path, m_project.command_line_arguments['constant_definitions'], 1, 1, False, command_line_arguments['seed'], command_line_arguments[
                      'permissive_input'],  m_project.command_line_arguments['disabled_features'], abstraction_input=m_project.command_line_arguments['abstract_features'],attack_config=command_line_arguments['attack_config'])
        m_project.create_agent(command_line_arguments,
                               env.observation_space, env.action_space)
        mdp_reward_result, model_size = env.storm_bridge.model_checker.induced_markov_chain(
            m_project.agent, env, command_line_arguments['constant_definitions'], command_line_arguments['prop'])
        print(command_line_arguments['prop'], ':', mdp_reward_result)
        print("Model Size", model_size)
        m_project.mlflow_bridge.log_best_property_result(mdp_reward_result)
        FrontEndPrinter.write_verification_result(m_project.mlflow_bridge.get_project_id(
        ), m_project.mlflow_bridge.get_run_id(), command_line_arguments['prop'], mdp_reward_result)
        if command_line_arguments['permissive_input'] == '':
            command_line_arguments['prop'] = command_line_arguments['prop'].replace(
                "max", "").replace("min", "")
        m_project.mlflow_bridge.set_property_query_as_run_name(
            command_line_arguments['prop'] + " for " + command_line_arguments['constant_definitions'])
        m_project.save()
        m_project.close()
        return [mdp_reward_result]
    elif command_line_arguments['constant_definitions'].count('[') == 1 and command_line_arguments['constant_definitions'].count(']') == 1:
        # For each step make model checking and save results in list
        all_constant_definitions, range_tuple, range_state_variable = ConstantDefinitionParser.parse_constant_definition(
            command_line_arguments['constant_definitions'])
        all_prop_results = []
        first = True
        for constant_definitions in all_constant_definitions:
            if command_line_arguments['permissive_input']=="" and first:
                query = command_line_arguments['prop']
                # Insert min at second position
                operator_str = query[:1]
                min_part = "min"
                command_line_arguments['prop'] = operator_str + min_part + query[1:]
                first = False
            env = SafeGym(prism_file_path, constant_definitions, 1, 1, False, command_line_arguments['seed'], command_line_arguments[
                          'permissive_input'],  m_project.command_line_arguments['disabled_features'], abstraction_input=m_project.command_line_arguments['abstract_features'],attack_config=command_line_arguments['attack_config'])
            m_project.create_agent(
                command_line_arguments, env.observation_space, env.action_space)
            mdp_reward_result, model_size = env.storm_bridge.model_checker.induced_markov_chain(
                m_project.agent, env, constant_definitions, command_line_arguments['prop'])
            print("Constant Definitions:", constant_definitions)
            print(command_line_arguments['prop'], ':', mdp_reward_result)

            all_prop_results.append(mdp_reward_result)
            # Plot results
            # Data for plotting
        if command_line_arguments['range_plotting']:
            x = np.arange(range_tuple[0], range_tuple[2], range_tuple[1])
            y = np.array(all_prop_results)

            fig, ax = plt.subplots()
            ax.plot(x, y)

            ax.set(xlabel=range_state_variable, ylabel=command_line_arguments['prop'].replace("max", "").replace(
                "min", ""), title='Property Results over a range of different constant assignments (' + str(range_state_variable) + ')')
            ax.grid()

            #fig.savefig(os.path.join(project_folder, "properties.png"))
            plt.show()
        m_project.close()
        return all_prop_results
    else:
        m_project.close()
        raise ValueError("We only support plotting for one state variable...")
