import sys
sys.path.insert(0, '../')
from common.tasks.helper import *
from common.tasks.verify_rl_agent import *
from common.tasks.safe_gym_training import run_safe_gym_training
from common.tasks.helper import *
from helper import *
import time

def empirical_static_reward_impact_map(command_line_arguments):
    results = {"task":"empirical_static_reward_impact_map"}
    start_time = time.time()
    start_timestemp = get_current_timestemp()
    results["env"] = command_line_arguments['prism_file_path']
    results['attack_strings'] = command_line_arguments['abstract_features']
    sample_size = command_line_arguments['range_plotting']
    attack_strs = command_line_arguments['abstract_features'].split(';')
    command_line_arguments['abstract_features'] = ""
    rewards = []
    for i in range(sample_size):
        command_line_arguments['attack_config'] = ""
        command_line_arguments['abstract_features'] = ""
        command_line_arguments['task'] = "safe_training"
        command_line_arguments['num_episodes'] = 1
        command_line_arguments['deploy'] = 1
        command_line_arguments['prop'] = ""
        #print(command_line_arguments['attack_config'])
        #command_line_arguments['attack_config'] = ""
        run_id = run_safe_gym_training(command_line_arguments)
        rewards.append(get_metric_from_experiment_run(command_line_arguments['project_name'], run_id))
        clean_folder("../mlruns",start_timestemp)
        #rs.append(last_reward)
        #print(last_reward)
    expected_reward = sum(rewards)/len(rewards)
    results['expected_reward'] = expected_reward
    for attack_str in attack_strs:
        adv_rewards = []
        for i in range(sample_size):
            command_line_arguments['attack_config'] = attack_str
            command_line_arguments['abstract_features'] = ""
            command_line_arguments['task'] = "safe_training"
            command_line_arguments['num_episodes'] = 1
            command_line_arguments['deploy'] = 1
            command_line_arguments['prop'] = ""
            #command_line_arguments['reward_flag'] = 1
            #print(command_line_arguments['attack_config'])
            #command_line_arguments['attack_config'] = ""
            run_id = run_safe_gym_training(command_line_arguments)
            adv_rewards.append(get_metric_from_experiment_run(command_line_arguments['project_name'], run_id))
            clean_folder("../mlruns",start_timestemp)
            #rs.append(last_reward)
            #print(last_reward)
        adv_expected_reward = sum(adv_rewards)/len(adv_rewards)
        results[attack_str+"_distance"] = abs(expected_reward - adv_expected_reward)
    print(results)
    results['sample_size'] = sample_size
    results['running_time'] = time.time() - start_time
    save_dictionary_and_integer_into_file(str(get_current_timestemp()) + "_results.csv", results)

def analytical_static_reward_impact_map(command_line_arguments):
    results = {"task":"analytical_static_reward_impact_map"}
    start_time = time.time()
    start_timestemp = get_current_timestemp()
    results['attack_strings'] = command_line_arguments['abstract_features']
    sample_size = command_line_arguments['range_plotting']
    attack_strs = command_line_arguments['abstract_features'].split(';')
    command_line_arguments['abstract_features'] = ""
    
    r = run_verify_rl_agent(command_line_arguments)
    #print(r[0])
    #exit(0)
    results["original_result"] = r[0]
    states, transitions = read_meta_data_from_drn("test.drn")
    results["original_model_size"] = states
    results["original_transition_size"] = transitions
    results["prop"] = command_line_arguments['prop']
    
    model_sizes = []
    transition_sizes = []
    for attack_str in attack_strs:
        print("ATTACK", attack_str)
        command_line_arguments['attack_config'] = attack_str
        r = run_verify_rl_agent(command_line_arguments)
        clean_folder("../mlruns",start_timestemp)
        prop_result = r[0]
        states, transitions = read_meta_data_from_drn("test.drn")
        model_sizes.append(states)
        transition_sizes.append(transitions)
        results[attack_str+"_distance"] = results["original_result"] - prop_result
    results['running_time'] = time.time() - start_time
    results['average_model_size'] = sum(model_sizes)/len(model_sizes)
    results['average_transitions'] = sum(transition_sizes)/len(transition_sizes)
    save_dictionary_and_integer_into_file(str(get_current_timestemp()) + "_results.csv", results)


if __name__ == '__main__':
    command_line_arguments = get_arguments()
    method = command_line_arguments['permissive_input']
    command_line_arguments['permissive_input'] = ""
    if method == "empirical_srim":
        empirical_static_reward_impact_map(command_line_arguments)
    elif method == "analytical_srim":
        analytical_static_reward_impact_map(command_line_arguments)
    else:
        raise "Unknown method"
    