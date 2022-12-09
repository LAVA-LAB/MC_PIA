import sys
sys.path.insert(0, '../')
from common.tasks.verify_rl_agent import *
from common.tasks.helper import *
import time



if __name__ == '__main__':
    command_line_arguments = get_arguments()
    if command_line_arguments['permissive_input'].startswith("robustness,"):
        start_time = time.time()
        command_line_arguments['prop'] = command_line_arguments['prop'].replace("min", "max")
        max_result = run_verify_rl_agent(command_line_arguments)[0]
        command_line_arguments['permissive_input'] = ""
        command_line_arguments['prop'] = command_line_arguments['prop'].replace("max", "")
        print(command_line_arguments['prop'])
        original_result = run_verify_rl_agent(command_line_arguments)[0]
        end_time = time.time() - start_time
        print("Original:", original_result)
        print("MAX:", max_result)
        print("Robustness:", max_result - original_result)
        print("Time:", end_time)
    elif command_line_arguments['permissive_input'].startswith("robustness_min,"):
        start_time = time.time()
        #command_line_arguments['prop'] = command_line_arguments['prop'].replace("max", "min")
        min_result = run_verify_rl_agent(command_line_arguments)[0]
        command_line_arguments['permissive_input'] = ""
        command_line_arguments['prop'] = command_line_arguments['prop'].replace("min", "")
        print(command_line_arguments['prop'])
        original_result = run_verify_rl_agent(command_line_arguments)[0]
        end_time = time.time() - start_time
        print("Original:", original_result)
        print("MIN:", min_result)
        print("Robustness:", original_result-min_result)
        print("Time:", end_time)
    
    else:
        run_verify_rl_agent(command_line_arguments)
