import time
import json
import sys
import os
import shutil
import datetime
import mlflow
from mlflow.tracking import MlflowClient
def get_current_timestemp():
    return datetime.datetime.now().timestamp()

def delete_folder_recursively(path, ts):
    """"
    Delete folder if it was created before timestemp ts
    """
    if os.path.exists(path):
        if os.path.getmtime(path) >= ts:
            shutil.rmtree(path)

def get_sub_directory_paths_of_folder(path):
    """
    Get sub directory paths of folder
    """
    return [os.path.join(path, name) for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]

def clean_folder(path, ts):
    for root_folder_path in get_sub_directory_paths_of_folder(path):
        for folder_path in os.listdir(root_folder_path):
            
            full_path = os.path.join(root_folder_path, folder_path)
            
            # Check if full_path is a folder
            if os.path.isdir(full_path):
                delete_folder_recursively(full_path, ts)

def get_metric_from_experiment_run(experiment_name, run_id, metric_name="episode_reward"):
    """
    Load metric from experiment run
    """

    #client = MlflowClient()
    #experiment_id = client.get_experiment_by_name(experiment_name).experiment_id
    client = MlflowClient()
    experiment_id = client.get_experiment_by_name(experiment_name).experiment_id
    metric_path = os.path.join("..", "mlruns", str(experiment_id), str(run_id), "metrics", metric_name)
    # Read first line in file
    with open(metric_path, "r") as f:
        line = f.readline()
        try:
            return float(line.split(" ")[1])
        except:
            raise Exception("Could not find metric in experiment run" + str(experiment_id) +  str(run_id))

def save_dictionary_and_integer_into_file(path: str, dictionary: dict):
    # save dictionary into csv
    with open(path, 'w') as fp:
        for key, value in dictionary.items():
            fp.write(str(key) + "," + str(value) + "\n")


def read_meta_data_from_drn(file_path: str):
    # Read lines from file file_path
    with open(file_path, "r") as f:
        lines = f.readlines()
        for idx, line in enumerate(lines):
            if line.startswith('@nr_states'):
                nr_states = int(lines[idx+1].replace('\n', ''))
            elif line.startswith('@nr_choices'):
                nr_transitions = int(lines[idx+1].replace('\n', ''))
                break
        return nr_states, nr_transitions

    
#print(load_metric_from_experiment_run(3,"b2fa282396584151a1595016f38b7a1e"))