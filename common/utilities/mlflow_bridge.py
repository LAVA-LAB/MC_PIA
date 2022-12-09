import os
import random
import string
import shutil
import json
import mlflow
import time
from mlflow.tracking import MlflowClient
from distutils.dir_util import copy_tree


class MlFlowBridge:

    def __init__(self, project_name, task, parent_run_id):
        self.experiment_name = project_name
        self.experiment = None
        self.task = task
        self.parent_run_id = parent_run_id
        self.client = MlflowClient()
        #mlflow.set_tracking_uri(self.project_dir)
        try:
            experiment_id = self.client.create_experiment(self.experiment_name)
        except:
            experiment_id = self.client.get_experiment_by_name(self.experiment_name).experiment_id
        self.experiment = self.client.get_experiment(experiment_id)
        self.create_new_run(self.task, self.parent_run_id)
        

    def create_new_run(self, task, run_id=None):
        if run_id == None or run_id == '':
            # Create new Run
            print("Create new run.")
            self.run = self.client.create_run(self.experiment.experiment_id)
            self.client.set_tag(self.run.info.run_id, "task", task)
        else:
            # Choose already existing run
            print("Choose already existing run.")
            run = mlflow.get_run(run_id)
            self.run = self.__copy_run(self.experiment, run)
            
        
        mlflow.start_run(self.run.info.run_id)
       

    def set_property_query_as_run_name(self, prop):
        mlflow.tracking.MlflowClient().set_tag(self.run.info.run_id, "mlflow.runName", prop)

    def __copy_run(self, experiment, run):
        # Find unique run_id
        exists = True
        new_run_id = ''.join(random.choices(string.ascii_lowercase + string.digits, k=len(run.info.run_id)+10))
        run_path = os.path.join('../mlruns', experiment.experiment_id, run.info.run_id)
        new_run_path = os.path.join('../mlruns', experiment.experiment_id, new_run_id)
        while exists:
            new_run_id = ''.join(random.choices(string.ascii_lowercase + string.digits, k=len(run.info.run_id)+10))
            new_run_path = os.path.join('../mlruns', experiment.experiment_id, new_run_id)
            exists = os.path.exists(new_run_path)
        copy_tree(run_path, new_run_path)
        # Modify Meta
        f = open(os.path.join(new_run_path,'meta.yaml'),'r')
        lines = f.readlines()
        f.close()
        lines[0] = lines[0].replace(run.info.run_id, new_run_id)
        lines[6] = lines[6].replace(run.info.run_id, new_run_id)
        lines[7] = lines[7].replace(run.info.run_id, new_run_id)
        lines[7] = lines[7].replace(run.info.run_id, new_run_id)
        lines[11] = 'start_time: ' + str(time.time()*1000).split('.')[0] + '\n'
        f = open(os.path.join(new_run_path,'meta.yaml'),'w')
        f.writelines(lines)
        f.close()
        # Update task
        f = open(os.path.join(new_run_path,'tags','task'),'w')
        f.write(self.task)
        f.close()
        # Delete already existing metrics
        metrics_path = os.path.join(new_run_path, 'metrics')
        shutil.rmtree(metrics_path)
        os.mkdir(metrics_path)
        # Delete already existing params
        params_path = os.path.join(new_run_path, 'params')
        shutil.rmtree(params_path)
        os.mkdir(params_path)
        # Get run
        run = mlflow.get_run(new_run_id)
        return run


    def save_command_line_arguments(self, command_line_arguments):
        with open("command_line_arguments.json", 'w') as f:
            json.dump(command_line_arguments, f, indent=2)
        # Command Line Arguments
        mlflow.log_artifact("command_line_arguments.json", artifact_path="meta")
        os.remove('command_line_arguments.json')


    def load_command_line_arguments(self):
        meta_folder_path = mlflow.get_artifact_uri(artifact_path="meta").replace('/file:/','')
        # If rerun, take all the command line arguments from previous run into account except the following:
        command_line_arguments_file_path = os.path.join(meta_folder_path, 'command_line_arguments.json')[5:]
        if os.path.exists(command_line_arguments_file_path):
            with open(command_line_arguments_file_path) as json_file:
                command_line_arguments = json.load(json_file)
                #print(command_line_arguments)
                return command_line_arguments
        return None

    def get_agent_path(self):
        model_folder_path = mlflow.get_artifact_uri(artifact_path="model").replace('file:///home/','/home/')
        return model_folder_path

    def get_run_id(self):
        return self.get_agent_path().split('/')[-3]

    def get_project_id(self):
        return self.get_agent_path().split('/')[-4] 

    def log_reward(self, reward, episode):
        mlflow.log_metric(key='episode_reward', value=reward, step= episode)

    def log_best_reward(self, reward):
        mlflow.log_param("best_sliding_window_reward", reward)

    def log_avg_reward(self, avg_reward, episode):
        mlflow.log_metric(key='avg_reward', value=avg_reward, step=episode)

    def log_property(self, property_result, property_query, episode):
        mlflow.log_metric(key=property_query, value=property_result, step= episode)

    def log_best_property_result(self, best_property_result, prop=None):
        if prop == None:
            mlflow.log_param("Best_Property_Result", best_property_result)
        else:
            mlflow.log_param(prop, best_property_result)


    def log_accuracy(self, acc):
        mlflow.log_param("Decision_Tree_Accuracy", acc)

    def close(self):
        mlflow.end_run() 

