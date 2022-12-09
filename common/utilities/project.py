from common.rl_agents.agent_builder import AgentBuilder
from common.utilities.mlflow_bridge import MlFlowBridge

class Project():

    def __init__(self, command_line_arguments):
        self.command_line_arguments = command_line_arguments
        self.mlflow_bridge = None
        self.agent = None

    def init_mlflow_bridge(self, project_name, task, parent_run_id):
        self.mlflow_bridge = MlFlowBridge(project_name, task, parent_run_id)

    def load_saved_command_line_arguments(self):
        saved_command_line_arguments = self.mlflow_bridge.load_command_line_arguments()
        if saved_command_line_arguments != None:
            old_task = saved_command_line_arguments['task']
            try:
                del saved_command_line_arguments['prop']
            except:
                pass
            del saved_command_line_arguments['task']
            del saved_command_line_arguments['parent_run_id']
            try:
                del saved_command_line_arguments['constant_definitions']
            except:
                pass
            try:
                del saved_command_line_arguments['permissive_input']
            except:
                pass
            try:
                del saved_command_line_arguments['epsilon']
            except:
                pass
            try:
                del saved_command_line_arguments['epsilon_dec']
            except:
                pass
            try:
                del saved_command_line_arguments['epsilon_min']
            except:
                pass
            try:
                del saved_command_line_arguments['seed']
            except:
                pass
            try:
                del saved_command_line_arguments['deploy']
            except:
                pass
            try:
                del saved_command_line_arguments['num_episodes']
            except:
                pass
            try:
                del saved_command_line_arguments['eval_interval']
            except:
                pass
            try:
                del saved_command_line_arguments['prop_type']
            except:
                pass
            try:
                del saved_command_line_arguments['attack_config']
            except:
                pass
            try:
                del saved_command_line_arguments['abstract_features']
            except:
                pass
            try:
                del saved_command_line_arguments['range_plotting']
            except:
                pass

            if old_task == 'openai_gym_training':
                try:
                    del saved_command_line_arguments['prism_dir']
                except:
                    pass
                try:
                    del saved_command_line_arguments['prism_file_path']
                except:
                    pass

            for key in saved_command_line_arguments.keys():
                self.command_line_arguments[key] = saved_command_line_arguments[key]
        

    def create_agent(self, command_line_arguments, observation_space, number_of_actions):
        agent = None
        try:
            model_folder_path = self.mlflow_bridge.get_agent_path()
            # Build agent with the model and the hyperparameters
            agent = AgentBuilder.build_agent(model_folder_path, command_line_arguments, observation_space, number_of_actions)
        except Exception as msg:
            # If Model was not saved
            agent = AgentBuilder.build_agent(None, command_line_arguments, observation_space, number_of_actions)
        self.agent = agent

    def save(self):
        # Agent
        self.agent.save()
        # Save Command Line Arguments
        self.mlflow_bridge.save_command_line_arguments(self.command_line_arguments)

    def close(self):
        self.mlflow_bridge.close()
    