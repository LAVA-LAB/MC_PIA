"""This module checks the user input before the execution of COOL-MC."""
import os
from click import command
import gym

class ErrorHandler:

    def __init__(self) -> None:
        pass

    def __check_prism_file_path(self, root_dir: str, file_name: str):
        """Check if file exists

        Args:
            root_dir (str): Folder Path
            file_name (str): Filename with extension

        Raises:
            FileNotFoundError: If file does not exist.
        """
        path = os.getcwd()
        full_path = os.path.join(path,"safe_gym_training", root_dir, file_name)
        if os.path.exists(full_path) == False:
            raise FileNotFoundError("PRISM File path is not correct. You entered:" + str(full_path))


    def __check_if_openai_environment_exists(self, env: str):
        """Check if the openai environment exists

        Args:
            env (str): OpenAI Gym Environment name

        """
        try:
            gym.make(env)
        except:
            raise ValueError("OpenAI-Gym does not exist. You entered: " + str(env))


    def check_command_line_arguments_for_safe_training(self, command_line_arguments: dict) -> None:
        """Check user command line arguments for safe training

        Args:
            command_line_arguments (dict): User Command Line Arguments
        """
        # Check if PRISM File exists
        self.__check_prism_file_path(command_line_arguments['prism_dir'], command_line_arguments['prism_file_path'])


    def check_command_line_arguments_for_rl_model_checking(self, command_line_arguments: dict) -> None:
        """Check user command line arguments for safe training

        Args:
            command_line_arguments (dict): User Command Line Arguments
        """
        # Check if PRISM File exists
        self.__check_prism_file_path(command_line_arguments['prism_dir'], command_line_arguments['prism_file_path'])


    def check_command_line_arguments_for_openai_training(self, command_line_arguments: dict) -> None:
        """Check user command line arguments for open ai gym training

        Args:
            command_line_arguments (dict): User Command Line Arguments
        """
        self.__check_if_openai_environment_exists(command_line_arguments['env'])
