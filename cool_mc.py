import mlflow
from common.tasks.helper import *
from common.utilities.error_handler import *


if __name__ == '__main__':
    args = get_arguments()
    m_error_handler = ErrorHandler()
    set_random_seed(args['seed'])


    if args['task'] == 'safe_training':
        m_error_handler.check_command_line_arguments_for_safe_training(dict(args))
        mlflow.run(
            "safe_gym_training",
            use_conda=False,
            parameters=dict(args)
        )
    elif args['task'] == 'openai_training':
        m_error_handler.check_command_line_arguments_for_openai_training(dict(args))
        mlflow.run(
            "openai_gym_training",
            parameters=dict(args),
            use_conda=False
        )
    elif args['task'] == 'rl_model_checking':
        m_error_handler.check_command_line_arguments_for_rl_model_checking(dict(args))
        mlflow.run(
            "verify_rl_agent",
            parameters=dict(args),
            use_conda=False
        )