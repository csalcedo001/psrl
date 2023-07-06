import os
import random
import numpy as np
import torch
from accelerate import Accelerator

from psrl.config import get_env_config, get_agent_config
from psrl.utils import env_name_map, agent_name_map

from file_system import load_yaml
from arg_utils import get_parser


def setup_script():# Get experiment configuration
    # Get args
    parser = get_parser()
    args = parser.parse_args()

    # Get experiment config
    exp_config = get_experiment_config(args)

    # Setup experiment given a configuration
    accelerator = setup_experiment(exp_config)

    # Get env and agent
    env = get_environment(exp_config)
    agent = get_agent(exp_config, env)

    return exp_config, env, agent, accelerator


def get_experiment_config(args):
    config_path = args.config
    exp_config = load_yaml(config_path)

    if args.seed is not None:
        exp_config.seed = args.seed

    if args.goal_reward is not None:
        exp_config.no_goal = args.goal_reward == 0

    if not exp_config.no_goal:
        exp_config.save_path = os.path.join(exp_config.save_path, 'regret_plot')
        exp_config.plots_path = os.path.join(exp_config.plots_path, 'regret_plot')

    return exp_config


def setup_experiment(exp_config):
    # Setup seed for reproducibility
    setup_seed(exp_config.seed)
    print("*** SEED:", exp_config.seed)

    # Setup device (automated by accelerate)
    accelerator = Accelerator(project_dir=exp_config.data_dir)

    return accelerator


def setup_seed(seed: int = 0) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


def get_environment(exp_config):
    # Get environment class and configuration
    env_class = env_name_map[exp_config.env]
    env_config = get_env_config(exp_config.env)

    # Fix some parameters of the environment
    env_config['gamma'] = exp_config.gamma
    env_config['no_goal'] = exp_config.no_goal

    # Setup environment
    env = env_class(env_config)

    return env


def get_agent(exp_config, env):
    # Get agent class and configuration
    agent_class = agent_name_map[exp_config.agent]
    agent_config = get_agent_config(exp_config.agent)

    # Setup agent
    agent = agent_class(env, agent_config)

    return agent