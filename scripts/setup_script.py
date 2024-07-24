import os
import re
import json
import random
import numpy as np
import torch
from dotenv import load_dotenv
from accelerate import Accelerator
import wandb
import hashlib

from psrl.config import get_env_config, get_agent_config
from psrl.utils import env_name_map, agent_name_map

from file_system import load_yaml
from arg_utils import get_parser


def setup_script(mode='run'):
    # Get args
    parser = get_parser()
    args = parser.parse_args()

    # Get experiment config
    exp_config = get_experiment_config(args)

    # Setup experiment given a configuration
    accelerator = setup_experiment(exp_config, mode=mode)

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
    
    if args.batch_size is not None:
        exp_config.batch_size = args.batch_size

    if args.lr is not None:
        exp_config.lr = args.lr
    
    if args.max_lr is not None:
        exp_config.lr_scheduler.max_lr = args.max_lr
    
    if args.pct_start is not None:
        exp_config.lr_scheduler.pct_start = args.pct_start
    
    if args.anneal_strategy is not None:
        exp_config.lr_scheduler.anneal_strategy = args.anneal_strategy
    
    if args.beta1 is not None:
        exp_config.adam.betas[0] = args.beta1
    
    if args.beta2 is not None:
        exp_config.adam.betas[1] = args.beta2
    
    if args.weight_decay is not None:
        exp_config.adam.weight_decay = args.weight_decay
    

    
    if args.exp_naming_strategy is None:
        args.exp_naming_strategy = 'default'
    if args.exp_naming_strategy not in ['default', 'sweep']:
        raise ValueError("Invalid experiment naming strategy. Must be one of: ['default', 'sweep']")
    exp_config.exp_naming_strategy = args.exp_naming_strategy

    
    data_dir, plots_dir = load_dirs_from_env()

    exp_config.data_dir = data_dir
    exp_config.plots_dir = plots_dir

    if not exp_config.no_goal:
        exp_config.data_dir = os.path.join(exp_config.data_dir, 'regret_plot')
        exp_config.plots_dir = os.path.join(exp_config.plots_dir, 'regret_plot')
    

    # Unique config identifier
    exp_config.hash = hashlib.md5(json.dumps(exp_config.toDict()).encode()).hexdigest()[:8]


    return exp_config


def setup_experiment(exp_config, mode='run'):
    # Setup seed for reproducibility
    setup_seed(exp_config.seed)
    print("*** SEED:", exp_config.seed)

    # Setup device (automated by accelerate)
    accelerator = Accelerator(project_dir=exp_config.data_dir)

    # Init wandb only if env variables provided
    env_vars = list(dict(os.environ).keys())
    wandb_env_var_pattern = re.compile("WANDB_*")
    wandb_env_var_matches = list(filter(wandb_env_var_pattern.match, env_vars))

    has_wandb_envs = len(wandb_env_var_matches) != 0
    if has_wandb_envs and mode == 'run':
        wandb.init(config=exp_config.toDict())
    else:
        wandb.init(mode="disabled")

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

def load_dirs_from_env():
    load_dotenv()

    if 'DATA_DIR' not in os.environ:
        raise ValueError("DATA_DIR must be set as environment variable")

    if 'PLOTS_DIR' not in os.environ:
        raise ValueError("PLOTS_DIR must be set as environment variable")

    # Directories MUST be set as environment variables
    data_dir = os.getenv("DATA_DIR")
    plots_dir = os.getenv("PLOTS_DIR")

    return data_dir, plots_dir