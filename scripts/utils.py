import os
import random
import numpy as np
import torch
import yaml
from dotmap import DotMap

def choose_gridworld_color(symbol):
    if symbol == ' ':
        color = 'w'
    elif symbol == '#':
        color = 'k'
    elif symbol == 'S':
        color = 'b'
    elif symbol == 'T':
        color = 'g'
    elif symbol == '.':
        color = '#7f7f7f'
    else:
        color = None
    
    return color

def load_experiment_config(path):
    with open(path, 'r') as f:
        config = yaml.load(f, yaml.FullLoader)
    
    return DotMap(config)

def set_seed(seed: int = 0) -> None:
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


def get_experiment_path_from_config(exp_config, mkdir=False, root_type='data'):
    if root_type == 'data':
        root = exp_config.save_path
    elif root_type == 'plots':
        root = exp_config.plots_path
    else:
        raise ValueError(f"Unknown root directory type {root_type}. Choose from 'data' or 'plots'")

    experiment_dir = '{env}_{agent}_{training_steps}_{seed:0>4}'.format(**exp_config)

    experiment_path = os.path.join(root, experiment_dir)
    if not os.path.exists(experiment_path):
        if not mkdir:
            raise ValueError(f"Experiment path {experiment_path} does not exist")
        
        os.makedirs(experiment_path, exist_ok=True)
    
    return experiment_path

def get_file_path_from_config(filename, exp_config, mkdir=False, root_type='data'):
    experiment_path = get_experiment_path_from_config(exp_config, mkdir, root_type)

    file_path = os.path.join(experiment_path, filename)

    return file_path