import os
import yaml
import json
import pickle
from dotmap import DotMap


def get_env_config(env_name):
    config_path = os.path.join(os.path.dirname(__file__), 'envs', env_name + '.yaml')

    config = load_config(config_path)

    return DotMap(config)


def get_agent_config(env_name):
    config_path = os.path.join(os.path.dirname(__file__), 'agents', env_name + '.yaml')

    config = load_config(config_path)

    return DotMap(config)


def load_config(config_path):
    _, extension = os.path.splitext(config_path)

    if extension not in ['.json', '.yaml', '.yml', '.pkl']:
        raise ValueError('Config file must be either json, yaml, or pickle')
    
    if extension == '.json':
        with open(config_path, 'r') as f:
            config = json.load(f)
    elif extension in ['.yaml', '.yml']:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    elif extension == '.pkl':
        with open(config_path, 'rb') as f:
            config = pickle.load(f)

    return config

def save_config(config, config_dir, extension='yaml'):
    extension = '.' + extension.lstrip('.')

    if extension not in ['.json', '.yaml', '.yml', '.pkl']:
        raise ValueError('Config file must be either json, yaml, or pickle')
    
    config_path = os.path.join(config_dir, 'config' + extension)
    
    # TODO: sort keys for readibility
    
    if extension == '.json':
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
    elif extension in ['.yaml', '.yml']:
        with open(config_path, 'w') as f:
            yaml.safe_dump(config, f)
    elif extension == '.pkl':
        with open(config_path, 'wb') as f:
            pickle.dump(config, f)