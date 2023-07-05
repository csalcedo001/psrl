import yaml
import json
import pickle
from dotmap import DotMap

def load_yaml(path):
    with open(path, 'r') as f:
        obj = yaml.load(f, yaml.FullLoader)
    
    return DotMap(obj)

def save_yaml(obj, path):
    with open(path, 'w') as f:
        yaml.dump(obj, f)

def load_json(path):
    with open(path, 'r') as f:
        config = json.load(f)
    
    return DotMap(config)

def save_json(obj, path):
    with open(path, 'w') as f:
        json.dump(obj, f, indent=4)

def load_pickle(path):
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    
    return obj

def save_pickle(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def save_obj(obj, path):
    if path.endswith('.yaml'):
        save_yaml(obj, path)
    elif path.endswith('.json'):
        save_json(obj, path)
    elif path.endswith('.pkl'):
        save_pickle(obj, path)
    else:
        raise ValueError(f"Unknown file format for path {path}")

def load_obj(path):
    if path.endswith('.yaml'):
        obj = load_yaml(path)
    elif path.endswith('.json'):
        obj = load_json(path)
    elif path.endswith('.pkl'):
        obj = load_pickle(path)
    else:
        raise ValueError(f"Unknown file format for path {path}")
    
    return obj