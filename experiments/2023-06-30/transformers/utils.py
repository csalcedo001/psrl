import yaml
from dotmap import DotMap

def load_experiment_config(path):
    with open(path, 'r') as f:
        config = yaml.load(f, yaml.FullLoader)
    
    return DotMap(config)