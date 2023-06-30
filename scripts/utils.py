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