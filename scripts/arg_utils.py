import argparse
from dotmap import DotMap

from psrl.config import get_env_config, get_agent_config
from psrl.utils import env_name_map, agent_name_map

from runs import get_experiment_dir


def get_parser(envs=None, agents=None):
    if envs == None:
        envs = list(env_name_map.keys())
        
    if agents == None:
        agents = list(agent_name_map.keys())


    parser = argparse.ArgumentParser()

    parser.add_argument('--experiment_name', type=str, default=None, help='Experiment name to save folder')
    parser.add_argument('--agent', type=str, default=agents[0], help='Agent to use')
    parser.add_argument('--env', type=str, default=envs[0], help='Environment to use')
    parser.add_argument('--max_steps', type=int, default=10000, help='Number of episodes to run')
    parser.add_argument('--render', action='store_true', default=False, help='Render environment')
    parser.add_argument('--no-render', action='store_false', dest='render', help='Do not render environment')
    parser.add_argument('--verbose', action='store_true', default=False, help='Render environment')

    return parser


def get_config(args, envs=None, agents=None):
    if envs == None:
        envs = env_name_map.keys()
        
    if agents == None:
        agents = agent_name_map.keys()
        

    if args.env not in envs:
        raise ValueError(f'Environment not supported. Choose from {envs}.')

    if args.agent not in agents:
        raise ValueError(f'Agent not supported. Choose from {agents}.')
    
    if args.max_steps < 1:
        raise ValueError('Max number of steps must be at least 1')
    

    config = dict(vars(args))

    config['env_config'] = get_env_config(args.env)
    config['agent_config'] = get_agent_config(args.agent)

    experiment_dir = get_experiment_dir(config)
    config['experiment_dir'] = experiment_dir

    return DotMap(config)