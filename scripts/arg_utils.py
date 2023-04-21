import argparse
from dotmap import DotMap

from psrl.config import get_env_config, get_agent_config
from psrl.utils import env_name_map, agent_name_map


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--agent', type=str, default='random_agent', help='Agent to use')
    parser.add_argument('--env', type=str, default='riverswim', help='Environment to use')
    parser.add_argument('--max_steps', type=int, default=10000, help='Number of episodes to run')
    parser.add_argument('--render', action='store_true', default=False, help='Render environment')
    parser.add_argument('--no-render', action='store_false', dest='render', help='Do not render environment')
    parser.add_argument('--verbose', action='store_true', default=False, help='Render environment')

    return parser


def get_config(args):
    if args.env not in env_name_map:
        raise ValueError('Environment not supported')

    if args.agent not in agent_name_map:
        raise ValueError('Agent not supported')
    
    if args.max_steps < 1:
        raise ValueError('Max number of steps must be at least 1')
    

    config = dict(vars(args))

    config['env_config'] = get_env_config(args.env)
    config['agent_config'] = get_agent_config(args.agent)

    return DotMap(config)