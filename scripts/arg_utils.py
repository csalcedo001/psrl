import argparse
from dotmap import DotMap

from psrl.config import get_env_config, get_agent_config
from psrl.utils import env_name_map, agent_name_map


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--agent', type=str, default='random_agent', help='Agent to use')
    parser.add_argument('--env', type=str, default='riverswim', help='Environment to use')
    parser.add_argument('--episodes', type=int, default=100, help='Number of episodes to run')

    return parser


def get_config(args):
    config = {}

    if args.env not in env_name_map:
        raise ValueError('Environment not supported')

    if args.agent not in agent_name_map:
        raise ValueError('Agent not supported')
    
    if args.episodes < 1:
        raise ValueError('Number of episodes must be at least 1')

    config['env'] = args.env
    config['agent'] = args.agent
    config['episodes'] = args.episodes

    config['env_config'] = get_env_config(args.env)
    config['agent_config'] = get_agent_config(args.agent)

    return DotMap(config)