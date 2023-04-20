
import argparse
from dotmap import DotMap
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

    config['agent'] = args.agent
    config['env'] = args.env
    config['episodes'] = args.episodes

    return DotMap(config)