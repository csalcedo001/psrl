import os
import argparse
from dotmap import DotMap

from psrl.config import get_env_config, get_agent_config
from psrl.utils import env_name_map, agent_name_map

from runs import get_experiment_config


def get_parser(envs=None, agents=None, data_dir=False):
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
    parser.add_argument('--no_render', action='store_false', dest='render', help='Do not render environment')
    parser.add_argument('--verbose', action='store_true', default=False, help='Render environment')

    if data_dir:
        parser.add_argument('--data_dir', type=str, default=None, help='Data dir to load')

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
    

    if type(args) == argparse.Namespace:
        config = dict(vars(args))
    elif type(args) == DotMap:
        config = args.toDict()
    elif type(args) == dict:
        config = args

    config['env_config'] = get_env_config(args.env)
    config['agent_config'] = get_agent_config(args.agent)
    config['agent_config'] = get_agent_config(args.agent)

    experiment_conf = get_experiment_config(config)
    config['experiment_name'] = experiment_conf['experiment_name']
    config['experiment_dir'] = experiment_conf['experiment_dir']


    return DotMap(config)

def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str, default=None, help='Path to experiment configuration file', required=True)
    parser.add_argument('--exp-naming-strategy', type=str, default=None, help='Strategy to choose naming convention for experiments')
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    parser.add_argument('--goal-reward', type=int, default=None, help='Reward for achieving goal')
    parser.add_argument('--batch-size', type=int, default=None, help='Training batch size')
    parser.add_argument('--lr', type=float, default=None, help='Training learning rate')
    parser.add_argument('--max-lr', type=float, default=None, help='Training maximum learning rate for OneCycleLR')
    parser.add_argument('--anneal-strategy', type=str, default=None, help='Training anneal strategy for OneCycleLR')
    parser.add_argument('--beta1', type=float, default=None, help='Training beta1 for ADAM optimizer')
    parser.add_argument('--beta2', type=float, default=None, help='Training beta2 for ADAM optimizer')
    parser.add_argument('--weight-decay', type=float, default=None, help='Training weight decay ADAM optimizer')

    return parser


def process_experiment_config(args, exp_config):
    if args.seed is not None:
        exp_config.seed = args.seed

    if args.goal_reward is not None:
        exp_config.no_goal = args.goal_reward == 0

    if not exp_config.no_goal:
        exp_config.data_dir = os.path.join(exp_config.data_dir, 'regret_plot')
        exp_config.plots_dir = os.path.join(exp_config.plots_dir, 'regret_plot')

    return exp_config