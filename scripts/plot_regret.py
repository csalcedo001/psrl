import copy
import numpy as np
from tqdm import tqdm

from psrl.agents import OptimalAgent
from psrl.config import get_agent_config
from psrl.utils import train_episode, rollout_episode, env_name_map, agent_name_map

from parallel import ParallelRunManager
from arg_utils import get_parser, get_config


class RegretBenchmarkExperiment:
    def __init__(self, run_config, num_runs=1):
        self.run_config = run_config
        self.num_runs = num_runs
        self.results = [None] * num_runs

        self.manager = ParallelRunManager(num_runs)

    def run(self):
        for i in range(self.num_runs):
            func_args = {
                'instance_id': i
            }
            self.manager.queue(self.run_instance, func_args)
        
        self.manager.finish()

        return self.results

    def run_instance(self, func_args):
        config = self.run_config

        # Get environment
        env_class = env_name_map[config.env]
        env = env_class(config.env_config)

        # Get agent
        agent_class = agent_name_map[config.agent]
        agent = agent_class(env, config.agent_config)

        # Get optimal policy for environment
        oracle_config = get_agent_config('optimal')
        oracle = OptimalAgent(env, oracle_config)


        if config.env_config.max_steps:
            episodes = config.max_steps // env.max_steps
        else:
            episodes = config.max_steps // 10


        oracle_env = copy.deepcopy(env)

        agent_trajectories = []
        for episode in tqdm(range(episodes)):
            agent_trajectory = train_episode(env, agent)
            agent_trajectories += agent_trajectory

        oracle_trajectories = []
        for _ in tqdm(range(episodes)):
            oracle_trajectory = rollout_episode(oracle_env, oracle)
            oracle_trajectories += oracle_trajectory

        regrets = []
        regret = 0
        for t in range(len(agent_trajectories)):
            agent_reward = agent_trajectories[min(t, len(agent_trajectories) - 1)][2]
            oracle_reward = oracle_trajectories[min(t, len(oracle_trajectories) - 1)][2]

            regret += oracle_reward - agent_reward
            regrets.append(regret)

        
        self.results[func_args['instance_id']] = regrets





parser = get_parser()
args = parser.parse_args()
config = get_config(args)

num_runs = 2
experiment = RegretBenchmarkExperiment(config, num_runs)

results = experiment.run()

print(np.array(results).shape)