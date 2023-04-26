import os
import copy
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from psrl.agents import OptimalAgent
from psrl.config import get_agent_config, save_config
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





# Get experiment config
parser = get_parser()
args = parser.parse_args()
config = get_config(args)
save_config(config.toDict(), config.experiment_dir)


# Run experiment and get results
num_runs = 2
experiment = RegretBenchmarkExperiment(config, num_runs)

results = experiment.run()


# Make plots from results
filename = os.path.join(config.experiment_dir, 'regret.png')

mean_regret = np.mean(results, axis=0)
min_regret = np.min(results, axis=0)
max_regret = np.max(results, axis=0)

x_index = np.arange(len(mean_regret))

figure = plt.figure()
plt.fill_between(x_index, min_regret, max_regret, alpha=0.5, label='min/max regret')
plt.plot(x_index, mean_regret, label='mean regret')
plt.savefig(filename)
plt.close()