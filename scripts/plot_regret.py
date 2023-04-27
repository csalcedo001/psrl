import os
import copy
import time
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


from psrl import train, rollout
from psrl.agents import OptimalAgent
from psrl.config import get_agent_config, save_config
from psrl.utils import train_episode, rollout_episode, env_name_map, agent_name_map

from parallel import ParallelRunManager
from arg_utils import get_parser, get_config
from plot_utils import env_plot_name_map, agent_plot_name_map


class RegretBenchmarkExperiment:
    def __init__(
            self,
            run_config,
            agents,
            runs_per_agent=1,
            max_parallel_runs=1,
            retry_freq=5
        ):

        self.run_config = run_config
        self.agents = agents
        self.num_runs = runs_per_agent
        self.results = {agent: [None] * runs_per_agent for agent in agents}
        self.retry_freq = retry_freq

        self.manager = ParallelRunManager(max_parallel_runs)

    def run(self):
        for agent in agents:
            i = 0
            while True:
                if i >= self.num_runs:
                    break

                func_args = {
                    'agent': agent,
                    'run_id': i,
                }

                if not self.manager.queue(self.run_instance, func_args):
                    time.sleep(self.retry_freq)
                else:
                    i += 1
        
        self.manager.finish()

        return self.results

    def run_instance(self, func_args):
        config = self.run_config
        config['agent'] = func_args['agent']

        # Get environment
        env_class = env_name_map[config.env]
        env = env_class(config.env_config)

        # Get agent
        agent_class = agent_name_map[config.agent]
        agent = agent_class(env, config.agent_config)

        # Get optimal policy for environment
        oracle_env = copy.deepcopy(env)
        oracle_config = get_agent_config('optimal')
        oracle = OptimalAgent(oracle_env, oracle_config)


        agent_trajectories = train(env, agent, config)
        oracle_trajectories = rollout(env, oracle, config)

        regrets = []
        regret = 0
        for t in range(len(agent_trajectories)):
            agent_reward = agent_trajectories[min(t, len(agent_trajectories) - 1)][2]
            oracle_reward = oracle_trajectories[min(t, len(oracle_trajectories) - 1)][2]

            regret += oracle_reward - agent_reward
            regrets.append(regret)

        
        self.results[func_args['agent']][func_args['run_id']] = regrets





# Get experiment config
parser = get_parser()
args = parser.parse_args()
config = get_config(args)
save_config(config.toDict(), config.experiment_dir)


# Run experiment and get results
agents = ['psrl', 'random_agent']
runs_per_agent = 2
max_parallel_runs = 1

# TODO: fix parallelization, it's slower than serial execution...
# do so by using multiprocessing instead of multithreading
experiment = RegretBenchmarkExperiment(config, agents, runs_per_agent, max_parallel_runs)

results = experiment.run()



# Make plots from results
filename = os.path.join(config.experiment_dir, 'regret.png')

figure = plt.figure()

plt.title(f'Regret for {env_plot_name_map[config.env]} environment')

for agent in agents:
    mean_regret = np.mean(results[agent], axis=0)
    min_regret = np.min(results[agent], axis=0)
    max_regret = np.max(results[agent], axis=0)

    x_index = np.arange(len(mean_regret))

    plt.fill_between(x_index, min_regret, max_regret, alpha=0.5)
    plt.plot(x_index, mean_regret, label=agent_plot_name_map[agent])

plt.legend()
plt.savefig(filename)
plt.close()