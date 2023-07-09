import os
import copy
import pickle

from psrl.agents import OptimalAgent
from psrl.rollout import rollout
from psrl.config import get_agent_config

from setup_script import get_parser, get_experiment_config, setup_experiment, get_environment
from plotting import save_regret_plot
from utils import get_file_path_from_config





# Get args
parser = get_parser()
args = parser.parse_args()

# Get experiment config
exp_config = get_experiment_config(args)

# Setup experiment given a configuration
_ = setup_experiment(exp_config, mode='plot')

# Get env and agent
env = get_environment(exp_config)
print(exp_config)




# Get agent data
agents = ['ucrl2', 'kl_ucrl']

agent_regrets = {}
for agent in agents:
    exp_config.agent = agent

    agent_regrets[agent] = []

    for seed in range(10):
        exp_config.seed = seed


        # Get optimal policy for environment
        oracle_env = copy.deepcopy(env)
        oracle_config = get_agent_config('optimal')
        oracle = OptimalAgent(oracle_env, oracle_config)
        oracle_trajectories = rollout(oracle_env, oracle, exp_config, max_steps=exp_config.plot_steps)



        # Load agent trajectories
        agent_trajectories_path = get_file_path_from_config('trajectories.pkl', exp_config)
        with open(agent_trajectories_path, 'rb') as f:
            agent_trajectories = pickle.load(f)[:exp_config.plot_steps]
        


        # Compute regret
        agent_rewards = [t[2] for t in agent_trajectories]
        oracle_rewards = [t[2] for t in oracle_trajectories]

        regrets = []
        regret = 0
        for t in range(exp_config.plot_steps):
            regret += oracle_rewards[t] - agent_rewards[t]
            regrets.append(regret)

        
        agent_regrets[agent].append(regrets)



# Plot regret
root = exp_config.plots_dir
experiment_dir = '{env}_{plot_steps}'.format(**exp_config)
file_dir = os.path.join(root, experiment_dir)
os.makedirs(file_dir, exist_ok=True)

save_regret_plot(
    agent_regrets,
    os.path.join(file_dir, 'regret.png'),
    title='Regret for Exploration Algorithms',
)
save_regret_plot(
    agent_regrets,
    os.path.join(file_dir, 'regret_log_log.png'),
    title='Regret for Exploration Algorithms',
    log_scale=True,
)