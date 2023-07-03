import os
import copy
import pickle

from psrl.agents import OptimalAgent
from psrl.rollout import rollout
from psrl.config import get_env_config, get_agent_config
from psrl.utils import env_name_map

from arg_utils import get_experiment_parser, process_experiment_config
from plotting import save_regret_plot
from utils import load_experiment_config, set_seed, get_file_path_from_config






# Get experiment configuration
parser = get_experiment_parser()
args = parser.parse_args()
config_path = args.config
exp_config = load_experiment_config(config_path)
exp_config = process_experiment_config(args, exp_config)



# Setup experiment
set_seed(exp_config.seed)
print("*** SEED:", exp_config.seed)



# Get environment
env_class = env_name_map[exp_config.env]
env_config = get_env_config(exp_config.env)
env_config['gamma'] = exp_config.gamma
env_config['no_goal'] = exp_config.no_goal
env = env_class(env_config)








# Get agent data
agents = ['psrl', 'ucrl2', 'kl_ucrl']

agent_regrets = {
    'psrl': [],
    'ucrl2': [],
    'kl_ucrl': []
}
for agent in agents:
    exp_config.agent = agent

    for seed in range(3):
        exp_config.seed = seed


        # Get optimal policy for environment
        oracle_env = copy.deepcopy(env)
        oracle_config = get_agent_config('optimal')
        oracle = OptimalAgent(oracle_env, oracle_config)
        oracle_trajectories = rollout(oracle_env, oracle, exp_config, max_steps=exp_config.training_steps)



        # Load agent trajectories
        agent_trajectories_path = get_file_path_from_config('trajectories.pkl', exp_config)
        with open(agent_trajectories_path, 'rb') as f:
            agent_trajectories = pickle.load(f)
        


        # Compute regret
        agent_rewards = [t[2] for t in agent_trajectories]
        oracle_rewards = [t[2] for t in oracle_trajectories]

        regrets = []
        regret = 0
        for t in range(exp_config.training_steps):
            regret += oracle_rewards[t] - agent_rewards[t]
            regrets.append(regret)

        
        agent_regrets[agent].append(regret)



# Plot regret
filename = 'regret.png'
root = exp_config.plots_path
experiment_dir = '{env}_{training_steps}'.format(**exp_config)

file_path = os.path.join(root, experiment_dir, filename)

save_regret_plot(
    agent_trajectories,
    file_path,
    title='Exploration Regret',
)