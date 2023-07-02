import os
from tqdm import tqdm
import pickle

from psrl.config import get_env_config, get_agent_config
from psrl.train import train
from psrl.utils import env_name_map, agent_name_map

from arg_utils import get_experiment_parser
from utils import load_experiment_config, set_seed, get_file_path_from_config




# Get experiment configuration
parser = get_experiment_parser()
args = parser.parse_args()
config_path = args.config
exp_config = load_experiment_config(config_path)



# Setup experiment
seed = exp_config.seed
if args.seed is not None:
    seed = args.seed
set_seed(seed)
print("*** SEED:", seed)

no_goal = exp_config.no_goal
if args.goal_reward is not None:
    no_goal = args.goal_reward == 0
    
if not no_goal:
    exp_config.save_dir = os.path.join(exp_config.save_dir, 'regret_plot')



# Get environment
env_class = env_name_map[exp_config.env]
env_config = get_env_config(exp_config.env)
env_config['gamma'] = exp_config.gamma
env_config['no_goal'] = no_goal
env = env_class(env_config)



# Get agent
agent_class = agent_name_map[exp_config.agent]
agent_config = get_agent_config(exp_config.agent)
agent = agent_class(env, agent_config)



trajectory = train(env, agent, exp_config, max_steps=exp_config.training_steps)




# Save agent
agent_path = get_file_path_from_config('agent.pkl', exp_config, mkdir=True)
agent.save(agent_path)

# Save trajectories
trajectories_path = get_file_path_from_config('trajectories.pkl', exp_config, mkdir=True)
with open(trajectories_path, 'wb') as f:
    pickle.dump(trajectory, f)