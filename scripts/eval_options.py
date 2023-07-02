import os
import torch
from torch.utils.data import DataLoader
from transformers import GPT2Config, GPT2LMHeadModel
import pickle
from tqdm import tqdm
from accelerate import Accelerator

from psrl.agents import RandomAgent
from psrl.config import get_env_config, get_agent_config
from psrl.utils import env_name_map, agent_name_map
from psrl.rollout import rollout

from arg_utils import get_experiment_parser
from options import OptionEnvWrapper
from utils import load_experiment_config, set_seed, get_file_path_from_config, get_experiment_path_from_config






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
    exp_config.save_path = os.path.join(exp_config.save_path, 'regret_plot')
    exp_config.plots_path = os.path.join(exp_config.plots_path, 'regret_plot')

data_dir = get_experiment_path_from_config(exp_config, mkdir=True, root_type='data')
accelerator = Accelerator(project_dir=data_dir)



# Get environment
env_class = env_name_map[exp_config.env]
env_config = get_env_config(exp_config.env)
env_config['gamma'] = exp_config.gamma
env_config['no_goal'] = False
env = env_class(env_config)




# Get model
vocab_size = env.observation_space.n + env.action_space.n + 1
model_config = GPT2Config()

model_config.vocab_size = vocab_size
model_config.n_positions = exp_config.seq_len
model_config.n_ctx = exp_config.seq_len

model = GPT2LMHeadModel(model_config)

checkpoints_dir = get_file_path_from_config('checkpoints', exp_config)
model = accelerator.prepare(model)
accelerator.load_state(checkpoints_dir)




# Get option wrapper over model
seq_len = model.config.n_ctx
option_env = OptionEnvWrapper(env, model, seq_len, vocab_size, exp_config.num_options)



# Get agent
# agent_class = agent_name_map[exp_config.agent]
# agent_config = get_agent_config(exp_config.agent)
# agent = agent_class(option_env, agent_config)
agent = RandomAgent(option_env, {})



# Evaluation loop
trajectory = rollout(env, agent, exp_config, max_steps=100000)

total_reward = 0
for transition in trajectory:
    total_reward += transition[2]

print("Total reward:", total_reward)





# # Save agent
# agent_path = get_file_path_from_config('agent.pkl', exp_config, mkdir=True)
# agent.save(agent_path)

# Save trajectories
trajectories_path = get_file_path_from_config('option_trajectories.pkl', exp_config, mkdir=True)
with open(trajectories_path, 'wb') as f:
    pickle.dump(trajectory, f)