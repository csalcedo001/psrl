import os
from tqdm import tqdm
import pickle

from psrl.config import get_env_config, get_agent_config
from psrl.train import train
from psrl.utils import env_name_map, agent_name_map

from utils import load_experiment_config




# Get experiment configuration
config_path = os.path.join(os.path.dirname(__file__), 'configs', 'exp_config.yaml')
exp_config = load_experiment_config(config_path)


# Get environment
env_class = env_name_map[exp_config.env]
env_config = get_env_config(exp_config.env)
env_config['gamma'] = exp_config.gamma
env_config['no_goal'] = exp_config.no_goal
env = env_class(env_config)




# Get agent
agent_class = agent_name_map[exp_config.agent]
agent_config = get_agent_config(exp_config.agent)
agent = agent_class(env, agent_config)




trajectory = train(env, agent, exp_config, max_steps=exp_config.training_steps)


# Save agent checkpoints
checkpoints_path = os.path.join(os.path.dirname(__file__), exp_config.save_path)
os.makedirs(checkpoints_path, exist_ok=True)

# Save agent
agent_path = os.path.join(checkpoints_path, f'agent_{exp_config.training_steps}.pkl')
agent.save(agent_path)

# Save trajectories
trajectories_path = os.path.join(checkpoints_path, f'trajectories_{exp_config.training_steps}.pkl')
with open(trajectories_path, 'wb') as f:
    pickle.dump(trajectory, f)