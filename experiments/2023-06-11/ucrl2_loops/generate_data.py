import os
from tqdm import tqdm

from psrl.envs.gridworld import TwoRoomGridworldEnv
from psrl.config import get_env_config, get_agent_config
from psrl.train import train_episode

from ucrl2 import UCRL2Agent
from utils import load_experiment_config



# Get experiment configuration
exp_config = load_experiment_config('exp_config.yaml')


env_config = get_env_config('tworoom')
env = TwoRoomGridworldEnv(env_config)



agent_config = get_agent_config('ucrl2')
agent = UCRL2Agent(env, agent_config)




for step in tqdm(range(exp_config.train_episodes)):
    train_episode(env, agent)


# Save agent checkpoints
os.makedirs(exp_config.save_path, exist_ok=True)


for step in tqdm(range(exp_config.train_debug_episodes)):
    train_episode(env, agent, max_steps=exp_config.train_episodes)
    checkpoint_path = os.path.join(exp_config.save_path, 'checkpoint_{}.pkl'.format(str(step).zfill(4)))
    agent.save(checkpoint_path)