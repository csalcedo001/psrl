import os
from tqdm import tqdm

from psrl.envs.gridworld import TwoRoomGridworldEnv
from psrl.config import get_env_config, get_agent_config

from ucrl2 import UCRL2Agent
from plotting import save_policy_plot
from utils import load_experiment_config



# Get experiment configuration
exp_config = load_experiment_config('exp_config.yaml')

policy_path = os.path.join(exp_config.plot_path, 'policy')


env_config = get_env_config('tworoom')
env = TwoRoomGridworldEnv(env_config)



agent_config = get_agent_config('ucrl2')
agent = UCRL2Agent(env, agent_config)



state_to_pos = {}
for i in range(env.rows):
    for j in range(env.cols):
        state_to_pos[env.state_id[i, j]] = [i, j]




os.makedirs(policy_path, exist_ok=True)

for step in tqdm(range(exp_config.train_debug_episodes)):
    agent = UCRL2Agent(env, agent_config)

    checkpoint_path = os.path.join(exp_config.save_path, 'checkpoint_{}.pkl'.format(str(step).zfill(4)))
    agent.load(checkpoint_path)
    agent.new_episode()

    print('Saving policy plot for episode', step)
    save_policy_plot(env, agent, state_to_pos, 'policy_' + str(step).zfill(4), title=f'Policy at episode {step}', path=policy_path)