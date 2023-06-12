import os
import copy
from tqdm import tqdm

from psrl.envs.gridworld import TwoRoomGridworldEnv
from psrl.config import get_env_config, get_agent_config
from psrl.train import train_episode

from ucrl2 import UCRL2Agent
from plotting import save_policy_plot



train_iter = 1000
train_debug_iter = 100
save_path = 'checkpoints'
plot_path = 'plots/policy'


env_config = get_env_config('tworoom')
env = TwoRoomGridworldEnv(env_config)



agent_config = get_agent_config('ucrl2')
agent = UCRL2Agent(env, agent_config)



state_to_pos = {}
for i in range(env.rows):
    for j in range(env.cols):
        state_to_pos[env.state_id[i, j]] = [i, j]




os.makedirs(plot_path, exist_ok=True)

for step in tqdm(range(train_debug_iter)):
    agent = UCRL2Agent(env, agent_config)

    checkpoint_path = os.path.join(save_path, 'checkpoint_{}.pkl'.format(str(step).zfill(4)))
    agent.load(checkpoint_path)

    print('Saving policy plot for episode', step)
    save_policy_plot(env, agent, state_to_pos, 'policy_' + str(step).zfill(4), title=f'Policy at episode {step}', path=plot_path)