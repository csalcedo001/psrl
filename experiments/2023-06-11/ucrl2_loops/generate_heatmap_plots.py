import os
import numpy as np
from tqdm import tqdm

from psrl.envs.gridworld import TwoRoomGridworldEnv
from psrl.config import get_env_config, get_agent_config

from ucrl2 import UCRL2Agent
from plotting import (
    save_expected_reward_heatmap_plot,
    save_state_value_heatmap_plot,
    save_action_value_heatmap_plot,
    save_empirical_state_visitation_heatmap_plot,
    save_reward_count_heatmap_plot,
)
from utils import load_experiment_config



# Get experiment configuration
exp_config = load_experiment_config('exp_config.yaml')

expected_reward_path = os.path.join(exp_config.plot_path, 'expected_reward')
action_value_path = os.path.join(exp_config.plot_path, 'action_value')
state_value_path = os.path.join(exp_config.plot_path, 'state_value')
empirical_state_visitation_path = os.path.join(exp_config.plot_path, 'empirical_state_visitation')
reward_count_path = os.path.join(exp_config.plot_path, 'reward_count')


env_config = get_env_config('tworoom')
env = TwoRoomGridworldEnv(env_config)



agent_config = get_agent_config('ucrl2')
agent = UCRL2Agent(env, agent_config)



state_to_pos = {}
for i in range(env.rows):
    for j in range(env.cols):
        state_to_pos[env.state_id[i, j]] = [i, j]




os.makedirs(exp_config.plot_path, exist_ok=True)
os.makedirs(expected_reward_path, exist_ok=True)
os.makedirs(action_value_path, exist_ok=True)
os.makedirs(state_value_path, exist_ok=True)
os.makedirs(empirical_state_visitation_path, exist_ok=True)
os.makedirs(reward_count_path, exist_ok=True)


for step in tqdm(range(exp_config.train_debug_episodes)):
    agent = UCRL2Agent(env, agent_config)

    checkpoint_path = os.path.join(exp_config.save_path, 'checkpoint_{}.pkl'.format(str(step).zfill(4)))
    agent.load(checkpoint_path)


    p_hat = agent.Pk / np.clip(agent.Pk.sum(axis=2, keepdims=True), 1, None) + np.expand_dims(agent.p_distances, axis=2)
    r_hat = agent.Rk / np.clip(agent.Nk, 1, None) + agent.r_distances
    p_count = agent.Pk
    r_count = agent.Rk
    v = agent.u
    q = agent.q

    state_visitation = np.sum(agent.Nk + agent.vk, axis=1)

    print('Saving plots for episode', step)

    save_expected_reward_heatmap_plot(
        env,
        state_to_pos,
        r_hat,
        'expected_reward_' + str(step).zfill(4),
        title=f'Expected Reward at episode {step}',
        path=expected_reward_path,
    )
    save_action_value_heatmap_plot(
        env,
        state_to_pos,
        q,
        'action_value_' + str(step).zfill(4),
        title=f'Action Value at episode {step}',
        path=action_value_path,
    )
    save_state_value_heatmap_plot(
        env,
        state_to_pos,
        v,
        'state_value_' + str(step).zfill(4),
        title=f'State Value at episode {step}',
        path=state_value_path,
    )
    save_empirical_state_visitation_heatmap_plot(
        env,
        state_to_pos,
        state_visitation,
        'empirical_state_visitation_' + str(step).zfill(4),
        title=f'Empirical State Visitation at episode {step}',
        path=empirical_state_visitation_path,
    )
    save_reward_count_heatmap_plot(
        env,
        state_to_pos,
        r_count,
        p_count,
        'reward_count_' + str(step).zfill(4),
        title=f'Empirical reward Count at episode {step}',
        path=reward_count_path,
    )
