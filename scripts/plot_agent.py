import torch
import torch.distributions as dist
import numpy as np

from setup_script import setup_script
from plotting import (
    save_policy_plot,
    save_expected_reward_heatmap_plot,
    save_state_value_heatmap_plot,
    save_action_value_heatmap_plot,
    save_empirical_state_visitation_heatmap_plot,
    save_reward_count_heatmap_plot,
)
from utils import get_file_path_from_config




# Setup script
exp_config, env, agent, accelerator = setup_script()
agent_path = get_file_path_from_config('agent.pkl', exp_config)
agent.load(agent_path)



# Get data from agent
if exp_config.agent in ['ucrl2', 'kl_ucrl']:
    p_hat = agent.Pk / np.clip(agent.Pk.sum(axis=2, keepdims=True), 1, None) + np.expand_dims(agent.p_distances, axis=2)
    r_hat = agent.Rk / np.clip(agent.Nk, 1, None) + agent.r_distances
    p_count = agent.Pk
    r_count = agent.Rk
    v = agent.u
    q = agent.q
    state_visitation = np.sum(agent.Nk + agent.vk, axis=1)
elif exp_config.agent in ['psrl']:
    samples = 10

    p_dist = agent.p_dist
    mu0, lambd, alpha, beta = agent.r_dist

    p_hats = []
    r_hats = []
    for _ in range(samples):
        p_hat = dist.Dirichlet(p_dist).sample()

        tau = dist.Gamma(alpha, 1. / beta).sample()
        mu = dist.Normal(mu0, 1. / torch.sqrt(lambd * tau)).sample()
        r_hat = mu

        p_hats.append(p_hat)
        r_hats.append(r_hat)

    p_hat = torch.stack(p_hats).mean(dim=0).numpy()
    r_hat = torch.stack(r_hats).mean(dim=0).numpy()
    p_count = np.array(agent.p_count)
    r_count = np.array(agent.r_total)

    v = None
    q = None
    state_visitation = None
else:
    raise Exception(f'Agent {exp_config.agent} not supported.')



# Save plots
print('Saving plots...')

save_policy_plot(
    env,
    agent,
    get_file_path_from_config('policy.png', exp_config, mkdir=True, root_type='plots'),
    title='Policy',
)
save_expected_reward_heatmap_plot(
    env,
    r_hat,
    get_file_path_from_config('expected_reward.png', exp_config, root_type='plots'),
    title='Expected Reward',
)
if q is not None:
    save_action_value_heatmap_plot(
        env,
        q,
        get_file_path_from_config('action_value_function.png', exp_config, root_type='plots'),
        title='Action Value Function',
    )
if v is not None:
    save_state_value_heatmap_plot(
        env,
        v,
        get_file_path_from_config('state_value.png', exp_config, root_type='plots'),
        title='State Value Function',
    )
if state_visitation is not None:
    save_empirical_state_visitation_heatmap_plot(
        env,
        state_visitation,
        get_file_path_from_config('empirical_state_visitation.png', exp_config, root_type='plots'),
        title='Empirical State Visitation',
    )
save_reward_count_heatmap_plot(
    env,
    r_count,
    get_file_path_from_config('empirical_total_reward.png', exp_config, root_type='plots'),
    title='Empirical Total Reward',
)
