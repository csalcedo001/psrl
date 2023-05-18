import os
import torch
import torch.distributions as dist
import matplotlib as mpl
import matplotlib.pyplot as plt

from utils import load_experiment_config, setup_env_and_agent, init_plt_grid



# Get experiment configuration
exp_config = load_experiment_config('exp_config.yaml')

# Get environment and agent
env, agent = setup_env_and_agent(exp_config)

# Load agent weights
agent.load('agent_weights.pkl')



# Sample r_hat from agent's reward function distribution
r_hats = []

mu0, lambd, alpha, beta = torch.moveaxis(agent.r_dist, 2, 0)

for _ in range(exp_config.psrl_heatmap_samples):
    tau = dist.Gamma(alpha, 1. / beta).sample()
    mu = dist.Normal(mu0, 1. / torch.sqrt(lambd * tau)).sample()

    r_hat = mu

    r_hats.append(r_hat)

r_hat = torch.stack(r_hats).mean(dim=0)




state_to_pos = {}
for i in range(env.rows):
    for j in range(env.cols):
        state_to_pos[env.state_id[i, j]] = [i, j]



# Plot heatmap
r = r_hat.sum(axis=1)
r_min = r.min()
r_max = r.max()

cmap = mpl.colormaps['plasma']
norm = mpl.colors.Normalize(vmin=r_min, vmax=r_max)
cmap = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)

fig, ax = plt.subplots()
init_plt_grid(ax, env)

for state in range(env.observation_space.n):
    i, j = state_to_pos[state]
        
    x = j
    y = env.rows - i - 1

    ax.add_patch(plt.Rectangle((x, y), 1, 1, color=cmap.to_rgba(r[state])))

fig.colorbar(cmap, ax=ax)
    
file_path = os.path.join('reward_heatmap.png')
plt.savefig(file_path)