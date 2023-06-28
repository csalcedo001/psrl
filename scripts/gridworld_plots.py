import os
import imageio
import numpy as np
import torch
import torch.distributions as dist
import matplotlib as mpl
import matplotlib.pyplot as plt

from psrl.agents import PSRLAgent, UCRL2Agent, KLUCRLAgent
from psrl.agents.utils import policy_evaluation
from psrl.train import train
from psrl.rollout import rollout_episode
from psrl.utils import env_name_map, agent_name_map
from psrl.config import save_config, get_env_config, get_agent_config

from arg_utils import get_parser, get_config
from utils import choose_gridworld_color




def init_plt_grid(ax, env):
    plt.xlim(0, env.cols)
    plt.ylim(0, env.rows)

    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])

    for i in range(env.rows):
        for j in range(env.cols):
            color = choose_gridworld_color(env.grid[i][j])
            
            x = j
            y = env.rows - i - 1

            ax.add_patch(plt.Rectangle((x, y), 1, 1, color=color))



### Get trained policy

envs = ['tworoom', 'fourroom']

parser = get_parser(envs=envs, data_dir=True)
args = parser.parse_args()
config = get_config(args, envs=envs)
save_config(config.toDict(), config.experiment_dir)


# Get environment
env_config = get_env_config(config.env)
env_class = env_name_map[config.env]
env = env_class(env_config)

# Get agent
agent_config = get_agent_config(config.agent)
agent_class = agent_name_map[config.agent]
agent = agent_class(env, agent_config)

if config.data_dir:
    weights_path = os.path.join(config.data_dir, 'weights.pkl')
    agent.load(weights_path)
else:
    train_trajectory = train(env, agent, config, max_steps=config.max_steps)

rollout_trajectory = rollout_episode(env, agent, max_steps=120, render=config.render, verbose=True)
trajectory = rollout_trajectory


train_states = [t[0] for t in train_trajectory]
train_states += [train_trajectory[-1][3]]

states = [t[0] for t in trajectory]
states += [trajectory[-1][3]]



### Make video
root = config.experiment_dir
frame_filename = f'{root}/frame.png'


fig, ax = plt.subplots()
init_plt_grid(ax, env)

frames = []
for t, state in enumerate(states):
    print(f"Processing frame {t}")

    i, j = env._get_pos_from_state(state)
        
    x = j
    y = env.rows - i - 1

    # Draw agent and save figure
    color = choose_gridworld_color(env.grid[i][j])
    ax.add_patch(plt.Circle((x + 0.5, y + 0.5), 0.25, color='r'))

    plt.savefig(frame_filename)

    # Return to previous state by covering it back
    color = choose_gridworld_color(env.grid[i][j])
    ax.add_patch(plt.Rectangle((x, y), 1, 1, color=color))

    # Load plot as image
    image = imageio.v2.imread(frame_filename)
    frames.append(image)

os.remove(frame_filename)
    

print("Saving video...")
imageio.mimsave(f'{root}/trajectory.mp4', frames, fps=2)




### Plot policy
print("Processing policy plot...")

# Get vectors for each state
origins = []
vectors = []
for state in range(env.observation_space.n):
    action = agent.act(state)

    # up: 0, right: 1, down: 2, left: 3
    axis = action % 2
    direction = action // 2

    
    # Correction for plot
    if axis == 0:
        direction = 1 - direction


    pos = np.array(env._get_pos_from_state(state))

    next_pos = pos.copy()
    next_pos[axis] += 1 if direction == 0 else -1


    # Correction for plot
    pos = pos[::-1]
    pos[1] = env.rows - pos[1] - 1
    next_pos = next_pos[::-1]
    next_pos[1] = env.rows - next_pos[1] - 1


    dir_vec = next_pos - pos
    
    origins.append(pos + 0.5 - dir_vec * 0.4)
    vectors.append(dir_vec * 0.8)

origins = np.array(origins).T
vectors = np.array(vectors).T


fig, ax = plt.subplots()
plt.title('Policy')
init_plt_grid(ax, env)

plt.quiver(*origins, *vectors, color='#000000', scale=1, scale_units='xy', angles='xy')

file_path = os.path.join(root, 'policy.png')
plt.savefig(file_path)



### Get p and r estimates
if isinstance(agent, UCRL2Agent) or isinstance(agent, KLUCRLAgent):
    p_hat = agent.Pk / np.clip(agent.Pk.sum(axis=2, keepdims=True), 1, None) + np.expand_dims(agent.p_distances, axis=2)
    r_hat = agent.Rk / np.clip(agent.Nk, 1, None) + agent.r_distances
    p_count = agent.Pk
    r_count = agent.Rk
    v = agent.u
elif isinstance(agent, PSRLAgent):
    samples = 100


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
else:
    p_hat = None
    r_hat = None
    p_count = None
    r_count = None
    v = None



if r_hat is not None:
    print("Processing expected reward heatmap plot...")

    ### Plot expected reward heatmap
    r = r_hat.sum(axis=1)
    r_min = r.min()
    r_max = r.max()
    

    cmap = mpl.colormaps['plasma']
    norm = mpl.colors.Normalize(vmin=r_min, vmax=r_max)
    cmap = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)

    fig, ax = plt.subplots()
    plt.title('Expected reward heatmap')
    init_plt_grid(ax, env)
    
    for state in range(env.observation_space.n):
        i, j = env._get_pos_from_state(state)
            
        x = j
        y = env.rows - i - 1

        ax.add_patch(plt.Rectangle((x, y), 1, 1, color=cmap.to_rgba(r[state])))
    
    fig.colorbar(cmap, ax=ax)
        
    file_path = os.path.join(root, 'reward_heatmap.png')
    plt.savefig(file_path)


if v is not None:
    print("Processing state value function heatmap plot...")

    ### Plot value function
    v_min = v.min()
    v_max = v.max()

    cmap = mpl.colormaps['plasma']
    norm = mpl.colors.Normalize(vmin=v_min, vmax=v_max)
    cmap = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)

    fig, ax = plt.subplots()
    plt.title('State value function heatmap')
    init_plt_grid(ax, env)
    
    for state in range(env.observation_space.n):
        i, j = env._get_pos_from_state(state)
            
        x = j
        y = env.rows - i - 1

        ax.add_patch(plt.Rectangle((x, y), 1, 1, color=cmap.to_rgba(v[state])))
    
    fig.colorbar(cmap, ax=ax)
        
    file_path = os.path.join(root, 'state_value_heatmap.png')
    plt.savefig(file_path)





print("Processing state distance plot...")

### Plot state distance
train_coords = [env._get_pos_from_state(t[0]) for t in train_trajectory]
train_coords += [env._get_pos_from_state(train_trajectory[-1][3])]
train_coords = np.array(train_coords)


k = 10

distances = np.abs(train_coords[k:] - train_coords[:-k]).sum(axis=1)

fig = plt.figure()
plt.title(f'Distance between states k={k} timesteps apart')
plt.plot(np.arange(len(distances)), distances, color='r', label=f'k={k}')
plt.legend()
        
file_path = os.path.join(root, 'state_distance.png')
plt.savefig(file_path)

plt.close()




### Plot empirical count of visited states
print("Processing empirical count of visited states plot...")

state_count = [0] * len(train_states)
for state in train_states:
    state_count[state] += 1
state_count = np.array(state_count)

sc_min = state_count.min()
sc_max = state_count.max()


cmap = mpl.colormaps['plasma']
norm = mpl.colors.Normalize(vmin=sc_min, vmax=sc_max)
cmap = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)

fig, ax = plt.subplots()
plt.title('Empirical state count heatmap')
init_plt_grid(ax, env)

for state in range(env.observation_space.n):
    i, j = env._get_pos_from_state(state)
        
    x = j
    y = env.rows - i - 1

    ax.add_patch(plt.Rectangle((x, y), 1, 1, color=cmap.to_rgba(state_count[state])))

fig.colorbar(cmap, ax=ax)

file_path = os.path.join(root, 'emp_state_count_heatmap.png')
plt.savefig(file_path)



if r_count is not None and p_count is not None:
    ### Plot empirical count of visited states
    print("Processing empirical total reward plot...")

    r_emp = r_count.sum(axis=1) / np.clip(p_count.sum(axis=(1, 2)), 1, None)
    r_min = r_emp.min()
    r_max = r_emp.max()

    cmap = mpl.colormaps['plasma']
    norm = mpl.colors.Normalize(vmin=r_min, vmax=r_max)
    cmap = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)

    fig, ax = plt.subplots()
    plt.title('Empirical total reward heatmap')
    init_plt_grid(ax, env)

    for state in range(env.observation_space.n):
        i, j = env._get_pos_from_state(state)
            
        x = j
        y = env.rows - i - 1

        ax.add_patch(plt.Rectangle((x, y), 1, 1, color=cmap.to_rgba(r_emp[state])))

    fig.colorbar(cmap, ax=ax)

    file_path = os.path.join(root, 'emp_total_reward_heatmap.png')
    plt.savefig(file_path)


p, r = env.get_p_and_r()

# Get value function via policy evaluation
v_eval = policy_evaluation(p, r, agent.pi, gamma=0.99, epsilon=1e-6, max_iter=1000)

print("Processing policy evaluation state value function heatmap plot...")
### Plot value function
v_min = v_eval.min()
v_max = v_eval.max()

cmap = mpl.colormaps['plasma']
norm = mpl.colors.Normalize(vmin=v_min, vmax=v_max)
cmap = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)

fig, ax = plt.subplots()
plt.title('Policy evaluation state value function heatmap')
init_plt_grid(ax, env)

for state in range(env.observation_space.n):
    i, j = env._get_pos_from_state(state)
        
    x = j
    y = env.rows - i - 1

    ax.add_patch(plt.Rectangle((x, y), 1, 1, color=cmap.to_rgba(v_eval[state])))

fig.colorbar(cmap, ax=ax)

file_path = os.path.join(root, 'pe_state_value_heatmap.png')
plt.savefig(file_path)

plt.close(fig)