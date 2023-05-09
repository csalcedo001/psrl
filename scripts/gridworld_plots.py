import os
import imageio
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from psrl.agents import PSRLAgent, UCRL2Agent
from psrl.train import train
from psrl.rollout import rollout_episode
from psrl.utils import env_name_map, agent_name_map
from psrl.config import save_config

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
env_class = env_name_map[config.env]
env = env_class(config.env_config)

# Get agent
agent_class = agent_name_map[config.agent]
agent = agent_class(env, config.agent_config)

if config.data_dir:
    weights_path = os.path.join(config.data_dir, 'weights.pkl')
    agent.load(weights_path)
else:
    train(env, agent, config)

trajectory = rollout_episode(env, agent, max_steps=120, render=config.render, verbose=True)

states = [t[0] for t in trajectory]
states += [trajectory[-1][3]]



### Make video

state_to_pos = {}
for i in range(env.rows):
    for j in range(env.cols):
        state_to_pos[env.state_id[i, j]] = [i, j]


root = config.experiment_dir
frame_filename = f'{root}/frame.png'


fig, ax = plt.subplots()
init_plt_grid(ax, env)

frames = []
for t, state in enumerate(states):
    print(f"Processing frame {t}")

    i, j = state_to_pos[state]
        
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


    pos = np.array(state_to_pos[state])

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
init_plt_grid(ax, env)

plt.quiver(*origins, *vectors, color='#000000', scale=1, scale_units='xy', angles='xy')

file_path = os.path.join(root, 'policy.png')
plt.savefig(file_path)



### Plot heatmap
if isinstance(agent, UCRL2Agent):
    # r_hat = agent.total_rewards / np.clip(agent.total_visitations, 1, None)
    r_hat = agent.Rk
elif isinstance(agent, PSRLAgent):
    r_hat = np.array(agent.r_total) / np.clip(np.array(agent.p_count).sum(axis=2), 1, None)
else:
    r_hat = None



if r_hat is not None:
    r = r_hat.sum(axis=1)
    r_min = r.min()
    r_max = r.max()
    print(r)
    print(r_min, r_max)
    

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
        
    file_path = os.path.join(root, 'reward_heatmap.png')
    plt.savefig(file_path)