import os
import wandb
import numpy as np
import imageio
import matplotlib.pyplot as plt
from tqdm import tqdm

from psrl.utils import train_episode, rollout_episode, env_name_map, agent_name_map
from psrl.config import save_config

from arg_utils import get_parser, get_config
from utils import choose_gridworld_color



envs = ['tworoom', 'fourroom']

parser = get_parser(envs=envs)
args = parser.parse_args()
config = get_config(args, envs=envs)
save_config(config.toDict(), config.experiment_dir)


# Get environment
env_class = env_name_map[args.env]
env = env_class(config.env_config)

# Get agent
agent_class = agent_name_map[args.agent]
agent = agent_class(env, config.agent_config)

episodes = 100
for episode in tqdm(range(episodes)):
    train_episode(env, agent)

trajectory = rollout_episode(env, agent, max_steps=1000, render=config.render, verbose=True)

states = [t[0] for t in trajectory]
states += [trajectory[-1][3]]



root = config.experiment_dir

fig, ax = plt.subplots()

plt.xlim(0, env.cols)
plt.ylim(0, env.rows)

ax.set_aspect('equal')
ax.set_xticks([])
ax.set_yticks([])

state_to_pos = {}

for i in range(env.rows):
    for j in range(env.cols):
        color = choose_gridworld_color(env.grid[i][j])
        
        x = j
        y = env.rows - i - 1

        ax.add_patch(plt.Rectangle((x, y), 1, 1, color=color))

        state_to_pos[env.state_id[i, j]] = [i, j]


frame_filename = f'{root}/frame.png'

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