import os
import wandb
import numpy as np
import imageio
import matplotlib.pyplot as plt
from tqdm import tqdm

from psrl.utils import rollout_episode, env_name_map, agent_name_map

from arg_utils import get_parser, get_config




def choose_color(symbol):
    if symbol == ' ':
        color = 'w'
    elif symbol == '#':
        color = 'k'
    elif symbol == 'S':
        color = 'b'
    elif symbol == 'T':
        color = 'g'
    elif symbol == '.':
        color = '#7f7f7f'
    else:
        color = None
    
    return color


parser = get_parser()
args = parser.parse_args()
config = get_config(args)

# Initialize wandb
# wandb.init(
#     entity='cesar-salcedo',
#     project='psrl',
#     config=config.toDict()
# )


# Get environment
env_class = env_name_map[args.env]
env = env_class(config.env_config)

# Get agent
agent_class = agent_name_map[args.agent]
agent = agent_class(env, config.agent_config)


trajectory = rollout_episode(env, agent, render=config.render, verbose=True)

states = [t[0] for t in trajectory]
states += [trajectory[-1][3]]

root = os.path.dirname(__file__)
os.makedirs(f'{root}/frames', exist_ok=True)



fig, ax = plt.subplots()

plt.xlim(0, env.cols)
plt.ylim(0, env.rows)

ax.set_aspect('equal')
ax.set_xticks([])
ax.set_yticks([])

state_to_pos = {}

for i in range(env.rows):
    for j in range(env.cols):
        color = choose_color(env.grid[i][j])
        
        x = j
        y = env.rows - i - 1

        ax.add_patch(plt.Rectangle((x, y), 1, 1, color=color))

        state_to_pos[env.state_id[i, j]] = [i, j]


for t, state in enumerate(states):
    print(f"Processing frame {t}")

    i, j = state_to_pos[state]
        
    x = j
    y = env.rows - i - 1

    # Draw agent and save figure
    color = choose_color(env.grid[i][j])
    ax.add_patch(plt.Circle((x + 0.5, y + 0.5), 0.25, color='r'))

    plt.savefig(f'{root}/frames/img_{t}.png')

    # Return to previous state by covering it back
    color = choose_color(env.grid[i][j])
    ax.add_patch(plt.Rectangle((x, y), 1, 1, color=color))
        


frames = []
for t in range(len(states)):
    image = imageio.v2.imread(f'{root}/frames/img_{t}.png')
    frames.append(image)

print("Saving gif...")
imageio.mimsave(f'{root}/trajectory.gif', frames, fps = 2)