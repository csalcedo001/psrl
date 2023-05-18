import yaml
from dotmap import DotMap
import matplotlib.pyplot as plt

from psrl.agents import PSRLAgent
from psrl.envs import GridworldEnv
from psrl.config import get_agent_config



def load_experiment_config(path):
    with open(path, 'r') as f:
        config = yaml.load(f, yaml.FullLoader)
    
    return DotMap(config)

def setup_env_and_agent(exp_config):
    # Get arguments
    side = exp_config.side
    training_steps = exp_config.training_steps


    # Get environment
    grid = []
    grid.append(['#'] * (side + 2))
    for _ in range(side):
        grid.append(['#'] + [' '] * side + ['#'])
    grid.append(['#'] * (side + 2))
    grid[1][1] = 'S'
    grid[-2][-2] = 'T'

    env_config = DotMap({'grid': grid})
    env = GridworldEnv(env_config)


    # Get agent
    agent_config = get_agent_config('psrl')
    agent_config.max_steps = training_steps
    agent = PSRLAgent(env, agent_config)

    
    return env, agent




def choose_gridworld_color(symbol):
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