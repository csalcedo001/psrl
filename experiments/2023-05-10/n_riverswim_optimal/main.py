import time
import copy
import numpy as np
from dotmap import DotMap
import matplotlib.pyplot as plt

from psrl.envs import RiverSwimEnv
from psrl.agents import OptimalAgent
from psrl.config import get_agent_config, get_env_config
from psrl.train import train
from psrl.rollout import rollout



### Get metrics

max_steps = 1
num_states_list = np.arange(5, 105, 5)

actions = []
for num_states in num_states_list:
    env_config = get_env_config('riverswim')
    env_config.n = num_states
    env = RiverSwimEnv(env_config)


    # Get optimal agent
    config = get_agent_config('optimal')
    agent = OptimalAgent(env, config)



    # Start training
    print(f'Running for {num_states} states')

    config = DotMap({
        'max_steps': max_steps,
    })

    trajectories = rollout(env, agent, config, max_steps=1)

    actions.append(trajectories[0][1])



### Make plots

fig, ax = plt.subplots(1,1) 
plt.title('RiverSwim optimal action vs. number of states (fixed reward)')
plt.ylabel('Optimal Action')
plt.xlabel('Number of States')
plt.ylim(-0.5, 1.5)
ax.set_yticks([0, 1])
ax.set_yticklabels(['left', 'right'])
plt.scatter(num_states_list, actions, c='b', marker='x')
plt.savefig('optimal_action_vs_states.png')