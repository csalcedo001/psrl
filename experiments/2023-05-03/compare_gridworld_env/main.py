import tqdm
import time
import matplotlib.pyplot as plt
from dotmap import DotMap

from psrl.config import get_env_config

from naive_mdp import GridworldEnvNaive
from numpy_mdp import GridworldEnvNumpy


max_steps = 100000

metrics = {}
for side in range(2, 40):
    ### Generate grid
    grid = []

    grid.append(['#'] * (side + 2))
    for _ in range(side):
        grid.append(['#'] + [' '] * side + ['#'])
    grid.append(['#'] * (side + 2))

    grid[1][1] = 'S'

    number_of_states = side ** 2

    print(f'Running for {number_of_states} states')

    config = DotMap({'grid': grid})

    env_dict = {
        'naive': GridworldEnvNaive(config),
        'numpy': GridworldEnvNumpy(config),
    }

    action_space = env_dict['naive'].action_space

    loop_metrics = {}
    for env_name in env_dict:
        print(f'- Running {env_name}')

        env = env_dict[env_name]
        env.reset()

        t0 = time.time()
        for _ in tqdm.tqdm(range(max_steps)):
            action = action_space.sample()
            env.step(action)
        t1 = time.time()

        loop_metrics[env_name] = t1 - t0
    
    metrics[number_of_states] = loop_metrics


fig = plt.figure()
plt.title('Compare execution time of GridWorld env implementations')
plt.ylabel('Execution time (s)')
plt.xlabel('Number of states')
# plt.xscale('log')
for env_name in env_dict:
    plt.plot(metrics.keys(), [metrics[k][env_name] for k in metrics], label=env_name)
plt.legend()
plt.savefig('benchmark.png')
plt.close()