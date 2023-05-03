import tqdm
import time
import matplotlib.pyplot as plt

from psrl.config import get_env_config

from naive_mdp import RiverSwimEnvNaive
from numpy_mdp import RiverSwimEnvNumpy
from dict_mdp import make_riverSwim


max_steps = 100000

metrics = {}
for p in range(2, 15):
    number_of_states = 2 ** p

    print(f'Running for {number_of_states} states')

    config = get_env_config('riverswim')
    config['max_steps'] = max_steps
    config['n'] = number_of_states

    env_dict = {
        'naive': RiverSwimEnvNaive(config),
        'numpy': RiverSwimEnvNumpy(config),
        'dict': make_riverSwim(epLen=max_steps, nState=number_of_states)
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
plt.title('Compare execution time of RiverSwim env implementations')
plt.ylabel('Execution time (s)')
plt.xlabel('Number of states')
plt.xscale('log')
for env_name in env_dict:
    plt.plot(metrics.keys(), [metrics[k][env_name] for k in metrics], label=env_name)
plt.legend()
plt.savefig('benchmark.png')
plt.close()