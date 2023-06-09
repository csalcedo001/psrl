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

from ucrl2_v1 import UCRL2_v1
from ucrl2_v2 import UCRL2_v2



### Get metrics

max_steps = 100000
num_states_list = np.arange(5, 155, 5)

metrics = {
    'ucrl2_v1': {},
    'ucrl2_v2': {}
}
for agent_name in metrics:
    metrics[agent_name]['time'] = []
    metrics[agent_name]['regret'] = []

    for num_states in num_states_list:
        env_config = get_env_config('riverswim')
        env_config.n = num_states
        env = RiverSwimEnv(env_config)


        # Get agent
        config = get_agent_config('ucrl2')
        if agent_name == 'ucrl2_v1':
            UCRL2 = UCRL2_v1
        else:
            UCRL2 = UCRL2_v2

        agent = UCRL2(env, config)
        

        # Get optimal policy for environment
        oracle_env = copy.deepcopy(env)
        oracle_config = get_agent_config('optimal')
        oracle = OptimalAgent(oracle_env, oracle_config)



        # Start training
        print(f'Running {agent_name} for {num_states} states')

        config = DotMap({
            'max_steps': max_steps,
        })

        t0 = time.time()
        agent_trajectories = train(env, agent, config, max_steps=max_steps)
        t1 = time.time()
        metrics[agent_name]['time'].append(t1 - t0)

        oracle_trajectories = rollout(oracle_env, oracle, config)


        # Calculate regret
        regrets = []
        regret = 0
        for t in range(len(agent_trajectories)):
            agent_reward = agent_trajectories[min(t, len(agent_trajectories) - 1)][2]
            oracle_reward = oracle_trajectories[min(t, len(oracle_trajectories) - 1)][2]

            regret += oracle_reward - agent_reward
            regrets.append(regret)
        
        metrics[agent_name]['regret'].append(regrets)



### Make plots

# Plot time vs. number of states
fig = plt.figure()
plt.title('Execution time vs. Number of States')
plt.ylabel('Time (s)')
plt.xlabel('Number of States')
for agent_name in metrics:
    plt.plot(num_states_list, metrics[agent_name]['time'], label=agent_name)
plt.legend()
plt.savefig('time_vs_states.png')



# Plot final regret vs. number of states
fig = plt.figure()
plt.title('Regret vs. Number of States')
plt.ylabel('Regret')
plt.xlabel('Number of States')
for agent_name in metrics:
    regrets = np.array(metrics[agent_name]['regret'])
    plt.plot(num_states_list, regrets[:, -1], label=agent_name)
plt.legend()
plt.savefig('regret_vs_states.png')



# Plot regret vs. steps
fig = plt.figure()
plt.title('Regret throughout an episode')
plt.ylabel('Regret')
plt.xlabel('Steps')
for agent_name in metrics:
    regrets = np.array(metrics[agent_name]['regret'])
    plt.plot(np.arange(regrets.shape[1]), regrets[-1], label=agent_name)
plt.legend()
plt.savefig('regret_vs_step.png')