import argparse
from tqdm import tqdm

from psrl.envs import RiverSwimEnv, RandomMDPEnv
from psrl.agents import PSRLAgent, RandomAgent, OptimalAgent


# Define argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--agent', type=str, default='random', help='Agent to use')
parser.add_argument('--env', type=str, default='river_swim', help='Environment to use')
parser.add_argument('--episodes', type=int, default=10, help='Number of episodes to run')



args = parser.parse_args()

# Validate arguments
# TODO: provide env arguments via env_config
if args.env == 'random':
    env = RandomMDPEnv(
        n_states=10,
        n_actions=5,
        max_steps=100,
    )
elif args.env == 'river_swim':
    env = RiverSwimEnv()
else:
    raise ValueError('Environment not supported')

# TODO: provide agent arguments via agent_config
if args.agent == 'random':
    agent = RandomAgent(env)
elif args.agent == 'psrl':
    agent = PSRLAgent(
        env,
        gamma=0.9,
        kappa=1,
        mu=0,
        lambd=1,
        alpha=1,
        beta=1,
        max_iter=1000,
    )
elif args.agent == 'optimal':
    agent = OptimalAgent(env, gamma=0.9, max_iter=10000)
else:
    raise ValueError('Agent not supported')



iteration = 0

trajectories = []
for episode in tqdm(range(args.episodes)):
    state = env.reset()

    trajectory = []
    while True:
        action = agent.act(state)

        next_state, reward, done, _ = env.step(action)

        transition = (state, action, reward, next_state)
        trajectory.append(transition)

        print('[{}/{}] Iteration: {}, State: {}, Action: {}, Next State: {}, Reward: {}, Done: {}'.format(episode, args.episodes, iteration, state, action, next_state, reward, done))

        if done:
            break
            
        iteration += 1
        state = next_state
    
    trajectories.append(trajectory)
