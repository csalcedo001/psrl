import argparse

from psrl.envs import RiverSwimEnv, RandomMDPEnv
from psrl.agents import PSRLAgent, RandomAgent, OptimalAgent
from psrl.config import get_env_config, get_agent_config
from psrl.utils import rollout, env_name_map



# Define argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--agent', type=str, default='random', help='Agent to use')
parser.add_argument('--env', type=str, default='riverswim', help='Environment to use')
parser.add_argument('--episodes', type=int, default=100, help='Number of episodes to run')



args = parser.parse_args()

# Validate arguments
# TODO: provide env arguments via env_config

if args.env not in env_name_map:
    raise ValueError('Environment not supported')

env_class = env_name_map[args.env]
env_config = get_env_config(args.env)
print(env_config)
env = env_class(env_config)


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
else:
    raise ValueError('Agent not supported')



print("Observation_space:", env.observation_space)
print("Action space:", env.action_space)


iteration = 0

for episode in range(args.episodes):
    state = env.reset()

    while True:
        action = agent.act(state)

        next_state, reward, done, _ = env.step(action)

        transition = (state, action, reward, next_state)
        agent.observe(transition)

        print('[{}/{}] Iteration: {}, State: {}, Action: {}, Next State: {}, Reward: {}, Done: {}'.format(episode, args.episodes, iteration, state, action, next_state, reward, done))

        if done:
            agent.update()
            break
            
        iteration += 1



# Get optimal policy
oracle = OptimalAgent(env, gamma=0.9, max_iter=1000)


# Rollouts policies
rollout_episodes = 100

agent_trajectories = rollout(env, agent, episodes=rollout_episodes)
oracle_trajectories = rollout(env, oracle, episodes=rollout_episodes)

# Compute regret
regret = 0
for k in range(rollout_episodes):
    agent_reward_per_episode = sum([r for _, _, r, _ in agent_trajectories[k]])
    oracle_reward_per_episode = sum([r for _, _, r, _ in oracle_trajectories[k]])

    regret += oracle_reward_per_episode - agent_reward_per_episode

print("REGRET:", regret)