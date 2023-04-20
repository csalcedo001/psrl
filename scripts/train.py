import argparse
from dotmap import DotMap

from psrl.envs import RiverSwimEnv, RandomMDPEnv
from psrl.agents import PSRLAgent, RandomAgent, OptimalAgent
from psrl.config import get_env_config, get_agent_config
from psrl.utils import rollout, env_name_map, agent_name_map



def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--agent', type=str, default='random_agent', help='Agent to use')
    parser.add_argument('--env', type=str, default='riverswim', help='Environment to use')
    parser.add_argument('--episodes', type=int, default=100, help='Number of episodes to run')

    return parser


def get_config(args):
    config = {}

    if args.env not in env_name_map:
        raise ValueError('Environment not supported')

    if args.agent not in agent_name_map:
        raise ValueError('Agent not supported')
    
    if args.episodes < 1:
        raise ValueError('Number of episodes must be at least 1')

    config['agent'] = args.agent
    config['env'] = args.env
    config['episodes'] = args.episodes

    return DotMap(config)


parser = get_parser()
args = parser.parse_args()
config = get_config(args)


env_class = env_name_map[args.env]
env_config = get_env_config(args.env)
env = env_class(env_config)


agent_class = agent_name_map[args.agent]
agent_config = get_agent_config(args.agent)
agent = agent_class(env, agent_config)



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
oracle_config = get_agent_config('optimal')
oracle = OptimalAgent(env, oracle_config)


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