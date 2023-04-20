import copy
import wandb
import numpy as np
from tqdm import tqdm

from psrl.agents import OptimalAgent
from psrl.config import get_agent_config
from psrl.utils import train_episode, rollout_episode, env_name_map, agent_name_map

from arg_utils import get_parser, get_config



parser = get_parser()
args = parser.parse_args()
config = get_config(args)

# Initialize wandb
wandb.init(
    entity='cesar-salcedo',
    project='psrl',
    config=config.toDict()
)


# Get environment
env_class = env_name_map[args.env]
env = env_class(config.env_config)

# Get agent
agent_class = agent_name_map[args.agent]
agent = agent_class(env, config.agent_config)

# Get optimal policy for environment
oracle_config = get_agent_config('optimal')
oracle = OptimalAgent(env, oracle_config)



print("Observation_space:", env.observation_space)
print("Action space:", env.action_space)



oracle_trajectories = []
agent_trajectories = []
for episode in tqdm(range(args.episodes)):
    oracle_env = copy.deepcopy(env)

    agent_trajectory = train_episode(env, agent)
    agent_trajectories.append(agent_trajectory)

    oracle_trajectory = rollout_episode(oracle_env, oracle)
    oracle_trajectories.append(oracle_trajectory)


# Compute regret
regrets = []
for episode in range(args.episodes):
    episode_regrets = []
    regret = 0

    for t in range(len(agent_trajectories[episode])):
        agent_reward = agent_trajectories[episode][t][2]
        oracle_reward = oracle_trajectories[episode][t][2]

        regret += oracle_reward - agent_reward
        episode_regrets.append(regret)
    
    regrets.append(episode_regrets)

regrets = np.array(regrets)

print("REGRET:", regrets[:, -1])

    # state = env.reset()

    # while True:
    #     oracle_env = copy.deepcopy(env)

    #     # Get agent's action
    #     action = agent.act(state)
    #     next_state, reward, done, _ = env.step(action)

    #     # Get oracle's action
    #     oracle_action = oracle.act(state)
    #     _, oracle_reward, _, _ = oracle_env.step(oracle_action)

    #     regret += oracle_reward - reward

    #     transition = (state, action, reward, next_state)
    #     agent.observe(transition)

    #     print('[{}/{}] Iteration: {}, State: {}, Action: {}, Next State: {}, Reward: {}, Done: {}, Regret: {:.2f}'.format(episode, args.episodes, iteration, state, action, next_state, reward, done, regret))
    #     # wandb.log({'iteration': iteration, 'state': state, 'action': action, 'next_state': next_state, 'reward': reward, 'done': done, 'regret': regret})
    #     wandb.log({'regret': regret})

    #     if done:
    #         agent.update()
    #         break
            
    #     iteration += 1
    #     state = next_state




# # Rollouts policies
# rollout_episodes = 100

# agent_trajectories = rollout(env, agent, episodes=rollout_episodes)
# oracle_trajectories = rollout(env, oracle, episodes=rollout_episodes)

# # Compute regret
# regret = 0
# for k in range(rollout_episodes):
#     agent_reward_per_episode = sum([r for _, _, r, _ in agent_trajectories[k]])
#     oracle_reward_per_episode = sum([r for _, _, r, _ in oracle_trajectories[k]])

#     regret += oracle_reward_per_episode - agent_reward_per_episode

# print("REGRET:", regret)