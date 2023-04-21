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


episodes = args.max_steps // env.max_steps


oracle_trajectories = []
agent_trajectories = []
for _ in tqdm(range(episodes)):
    oracle_env = copy.deepcopy(env)

    agent_trajectory = train_episode(env, agent)
    agent_trajectories += agent_trajectory

    oracle_trajectory = rollout_episode(oracle_env, oracle)
    oracle_trajectories += oracle_trajectory


# Compute regret
regrets = []
regret = 0
for t in range(len(agent_trajectories)):
    agent_reward = agent_trajectories[t][2]
    oracle_reward = oracle_trajectories[t][2]

    regret += oracle_reward - agent_reward
    regrets.append(regret)

    wandb.log({'regret': regret})

print("REGRET:", regrets[-1])