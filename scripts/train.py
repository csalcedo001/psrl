import copy
import wandb
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




agent_trajectories = []
remaining_steps = config.max_steps
pbar = tqdm(total=config.max_steps)
while remaining_steps > 0:
    agent_trajectory = train_episode(env, agent)
    agent_trajectories += agent_trajectory
    elapsed_steps = len(agent_trajectory)
    remaining_steps -= elapsed_steps
    pbar.update(elapsed_steps)



oracle_env = copy.deepcopy(env)

oracle_trajectories = []
remaining_steps = config.max_steps
pbar = tqdm(total=config.max_steps)
while remaining_steps > 0:
    oracle_trajectory = rollout_episode(oracle_env, oracle)
    oracle_trajectories += oracle_trajectory
    elapsed_steps = len(oracle_trajectory)
    remaining_steps -= elapsed_steps
    pbar.update(elapsed_steps)


regrets = []
regret = 0
for t in range(len(agent_trajectories)):
    agent_reward = agent_trajectories[min(t, len(agent_trajectories) - 1)][2]
    oracle_reward = oracle_trajectories[min(t, len(oracle_trajectories) - 1)][2]

    regret += oracle_reward - agent_reward
    regrets.append(regret)

    wandb.log({'regret': regret})


if config.render or config.verbose:
    rollout_episode(env, agent, render=config.render, verbose=config.verbose)

print("Regret:", regrets[-1])