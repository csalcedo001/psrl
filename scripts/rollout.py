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



print("Observation_space:", env.observation_space)
print("Action space:", env.action_space)


state = env.reset()
env.render()

trajectory = []
while True:
    action = agent.act(state)

    next_state, reward, done, _ = env.step(action)

    transition = (state, action, reward, next_state)
    trajectory.append(transition)

    env.render()

    if done:
        break

    state = next_state