import os
import wandb
import pickle

from psrl.rollout import rollout
from psrl.utils import env_name_map, agent_name_map

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

weights_path = os.path.join(config.data_dir, 'weights.pkl')
agent.load(weights_path)


trajectories = rollout(
    env,
    agent,
    config,
    render=config.render,
    verbose=True,
    max_steps=config.max_steps
)

traj_path = os.path.join(config.experiment_dir, 'trajectories.pkl')

with open(traj_path, 'wb') as out_file:
    pickle.dump(trajectories, out_file)