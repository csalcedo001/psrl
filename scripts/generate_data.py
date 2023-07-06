from psrl.train import train

from setup_script import setup_script
from file_system import save_pickle

from utils import get_file_path_from_config




# Setup script
exp_config, env, agent, _ = setup_script()


# Train agent and get trajectories
trajectory = train(env, agent, exp_config, max_steps=exp_config.training_steps)



# Save agent and trajectories
agent_path = get_file_path_from_config('agent.pkl', exp_config, mkdir=True)
agent.save(agent_path)

trajectories_path = get_file_path_from_config('trajectories.pkl', exp_config, mkdir=True)
save_pickle(trajectory, trajectories_path)