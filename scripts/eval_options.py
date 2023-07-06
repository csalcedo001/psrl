from transformers import GPT2Config, GPT2LMHeadModel

from psrl.agents import RandomAgent
from psrl.rollout import rollout

from setup_script import setup_script
from file_system import save_pickle
from options import OptionEnvWrapper
from utils import get_file_path_from_config




# Setup script
exp_config, env, _, accelerator = setup_script()




# Get model
vocab_size = env.observation_space.n + env.action_space.n + 1
model_config = GPT2Config()

model_config.vocab_size = vocab_size
model_config.n_positions = exp_config.seq_len
model_config.n_ctx = exp_config.seq_len

model = GPT2LMHeadModel(model_config)

checkpoints_dir = get_file_path_from_config('checkpoints', exp_config)
model = accelerator.prepare(model)
accelerator.load_state(checkpoints_dir)




# Get option wrapper over model
seq_len = model.config.n_ctx
option_env = OptionEnvWrapper(env, model, seq_len, vocab_size, exp_config.num_options)



# Get agent
# agent_class = agent_name_map[exp_config.agent]
# agent_config = get_agent_config(exp_config.agent)
# agent = agent_class(option_env, agent_config)
agent = RandomAgent(option_env, {})



# Evaluation loop
trajectory = rollout(option_env, agent, exp_config, max_steps=1000, add_info=True)

total_option_reward = 0
total_env_reward = 0
for transition in trajectory:
    _, _, option_reward, _, info = transition
    total_option_reward += transition[2]
    for env_reward in info['rewards']:
        total_env_reward += env_reward


print("Total option reward:", total_option_reward)
print("Total env reward:", total_env_reward)





# # Save agent
# agent_path = get_file_path_from_config('agent.pkl', exp_config, mkdir=True)
# agent.save(agent_path)

# Save trajectories
trajectories_path = get_file_path_from_config('option_trajectories.pkl', exp_config, mkdir=True)
save_pickle(trajectory, trajectories_path)