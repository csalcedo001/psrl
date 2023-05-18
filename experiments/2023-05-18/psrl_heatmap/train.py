from psrl.train import train

from utils import load_experiment_config, setup_env_and_agent



# Get experiment configuration
exp_config = load_experiment_config('exp_config.yaml')

# Get environment and agent
env, agent = setup_env_and_agent(exp_config)

# Train and save agent weights
train(env, agent, agent.config)
agent.save('agent_weights.pkl')