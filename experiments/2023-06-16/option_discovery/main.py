from option_discovery import OptionDiscoveryAgent, DummyEnvironment

# Define your environment and its parameters
env = DummyEnvironment()
max_options = 3
max_option_length = 10
exploration_budget = 1000

# Create the option discovery agent
agent = OptionDiscoveryAgent(env, max_options, max_option_length, exploration_budget)

# Discover options
discovered_options = agent.discover_options()

# Print the discovered options
for i, option in enumerate(discovered_options):
    print(f"Option {i+1}:")
    print(f"Initial state: {option.initial_state}")
    # Add more details about the option if necessary

# Optionally, you can use the discovered options for further tasks
# For example, you can use them in a higher-level policy for hierarchical RL
