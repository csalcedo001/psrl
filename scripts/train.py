import copy

from psrl.agents import OptimalAgent
from psrl.config import get_env_config, get_agent_config
from psrl.utils import rollout, env_name_map, agent_name_map

from arg_utils import get_parser, get_config



parser = get_parser()
args = parser.parse_args()
config = get_config(args)


# Get environment
env_class = env_name_map[args.env]
env_config = get_env_config(args.env)
env = env_class(env_config)

# Get agent
agent_class = agent_name_map[args.agent]
agent_config = get_agent_config(args.agent)
agent = agent_class(env, agent_config)

# Get optimal policy for environment
oracle_config = get_agent_config('optimal')
oracle = OptimalAgent(env, oracle_config)



print("Observation_space:", env.observation_space)
print("Action space:", env.action_space)


iteration = 0
regret = 0

for episode in range(args.episodes):
    state = env.reset()

    while True:
        oracle_env = copy.deepcopy(env)

        # Get agent's action
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)

        # Get oracle's action
        oracle_action = oracle.act(state)
        _, oracle_reward, _, _ = oracle_env.step(oracle_action)

        regret += oracle_reward - reward

        transition = (state, action, reward, next_state)
        agent.observe(transition)

        print('[{}/{}] Iteration: {}, State: {}, Action: {}, Next State: {}, Reward: {}, Done: {}, Regret: {:.2f}'.format(episode, args.episodes, iteration, state, action, next_state, reward, done, regret))

        if done:
            agent.update()
            break
            
        iteration += 1





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