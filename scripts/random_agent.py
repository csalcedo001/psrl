from psrl.envs.riverswim import RiverSwimEnv

env = RiverSwimEnv()

print("Observation_space:", env.observation_space)
print("Action space:", env.action_space)

iterations = 20

state = env.reset()
for iteration in range(iterations):
    action = env.action_space.sample()

    next_state, reward, done, _ = env.step(action)

    print('(iteration: {}) State: {}, Action: {}, Next State: {}, Reward: {}, Done: {}'.format(iteration, state, action, next_state, reward, done))

    if done:
        state = env.reset()
    else:
        state = next_state