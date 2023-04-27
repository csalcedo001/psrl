from tqdm import tqdm

def train_episode(env, agent, render=False, verbose=False, max_steps=100):
    state = env.reset()
    if render:
        env.render()

    trajectory = []
    for i in range(max_steps):
        action = agent.act(state)

        next_state, reward, done, _ = env.step(action)

        transition = (state, action, reward, next_state)
        trajectory.append(transition)

        agent.observe(transition)

        if verbose:
            print('(i:{}) state: {}, action: {}, reward: {}, next_state: {}'.format(
                i, state, action, reward, next_state
            ))


        if render:
            env.render()


        if done:
            agent.update()
            break
            
        state = next_state
    
    return trajectory


def train(env, agent, config, render=False, verbose=False, max_steps=100):
    agent_trajectories = []
    remaining_steps = config.max_steps

    pbar = tqdm(total=config.max_steps)

    while remaining_steps > 0:
        agent_trajectory = train_episode(env, agent, render, verbose, max_steps)
        agent_trajectories += agent_trajectory

        elapsed_steps = len(agent_trajectory)
        remaining_steps -= elapsed_steps

        pbar.update(elapsed_steps)
    
    return agent_trajectories