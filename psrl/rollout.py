from tqdm import tqdm

def rollout_episode(env, agent, render=False, verbose=False, max_steps=100, pbar=None):
    state = env.reset()
    if render:
        env.render()

    if pbar is None:
        pbar = tqdm(total=max_steps)

    trajectory = []
    while pbar.n < max_steps:
        action = agent.act(state)

        next_state, reward, done, _ = env.step(action)

        transition = (state, action, reward, next_state)
        trajectory.append(transition)

        if verbose:
            print('(i:{}) state: {}, action: {}, reward: {}, next_state: {}'.format(
                pbar.n, state, action, reward, next_state
            ))


        if render:
            env.render()

        pbar.update(1)

        if done:
            break

        state = next_state
    
    pbar.close()
    
    return trajectory


def rollout(env, agent, config, render=False, verbose=False, max_steps=100):
    agent_trajectories = []
    
    pbar = tqdm(total=max_steps)
    
    while pbar.n < max_steps:
        agent_trajectory = rollout_episode(env, agent, render, verbose, max_steps, pbar=pbar)
        agent_trajectories += agent_trajectory
    
    return agent_trajectories