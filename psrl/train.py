from tqdm import tqdm

def train_episode(env, agent, render=False, verbose=False, max_steps=100, pbar=None, add_info=False):
    state = env.reset()
    if render:
        env.render()
    
    agent.reset(state)

    external_pbar = pbar is not None
    if pbar is None:
        pbar = tqdm(total=max_steps)

    trajectory = []
    while pbar.n < max_steps:
        action = agent.act(state)

        next_state, reward, done, info = env.step(action)

        transition = (state, action, reward, next_state)
        if add_info:
            transition = transition + (info,)
        trajectory.append(transition)

        agent.observe(transition)

        if verbose:
            print('(i:{}) state: {}, action: {}, reward: {}, next_state: {}'.format(
                pbar.n, state, action, reward, next_state
            ))


        if render:
            env.render()
        
        agent.update()

        pbar.update(1)

        if done:
            break
            
        state = next_state
    
    if not external_pbar:
        pbar.close()
    
    return trajectory


def train(env, agent, config, render=False, verbose=False, max_steps=100, add_info=False):
    agent_trajectories = []

    pbar = tqdm(total=max_steps)

    while pbar.n < max_steps:
        agent_trajectory = train_episode(env, agent, render, verbose, max_steps, pbar=pbar, add_info=add_info)
        agent_trajectories += agent_trajectory
    
    pbar.close()
    
    return agent_trajectories