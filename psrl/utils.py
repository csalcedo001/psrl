from .envs import (
    RiverSwimEnv,
    RandomMDPEnv,
    GridworldEnv,
    TwoRoomGridworldEnv,
    FourRoomGridworldEnv,
)

from .agents import (
    RandomAgent,
    PSRLAgent,
    OptimalAgent,
)


def rollout_episode(env, agent, render=False, verbose=False, max_steps=100):
    state = env.reset()
    if render:
        env.render()

    trajectory = []
    for i in range(max_steps):
        action = agent.act(state)

        next_state, reward, done, _ = env.step(action)

        transition = (state, action, reward, next_state)
        trajectory.append(transition)

        if verbose:
            print('(i:{}) state: {}, action: {}, reward: {}, next_state: {}'.format(
                i, state, action, reward, next_state
            ))


        if render:
            env.render()

        if done:
            break

        state = next_state
    
    return trajectory


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


env_name_map = {
    'gridworld': GridworldEnv,
    'tworoom': TwoRoomGridworldEnv,
    'fourroom': FourRoomGridworldEnv,
    'riverswim': RiverSwimEnv,
    'randommdp': RandomMDPEnv,
}

agent_name_map = {
    'random_agent': RandomAgent,
    'psrl': PSRLAgent,
    'optimal': OptimalAgent,
}