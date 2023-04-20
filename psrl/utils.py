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


def rollout(env, agent, episodes):
    trajectories = []
    for episode in range(episodes):
        state = env.reset()

        trajectory = []
        while True:
            action = agent.act(state)

            next_state, reward, done, _ = env.step(action)

            transition = (state, action, reward, next_state)
            trajectory.append(transition)

            if done:
                break
        
        trajectories.append(trajectory)
    
    return trajectories



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