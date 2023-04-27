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


env_name_map = {
    'riverswim': RiverSwimEnv,
    'randommdp': RandomMDPEnv,
    'tworoom': TwoRoomGridworldEnv,
    'fourroom': FourRoomGridworldEnv,
    'gridworld': GridworldEnv,
}

agent_name_map = {
    'psrl': PSRLAgent,
    'random_agent': RandomAgent,
    'optimal': OptimalAgent,
}