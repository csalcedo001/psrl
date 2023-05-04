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
    UCRL2Agent,
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
    'ucrl2': UCRL2Agent,
    'random_agent': RandomAgent,
    'optimal': OptimalAgent,
}