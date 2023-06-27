from psrl.envs import GridworldEnv
from psrl.config import get_env_config


def count_in_grid(grid, items):
    count = 0
    for row in grid:
        for item in row:
            if item in items:
                count += 1
    return count

def get_gridworld_data(raw_gridworld_data):
    gridworld_data = {}
    for shape, grid in raw_gridworld_data.items():
        n_rows = len(grid)
        n_cols = len(grid[0])
        n_walls = count_in_grid(grid, ['#'])

        shape_data = {
            'grid': grid,
            'n_rows': n_rows,
            'n_cols': n_cols,
            'n_walls': n_walls,
            'n_s': n_rows * n_cols - n_walls,
            'n_starts': count_in_grid(grid, ['S']),
            'n_goals': count_in_grid(grid, ['G']),
        }

        gridworld_data[shape] = shape_data
    
    return gridworld_data


# Gridworld test maps


simple = [
    ['S', 'G'],
]

square = [
    ['S', ' '],
    [' ', 'G'],
]

two_goals = [
    [' ', ' ', 'G'],
    ['S', ' ', ' '],
    [' ', ' ', 'G'],
]

no_goal = [
    [' ', ' ', ' '],
    ['S', ' ', ' '],
    [' ', ' ', ' '],
]

two_starts = [
    ['S', ' ', ' '],
    [' ', ' ', ' '],
    ['S', ' ', ' '],
]

middle_wall = [
    [' ', ' ', ' '],
    ['S', '#', 'G'],
    [' ', ' ', ' '],
]

zig = [
    ['S', ' ', ' '],
    ['#', '#', ' '],
    [' ', ' ', ' '],
    [' ', '#', '#'],
    [' ', ' ', 'G'],
]

big = [[' '] * 20 for _ in range(20)]
big[0][0] = 'S'
big[-1][-1] = 'G'


# Mapping from names to grids
shape_to_grid = {
    'simple': simple,
    'square': square,
    'two_goals': two_goals,
    'no_goal': no_goal,
    'two_starts': two_starts,
    'middle_wall': middle_wall,
    'zig': zig,
    'big': big
}

# Fill up data and make parameterized test cases
gridworld_data = get_gridworld_data(shape_to_grid)


# Set environment according to parameterized test
def setup_env(shape: str, episodic: bool):
    if shape not in gridworld_data:
        raise ValueError('Invalid shape: {}'.format(shape))
    
    # Setup environment
    grid = gridworld_data[shape]['grid']

    env_config = get_env_config('gridworld')
    env_config.grid = grid
    env_config.episodic = episodic

    env = GridworldEnv(env_config)

    return env