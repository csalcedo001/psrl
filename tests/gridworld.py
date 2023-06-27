import unittest
from parameterized import parameterized
import numpy as np
from dotmap import DotMap
import gym

from psrl import agents
from psrl.agents import UCRL2Agent
from psrl.envs import GridworldEnv
from psrl.config import get_env_config

from testcase import TestCase
import utils


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



square = [
    [' ', ' '],
    [' ', ' '],
]

two_goals = [
    [' ', ' ', 'G'],
    ['S', ' ', ' '],
    [' ', ' ', 'G'],
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
    [' ', ' ', ' '],
    ['#', '#', ' '],
    [' ', ' ', ' '],
    [' ', '#', '#'],
    [' ', ' ', ' '],
]

big = [[' '] * 20 for _ in range(20)]


# Mapping from names to grids
shape_to_grid = {
    'square': square,
    'two_goals': two_goals,
    'two_starts': two_starts,
    'middle_wall': middle_wall,
    'zig': zig,
    'big': big
}

# Fill up data and make parameterized test cases
gridworld_data = get_gridworld_data(shape_to_grid)
gridworld_parameterized_tests = [
    (shape, episodic)
    for episodic in [True, False]
    for shape in gridworld_data
]


# Set environment according to parameterized test
def setup_env(shape: str, episodic: bool):
    if shape not in gridworld_data:
        raise ValueError('Invalid shape: {}'.format(shape))
    
    # Set up grid
    grid = gridworld_data[shape]['grid']
    grid[0][0] = 'S'
    if episodic:
        grid[-1][-1] = 'G'

    # Setup environment
    env_config = get_env_config('gridworld')
    env_config.grid = grid
    env = GridworldEnv(env_config)

    return env


class TestGridworld(TestCase):
    # Make sure that number of states and actions are correct
    @parameterized.expand(gridworld_parameterized_tests)
    def test_num_states_actions(self, shape, episodic):
        env = setup_env(shape=shape, episodic=episodic)

        n_s = env.observation_space.n
        n_a = env.action_space.n
        
        err_msg = utils.get_error_msg(n_s, gridworld_data[shape]['n_s'], context={
            'shape': shape,
            'episodic': episodic,
            'n_s': gridworld_data[shape]['n_s'],
        })
        self.assertEqual(n_s, gridworld_data[shape]['n_s'], msg=err_msg)
        self.assertEqual(n_a, 4, msg=n_a)
    

    # Check that the mappings from state id to grid position and
    # vice versa are correct
    @parameterized.expand(gridworld_parameterized_tests)
    def test_state_to_pos(self, shape, episodic):
        env = setup_env(shape=shape, episodic=episodic)

        n_s = env.observation_space.n
        n_rows = len(env.grid)
        n_cols = len(env.grid[0])

        for state in range(n_s):
            pos = env._get_pos_from_state(state)
            self.assertEqual(len(pos), 2, msg=pos)
            self.assertTrue(0 <= pos[0] < n_rows, msg=pos)
            self.assertTrue(0 <= pos[1] < n_cols, msg=pos)

            # Expect same state while mapping back
            state_ = env._get_state_from_pos(pos)
            self.assertEqual(state, state_, msg=(shape, state, pos, state_))


    # Check that the transition operator is correct by checking
    # that the values for any given (s, a) pair sum up to one
    @parameterized.expand(gridworld_parameterized_tests)
    def test_p_well_formed(self, shape, episodic):
        env = setup_env(shape=shape, episodic=episodic)

        p, r = env.get_p_and_r()

        n_s = env.observation_space.n
        n_a = env.action_space.n

        for s in range(n_s):
            for a in range(n_a):
                self.assertEqual(p[s, a].sum(), 1, msg=(shape, s, a, p[s, a]))


    # Check that the transition operator and step function give
    # the same results
    @parameterized.expand(gridworld_parameterized_tests)
    def test_step_eq_p(self, shape, episodic):
        env = setup_env(shape=shape, episodic=episodic)

        p, r = env.get_p_and_r()

        n_s = env.observation_space.n
        n_a = env.action_space.n

        for s in range(n_s):
            for a in range(n_a):
                pos = env._get_pos_from_state(s)
                next_pos = env._get_next_pos(pos=pos, action=a)
                next_s = env._get_state_from_pos(next_pos)

                # Environment is deterministic, so for any (s, a) there is
                # a dirac distribution to a single next state
                self.assertEqual(p[s, a, next_s], 1, msg=(shape, s, a, next_s, p[s, a, next_s]))


    # Check that the reward function is correct: only gives
    # reward while transitioning to goal state
    @parameterized.expand(gridworld_parameterized_tests)
    def test_step_eq_r(self, shape, episodic):
        env = setup_env(shape=shape, episodic=episodic)

        p, r = env.get_p_and_r()

        n_s = env.observation_space.n
        n_a = env.action_space.n

        for s in range(n_s):
            for a in range(n_a):
                pos = env._get_pos_from_state(s)
                next_pos = env._get_next_pos(pos=pos, action=a)
                next_s = env._get_state_from_pos(next_pos)

                # Deterministic reward only when transitioning to goal state
                if next_s in env.goal_states and s not in env.goal_states:
                    expected_r = 1
                else:
                    expected_r = 0
                
                self.assertEqual(r[s, a, next_s], expected_r, msg=(shape, s, a, next_s, r[s, a, next_s]))




if __name__ == '__main__':
    unittest.main()