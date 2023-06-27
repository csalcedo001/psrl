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

gridworld_data = {
    'square': {
        'grid': square,
        'n_s': 4,
    },
    'two_goals': {
        'grid': two_goals,
        'n_s': 9,
    },
    'two_starts': {
        'grid': two_starts,
        'n_s': 9,
    },
    'middle_wall': {
        'grid': middle_wall,
        'n_s': 8,
    },
    'zig': {
        'grid': zig,
        'n_s': 11,
    },
    'big': {
        'grid': big,
        'n_s': 400,
    }
}

gridworld_parameterized_tests = [
    (shape, episodic)
    for episodic in [True, False]
    for shape in gridworld_data
]



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

            # Return and expect same
            state_ = env._get_state_from_pos(pos)
            self.assertEqual(state, state_, msg=(state, pos, state_))


    @parameterized.expand(gridworld_parameterized_tests)
    def test_p_well_formed(self, shape, episodic):
        env = setup_env(shape=shape, episodic=episodic)

        p, r = env.get_p_and_r()

        n_s = env.observation_space.n
        n_a = env.action_space.n

        for s in range(n_s):
            for a in range(n_a):
                self.assertEqual(p[s, a].sum(), 1, msg=(s, a, p[s, a]))


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

                self.assertEqual(p[s, a, next_s], 1, msg=(s, a, next_s, p[s, a, next_s]))




if __name__ == '__main__':
    unittest.main()