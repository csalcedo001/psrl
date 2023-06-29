import unittest
from parameterized import parameterized

from testcase import TestCase
from env_utils import gridworld_data, setup_env
import utils




gridworld_parameterized_tests = [
    (shape, episodic)
    for episodic in [True, False]
    for shape in gridworld_data
]

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
        n_rows = gridworld_data[shape]['n_rows']
        n_cols = gridworld_data[shape]['n_cols']

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
    # consistent results
    @parameterized.expand(gridworld_parameterized_tests)
    def test_step_as_p(self, shape, episodic):
        env = setup_env(shape=shape, episodic=episodic)

        p, _ = env.get_p_and_r()

        n_s = env.observation_space.n
        n_a = env.action_space.n

        for s in range(n_s):
            for a in range(n_a):
                pos = env._get_pos_from_state(s)
                env.pos = pos

                next_s, _, _, _ = env.step(a)

                if s in env.goal_states:
                    expected_prob = 1.0 / len(env.start_states)
                else:
                    expected_prob = 1.0
                
                self.assertEqual(p[s, a, next_s], expected_prob, msg=(shape, s, a, next_s, p[s, a, next_s]))


    # Check that the reward function and step function give
    # the same reward (reward is deterministic)
    @parameterized.expand(gridworld_parameterized_tests)
    def test_step_as_r(self, shape, episodic):
        env = setup_env(shape=shape, episodic=episodic)

        _, r = env.get_p_and_r()

        n_s = env.observation_space.n
        n_a = env.action_space.n

        for s in range(n_s):
            for a in range(n_a):
                pos = env._get_pos_from_state(s)
                env.pos = pos
                _, reward, _, _ = env.step(a)

                self.assertEqual(r[s, a], reward, msg=(shape, s, a, r[s, a], reward))



    # Check that the reward function is correct: only gives
    # reward while transitioning to goal state
    @parameterized.expand(gridworld_parameterized_tests)
    def test_r_value(self, shape, episodic):
        env = setup_env(shape=shape, episodic=episodic)

        _, r = env.get_p_and_r()

        n_s = env.observation_space.n
        n_a = env.action_space.n

        for s in range(n_s):
            for a in range(n_a):
                pos = env._get_pos_from_state(s)
                next_pos = env._get_next_pos(pos, a)
                next_s = env._get_state_from_pos(next_pos)

                # Deterministic reward only when transitioning to goal state
                grid_char = gridworld_data[shape]['grid'][pos[0]][pos[1]]
                if s in env.goal_states or next_s in env.goal_states:
                    expected_r = 1
                elif grid_char == 'R':
                    expected_r = 1
                elif grid_char == '.':
                    expected_r = -1
                else:
                    expected_r = 0
            
                self.assertEqual(r[s, a], expected_r, msg=(shape, s, a, next_s, r[s, a]))




if __name__ == '__main__':
    unittest.main(verbosity=2)