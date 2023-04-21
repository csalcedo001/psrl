import os
from gym import spaces
import numpy as np
from dotmap import DotMap

from .env import Env
from .utils import get_grid_from_file


class GridworldEnv(Env):
    def __init__(self, config):
        Env.__init__(self)

        self._setup_from_grid(config.grid)

        self.reset()
    
    def reset(self):
        self.pos = self.start_states[np.random.randint(0, len(self.start_states))]

        return self._get_state_from_pos(self.pos)

    def step(self, action):
        next_pos = self._get_next_pos(action)
        self.pos = next_pos

        next_state = self._get_state_from_pos(next_pos)
        reward = self.reward_grid[next_pos[0], next_pos[1]]
        # reward = 6 - np.linalg.norm(self.goal_states[0] - self.pos)
        done = self.state_id[next_pos[0], next_pos[1]] == 0

        return next_state, reward, done, {}

    def render(self):
        pos_y, pos_x = self.pos

        print(u'\u250c' + u'\u2500' * self.cols + u'\u2510')
        for i in range(self.rows):
            print(u'\u2502', end='')
            for j in range(self.cols):
                if pos_y != i or pos_x != j:
                    print(self.grid[i][j], end='')
                else:
                    print('X', end='')
            print(u'\u2502')
        print(u'\u2514' + u'\u2500' * self.cols + u'\u2518')

    def get_p_and_r(self):
        n_s = self.observation_space.n
        n_a = self.action_space.n

        p = np.zeros((n_s, n_a, n_s))
        r = np.zeros((n_s, n_a, n_s))

        for i in range(self.rows):
            for j in range(self.cols):
                pos = np.array([i, j])
                state = self._get_state_from_pos(pos)

                if state <= 0:
                    continue

                for action in range(self.action_space.n):
                    next_pos = self._get_next_pos(action, pos)
                    next_state = self._get_state_from_pos(next_pos)

                    p[state, action, next_state] = 1
                    r[state, action, next_state] = self.reward_grid[next_pos[0], next_pos[1]]

        return p, r

    def _setup_from_grid(self, grid):
        self.grid = grid

        self.rows = len(grid)
        if self.rows == 0:
            raise ValueError('Empty grid')
        
        self.cols = len(grid[0])
        if self.cols == 0:
            raise ValueError('Empty grid')

        grid_shape = (self.rows, self.cols)


        self.reward_grid = np.zeros(grid_shape)
        self.state_id = -np.ones(grid_shape)

        self.start_states = []
        self.goal_states = []

        num_states = 1
        for i in range(grid_shape[0]):
            for j in range(grid_shape[1]):
                cell = grid[i][j]

                if cell == '#':
                    continue

                pos = (i, j)
                pos_id = self._get_pos_id(pos)


                # Terminal states
                if cell == 'T':
                    self.state_id[i, j] = 0     # Convention: 0 -> terminal state
                    self.reward_grid[i, j] = 1
                    self.goal_states.append(pos_id)

                    continue


                self.state_id[i, j] = num_states

                if cell == 'S':
                    self.start_states.append(np.array(pos))
                elif cell == '.':
                    self.reward_grid[i, j] = -1
                elif cell == ' ':
                    pass
                else:
                    raise ValueError('Invalid grid character')

                num_states += 1
                
        num_states += 1 - len(self.goal_states)
        
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Discrete(num_states)
    
    def _get_state_from_pos(self, pos):
        state_id = int(self.state_id[pos[0], pos[1]])

        return state_id

    def _get_pos_id(self, pos):
        return pos[0] * self.rows + pos[1]
    
    def _state_to_vec(self, state):
        vec = np.zeros(self.rows * self.cols)
        vec[state] = 1
        return vec
    
    def _get_next_pos(self, action, pos=None):
        # up: 0, right: 1, down: 2, left: 3
        axis = action % 2
        direction = action // 2

        # Correct for printing
        if axis == 0:
            direction = 1 - direction

        next_pos = self.pos.copy() if pos is None else pos.copy()
        next_pos[axis] += 1 if direction == 0 else -1

        is_outside_grid = np.any(np.minimum(np.maximum(next_pos, 0), np.array([self.rows, self.cols]) - 1) != next_pos)
        is_touching_wall = not is_outside_grid and self.state_id[next_pos[0], next_pos[1]] == -1
        if is_outside_grid or is_touching_wall:
            next_pos = self.pos.copy()

        return next_pos



class TwoRoomGridworldEnv(GridworldEnv):
    def __init__(self, config):
        gridworld_path = os.path.join(os.path.dirname(__file__), 'maps', 'two_room.txt')

        grid = get_grid_from_file(gridworld_path)

        config = DotMap({'grid': grid})

        GridworldEnv.__init__(self, config)


class FourRoomGridworldEnv(GridworldEnv):
    def __init__(self, config):
        gridworld_path = os.path.join(os.path.dirname(__file__), 'maps', 'four_room.txt')

        grid = get_grid_from_file(gridworld_path)

        config = DotMap({'grid': grid})

        GridworldEnv.__init__(self, config)