import os
import copy
from gym import spaces
import numpy as np
from dotmap import DotMap

from .env import Env
from .utils import get_grid_from_file


class GridworldEnv(Env):
    def __init__(self, config):
        Env.__init__(self)

        self.is_episodic = config.is_episodic

        self.grid = None
        self.rows = None
        self.cols = None
        self.p = None
        self.r = None
        self.state_from_pos = None
        self.pos_from_state = None
        self.start_positions = None
        self.goal_states = None

        self._setup_from_grid(config.grid)

        self.reset()
    
    def reset(self):
        self.pos = self._sample_start_pos()

        return self._get_state_from_pos(self.pos)

    def step(self, action):
        pos = self.pos
        state = self._get_state_from_pos(pos)

        attempted_next_pos = self._attempt_next_pos(self.pos, action)
        attempted_next_state = self._get_state_from_pos(attempted_next_pos)

        next_pos = self._get_next_pos(self.pos, action)
        next_state = self._get_state_from_pos(next_pos)

        self.pos = next_pos

        reward = self.r[state, action]
        done = self.is_episodic and attempted_next_state in self.goal_states

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
        return self.p[:], self.r[:]
    
    def get_grid(self):
        return copy.deepcopy(self.grid)
    

    def _setup_grid(self):
        ''' Set grid, rows and cols, while validating grid shape
        '''

        grid = self.grid

        self.rows = len(grid)
        if self.rows == 0:
            raise ValueError('Empty grid')
        
        self.cols = len(grid[0])
        if self.cols == 0:
            raise ValueError('Empty grid')

        # try:
        #     np.array(grid)
        # except:
        #     raise ValueError('Invalid grid. Check that all rows have the same length')
        
    def _setup_states(self):
        ''' Given a grid, setup state and action spaces as well
            as the mapping between states and positions (forward and
            backward).
        '''

        grid_shape = (self.rows, self.cols)

        self.state_from_pos = -np.ones(grid_shape)
        self.pos_from_state = []

        start_states = []
        goal_states = []
        remaining_states = []

        state = 0
        for i in range(self.rows):
            for j in range(self.cols):
                cell = self.grid[i][j]

                if cell not in [' ', '.', 'S', 'G', 'R', '#']:
                    raise ValueError('Invalid grid character')

                if cell == '#':
                    continue

                pos = np.array([i, j])

                # Map states to positions and vice versa
                self.state_from_pos[i, j] = state
                self.pos_from_state.append(pos)

                # Categorize states
                if cell == 'S':
                    start_states.append(state)
                elif cell == 'G':
                    goal_states.append(state)
                else:
                    remaining_states.append(state)

                state += 1
        
        # Up till this point, all states have been observed
        num_states = state
        
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Discrete(num_states)
        

        # In case there are no initial states, by default
        # we sample from any state that is not a terminal
        # state
        if len(start_states) != 0:
            self.start_states = start_states
        else:
            self.start_states = remaining_states
        
        self.goal_states = goal_states
        
    
    def _setup_p_and_r(self):
        n_s = self.observation_space.n
        n_a = self.action_space.n

        p = np.zeros((n_s, n_a, n_s))
        r = np.zeros((n_s, n_a))

        for state in range(n_s):
            pos = self._get_pos_from_state(state)
            
            for action in range(n_a):
                attempted_next_pos = self._attempt_next_pos(pos, action)
                next_pos = self._get_next_pos(pos, action)

                # If state is terminal, stay in place
                next_state = self._get_state_from_pos(next_pos)
                if next_state in self.goal_states:
                    next_pos = pos
                    next_state = self._get_state_from_pos(next_pos)
                
                attempted_next_state = self._get_state_from_pos(attempted_next_pos)
                
                if attempted_next_state in self.goal_states and state not in self.goal_states:
                    p[state, action, self.start_states] = 1. / len(self.start_states)
                else:
                    p[state, action, next_state] = 1


                # Receive reward only when i) next state is terminal ii) next
                # state is not terminal but is a reward state iii) next state
                # is not terminal and but is a penalty state
                grid_char = self.grid[next_pos[0]][next_pos[1]]
                if attempted_next_state in self.goal_states or grid_char == 'R':
                    r[state, action] = 1
                elif grid_char == '.':
                    r[state, action] = -1
        
        self.p = p
        self.r = r

    def _setup_from_grid(self, grid):
        self.grid = grid

        self._setup_grid()
        self._setup_states()
        self._setup_p_and_r()

        # Compute transition and reward operators
        self.get_p_and_r()

    def _sample_start_pos(self):
        state = self.start_states[np.random.randint(0, len(self.start_states))]
        pos = self._get_pos_from_state(state)
        return pos
    
    def _get_state_from_pos(self, pos):
        state_from_pos = int(self.state_from_pos[pos[0], pos[1]])

        return state_from_pos

    def _get_pos_id(self, pos):
        return pos[0] * self.rows + pos[1]
    
    def _attempt_next_pos(self, pos, action):
        ''' Move position according to action ignoring whether
            the state is terminal or not
        '''

        # Given an action, compute shift in grid
        # up: 0, right: 1, down: 2, left: 3
        axis = action % 2
        direction = action // 2

        # Note: This correction is made for convenience in plots
        if axis == 0:
            direction = 1 - direction

        next_pos = pos.copy()
        next_pos[axis] += 1 if direction == 0 else -1


        # Check if the next position is outside the grid or would
        # collide with a wall. If it collides, then the agent stays
        # in the same position
        is_outside_grid = np.any(np.minimum(np.maximum(next_pos, 0), np.array([self.rows, self.cols]) - 1) != next_pos)
        is_touching_wall = not is_outside_grid and self.state_from_pos[next_pos[0], next_pos[1]] == -1
        if is_outside_grid or is_touching_wall:
            next_pos = pos.copy()
        
        return next_pos


    
    def _get_next_pos(self, pos, action):
        # Get current state from pos
        state = self._get_state_from_pos(pos)

        # If the task is episodic and the agent is in a terminal
        # state (which shouldn't occur), for convenience we make
        # the agent not able to scape that state
        if state in self.goal_states:
            return pos.copy()

        next_pos = self._attempt_next_pos(pos, action)
        
        # If the task is continuous and the agent is in a terminal
        # state, then we sample a new start position
        next_state = self._get_state_from_pos(next_pos)
        if next_state in self.goal_states:
            next_pos = self._sample_start_pos()

        return next_pos

    def _get_pos_from_state(self, state):
        return self.pos_from_state[state][:]



class TwoRoomGridworldEnv(GridworldEnv):
    def __init__(self, config):
        gridworld_path = os.path.join(os.path.dirname(__file__), 'maps', 'two_room.txt')

        grid = get_grid_from_file(gridworld_path)

        config.grid = grid

        GridworldEnv.__init__(self, config)


class FourRoomGridworldEnv(GridworldEnv):
    def __init__(self, config):
        gridworld_path = os.path.join(os.path.dirname(__file__), 'maps', 'four_room.txt')

        grid = get_grid_from_file(gridworld_path)

        config.grid = grid

        GridworldEnv.__init__(self, config)