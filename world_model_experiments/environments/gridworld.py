
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt

class KeyDoorGridWorld(gym.Env):
    """
    10x10 GridWorld with Key-Door mechanics and partial observability.

    Grid layout:
    - Agent: 'A'
    - Key: 'K' 
    - Door: 'D'
    - Goal: 'G'
    - Walls: '#'
    - Empty: '.'

    Partial observability: Agent only sees 5x5 window around itself.
    """

    def __init__(self, grid_size=10, observation_window=5, max_steps=100):
        super().__init__()

        self.grid_size = grid_size
        self.observation_window = observation_window
        self.max_steps = max_steps

        # Action space: 4 discrete actions (up, down, left, right)
        self.action_space = spaces.Discrete(4)

        # Observation space: flattened 5x5 grid window
        self.observation_space = spaces.Box(
            low=0, high=1,
            shape=(observation_window * observation_window * 6,),  # 6 channels for one-hot encoding
            dtype=np.float32
        )

        # State encoding channels
        self.channels = {
            'agent': 0,
            'key': 1,
            'door': 2,
            'goal': 3,
            'wall': 4,
            'empty': 5
        }

        self.num_channels = len(self.channels)

        # Initialize state
        self.reset()

    def _create_grid(self):
        """Create the initial grid layout."""
        grid = np.full((self.grid_size, self.grid_size), '.', dtype='<U1')

        # Add walls around the border
        grid[0, :] = '#'
        grid[-1, :] = '#'
        grid[:, 0] = '#'
        grid[:, -1] = '#'

        # Add some internal walls
        for i in range(3, 7):
            grid[i, 5] = '#'

        # Place key
        grid[1, 1] = 'K'

        # Place door (initially closed)
        grid[5, 8] = 'D'

        # Place goal
        grid[8, 8] = 'G'

        # Place agent
        grid[8, 1] = 'A'

        return grid

    def _get_observation(self):
        """Get partial observation (5x5 window around agent)."""
        half_window = self.observation_window // 2

        # Create observation grid
        obs_grid = np.zeros((self.observation_window, self.observation_window, self.num_channels), 
                           dtype=np.float32)

        # Fill observation grid
        for i in range(self.observation_window):
            for j in range(self.observation_window):
                grid_i = self.agent_pos[0] - half_window + i
                grid_j = self.agent_pos[1] - half_window + j

                # Check bounds
                if 0 <= grid_i < self.grid_size and 0 <= grid_j < self.grid_size:
                    cell = self.grid[grid_i, grid_j]

                    if cell == 'A':
                        obs_grid[i, j, self.channels['agent']] = 1
                    elif cell == 'K' and not self.has_key:
                        obs_grid[i, j, self.channels['key']] = 1
                    elif cell == 'D' and not self.door_open:
                        obs_grid[i, j, self.channels['door']] = 1
                    elif cell == 'G':
                        obs_grid[i, j, self.channels['goal']] = 1
                    elif cell == '#':
                        obs_grid[i, j, self.channels['wall']] = 1
                    else:
                        obs_grid[i, j, self.channels['empty']] = 1
                else:
                    # Out of bounds is treated as wall
                    obs_grid[i, j, self.channels['wall']] = 1

        return obs_grid.flatten()

    def reset(self, seed=None):
        """Reset the environment."""
        super().reset(seed=seed)

        # Create new grid
        self.grid = self._create_grid()

        # Find agent position
        agent_pos = np.argwhere(self.grid == 'A')[0]
        self.agent_pos = tuple(agent_pos)

        # Initialize state variables
        self.has_key = False
        self.door_open = False
        self.step_count = 0
        self.prev_agent_pos = self.agent_pos  # Initialize previous position

        # Get initial observation
        obs = self._get_observation()
        info = {'agent_pos': self.agent_pos, 'has_key': self.has_key, 'door_open': self.door_open}

        return obs, info

    def step(self, action):
        """Take a step in the environment."""
        self.step_count += 1

        # Define movement directions
        movements = {
            0: (-1, 0),  # up
            1: (1, 0),   # down
            2: (0, -1),  # left
            3: (0, 1)    # right
        }

        # Calculate new position
        move = movements[action]
        new_pos = (self.agent_pos[0] + move[0], self.agent_pos[1] + move[1])

        # Initialize reward and done
        reward = -0.01  # small step penalty
        terminated = False
        truncated = self.step_count >= self.max_steps

        # Check if new position is valid
        if 0 <= new_pos[0] < self.grid_size and 0 <= new_pos[1] < self.grid_size:
            cell = self.grid[new_pos]

            if cell == '#':  # Wall
                # Can't move into wall
                new_pos = self.agent_pos
                reward = -0.1
            elif cell == 'K' and not self.has_key:
                # Collect key
                self.has_key = True
                self.grid[new_pos] = '.'  # Remove key from grid
                reward = 1.0
                self.agent_pos = new_pos
            elif cell == 'D' and not self.door_open:
                if self.has_key:
                    # Open door with key
                    self.door_open = True
                    self.grid[new_pos] = '.'  # Door becomes passable
                    reward = 1.0
                    self.agent_pos = new_pos
                else:
                    # Can't pass through closed door without key
                    new_pos = self.agent_pos
                    reward = -0.1
            elif cell == 'G':
                # Reach goal
                reward = 10.0
                terminated = True
                self.agent_pos = new_pos
            else:
                # Empty cell or open door
                self.agent_pos = new_pos

        # Update grid
        if self.agent_pos != self.prev_agent_pos:
            old_pos = self.prev_agent_pos
            if self.grid[old_pos] == 'A':
                self.grid[old_pos] = '.' if old_pos != (1, 1) or self.has_key else 'K'
            if self.grid[self.agent_pos] in ['.', 'K', 'D', 'G']:
                self.grid[self.agent_pos] = 'A'

        self.prev_agent_pos = self.agent_pos

        # Get observation
        obs = self._get_observation()
        info = {
            'agent_pos': self.agent_pos,
            'has_key': self.has_key,
            'door_open': self.door_open,
            'step': self.step_count
        }

        return obs, reward, terminated, truncated, info

    def render(self, mode='human'):
        """Render the environment."""
        if mode == 'human':
            print_grid = np.copy(self.grid)
            print(f"Step: {self.step_count}, Has Key: {self.has_key}, Door Open: {self.door_open}\n")
            for row in print_grid:
                print(' '.join(row))
            print()
        elif mode == 'rgb_array':
            # Create RGB array representation
            img = np.zeros((self.grid_size * 10, self.grid_size * 10, 3), dtype=np.uint8)

            color_map = {
                '.': (255, 255, 255),  # white
                '#': (100, 100, 100),  # gray
                'A': (0, 0, 255),      # blue
                'K': (255, 255, 0),    # yellow
                'D': (165, 42, 42),    # brown
                'G': (0, 255, 0),      # green
            }

            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    cell = self.grid[i, j]
                    color = color_map.get(cell, (255, 255, 255))
                    img[i*10:(i+1)*10, j*10:(j+1)*10] = color

            return img

    def get_full_state(self):
        """Get the full state for evaluation purposes."""
        return {
            'grid': np.copy(self.grid),
            'agent_pos': self.agent_pos,
            'has_key': self.has_key,
            'door_open': self.door_open,
            'step': self.step_count
        }


class PartialObservabilityWrapper(gym.Wrapper):
    """Wrapper to add observation masking for partial observability experiments."""

    def __init__(self, env, occlusion_level=0.0):
        super().__init__(env)
        self.occlusion_level = occlusion_level

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs = self._apply_occlusion(obs)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs = self._apply_occlusion(obs)
        return obs, reward, terminated, truncated, info

    def _apply_occlusion(self, obs):
        """Randomly mask observation channels based on occlusion level."""
        if self.occlusion_level > 0:
            mask = np.random.random(obs.shape) > self.occlusion_level
            obs = obs * mask
        return obs
