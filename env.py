import numpy as np
import gymnasium as gym
from gymnasium import spaces

class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, size=10, max_steps=200, seed=None):
        super().__init__()

        self.size = size
        self.max_steps = max_steps
        self._seed = seed

        self.action_space = spaces.Discrete(4)  # up, down, left, right
        self.observation_space = spaces.Box(
            low=0,
            high=size - 1,
            shape=(4,),
            dtype=np.int32
        )

        self.steps = 0
        self.reset(seed=seed)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if seed is not None:
            np.random.seed(seed)

        self.grid = np.zeros((self.size, self.size), dtype=np.int8)

        self.start = (0, 0)
        self.goal = (self.size - 1, self.size - 1)

        # Place obstacles
        for _ in range(15):
            x, y = np.random.randint(0, self.size, size=2)
            if (x, y) not in [self.start, self.goal]:
                self.grid[x, y] = 1

        self.agent_pos = np.array(self.start, dtype=np.int32)
        self.steps = 0

        return self._get_obs(), {}

    def _get_obs(self):
        return np.array([
            self.agent_pos[0],
            self.agent_pos[1],
            self.goal[0],
            self.goal[1]
        ], dtype=np.int32)

    def step(self, action):
        self.steps += 1

        moves = {
            0: (-1, 0),  # up
            1: (1, 0),   # down
            2: (0, -1),  # left
            3: (0, 1),   # right
        }

        move = np.array(moves[action])
        new_pos = self.agent_pos + move

        reward = -1  # step penalty
        terminated = False
        truncated = False

        # Valid move
        if (
            0 <= new_pos[0] < self.size and
            0 <= new_pos[1] < self.size and
            self.grid[new_pos[0], new_pos[1]] == 0
        ):
            self.agent_pos = new_pos
        else:
            reward = -10  # wall/obstacle penalty

        # Goal reached
        if tuple(self.agent_pos) == self.goal:
            reward = 100
            terminated = True

        # Max step limit
        if self.steps >= self.max_steps:
            truncated = True

        return self._get_obs(), reward, terminated, truncated, {}

    def render(self):
        grid = np.copy(self.grid)
        grid[self.agent_pos[0], self.agent_pos[1]] = 2
        grid[self.goal[0], self.goal[1]] = 3
        print(grid)
