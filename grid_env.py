"""
grid_env.py
Custom 10x10 grid environment for RL agents 
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame

"""
    Custom Grid Environment
    - 10x10 grid
    - Agent starts at (0,0) by default
    - Goal at (9,9)
    - 4 discrete actions: 0=Up, 1=Down, 2=Left, 3=Right
    - Returns agent position as observation
    - Renders grid using pygame
    """

class GridEnv(gym.Env):
    metadata = {"render_modes": ["human"]}
    """
        Initialize environment
        render_mode: None or "human" for visualization
        """
        
    def __init__(self, render_mode=None):
        super().__init__()
        self.grid_size = 10
        self.cell_size = 50
        self.render_mode = render_mode  # store render mode


        # Define observation and action spaces
        self.observation_space = spaces.Box(
            low=0, high=self.grid_size - 1, shape=(2,), dtype=np.int32
        )
        self.action_space = spaces.Discrete(4)  # 0: Up, 1: Down, 2: Left, 3: Right

        # Initialize pygame variables
        self.window = None
        self.clock = None

        self.agent_pos = np.array([0, 0], dtype=np.int32)
        self.goal_pos = np.array([self.grid_size - 1, self.grid_size - 1], dtype=np.int32)

    def reset(self, *, seed=None, options=None):
        """
        Reset the environment to the initial state
        Returns:
            obs (np.array): agent position
            info (dict): optional info (empty here)
        """
        super().reset(seed=seed)
        self.agent_pos = np.array([0, 0], dtype=np.int32)
        self.goal_pos = np.array([self.grid_size - 1, self.grid_size - 1], dtype=np.int32)
        return self.agent_pos.copy(), {}

    def step(self, action):
        """
        Take an action and update environment
        Actions:
            0: Up, 1: Down, 2: Left, 3: Right
        Returns:
            obs (np.array): new agent position
            reward (float): +1 if goal reached, else -0.01
            done (bool): True if goal reached
            truncated (bool): always False here
            info (dict): empty
        """
        if action == 0 and self.agent_pos[1] < self.grid_size - 1:  # Up
            self.agent_pos[1] += 1
        elif action == 1 and self.agent_pos[1] > 0:  # Down
            self.agent_pos[1] -= 1
        elif action == 2 and self.agent_pos[0] > 0:  # Left
            self.agent_pos[0] -= 1
        elif action == 3 and self.agent_pos[0] < self.grid_size - 1:  # Right
            self.agent_pos[0] += 1

        done = np.array_equal(self.agent_pos, self.goal_pos)
        reward = 1 if done else -0.01
        return self.agent_pos.copy(), reward, done, False, {}

    def render(self):
        """
        Render the environment using pygame
        """
        if self.render_mode != "human":
            return

        if self.window is None:
            pygame.init()
            self.window = pygame.display.set_mode((self.grid_size * self.cell_size,
                                                   self.grid_size * self.cell_size))
            pygame.display.set_caption("PPO Grid World")
            self.clock = pygame.time.Clock()

        # Colors
        white = (255, 255, 255)
        black = (0, 0, 0)
        blue = (50, 100, 255)
        green = (0, 200, 0)

        # Clear screen
        self.window.fill(white)

        # Draw grid lines
        for x in range(0, self.grid_size * self.cell_size, self.cell_size):
            pygame.draw.line(self.window, black, (x, 0), (x, self.grid_size * self.cell_size))
        for y in range(0, self.grid_size * self.cell_size, self.cell_size):
            pygame.draw.line(self.window, black, (0, y), (self.grid_size * self.cell_size, y))

        # Draw agent
        ax, ay = self.agent_pos
        agent_rect = pygame.Rect(
            ax * self.cell_size + 5, (self.grid_size - 1 - ay) * self.cell_size + 5,
            self.cell_size - 10, self.cell_size - 10
        )
        pygame.draw.rect(self.window, blue, agent_rect)

        # Draw goal
        gx, gy = self.goal_pos
        goal_rect = pygame.Rect(
            gx * self.cell_size + 5, (self.grid_size - 1 - gy) * self.cell_size + 5,
            self.cell_size - 10, self.cell_size - 10
        )
        pygame.draw.rect(self.window, green, goal_rect)

        pygame.display.flip()
        self.clock.tick(5)  # Control FPS

    def close(self):
        """
        Close the pygame window
        """
        if self.window is not None:
            pygame.quit()
            self.window = None
