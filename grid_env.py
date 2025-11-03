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
            low=0, high=self.grid_size - 1, shape=(4,), dtype=np.int32
        )
        self.action_space = spaces.MultiDiscrete([4, 4])
         #tuple of two actions: (action_Hero, action_Villain)  0: Up, 1: Down, 2: Left, 3: Right

        # Initialize pygame variables
        self.window = None
        self.clock = None

        self.Hero_pos = np.array([0, 0], dtype=np.int32)
        self.Villain_pos = np.array([0, 0], dtype=np.int32)
        self.Hero_goal_pos = np.array([self.grid_size - 3, self.grid_size - 1], dtype=np.int32)
        self.Villain_goal_pos = np.array([self.grid_size - 1, self.grid_size - 3], dtype=np.int32)

    def reset(self, *, seed=None, options=None):
        """
        Reset the environment to the initial state
        Returns:
            obs (np.array): agent position
            info (dict): optional info (empty here)
        """
        super().reset(seed=seed)
        self.Hero_pos = np.array([0, 0], dtype=np.int32)
        self.Villain_pos = np.array([0, 1], dtype=np.int32)
        obs = np.concatenate([self.Hero_pos, self.Villain_pos])
        return obs, {}
    
    def _move_agent(self, pos, action):
        """Helper to move a single agent safely within bounds."""
        if action == 0 and pos[1] < self.grid_size - 1:  # Up
            pos[1] += 1
        elif action == 1 and pos[1] > 0:  # Down
            pos[1] -= 1
        elif action == 2 and pos[0] > 0:  # Left
            pos[0] -= 1
        elif action == 3 and pos[0] < self.grid_size - 1:  # Right
            pos[0] += 1

    def _undo_move(self, pos, action):
        """Undo the last move if needed (e.g., collision)."""
        if action == 0:
            pos[1] -= 1
        elif action == 1:
            pos[1] += 1
        elif action == 2:
            pos[0] += 1
        elif action == 3:
            pos[0] -= 1

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
       
        """
        actions: tuple or list (a1, a2)
        Each action ∈ {0: Up, 1: Down, 2: Left, 3: Right}
        """
        a1, a2 = action
        # Check if agents have already reached their goals
        hero_done = np.array_equal(self.Hero_pos, self.Hero_goal_pos)
        villain_done = np.array_equal(self.Villain_pos, self.Villain_goal_pos)

        # Move agents only if they haven't reached their goals
        if not hero_done:
            self._move_agent(self.Hero_pos, a1)
        if not villain_done:
            self._move_agent(self.Villain_pos, a2)

        # Prevent both agents from occupying the same cell
        if np.array_equal(self.Hero_pos, self.Villain_pos):
            # simple rule: move agent 2 back to previous position
            self._undo_move(self.Villain_pos, a2)

        # Check for goal
        done1 = np.array_equal(self.Hero_pos, self.Hero_goal_pos)
        done2 = np.array_equal(self.Villain_pos, self.Villain_goal_pos)
        done = done1 or done2

        reward = 0
        if done1:
            reward += 1
        if done2:
            reward += 1
        if not done1 and not done2:
            reward -= 0.01

        obs = np.concatenate([self.Hero_pos, self.Villain_pos])
        
        # Add info about which agent reached the goal
        info = {
            "hero_done": done1,
            "villain_done": done2
        }

        return obs, reward, done, False, info
    
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
            pygame.display.set_caption("PPO Grid World with 2 Agents")
            self.clock = pygame.time.Clock()

        # Colors
        white = (255, 255, 255)
        black = (0, 0, 0)
        blue = (50, 100, 255)
        red = (255, 50, 50)
        green = (0, 200, 0)
        plum = (103, 49, 71)

        # Clear screen
        self.window.fill(white)

        # Draw grid lines
        for x in range(0, self.grid_size * self.cell_size, self.cell_size):
            pygame.draw.line(self.window, black, (x, 0), (x, self.grid_size * self.cell_size))
        for y in range(0, self.grid_size * self.cell_size, self.cell_size):
            pygame.draw.line(self.window, black, (0, y), (self.grid_size * self.cell_size, y))

        # Draw agent (blue)
        ax, ay = self.Hero_pos
        agent_rect = pygame.Rect(
            ax * self.cell_size + 5, (self.grid_size - 1 - ay) * self.cell_size + 5,
            self.cell_size - 10, self.cell_size - 10
        )
        pygame.draw.rect(self.window, blue, agent_rect)
        
        # Draw agent 2 (red)
        ax2, ay2 = self.Villain_pos
        rect2 = pygame.Rect(
            ax2 * self.cell_size + 5,
            (self.grid_size - 1 - ay2) * self.cell_size + 5,
            self.cell_size - 10,
            self.cell_size - 10,
        )
        pygame.draw.rect(self.window, red, rect2)

        # Draw goal for hero (green)
        hx, hy = self.Hero_goal_pos
        hero_goal_rect = pygame.Rect(
            hx * self.cell_size + 5, (self.grid_size - 1 - hy) * self.cell_size + 5,
            self.cell_size - 10, self.cell_size - 10
        )
        pygame.draw.rect(self.window, green, hero_goal_rect)

        # Draw goal for villain (purple)
        vx, vy = self.Villain_goal_pos
        villain_goal_rect = pygame.Rect(
            vx * self.cell_size + 5, (self.grid_size - 1 - vy) * self.cell_size + 5,
            self.cell_size - 10, self.cell_size - 10
        )
        pygame.draw.rect(self.window, plum, villain_goal_rect)

        pygame.display.flip()
        self.clock.tick(5)  # Control FPS

    def close(self):
        """
        Close the pygame window
        """
        if self.window is not None:
            pygame.quit()
            self.window = None
