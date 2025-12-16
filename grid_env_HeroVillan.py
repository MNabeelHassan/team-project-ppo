import numpy as np
import pygame
from ppo_config_HeroVillan import TRAINING_CONFIG

class GridEnv:
    def __init__(self, grid_size=10, render_mode="human"):
        self.grid_size = grid_size
        self.render_mode = render_mode

        # Positions initialized in reset
        self.hero_pos = [1,1] #np.zeros(2, dtype=int)
        self.villain_pos = np.array([grid_size-3, grid_size-3])#np.zeros(2, dtype=int)
        self.goal_pos = np.array([grid_size-1, grid_size-1]) #10-1 at 9x9 position

        # Rewards (symmetric)
        self.step_penalty = TRAINING_CONFIG["step_penalty"]
        self.win_reward = TRAINING_CONFIG["win_reward"]
        self.lose_reward = TRAINING_CONFIG["lose_reward"]

        self.done = False

        # Rendering
        if render_mode == "human":
            pygame.init()
            self.cell_size = 40
            self.screen = pygame.display.set_mode(
                (grid_size*self.cell_size, grid_size*self.cell_size)
            )
            pygame.display.set_caption("Hero vs Villain")

    def reset(self):
        # Random starting positions
        if TRAINING_CONFIG["random"]:
            self.hero_pos = np.random.randint(0, self.grid_size, size=2)
            self.villain_pos = np.random.randint(0, self.grid_size, size=2)
        else:
            self.hero_pos = np.array(TRAINING_CONFIG["hero_starting_pos"])
            self.villain_pos = np.array(TRAINING_CONFIG["villain_starting_pos"])
        self.done = False
        return self._get_state()

    def _get_state(self):
        return np.concatenate([self.hero_pos, self.villain_pos]).astype(np.float32)

    def _move(self, pos, action):
        if action == 0 and pos[1] < self.grid_size - 1:  # up
            pos[1] += 1
        elif action == 1 and pos[1] > 0:  # down
            pos[1] -= 1
        elif action == 2 and pos[0] > 0:  # left
            pos[0] -= 1
        elif action == 3 and pos[0] < self.grid_size - 1:  # right
            pos[0] += 1

    def step(self, actions):
        hero_action, villain_action = actions

        hero_reward =+ self.step_penalty
        villain_reward =+ self.step_penalty

        # Move agents
        self._move(self.hero_pos, hero_action)
        self._move(self.villain_pos, villain_action)

        info = {"hero_win": False, "villain_win": False}

        # Hero reaches goal
        if np.array_equal(self.hero_pos, self.goal_pos):
            hero_reward += self.win_reward
            villain_reward += self.lose_reward
            info["hero_win"] = True
            self.done = True

        # Villain catches hero
        elif np.array_equal(self.hero_pos, self.villain_pos):
            villain_reward += self.win_reward
            hero_reward += self.lose_reward
            info["villain_win"] = True
            self.done = True

        return self._get_state(), hero_reward, self.done, villain_reward, info

    def render(self):
        if self.render_mode != "human":
            return

        self.screen.fill((255, 255, 255))
        # Draw grid
        for x in range(0, self.grid_size*self.cell_size, self.cell_size):
            for y in range(0, self.grid_size*self.cell_size, self.cell_size):
                pygame.draw.rect(self.screen, (200,200,200),
                                 (x, y, self.cell_size, self.cell_size), 1)

        # Draw agents
        pygame.draw.rect(self.screen, (0,255,0),
                         (*self.hero_pos*self.cell_size, self.cell_size, self.cell_size))
        pygame.draw.rect(self.screen, (255,0,0),
                         (*self.villain_pos*self.cell_size, self.cell_size, self.cell_size))
        # Draw goal
        pygame.draw.rect(self.screen, (0,0,255),
                         (*self.goal_pos*self.cell_size, self.cell_size, self.cell_size))

        pygame.display.flip()

    def close(self):
        if hasattr(self, "screen"):
            pygame.quit()
