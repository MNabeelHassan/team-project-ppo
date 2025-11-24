import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame

from ppo_config_HeroVillan import(
    Reward_Structure,
    TRAINING_CONFIG
)

class GridEnv(gym.Env):

    def __init__(
        self,
        render_mode=None,
        grid_size=10,
        step_penalty=Reward_Structure["step_penalty"],
        win_reward=Reward_Structure["win_reward"],
        lose_penalty=Reward_Structure["lose_penalty"],
    ):
        super().__init__()
        self.grid_size = grid_size
        self.cell_size = TRAINING_CONFIG["cell_size"]
        self.render_mode = render_mode

        # Rewards
        self.step_penalty = float(step_penalty)
        self.win_reward = float(win_reward)
        self.lose_penalty = float(lose_penalty)

        # Spaces
        self.observation_space = spaces.Box(
            low=0, high=self.grid_size - 1, shape=(4,), dtype=np.int32
        )
        self.action_space = spaces.MultiDiscrete([4, 4])

        self.window = None
        self.clock = None

        # starts & goals
        self.a1_start = np.array([0, 0], dtype=np.int32)   # BLUE prey
        self.a2_start = np.array([8, 8], dtype=np.int32)   # RED predator
        self.a1_goal  = np.array([self.grid_size - 1, self.grid_size - 1], dtype=np.int32)  # (9,9) on 10x10

        # State
        self.agent1_pos = self.a1_start.copy()
        self.agent2_pos = self.a2_start.copy()
        self.done = False

    def _obs(self):
        return np.array(
            [self.agent1_pos[0], self.agent1_pos[1],
             self.agent2_pos[0], self.agent2_pos[1]],
            dtype=np.int32
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.agent1_pos[:] = self.a1_start
        self.agent2_pos[:] = self.a2_start
        self.done = False
        return self._obs(), {}

    def _move_one(self, pos, act):
        # 0 Up, 1 Down, 2 Left, 3 Right
        if act == 0 and pos[1] < self.grid_size - 1:
            pos[1] += 1
        elif act == 1 and pos[1] > 0:
            pos[1] -= 1
        elif act == 2 and pos[0] > 0:
            pos[0] -= 1
        elif act == 3 and pos[0] < self.grid_size - 1:
            pos[0] += 1

    def step(self, action):
        if self.done:
            return self._obs(), 0.0, True, False, {}

        a1_act, a2_act = int(action[0]), int(action[1])

        hero_reward = self.step_penalty
        villain_reward = self.step_penalty

        self._move_one(self.agent1_pos, a1_act)
        self._move_one(self.agent2_pos, a2_act)

        caught = np.array_equal(self.agent2_pos, self.agent1_pos)
        prey_at_goal = np.array_equal(self.agent1_pos, self.a1_goal)

        done = False
        info = {"caught": False, "prey_goal": False}

        if caught:
            # Predator wins
            hero_reward += self.lose_penalty
            villain_reward += self.win_reward
            done = True
            print("Caught")
            info["caught"] = True

        elif prey_at_goal:
            # Prey wins
            hero_reward += self.win_reward
            villain_reward += self.lose_penalty
            done = True
            print("Prey at goal")
            info["prey_goal"] = True

        self.done = done
        truncated = False

        reward = float(hero_reward)
        info["hero_reward"] = float(hero_reward)
        info["villain_reward"] = float(villain_reward)
        return self._obs(), reward, done, truncated, info

    def render(self):
        if self.render_mode != "human":
            return

        if self.window is None:
            pygame.init()
            self.window = pygame.display.set_mode(
                (self.grid_size * self.cell_size, self.grid_size * self.cell_size)
            )
            pygame.display.set_caption(
                "Hero Villan"
            )
            self.clock = pygame.time.Clock()

        # Colors
        white = (255, 255, 255)
        blue  = (50, 120, 255)   # Agent 1
        red   = (220, 40, 40)    # Agent 2
        yellow= (245, 200, 30)   # Prey's goal
        gray  = (200, 200, 200)

        # Clear & grid
        self.window.fill(white)
        #for i in range(self.grid_size + 1):
            #pygame.draw.line(self.window, gray, (i * self.cell_size, 0),
                             #(i * self.cell_size, self.grid_size * self.cell_size), 1)
            #pygame.draw.line(self.window, gray, (0, i * self.cell_size),
                             #(self.grid_size * self.cell_size, i * self.cell_size), 1)

        def cell_rect(x, y, pad=5):
            # y-up to screen coords conversion
            return pygame.Rect(
                x * self.cell_size + pad,
                (self.grid_size - 1 - y) * self.cell_size + pad,
                self.cell_size - 2 * pad, self.cell_size - 2 * pad
            )

        # GOAL
        gx, gy = self.a1_goal
        pygame.draw.rect(self.window, yellow, cell_rect(gx, gy, pad=4))

        # Agents as circles
        ax1, ay1 = self.agent1_pos
        ax2, ay2 = self.agent2_pos
        c1 = (ax1 * self.cell_size + self.cell_size // 2,
              (self.grid_size - 1 - ay1) * self.cell_size + self.cell_size // 2)
        c2 = (ax2 * self.cell_size + self.cell_size // 2,
              (self.grid_size - 1 - ay2) * self.cell_size + self.cell_size // 2)

        pygame.draw.circle(self.window, blue, c1, (self.cell_size - 10) // 2)  # Agent 1
        pygame.draw.circle(self.window, red,  c2, (self.cell_size - 10) // 2)  # Agent 2

        pygame.display.flip()
        self.clock.tick(10)

    def close(self):
        if self.window is not None:
            pygame.quit()
            self.window = None
