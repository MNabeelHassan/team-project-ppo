# env.py
import random
import pygame
import sys

class GridEnv:
    """Grid world with optional fast pygame rendering.

    State: [hero_x, hero_y, goal_x, goal_y, villain_x, villain_y]
    Actions: 0=Up,1=Down,2=Left,3=Right
    """

    def __init__(self, grid_size=8, max_steps=50, render=False, cell_pixels=40):
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.render_mode = render
        self.cell_pixels = cell_pixels

        if self.render_mode:
            pygame.init()
            self.screen_size = self.grid_size * self.cell_pixels
            self.screen = pygame.display.set_mode((self.screen_size, self.screen_size))
            pygame.display.set_caption("Hero vs Villain - PPO Coevolution")

        self.reset()

    def reset(self):
        positions = set()
        def sample():
            return (random.randint(0, self.grid_size-1), random.randint(0, self.grid_size-1))

        self.hero_pos = sample()
        positions.add(self.hero_pos)

        self.villain_pos = sample()
        while self.villain_pos in positions:
            self.villain_pos = sample()
        positions.add(self.villain_pos)

        self.goal_pos = sample()
        while self.goal_pos in positions:
            self.goal_pos = sample()

        self.steps = 0
        return self._get_state()

    def _get_state(self):
        return [
            self.hero_pos[0], self.hero_pos[1],
            self.goal_pos[0], self.goal_pos[1],
            self.villain_pos[0], self.villain_pos[1]
        ]

    def _clip_pos(self, pos):
        x, y = pos
        x = max(0, min(self.grid_size-1, x))
        y = max(0, min(self.grid_size-1, y))
        return (x, y)

    def step(self, hero_action, villain_action):
        hx, hy = self.hero_pos
        if hero_action == 0: hy -= 1
        elif hero_action == 1: hy += 1
        elif hero_action == 2: hx -= 1
        elif hero_action == 3: hx += 1
        self.hero_pos = self._clip_pos((hx, hy))

        vx, vy = self.villain_pos
        if villain_action == 0: vy -= 1
        elif villain_action == 1: vy += 1
        elif villain_action == 2: vx -= 1
        elif villain_action == 3: vx += 1
        self.villain_pos = self._clip_pos((vx, vy))

        self.steps += 1
        done = False
        hero_reward = -0.01
        villain_reward = -0.01

        if self.hero_pos == self.goal_pos:
            hero_reward += 1.0
            villain_reward -= 1.0
            done = True
        elif self.hero_pos == self.villain_pos:
            hero_reward -= 1.0
            villain_reward += 1.0
            done = True
        elif self.steps >= self.max_steps:
            done = True

        return self._get_state(), hero_reward, villain_reward, done

    def render(self, wait_ms=1000):
        """Render current grid using pygame. Minimal delay for fast training."""
        if not self.render_mode:
            return

        # handle quit events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        self.screen.fill((245, 245, 245))

        # draw grid
        for x in range(self.grid_size + 1):
            pygame.draw.line(self.screen, (200,200,200), (x*self.cell_pixels,0),(x*self.cell_pixels,self.grid_size*self.cell_pixels))
        for y in range(self.grid_size + 1):
            pygame.draw.line(self.screen, (200,200,200), (0,y*self.cell_pixels),(self.grid_size*self.cell_pixels,y*self.cell_pixels))

        # draw goal
        gx,gy = self.goal_pos
        pygame.draw.rect(self.screen, (50,200,50), pygame.Rect(gx*self.cell_pixels+4, gy*self.cell_pixels+4, self.cell_pixels-8, self.cell_pixels-8))

        # draw hero
        hx,hy = self.hero_pos
        pygame.draw.rect(self.screen, (40,120,220), pygame.Rect(hx*self.cell_pixels+8, hy*self.cell_pixels+8, self.cell_pixels-16, self.cell_pixels-16))

        # draw villain
        vx,vy = self.villain_pos
        pygame.draw.rect(self.screen, (200,40,40), pygame.Rect(vx*self.cell_pixels+8, vy*self.cell_pixels+8, self.cell_pixels-16, self.cell_pixels-16))

        pygame.display.flip()
        # minimal wait for speed
        pygame.time.delay(wait_ms)
