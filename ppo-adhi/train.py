# 24.10.2023
# Adithya Mohan - herrmohan1394@gmail.com

import pygame
import random
import torch
import time
from ppo import PPO
from torch.distributions import Categorical


# Constants
SCREEN_DIM = (1000, 1000)
AGENT_COLOR = (0, 255, 0)
ADVERSARY_COLOR = (255, 0, 0)
WAYPOINT_COLOR = (0, 0, 255)
BLOCK_SIZE = 10


class GridEnv:
    def __init__(self, grid_size=5):
        # Initialize pygame for rendering
        pygame.init()
        self.cell_size = 40  # size of each grid cell in pixels
        self.screen_size = grid_size * self.cell_size
        self.screen = pygame.display.set_mode(
            (self.screen_size, self.screen_size))
        pygame.display.set_caption('Co-Evolution Strategy Experiment')

        self.grid_size = grid_size
        self.agent_pos = [random.randint(
            0, grid_size - 1), random.randint(0, grid_size - 1)]
        self.adversary_pos = [random.randint(
            0, grid_size - 1), random.randint(0, grid_size - 1)]
        self.target_pos = [random.randint(
            0, grid_size - 1), random.randint(0, grid_size - 1)]

    def reset(self):
        self.agent_pos = [random.randint(
            0, self.grid_size - 1), random.randint(0, self.grid_size - 1)]
        self.adversary_pos = [random.randint(
            0, self.grid_size - 1), random.randint(0, self.grid_size - 1)]
        self.target_pos = [random.randint(
            0, self.grid_size - 1), random.randint(0, self.grid_size - 1)]
        return self.agent_pos + self.target_pos + self.adversary_pos

    def step(self, agent_action, adversary_action):
        # Update agent position
        if agent_action == 0:    # Up
            self.agent_pos[1] -= 1
        elif agent_action == 1:  # Down
            self.agent_pos[1] += 1
        elif agent_action == 2:  # Left
            self.agent_pos[0] -= 1
        elif agent_action == 3:  # Right
            self.agent_pos[0] += 1

        # Update adversary position
        if adversary_action == 0:    # Up
            self.adversary_pos[1] -= 1
        elif adversary_action == 1:  # Down
            self.adversary_pos[1] += 1
        elif adversary_action == 2:  # Left
            self.adversary_pos[0] -= 1
        elif adversary_action == 3:  # Right
            self.adversary_pos[0] += 1

        # Rewards and termination conditions
        if self.agent_pos == self.target_pos:
            return self.agent_pos + self.target_pos + self.adversary_pos, 1, -1, True
        if self.agent_pos == self.adversary_pos:
            return self.agent_pos + self.target_pos + self.adversary_pos, -1, 1, True

        return self.agent_pos + self.target_pos + self.adversary_pos, 0, 0, False

    def render(self):
        # Fill the screen with white
        self.screen.fill((255, 255, 255))

        # Draw the grid
        for x in range(0, self.screen_size, self.cell_size):
            for y in range(0, self.screen_size, self.cell_size):
                rect = pygame.Rect(x, y, self.cell_size, self.cell_size)
                pygame.draw.rect(self.screen, (200, 200, 200), rect, 1)

        # Draw agent, adversary, and target
        agent_rect = pygame.Rect(
            self.agent_pos[0] * self.cell_size, self.agent_pos[1] * self.cell_size, self.cell_size, self.cell_size)
        pygame.draw.rect(self.screen, (0, 0, 255), agent_rect)

        adversary_rect = pygame.Rect(
            self.adversary_pos[0] * self.cell_size, self.adversary_pos[1] * self.cell_size, self.cell_size, self.cell_size)
        pygame.draw.rect(self.screen, (255, 0, 0), adversary_rect)

        target_rect = pygame.Rect(
            self.target_pos[0] * self.cell_size, self.target_pos[1] * self.cell_size, self.cell_size, self.cell_size)
        pygame.draw.rect(self.screen, (0, 255, 0), target_rect)

        pygame.display.flip()

        # Introduce a delay for viewing convenience
        pygame.time.wait(100)


def coevolution_train():
    env = GridEnv()

    agent = PPO()
    adversary = PPO()

    for episode in range(10000):  # number of episodes
        s = env.reset()
        done = False

        # First, train the agent with a dummy adversary that doesn't try to stop the agent
        print("Agent training")
        while not done:
            agent_action_prob = agent.pi(torch.Tensor(s))
            agent_action = Categorical(agent_action_prob).sample().item()

            # Dummy adversary action, doesn't actually act against the agent
            adversary_action = random.choice([0, 1, 2, 3])

            s_prime, agent_reward, _, done = env.step(
                agent_action, adversary_action)

            # Rendering the environment
            env.render()

            agent.put_data((s, agent_action, agent_reward, s_prime,
                           agent_action_prob[agent_action].item(), done))

            s = s_prime
            # time.sleep(0.2)

        agent.train_net()

        s = env.reset()  # Reset environment for adversary training
        done = False

        # Now train the adversary based on how the agent acts
        print("Adversary training")
        while not done:
            agent_action_prob = agent.pi(torch.Tensor(s))
            agent_action = Categorical(agent_action_prob).sample().item()

            adversary_action_prob = adversary.pi(torch.Tensor(s))
            adversary_action = Categorical(
                adversary_action_prob).sample().item()

            s_prime, _, adversary_reward, done = env.step(
                agent_action, adversary_action)

            # Rendering the environment
            env.render()

            adversary.put_data((s, adversary_action, adversary_reward, s_prime,
                               adversary_action_prob[adversary_action].item(), done))

            s = s_prime
            # time.sleep(0.2)

        adversary.train_net()

        if episode % 10 == 0:  # print results every 100 episodes
            print(f"Episode {episode} Complete")


if __name__ == "__main__":
    coevolution_train()
