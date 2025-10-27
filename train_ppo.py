"""
train_ppo.py
Train PPO agent on 10x10 GridEnv with visualization

"""

import pygame
import time
from stable_baselines3 import PPO
from grid_env import GridEnv

# Initialize pygame **before using the env**
pygame.init()
cell_size = 50
screen = pygame.display.set_mode((10 * cell_size, 10 * cell_size))
pygame.display.set_caption("PPO Grid Environment")

# Create environment
env = GridEnv(render_mode="human")

# Create PPO model
model = PPO("MlpPolicy", env, verbose=0)

# Training (same as before)
num_episodes = 50
timesteps_per_episode = 200
model.learn(total_timesteps=num_episodes * timesteps_per_episode)

# Testing loop with visualization
for episode in range(1, 6):
    obs, _ = env.reset()
    done = False
    steps = 0
    goal_reached = False

    while not done and steps < timesteps_per_episode:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.close()
                exit()

        action, _ = model.predict(obs)
        obs, reward, done, _, _ = env.step(action)
        env.render()
        steps += 1
        time.sleep(0.1)

        if done:
            goal_reached = True

    print(f"Episode {episode}: {'Goal reached ✅' if goal_reached else 'Failed ❌'}")

env.close()
pygame.quit()
