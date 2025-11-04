import pygame
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from grid_env_HeroVillan import GridEnv
import numpy as np

from ppo_config_HeroVillan import (
    PlotCallback,
    PPO_CONFIG,
    TRAINING_CONFIG,
    build_ppo_kwargs,
    print_ppo_hyperparams,
    plot_training_curves
)

pygame.init()

cell_size = TRAINING_CONFIG["cell_size"]
screen = pygame.display.set_mode((10 * cell_size, 10 * cell_size))
pygame.display.set_caption("Hero Villan")

env = GridEnv(render_mode="human")

ppo_kwargs = build_ppo_kwargs(PPO_CONFIG)
model = PPO("MlpPolicy", env, **ppo_kwargs)

print_ppo_hyperparams(model)

callback = PlotCallback()

num_episodes = TRAINING_CONFIG["num_episodes"]
timesteps_per_episode = TRAINING_CONFIG["timesteps_per_episode"]
model.learn(total_timesteps=num_episodes * timesteps_per_episode, callback=callback)

episode_steps = []

for episode in range(num_episodes):
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
        if done:
            goal_reached = True
    episode_steps.append(steps)
    print(f"Episode {episode}: {'Goal reached ✅' if goal_reached else 'Failed ❌'} | Steps: {steps} | Reward: {reward}")

env.close()
pygame.quit()

plot_training_curves(callback, episode_steps, num_episodes)