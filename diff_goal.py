import pygame
import time
from stable_baselines3 import PPO
from grid_env import GridEnv

pygame.init()
cell_size = 50
screen = pygame.display.set_mode((10 * cell_size, 10 * cell_size))
pygame.display.set_caption("PPO Grid Environment")

env = GridEnv(render_mode="human")
model = PPO("MlpPolicy", env, verbose=0)

num_episodes = 50
timesteps_per_episode = 200
model.learn(total_timesteps=num_episodes * timesteps_per_episode)

for episode in range(1, 6):
    obs, _ = env.reset()
    done = False
    steps = 0
    hero_done = False
    villain_done = False

    while not (hero_done and villain_done) and steps < timesteps_per_episode:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.close()
                exit()

        action, _ = model.predict(obs)
        obs, reward, done, _, info = env.step(action)
        env.render()
        steps += 1
        time.sleep(0.1)

        hero_done = info.get("hero_done", False)
        villain_done = info.get("villain_done", False)

    if hero_done and villain_done:
        print(f"Episode {episode}: Both agents reached their goals!")
    elif hero_done:
        print(f"Episode {episode}: Hero reached the goal.")
    elif villain_done:
        print(f"Episode {episode}: Villain reached the goal.")
    else:
        print(f"Episode {episode}: Failed.")

env.close()
pygame.quit()
