import pygame
import torch
from torch.distributions import Categorical
from grid_env_HeroVillan import GridEnv
from ppo_hero_villan import PPOAgent
import matplotlib.pyplot as plt
import numpy as np

# ======= CONFIG =======
PPO_CONFIG_Hero = {
    "state_dim": 4,
    "action_dim": 4,
    "hidden_dim": 256,
    "lr": 0.0005,
    "gamma": 0.98,
    "lmbda": 0.95,
    "eps_clip": 0.1,
    "K_epoch": 3,
}

PPO_CONFIG_Villan = {
    "state_dim": 4,
    "action_dim": 4,
    "hidden_dim": 256,
    "lr": 0.0005,
    "gamma": 0.98,
    "lmbda": 0.95,
    "eps_clip": 0.1,
    "K_epoch": 3,
}

TRAINING_CONFIG = {
    "num_episodes": 1000,
    "timesteps_per_episode": 50,
    "T_horizon": 64,
    "sma_window_size": 10,
}

# ======= HELPER =======
def handle_events():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            raise SystemExit

def sample_action(agent, state_tensor):
    with torch.no_grad():
        probs = agent.pi(state_tensor, softmax_dim=-1) #how it gets prob 
    dist = Categorical(probs)
    a = dist.sample()
    return a.item(), probs[a].item()

def sma(values, window):
    if len(values) < window:
        return np.array(values)
    return np.convolve(values, np.ones(window)/window, mode='valid')

def plot_training_curves(hero_rewards, hero_steps, villain_rewards, villain_steps, window):
    fig, axs = plt.subplots(2,2, figsize=(12,8))

    axs[0, 0].plot(sma(hero_rewards, window), color='blue', label=f"SMA {window}")
    axs[0,0].set_title("Hero Rewards")

    axs[0, 1].plot(sma(villain_rewards, window), color='red', label=f"SMA {window}")
    axs[0,1].set_title("Villain Rewards")

    axs[1, 0].plot(sma(hero_steps, window), color='blue', label=f"SMA {window}")
    axs[1,0].set_title("Hero Steps")

    axs[1, 1].plot(sma(villain_steps, window), color='red', label=f"SMA {window}")
    axs[1,1].set_title("Villain Steps")
    
    for ax in axs.flat:
        ax.legend()
    plt.tight_layout()
    plt.show()
  #save plot and csv data
# ======= MAIN TRAINING =======
def main():
    pygame.init()
    env = GridEnv(render_mode="human")

    hero = PPOAgent(**PPO_CONFIG_Hero)
    villain = PPOAgent(**PPO_CONFIG_Villan)

    hero_rewards, villain_rewards = [], []
    hero_steps, villain_steps = [], []

    for ep in range(TRAINING_CONFIG["num_episodes"]):
        state, _ = env.reset()
        done = False
        hero_ep_r, villain_ep_r = 0,0
        steps = 0

        while not done and steps < TRAINING_CONFIG["timesteps_per_episode"]:
            handle_events()
            s_t = torch.tensor(state, dtype=torch.float32)

            hero_act, hero_prob = sample_action(hero, s_t)
            villain_act, villain_prob = sample_action(villain, s_t)

            next_state, hero_r, done, villain_r, info = env.step([hero_act, villain_act])
            env.render()

            hero.put_data((state, hero_act, hero_r, next_state, hero_prob, done))
            villain.put_data((state, villain_act, villain_r, next_state, villain_prob, done))

            hero_ep_r += hero_r
            villain_ep_r += villain_r
            state = next_state
            steps += 1

            if len(hero.data) >= TRAINING_CONFIG["T_horizon"]:
                hero.train_net()
            if len(villain.data) >= TRAINING_CONFIG["T_horizon"]:
                villain.train_net()

        # Final update per episode
        hero.train_net()
        villain.train_net()

        hero_rewards.append(hero_ep_r)
        villain_rewards.append(villain_ep_r)
        hero_steps.append(steps)
        villain_steps.append(steps)

        winner = "Hero" if info.get("hero_win") else ("Villain" if info.get("villain_win") else "None")
        print(f"EP {ep+1}/{TRAINING_CONFIG['num_episodes']}  Hero={hero_ep_r:.2f}  Villain={villain_ep_r:.2f}  Winner={winner}")

    env.close()
    window = TRAINING_CONFIG["sma_window_size"]
    plot_training_curves(hero_rewards, hero_steps, villain_rewards, villain_steps, window)

if __name__ == "__main__":
    main()
