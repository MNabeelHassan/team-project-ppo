import numpy as np
import matplotlib.pyplot as plt


TRAINING_CONFIG = {
    "num_episodes": 3000,
    "timesteps_per_episode": 50,
    "T_horizon": 64,
    "sma_window_size": 100,
    "random" : True,
    "hero_starting_pos": [1, 1],
    "villain_starting_pos": [7, 7],
    "step_penalty": -0.1,
    "win_reward": +5,
    "lose_reward": -5,
}


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

def moving_average(data, window=50):
    if len(data) < window:
        window = max(1, len(data) // 5)  # adjust window if too big
    return np.convolve(data, np.ones(window)/window, mode='valid')

def plot_training_curves(hero_rewards, hero_steps, villain_rewards, villain_steps, window=50):
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    # Hero Rewards
    axs[0, 0].plot(hero_rewards, color="lightcoral", alpha=0.4, label="Raw")
    axs[0, 0].plot(moving_average(hero_rewards, window), color="red", label=f"SMA ({window})")
    axs[0, 0].set_title("Hero Rewards (Smoothed)")
    axs[0, 0].set_xlabel("Episode")
    axs[0, 0].set_ylabel("Reward")
    axs[0, 0].legend()

    # Hero Steps
    axs[0, 1].plot(hero_steps, color="lightblue", alpha=0.4, label="Raw")
    axs[0, 1].plot(moving_average(hero_steps, window), color="blue", label=f"SMA ({window})")
    axs[0, 1].set_title("Hero Steps (Smoothed)")
    axs[0, 1].set_xlabel("Episode")
    axs[0, 1].set_ylabel("Steps")
    axs[0, 1].legend()

    # Villain Rewards
    axs[1, 0].plot(villain_rewards, color="lightgreen", alpha=0.4, label="Raw")
    axs[1, 0].plot(moving_average(villain_rewards, window), color="green", label=f"SMA ({window})")
    axs[1, 0].set_title("Villain Rewards (Smoothed)")
    axs[1, 0].set_xlabel("Episode")
    axs[1, 0].set_ylabel("Reward")
    axs[1, 0].legend()

    # Villain Steps
    axs[1, 1].plot(villain_steps, color="violet", alpha=0.4, label="Raw")
    axs[1, 1].plot(moving_average(villain_steps, window), color="purple", label=f"SMA ({window})")
    axs[1, 1].set_title("Villain Steps (Smoothed)")
    axs[1, 1].set_xlabel("Episode")
    axs[1, 1].set_ylabel("Steps")
    axs[1, 1].legend()

    plt.tight_layout()
    plt.show()



# import matplotlib.pyplot as plt

# def plot_training_curves(hero_rewards, hero_steps, villain_rewards, villain_steps):
#     fig, axs = plt.subplots(2, 2, figsize=(12, 10))

#     # Top-left: Hero Rewards
#     axs[0, 0].plot(hero_rewards, color="red ", label="Hero Steps")
#     axs[0, 0].set_title("Hero Rewards")
#     axs[0, 0].set_xlabel("Episode")
#     axs[0, 0].set_ylabel("Reward")
#     axs[0, 0].legend()

#     # Top-right: Hero Steps
#     axs[0, 1].plot(hero_steps, color="blue", label="Hero Steps")
#     axs[0, 1].set_title("Hero Steps")
#     axs[0, 1].set_xlabel("Episode")
#     axs[0, 1].set_ylabel("Steps")
#     axs[0, 1].legend()

#     # Bottom-left: Villain Rewards
#     axs[1, 0].plot(villain_rewards, color="red", label="Villain Reward")
#     axs[1, 0].set_title("Villain Rewards")
#     axs[1, 0].set_xlabel("Episode")
#     axs[1, 0].set_ylabel("Reward")
#     axs[1, 0].legend()

#     # Bottom-right: Villain Steps
#     axs[1, 1].plot(villain_steps, color="red", label="villan steps")
#     axs[1, 1].set_title("Villain Steps")
#     axs[1, 1].set_xlabel("Episode")
#     axs[1, 1].set_ylabel("Steps")
#     axs[1, 1].legend()

#     plt.tight_layout()
#     plt.show()

