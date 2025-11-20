TRAINING_CONFIG = {
    "num_episodes": 5000,
    "timesteps_per_episode": 100,
    "T_horizon": 64
}

PPO_CONFIG = {
    "state_dim": 4,         # hero_x, hero_y, villain_x, villain_y
    "action_dim": 4,        # up, down, left, right
    "hidden_dim": 128,      # smaller than 256 to stabilize learning
    "lr": 3e-4,             # slightly higher than 0.0005 for faster learning
    "gamma": 0.99,          # emphasizes long-term reward
    "lmbda": 0.95,          # GAE parameter
    "eps_clip": 0.2,        # PPO clipping parameter
    "K_epoch": 4,           # PPO update iterations per batch
    "T_horizon": 64,        # steps per batch
    "entropy_coef": 0.01,   # encourages exploration
    "value_loss_coef": 0.5  # balance policy vs value loss
}



import matplotlib.pyplot as plt

def plot_training_curves(hero_rewards, hero_steps, villain_rewards, villain_steps):
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    # Top-left: Hero Rewards
    axs[0, 0].plot(hero_rewards, color="red ", label="Hero Steps")
    axs[0, 0].set_title("Hero Rewards")
    axs[0, 0].set_xlabel("Episode")
    axs[0, 0].set_ylabel("Reward")
    axs[0, 0].legend()

    # Top-right: Hero Steps
    axs[0, 1].plot(hero_steps, color="blue", label="Hero Steps")
    axs[0, 1].set_title("Hero Steps")
    axs[0, 1].set_xlabel("Episode")
    axs[0, 1].set_ylabel("Steps")
    axs[0, 1].legend()

    # Bottom-left: Villain Rewards
    axs[1, 0].plot(villain_rewards, color="red", label="Villain Reward")
    axs[1, 0].set_title("Villain Rewards")
    axs[1, 0].set_xlabel("Episode")
    axs[1, 0].set_ylabel("Reward")
    axs[1, 0].legend()

    # Bottom-right: Villain Steps
    axs[1, 1].plot(villain_steps, color="red", label="villan steps")
    axs[1, 1].set_title("Villain Steps")
    axs[1, 1].set_xlabel("Episode")
    axs[1, 1].set_ylabel("Steps")
    axs[1, 1].legend()

    plt.tight_layout()
    plt.show()
