from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np

PPO_CONFIG: Dict[str, Any] = {
    "learning_rate": 3e-4,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "K_epoch": 4,
    "hidden_dim": 128,
    "T_horizon": 256,
}

TRAINING_CONFIG: Dict[str, Any] = {
    "num_episodes": 25,
    "timesteps_per_episode": 100,
    "cell_size": 25,
}

Reward_Structure: Dict[str, Any] = {
    "step_penalty": 0.0,
    "win_reward": 2.0,
    "lose_penalty": -2.0,
}


def plot_training_curves(
    episode_rewards: List[float],
    episode_steps: List[int],
) -> None:
    episodes = np.arange(1, len(episode_rewards) + 1)

    plt.figure(figsize=(10, 6))

    plt.subplot(2, 1, 1)
    plt.plot(episodes, episode_rewards, label="Episode Reward")
    plt.ylabel("Reward")
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(episodes, episode_steps, label="Steps per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Steps")
    plt.legend()

    plt.tight_layout()
    plt.show()