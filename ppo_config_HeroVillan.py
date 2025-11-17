from typing import Any, Dict, List, Optional

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
    hero_rewards: List[float],
    hero_steps: List[int],
    villain_rewards: Optional[List[float]] = None,
    villain_steps: Optional[List[int]] = None,
) -> None:
    rows = 2
    if villain_rewards is not None and villain_steps is not None:
        rows = 4

    plt.figure(figsize=(10, 4 * rows / 2))

    hero_eps = np.arange(1, len(hero_rewards) + 1)
    plt.subplot(rows, 1, 1)
    plt.plot(hero_eps, hero_rewards, label="Hero Reward")
    plt.ylabel("Reward")
    plt.legend()

    plt.subplot(rows, 1, 2)
    plt.plot(hero_eps, hero_steps, label="Hero Steps")
    plt.xlabel("Episode")
    plt.ylabel("Steps")
    plt.legend()

    if rows == 4:
        villain_eps = np.arange(1, len(villain_rewards) + 1)
        plt.subplot(rows, 1, 3)
        plt.plot(villain_eps, villain_rewards, label="Villain Reward", color="orange")
        plt.ylabel("Reward")
        plt.legend()

        plt.subplot(rows, 1, 4)
        plt.plot(villain_eps, villain_steps, label="Villain Steps", color="orange")
        plt.xlabel("Episode")
        plt.ylabel("Steps")
        plt.legend()

    plt.tight_layout()
    plt.show()