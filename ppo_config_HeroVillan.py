from typing import Any, Callable, Dict, Tuple, Union
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3.common.callbacks import BaseCallback



PPO_CONFIG: Dict[str, Any] = {
    "learning_rate": ('linear', 3e-4, 1e-5),  # or 3e-4
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "clip_range_vf": None,
    "ent_coef": 0.0,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "target_kl": None,
    "seed": None,
    "device": "auto",
    "verbose": 0,
    "use_sde": False,
    "sde_sample_freq": -1,
    "normalize_advantage": True,
    "policy_kwargs": {},
}
TRAINING_CONFIG: Dict[str, Any] = {
    "num_episodes": 50,
    "timesteps_per_episode": 200,
    "cell_size": 25,
}
Reward_Structure: Dict[str, Any] = {
    "step_penalty": -0.01,
    "win_reward": 2.0,
    "lose_penalty": -3.0,
}

class PlotCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.policy_loss = []
        self.value_loss = []
        self.entropy = []
        self.kl_divergence = []
        self.timesteps = []

    def _on_step(self) -> bool:
        logs = self.model.logger.name_to_value
        if "train/policy_gradient_loss" in logs:
            self.policy_loss.append(logs["train/policy_gradient_loss"])
            self.value_loss.append(logs["train/value_loss"])
            self.entropy.append(logs["train/entropy_loss"])
            if "train/approx_kl" in logs:
                self.kl_divergence.append(logs["train/approx_kl"])
            else:
                self.kl_divergence.append(np.nan)
            self.timesteps.append(self.num_timesteps)
        return True

# ---------- Helpers ----------
def make_linear_schedule(start_value: float, end_value: float) -> Callable[[float], float]:
    """
    SB3 schedule signature: f(progress_remaining) where progress_remaining in [0, 1].
    At start of training: progress_remaining ≈ 1 -> returns start_value
    At end of training:   progress_remaining ≈ 0 -> returns end_value
    """
    def schedule(progress_remaining: float) -> float:
        return end_value + (start_value - end_value) * progress_remaining
    return schedule

def resolve_schedule_or_value(
    val: Union[None, float, int, Tuple[str, float, float], Callable[[float], float]]
):
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return val
    if isinstance(val, tuple) and len(val) == 3 and str(val[0]).lower() == "linear":
        _, start, end = val
        return make_linear_schedule(float(start), float(end))
    if callable(val):
        return val
    return val

def build_ppo_kwargs(cfg: Dict[str, Any]) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {}
    for key in ["learning_rate", "clip_range", "clip_range_vf"]:
        if key in cfg:
            kwargs[key] = resolve_schedule_or_value(cfg[key])

    passthrough = [
        "n_steps", "batch_size", "n_epochs", "gamma", "gae_lambda",
        "ent_coef", "vf_coef", "max_grad_norm", "target_kl",
        "seed", "device", "verbose", "use_sde", "sde_sample_freq",
        "normalize_advantage"
    ]
    for k in passthrough:
        if k in cfg:
            kwargs[k] = cfg[k]
    kwargs["policy_kwargs"] = cfg.get("policy_kwargs", {})
    return kwargs


def _resolve_value_for_print(val):
    if callable(val):
        try:
            return val(1.0)
        except Exception:
            return str(val)
    return val


def print_ppo_hyperparams(model) -> None:
    print("\n================= PPO Hyperparameters =================")
    keys = [
        "learning_rate", "n_steps", "batch_size", "n_epochs", "gamma",
        "gae_lambda", "clip_range", "clip_range_vf", "ent_coef", "vf_coef",
        "max_grad_norm", "target_kl", "seed"
    ]
    for k in keys:
        if hasattr(model, k):
            v = getattr(model, k)
            print(f"{k:>16}: {_resolve_value_for_print(v)}")
        else:
            print(f"{k:>16}: N/A")
    print("=====================================================\n")

def plot_training_curves(callback: PlotCallback, episode_steps, num_episodes: int) -> None:

    plt.figure(figsize=(12, 8))

    # Policy Loss
    plt.subplot(3, 2, 1)
    plt.plot(callback.timesteps, callback.policy_loss, label="Policy Loss")
    plt.ylabel("Policy Loss")
    plt.legend()

    # Value Loss
    plt.subplot(3, 2, 2)
    plt.plot(callback.timesteps, callback.value_loss, label="Value Loss")
    plt.ylabel("Value Loss")
    plt.legend()

    # Entropy
    plt.subplot(3, 2, 3)
    plt.plot(callback.timesteps, callback.entropy, label="Entropy")
    plt.ylabel("Entropy")
    plt.legend()

    # Average Steps per Episode
    plt.subplot(3, 2, 4)
    episodes = np.arange(1, num_episodes + 1)
    plt.plot(episodes, episode_steps, label="Steps per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Steps")
    plt.legend()

    # KL Divergence
    plt.subplot(3, 2, 5)
    plt.plot(callback.timesteps, callback.kl_divergence, label="KL Divergence")
    plt.ylabel("KL Divergence")
    plt.legend()

    plt.tight_layout()
    plt.show()