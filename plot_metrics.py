# plot_metrics.py
import numpy as np
import matplotlib.pyplot as plt

data = np.load("logs.npz")

episodes = np.arange(1, len(data["hero_reward"]) + 1)

plt.figure(figsize=(12, 10))

plt.subplot(3, 2, 1)
plt.plot(episodes, data["hero_reward"], label="Hero")
plt.plot(episodes, data["villain_reward"], label="Villain")
plt.title("Rewards")
plt.legend()

plt.subplot(3, 2, 2)
plt.plot(episodes, data["hero_kl"], label="Hero")
plt.plot(episodes, data["villain_kl"], label="Villain")
plt.title("KL Divergence")
plt.legend()

plt.subplot(3, 2, 3)
plt.plot(episodes, data["hero_policy_loss"], label="Hero")
plt.plot(episodes, data["villain_policy_loss"], label="Villain")
plt.title("Policy Loss")
plt.legend()

plt.subplot(3, 2, 4)
plt.plot(episodes, data["hero_value_loss"], label="Hero")
plt.plot(episodes, data["villain_value_loss"], label="Villain")
plt.title("Value Loss")
plt.legend()

plt.subplot(3, 2, 5)
plt.plot(episodes, data["hero_entropy"], label="Hero")
plt.plot(episodes, data["villain_entropy"], label="Villain")
plt.title("Entropy")
plt.legend()

plt.tight_layout()
plt.show()
