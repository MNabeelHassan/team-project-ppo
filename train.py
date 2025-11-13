# train.py
import numpy as np
from env import GridEnv
from ppo_multi import PPOAgent
import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--render", action="store_true", help="Enable pygame visualization")
parser.add_argument("--episodes", type=int, default=2000)
parser.add_argument("--grid", type=int, default=6)
parser.add_argument("--max_steps", type=int, default=50)
parser.add_argument("--wait_ms", type=int, default=1, help="ms between rendered frames")
parser.add_argument("--render_every", type=int, default=1, help="render every N steps")
args = parser.parse_args()

RENDER = args.render
GRID_SIZE = args.grid
EPISODES = args.episodes
MAX_STEPS = args.max_steps
RENDER_EVERY = args.render_every

env = GridEnv(grid_size=GRID_SIZE, max_steps=MAX_STEPS, render=RENDER, cell_pixels=60)

hero = PPOAgent()
villain = PPOAgent()

metrics = {
    "hero_reward": [], "villain_reward": [],
    "hero_kl": [], "villain_kl": [],
    "hero_policy_loss": [], "villain_policy_loss": [],
    "hero_value_loss": [], "villain_value_loss": [],
    "hero_entropy": [], "villain_entropy": []
}

print(f"Starting training. Render={RENDER}, Episodes={EPISODES}, Grid={GRID_SIZE}")

for ep in range(1, EPISODES + 1):
    s = env.reset()
    done = False
    hr = vr = 0.0
    step_count = 0

    while not done:
        ha, hp = hero.get_action(s)
        va, vp = villain.get_action(s)
        sp, h_r, v_r, done = env.step(ha, va)

        hero.put_data((s, ha, h_r, sp, hp, done))
        villain.put_data((s, va, v_r, sp, vp, done))

        hr += h_r
        vr += v_r
        s = sp
        step_count += 1

        if RENDER and step_count % RENDER_EVERY == 0:
            env.render(wait_ms=args.wait_ms)

    hero_m = hero.train_net()
    villain_m = villain.train_net()

    # logging
    metrics["hero_reward"].append(hr)
    metrics["villain_reward"].append(vr)
    metrics["hero_kl"].append(hero_m.get("kl", 0))
    metrics["villain_kl"].append(villain_m.get("kl", 0))
    metrics["hero_policy_loss"].append(hero_m.get("policy_loss", 0))
    metrics["villain_policy_loss"].append(villain_m.get("policy_loss", 0))
    metrics["hero_value_loss"].append(hero_m.get("value_loss", 0))
    metrics["villain_value_loss"].append(villain_m.get("value_loss", 0))
    metrics["hero_entropy"].append(hero_m.get("entropy", 0))
    metrics["villain_entropy"].append(villain_m.get("entropy", 0))

    if ep % 50 == 0:
        print(f"Ep {ep} | Hero avg R: {np.mean(metrics['hero_reward'][-50:]):.3f} "
              f"Villain avg R: {np.mean(metrics['villain_reward'][-50:]):.3f}")

# Save models and logs
np.savez("logs.npz", **metrics)
torch.save(hero.state_dict(), "hero_model.pt")
torch.save(villain.state_dict(), "villain_model.pt")
print("Training complete. Logs + models saved.")
