import random

import pygame
import torch
from torch.distributions import Categorical

from grid_env_HeroVillan import GridEnv
from ppo_config_HeroVillan import (
    PPO_CONFIG,
    TRAINING_CONFIG,
    plot_training_curves,
)
from ppo_hero_villan import PPOAgent


def handle_events():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            raise SystemExit


def sample_action(agent: PPOAgent, state_tensor: torch.Tensor) -> tuple[int, float]:
    with torch.no_grad():
        probs = agent.pi(state_tensor, softmax_dim=-1)
    dist = Categorical(probs)
    action = dist.sample()
    return action.item(), probs[action].item()


def main():
    pygame.init()
    cell_size = TRAINING_CONFIG["cell_size"]
    pygame.display.set_mode((10 * cell_size, 10 * cell_size))
    pygame.display.set_caption("Hero Villan")

    env = GridEnv(render_mode="human")

    hero_agent = PPOAgent(
        state_dim=4,
        action_dim=4,
        hidden_dim=PPO_CONFIG["hidden_dim"],
        learning_rate=PPO_CONFIG["learning_rate"],
        gamma=PPO_CONFIG["gamma"],
        gae_lambda=PPO_CONFIG["gae_lambda"],
        clip_eps=PPO_CONFIG["clip_range"],
        K_epoch=PPO_CONFIG["K_epoch"],
    )

    villain_agent = PPOAgent(
        state_dim=4,
        action_dim=4,
        hidden_dim=PPO_CONFIG["hidden_dim"],
        learning_rate=PPO_CONFIG["learning_rate"],
        gamma=PPO_CONFIG["gamma"],
        gae_lambda=PPO_CONFIG["gae_lambda"],
        clip_eps=PPO_CONFIG["clip_range"],
        K_epoch=PPO_CONFIG["K_epoch"],
    )

    num_episodes = TRAINING_CONFIG["num_episodes"]
    timesteps_per_episode = TRAINING_CONFIG["timesteps_per_episode"]
    T_horizon = PPO_CONFIG["T_horizon"]

    hero_episode_rewards = []
    hero_episode_steps = []
    villain_episode_rewards = []
    villain_episode_steps = []

    for episode in range(num_episodes):
        # Hero training phase (villain acts randomly)
        state, _ = env.reset()
        done = False
        steps = 0
        cumulative_reward = 0.0

        while not done and steps < timesteps_per_episode:
            handle_events()

            state_tensor = torch.tensor(state, dtype=torch.float32)
            hero_action, hero_prob = sample_action(hero_agent, state_tensor)
            villain_action = random.randint(0, 3)

            next_state, reward, done, _, info = env.step([hero_action, villain_action])
            env.render()

            hero_agent.put_data(
                (
                    state,
                    hero_action,
                    reward,
                    next_state,
                    hero_prob,
                    done,
                )
            )

            state = next_state
            steps += 1
            cumulative_reward += reward

            if len(hero_agent.data) >= T_horizon:
                hero_agent.train_net()

        hero_agent.train_net()
        hero_episode_rewards.append(cumulative_reward)
        hero_episode_steps.append(steps)

        # Villain training phase (hero uses its learned policy)
        state, _ = env.reset()
        done = False
        steps = 0
        cumulative_villain_reward = 0.0

        while not done and steps < timesteps_per_episode:
            handle_events()

            state_tensor = torch.tensor(state, dtype=torch.float32)
            hero_action, _ = sample_action(hero_agent, state_tensor)
            villain_action, villain_prob = sample_action(villain_agent, state_tensor)

            next_state, reward, done, _, info = env.step([hero_action, villain_action])
            env.render()

            villain_reward = info.get("villain_reward", -reward)
            villain_agent.put_data(
                (
                    state,
                    villain_action,
                    villain_reward,
                    next_state,
                    villain_prob,
                    done,
                )
            )

            state = next_state
            steps += 1
            cumulative_villain_reward += villain_reward

            if len(villain_agent.data) >= T_horizon:
                villain_agent.train_net()

        villain_agent.train_net()
        villain_episode_rewards.append(cumulative_villain_reward)
        villain_episode_steps.append(steps)

        print(
            f"Episode {episode + 1}/{num_episodes} | "
            f"Hero Reward: {hero_episode_rewards[-1]:.2f} | "
            f"Villain Reward: {villain_episode_rewards[-1]:.2f}"
        )

    env.close()
    pygame.quit()

    plot_training_curves(
        hero_episode_rewards,
        hero_episode_steps,
        villain_episode_rewards,
        villain_episode_steps,
    )


if __name__ == "__main__":
    main()