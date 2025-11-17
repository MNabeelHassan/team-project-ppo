import pygame

from grid_env_HeroVillan import GridEnv
from ppo_config_HeroVillan import (
    PPO_CONFIG,
    TRAINING_CONFIG,
    plot_training_curves,
)
from ppo_hero_villan import HeroVillainPPO


def handle_events():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            raise SystemExit


def main():
    pygame.init()
    cell_size = TRAINING_CONFIG["cell_size"]
    pygame.display.set_mode((10 * cell_size, 10 * cell_size))
    pygame.display.set_caption("Hero Villan")

    env = GridEnv(render_mode="human")

    agent = HeroVillainPPO(
        obs_dim=4,
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

    episode_rewards = []
    episode_steps = []

    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        steps = 0
        cumulative_reward = 0.0

        while not done and steps < timesteps_per_episode:
            handle_events()

            action_agent, action_adv, logprob_agent, logprob_adv = agent.act(state)
            next_state, reward, done, _, _ = env.step([action_agent, action_adv])

            env.render()

            agent.put_data(
                (
                    state,
                    action_agent,
                    action_adv,
                    reward,
                    next_state,
                    float(done),
                    logprob_agent,
                    logprob_adv,
                )
            )

            state = next_state
            steps += 1
            cumulative_reward += reward

            if len(agent.memory) >= T_horizon:
                agent.train_net()

        agent.train_net()

        episode_rewards.append(cumulative_reward)
        episode_steps.append(steps)
        print(
            f"Episode {episode + 1}/{num_episodes} | Reward: {cumulative_reward:.2f} | Steps: {steps}"
        )

    env.close()
    pygame.quit()

    plot_training_curves(episode_rewards, episode_steps)


if __name__ == "__main__":
    main()