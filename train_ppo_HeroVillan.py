import pygame
import torch
from torch.distributions import Categorical
from grid_env_HeroVillan import GridEnv
from ppo_hero_villan import PPOAgent
import matplotlib.pyplot as plt
import numpy as np
import csv
import os
from datetime import datetime
from ppo_config_HeroVillan import PPO_CONFIG_Hero, PPO_CONFIG_Villan, TRAINING_CONFIG

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

def create_results_directory():
    """Create a directory for saving results"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name_inp = input("Enter a name for the training session (or press Enter to use timestamp): ").strip()

    base_dir = "results"

    if name_inp:
        dir_name = os.path.join(base_dir, f"training_results_{name_inp}")
    else:
        dir_name = os.path.join(base_dir, f"training_results_{timestamp}")
        
    os.makedirs(dir_name, exist_ok=True)
    return dir_name

def save_training_plots(hero_rewards, hero_steps, villain_rewards, villain_steps, 
                        window, save_dir=".", episode_data=None):
    """Save training plots to file"""
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    
    # Plot 1: Hero Rewards
    axs[0, 0].plot(sma(hero_rewards, window), color='blue', label=f"SMA {window}")
    axs[0, 0].set_title("Hero Rewards")
    axs[0, 0].set_xlabel("Episode")
    axs[0, 0].set_ylabel("Reward")
    axs[0, 0].grid(True, alpha=0.3)
    axs[0, 0].legend()
    
    # Plot 2: Villain Rewards
    axs[0, 1].plot(sma(villain_rewards, window), color='red', label=f"SMA {window}")
    axs[0, 1].set_title("Villain Rewards")
    axs[0, 1].set_xlabel("Episode")
    axs[0, 1].set_ylabel("Reward")
    axs[0, 1].grid(True, alpha=0.3)
    axs[0, 1].legend()
    
    # Plot 3: Episode Steps
    axs[1, 0].plot(sma(hero_steps, window), color='blue', label=f"SMA {window}")
    axs[1, 0].set_title("Episode Steps")
    axs[1, 0].set_xlabel("Episode")
    axs[1, 0].set_ylabel("Steps")
    axs[1, 0].grid(True, alpha=0.3)
    axs[1, 0].legend()
    
    # Plot 4: Win Counts Bar Graph 
    hero_wins = sum(1 for d in episode_data if d['winner'] == 'Hero')
    villain_wins = sum(1 for d in episode_data if d['winner'] == 'Villain')
    draws = sum(1 for d in episode_data if d['winner'] == 'None')
    
    labels = ['Hero Wins', 'Villain Wins', 'Draws']
    values = [hero_wins, villain_wins, draws]
    colors = ['blue', 'red', 'gray']
    
    bars = axs[1, 1].bar(labels, values, color=colors)
    axs[1, 1].set_title("Win Counts")
    axs[1, 1].set_ylabel("Number of Episodes")
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        axs[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom')
    
    axs[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(save_dir, "training_plots.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Plots saved to: {plot_path}")
    
    # Also save individual plots
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(sma(hero_rewards, window), color='blue', label=f"Hero Rewards (SMA {window})")
    ax1.plot(sma(villain_rewards, window), color='red', label=f"Villain Rewards (SMA {window})")
    ax1.set_title("Rewards Comparison")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Reward")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "rewards_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close(fig1)
    
    plt.close(fig)
    return plot_path

def save_training_data_to_csv(episode_data, save_dir="."):
    """Save training data to CSV file"""
    csv_path = os.path.join(save_dir, "training_data.csv")
    
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = ['episode', 'hero_reward', 'villain_reward', 'steps', 
                      'winner', 'hero_avg_prob', 'villain_avg_prob']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for data in episode_data:
            writer.writerow(data)
    
    print(f"CSV data saved to: {csv_path}")
    return csv_path

def save_config_to_file(save_dir=".", hero_config=None, villain_config=None, training_config=None):
    """Save configuration parameters to a text file"""
    config_path = os.path.join(save_dir, "training_config.txt")
    
    with open(config_path, 'w') as f:
        f.write("=== TRAINING CONFIGURATION ===\n\n")
        f.write(f"Training completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("=== HERO AGENT CONFIG ===\n")
        for key, value in (hero_config or {}).items():
            f.write(f"{key}: {value}\n")
        
        f.write("\n=== VILLAIN AGENT CONFIG ===\n")
        for key, value in (villain_config or {}).items():
            f.write(f"{key}: {value}\n")
        
        f.write("\n=== TRAINING CONFIG ===\n")
        for key, value in (training_config or {}).items():
            f.write(f"{key}: {value}\n")
    
    print(f"Config saved to: {config_path}")
    return config_path

# ======= MAIN TRAINING =======
def main():
    pygame.init()
    env = GridEnv(render_mode="human")

    hero = PPOAgent(**PPO_CONFIG_Hero)
    villain = PPOAgent(**PPO_CONFIG_Villan)

    hero_rewards, villain_rewards = [], []
    hero_steps, villain_steps = [], []
    episode_data = []  # Store detailed episode data for CSV
    
    # Create directory for saving results
    results_dir = create_results_directory()
    
    for ep in range(TRAINING_CONFIG["num_episodes"]):
        state, _ = env.reset()
        done = False
        hero_ep_r, villain_ep_r = 0, 0
        steps = 0
        hero_probs, villain_probs = [], []  # Track action probabilities
        
        while not done and steps < TRAINING_CONFIG["timesteps_per_episode"]:
            handle_events()
            s_t = torch.tensor(state, dtype=torch.float32)

            hero_act, hero_prob = sample_action(hero, s_t)
            villain_act, villain_prob = sample_action(villain, s_t)
            
            hero_probs.append(hero_prob)
            villain_probs.append(villain_prob)

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
        
        # Determine winner
        winner = "Hero" if info.get("hero_win") else ("Villain" if info.get("villain_win") else "None")
        
        # Calculate average probabilities
        hero_avg_prob = np.mean(hero_probs) if hero_probs else 0
        villain_avg_prob = np.mean(villain_probs) if villain_probs else 0
        
        # Store episode data for CSV
        episode_data.append({
            'episode': ep + 1,
            'hero_reward': hero_ep_r,
            'villain_reward': villain_ep_r,
            'steps': steps,
            'winner': winner,
            'hero_avg_prob': hero_avg_prob,
            'villain_avg_prob': villain_avg_prob
        })

        print(f"EP {ep+1}/{TRAINING_CONFIG['num_episodes']}  "
              f"Hero={hero_ep_r:.2f}  Villain={villain_ep_r:.2f}  "
              f"Steps={steps}  Winner={winner}")

    env.close()
    
    # Save all results
    window = TRAINING_CONFIG["sma_window_size"]
    
    # Save plots
    save_training_plots(hero_rewards, hero_steps, villain_rewards, 
                        villain_steps, window, results_dir, episode_data)
    
    # Save CSV data
    save_training_data_to_csv(episode_data, results_dir)
    
    # Save configuration
    save_config_to_file(results_dir, PPO_CONFIG_Hero, PPO_CONFIG_Villan, TRAINING_CONFIG)
    
    # Show final summary
    print("\n=== TRAINING SUMMARY ===")
    print(f"Results saved in: {results_dir}")
    print(f"Total episodes: {TRAINING_CONFIG['num_episodes']}")
    print(f"Hero average reward: {np.mean(hero_rewards):.2f}")
    print(f"Villain average reward: {np.mean(villain_rewards):.2f}")
    print(f"Hero wins: {sum(1 for d in episode_data if d['winner'] == 'Hero')}")
    print(f"Villain wins: {sum(1 for d in episode_data if d['winner'] == 'Villain')}")
    print(f"Draws: {sum(1 for d in episode_data if d['winner'] == 'None')}")
    
    # Optionally show the plot (commented out to allow headless operation)
    # plt.show()

if __name__ == "__main__":
    main()