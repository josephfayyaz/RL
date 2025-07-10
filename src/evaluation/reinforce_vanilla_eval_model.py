import os
import sys
import argparse
import torch
import gym
import csv
import numpy as np
from datetime import datetime
import random

# Include parent directory in path for custom imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.env.custom_hopper import *  # Custom MuJoCo Hopper environments
from src.agents.agent_reinforce_normal import Agent, Policy  # REINFORCE agent without baseline

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='/home/parastoo/Desktop/new_yousef/RL/models/reinforce_vanilla/model_reinforce_vanilla_source_1M.mdl',
                        help='Path to the saved reinforce-vanilla model (.mdl)')
    parser.add_argument('--domain', type=str, default='target',
                        choices=['source', 'cdr', 'udr', 'target'], help='Environment domain to evaluate on')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility')
    parser.add_argument('--n_eval_episodes', type=int, default=50, help='Number of evaluation episodes')
    parser.add_argument('--algorithm', type=str, default='reinforce-vanilla', choices=['reinforce-vanilla'], help='Algorithm name for logging')
    parser.add_argument('--entropy_sched', type=bool, default=False, help='Whether entropy scheduling was used')
    parser.add_argument('--log_path', type=str, default='../../Logs/reinforce-vanilla_eval/', help='Folder to save evaluation logs')
    return parser.parse_args()


def make_eval_env(domain, seed):
    def _init():
        env = CustomHopper(domain=domain, total_timesteps=0)
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env
    return _init()


def evaluate_episode_rewards(agent, env, n_episodes, device):
    episode_rewards = []
    for ep in range(n_episodes):
        obs = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            action, _ = agent.get_action(obs, evaluation=True)
            obs, reward, done, _ = env.step(action.detach().cpu().numpy())
            total_reward += reward

        episode_rewards.append(total_reward)
        print(f"Episode {ep+1}/{n_episodes} - Reward: {total_reward:.2f}")
    return episode_rewards


def save_per_episode_rewards(csv_path, episode_rewards, model_path, domain, seed):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    write_header = not os.path.exists(csv_path)

    with open(csv_path, mode='a', newline='') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(['timestamp', 'model', 'domain', 'seed', 'episode', 'reward'])
        for i, reward in enumerate(episode_rewards):
            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                os.path.basename(model_path),
                domain,
                seed,
                i + 1,
                f"{reward:.2f}"
            ])
    print(f"✅ Per-episode rewards saved to {csv_path}")


def main():
    args = parse_args()

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"❌ Model file not found at: {args.model_path}")

    # Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🚀 Evaluating reinforce-vanilla on {args.domain} using device: {device}")
    print(f"📂 Loading model from: {args.model_path}")

    # Initialize environment and agent
    env = make_eval_env(args.domain, args.seed)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    policy = Policy(obs_dim, act_dim).to(device)
    policy.load_state_dict(torch.load(args.model_path, map_location=device), strict=True)
    agent = Agent(policy, device=device)

    # Evaluate agent
    episode_rewards = evaluate_episode_rewards(agent, env, args.n_eval_episodes, device)
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)

    print(f"\n📊 Evaluation Summary:")
    print(f"  Mean reward:    {mean_reward:.2f}")
    print(f"  Std deviation:  {std_reward:.2f}")

    # Save CSV
    model_name = os.path.basename(args.model_path).replace(".mdl", "")
    csv_filename = f"{args.algorithm}_{args.domain}_seed_{args.seed}_eval_full.csv"
    full_csv_path = os.path.join(args.log_path, csv_filename)

    save_per_episode_rewards(full_csv_path, episode_rewards, args.model_path, args.domain, args.seed)


if __name__ == "__main__":
    main()
