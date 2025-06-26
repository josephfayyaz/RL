import os
import sys
import argparse
import torch
import gym
import csv
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from datetime import datetime
import random

# Allow import from parent directory (custom_hopper)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from env.custom_hopper import CustomHopper


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='/home/joseph/python-proj/udr_ES/models/PPO/vanila/PPO_source_ES_False_seed_0_CustomHopper_source_v0_5000000.zip', help='Path to the saved PPO model (.zip)')
    parser.add_argument('--domain', type=str, default='-target-v0', choices=['-source-v0', 'cdr', 'udr','target'], help='Environment domain to evaluate on')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility')
    parser.add_argument('--n_eval_episodes', type=int, default=50, help='Number of evaluation episodes')
    parser.add_argument('--algorithm', type=str, default='PPO', choices=['PPO'], help='Algorithm used (for naming consistency)')
    parser.add_argument('--entropy_sched', type=bool, default=False, help='Whether entropy scheduling was used')
    parser.add_argument('--log_path', type=str, default='../../Logs/PPO_eval/', help='Folder to save evaluation CSV logs')

    return parser.parse_args()


def make_eval_env(domain, seed):
    def _init():
        env = CustomHopper(domain=domain, total_timesteps=0)
        # env = "CustomHopper-target-v0"
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return Monitor(env)
    return DummyVecEnv([_init])


def evaluate_episode_rewards(model, env, n_episodes):
    episode_rewards = []
    for ep in range(n_episodes):
        obs = env.reset()
        done, state = False, None
        total_reward = 0.0
        while not done:
            action, state = model.predict(obs, state=state, deterministic=True)
            obs, reward, done, _ = env.step(action)
            done = done[0] if isinstance(done, (list, np.ndarray)) else done
            total_reward += reward[0] if isinstance(reward, (list, np.ndarray)) else reward
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

    # Reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🚀 Evaluating on {args.domain} using device: {device}")
    print(f"📂 Loading model from: {args.model_path}")

    model = PPO.load(args.model_path, device=device)
    eval_env = make_eval_env(args.domain, args.seed)

    # Evaluate and collect per-episode rewards
    episode_rewards = evaluate_episode_rewards(model, eval_env, args.n_eval_episodes)
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)

    print(f"\n📊 Evaluation Summary:")
    print(f"  Mean reward:    {mean_reward:.2f}")
    print(f"  Std deviation:  {std_reward:.2f}")

    # Compose consistent CSV path
    model_name = os.path.basename(args.model_path).replace(".zip", "")
    csv_filename = f"{args.algorithm}_{args.domain}_ES_{args.entropy_sched}_seed_{args.seed}_eval_full.csv"
    full_csv_path = os.path.join(args.log_path, csv_filename)

    save_per_episode_rewards(full_csv_path, episode_rewards, args.model_path, args.domain, args.seed)


if __name__ == "__main__":
    main()
