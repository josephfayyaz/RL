import argparse
import csv
import os
import random
import sys
from pathlib import Path

import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from stable_baselines3 import PPO

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.env.custom_hopper import *  # noqa: F401,F403 - required for gym env registration
from project_paths import FIGURES_DIR, LOGS_DIR, MODELS_DIR


class ObsNoiseWrapper(gym.ObservationWrapper):
    def __init__(self, env, eps):
        super().__init__(env)
        self.eps = eps

    def observation(self, obs):
        noise = np.random.uniform(-self.eps, self.eps, size=obs.shape)
        return obs + noise


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained PPO policy under bounded observation noise."
    )
    parser.add_argument(
        "--model-path",
        default=str(
            MODELS_DIR
            / "PPO"
            / "cdr_es"
            / "PPO_cdr_ES_True_seed_42_CustomHopper_cdr_v0_5000000.zip"
        ),
        help="Path to the trained PPO checkpoint.",
    )
    parser.add_argument(
        "--algorithm-label",
        default="PPO_CDR_ES_seed_42",
        help="Short label used in the CSV and figure filenames.",
    )
    parser.add_argument(
        "--domain",
        default="target",
        choices=["source", "cdr", "udr", "target"],
        help="Environment domain to evaluate under noisy observations.",
    )
    parser.add_argument("--episodes", type=int, default=30, help="Episodes per noise level.")
    parser.add_argument("--noise-max", type=float, default=0.5, help="Maximum noise bound.")
    parser.add_argument("--points", type=int, default=15, help="Number of noise levels.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--csv-path", default="", help="Optional custom CSV output path.")
    parser.add_argument("--figure-path", default="", help="Optional custom figure output path.")
    return parser.parse_args()


def make_noisy_env(domain, eps, seed):
    env = gym.make(f"CustomHopper-{domain}-v0")
    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return ObsNoiseWrapper(env, eps)


def evaluate(model, domain, eps, n_episodes, seed):
    env = make_noisy_env(domain, eps, seed)
    returns = []
    try:
        for _ in range(n_episodes):
            obs = env.reset()
            done = False
            total_reward = 0.0
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, _ = env.step(action)
                total_reward += reward
            returns.append(total_reward)
    finally:
        env.close()
    return float(np.mean(returns))


def main():
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    csv_path = Path(args.csv_path) if args.csv_path else LOGS_DIR / "PPO_robustness" / f"robustness_results_{args.algorithm_label}.csv"
    figure_path = Path(args.figure_path) if args.figure_path else FIGURES_DIR / "robustness_curve" / f"noise_robustness_curve_{args.algorithm_label}.png"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    figure_path.parent.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PPO.load(str(model_path), device=device)

    epsilons = np.linspace(0.0, args.noise_max, args.points)
    mean_returns = []
    for eps in epsilons:
        mean_return = evaluate(model, args.domain, eps, args.episodes, args.seed)
        mean_returns.append(mean_return)
        print(f"epsilon={eps:.2f} -> mean return={mean_return:.1f}")

    with open(csv_path, mode="w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epsilon", "mean_return"])
        for eps, mean_return in zip(epsilons, mean_returns):
            writer.writerow([eps, mean_return])
    print(f"Saved robustness results to {csv_path}")

    plt.figure(figsize=(6, 4))
    plt.plot(epsilons, mean_returns, marker="o")
    plt.xlabel("Perturbation noise bound epsilon")
    plt.ylabel("Average return J(epsilon)")
    plt.title(f"Robustness curve for {args.algorithm_label}")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(figure_path, dpi=300)
    print(f"Saved robustness figure to {figure_path}")


if __name__ == "__main__":
    main()
