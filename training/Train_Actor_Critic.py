#!/usr/bin/env python3
"""
Train an Actor–Critic baseline on the **source** Hopper domain.

Usage
-----
$ python -m src.training.train_actor_critic \
        --total-steps 1_000_000 \
        --save-every 10_000 \
        --device cuda

• Check-points are written to  models/actor_critic/
• Raw logs  (one row per episode)   go to logs/csv/actor_critic/
• Final test statistics             go to logs/csv/actor_critic/test_log_*.csv
"""

from __future__ import annotations
import argparse
import csv
from pathlib import Path
from time import perf_counter

import gym
import numpy as np
import torch

# --------------------------------------------------------------------------- #
# Project-internal imports                                                    #
# (the trainer lives in  src/training/,   repo root is two levels higher)     #
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parents[2]          # repo root
import sys
sys.path.append(str(ROOT / "src"))                  # add "src" to PYTHONPATH

from envs.custom_hopper import *                    # registers CustomHopper-*

from agents.agent_ac import Agent_ac, Policy_ac


# --------------------------------------------------------------------------- #
# Helper: quick evaluation loop                                               #
# --------------------------------------------------------------------------- #
def evaluate_agent(
    env: gym.Env,
    agent: Agent_ac,
    episodes: int,
    success_threshold: float,
) -> tuple[float, float, float, float, list[float]]:
    """Run the policy for `episodes` and return summary statistics."""
    returns: list[float] = []
    for _ in range(episodes):
        state, _ = env.reset()
        done, ep_return = False, 0.0
        while not done:
            action, _ = agent.get_action(state, evaluation=True)
            state, reward, done, _, _ = env.step(action.detach().cpu().numpy())
            ep_return += reward
        returns.append(ep_return)

    returns_arr = np.asarray(returns, dtype=np.float32)
    mean_r = float(returns_arr.mean())
    std_r = float(returns_arr.std(ddof=0))
    p5_r = float(np.percentile(returns_arr, 5))
    success_rate = float((returns_arr >= success_threshold).mean())
    return mean_r, std_r, p5_r, success_rate, returns  # last one is the list


# --------------------------------------------------------------------------- #
# Main training loop                                                          #
# --------------------------------------------------------------------------- #
def train(config: dict) -> None:
    """Actor–Critic on source Hopper, saving checkpoints and CSV logs."""
    # ------------ folders ------------ #
    model_dir = ROOT / "models" / "actor_critic"
    log_dir   = ROOT / "logs"   / "csv" / "actor_critic"
    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    ckpt_final   = model_dir / "model_actor_critic_final.mdl"
    ckpt_pattern = model_dir / "model_actor_critic_step_{:d}.mdl"
    train_csv    = log_dir   / "training_actor_critic.csv"
    test_csv     = log_dir   / "test_summary.csv"

    # ------------ envs ------------ #
    env         = gym.make(config["env_id_source"])
    env_target  = gym.make(config["env_id_target"])

    obs_dim     = env.observation_space.shape[-1]
    act_dim     = env.action_space.shape[-1]

    # ------------ agent ------------ #
    policy = Policy_ac(obs_dim, act_dim)
    agent  = Agent_ac(policy, config["device"])

    # ------------ bookkeeping ------------ #
    total_timesteps   = 0
    ep_return_running = 0.0
    reached_threshold = False
    steps_to_thresh   = None
    episode_returns   = []

    t0 = perf_counter()

    with train_csv.open("w", newline="") as f_csv:
        csv_writer = csv.writer(f_csv)
        csv_writer.writerow(
            [
                "timesteps",
                "mean_return",
                "std_return",
                "steps_to_1000",
                "actor_loss",
                "critic_loss",
                "entropy",
            ]
        )

        state, _ = env.reset()

        # ================= main loop ================= #
        while total_timesteps < config["total_timesteps"]:
            action, act_logp = agent.get_action(state)
            prev_state = state
            state, reward, done, _, _ = env.step(action.detach().cpu().numpy())

            agent.store_outcome(prev_state, state, act_logp, reward, done)

            ep_return_running += reward
            total_timesteps   += 1

            if done:
                # --- policy update after every episode --- #
                actor_l, critic_l, entropy = agent.update_policy()

                episode_returns.append(ep_return_running)

                # when did we first hit ≥ 1 000 return?
                if (
                    not reached_threshold
                    and ep_return_running >= config["success_threshold"]
                ):
                    reached_threshold = True
                    steps_to_thresh   = total_timesteps
                    print(
                        f"🎉 solved at t={total_timesteps:,d} "
                        f"return={ep_return_running:.1f}"
                    )

                # ------------- CSV log row ------------- #
                csv_writer.writerow(
                    [
                        total_timesteps,
                        np.mean(episode_returns),
                        np.std(episode_returns, ddof=0),
                        steps_to_thresh or "",
                        actor_l,
                        critic_l,
                        entropy,
                    ]
                )

                # ------------- stdout banner ------------- #
                print(
                    f"[{total_timesteps:,d}] "
                    f"R={ep_return_running:7.1f}  "
                    f"μ={np.mean(episode_returns):7.1f}  "
                    f"σ={np.std(episode_returns, ddof=0):6.1f}  "
                    f"Ent={entropy:5.3f}  "
                    f"LA={actor_l:6.3f}  LC={critic_l:6.3f}"
                )

                # ------------- periodic checkpoint ------------- #
                if (
                    config["save_every"] > 0
                    and total_timesteps % config["save_every"] == 0
                ):
                    ckpt_path = ckpt_pattern.as_posix().format(total_timesteps)
                    torch.save(agent.policy.state_dict(), ckpt_path)
                    print(f"🔑 checkpoint saved → {ckpt_path}")

                state, _ = env.reset()
                ep_return_running = 0.0  # reset episode return

    t1 = perf_counter()
    print(f"✅ training finished in {(t1 - t0)/60:.1f} min")

    # ------------ save final model ------------ #
    torch.save(agent.policy.state_dict(), ckpt_final)
    print(f"💾 final weights saved to → {ckpt_final}")

    # ----------------------------------------------------------------------- #
    # Evaluation (source & target)                                            #
    # ----------------------------------------------------------------------- #
    with test_csv.open("w", newline="") as f_test:
        test_writer = csv.writer(f_test)
        test_writer.writerow(
            ["env", "mean", "std", "5th-perc", "success_rate"]
        )

        for tag, ev in [("source", env), ("target", env_target)]:
            m, s, p5, sr, _ = evaluate_agent(
                ev, agent, config["test_episodes"], config["success_threshold"]
            )
            print(
                f"[{tag}]  μ={m:.1f}  σ={s:.1f}  p5={p5:.1f}  "
                f"succ={sr*100:.1f}%"
            )
            test_writer.writerow([tag, m, s, p5, sr])


# --------------------------------------------------------------------------- #
# CLI entry-point                                                             #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Actor–Critic trainer for Custom Hopper (source domain)."
    )
    parser.add_argument(
        "--total-steps",
        type=int,
        default=1_000_000,
        help="training horizon (environment steps)",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=10_000,
        help="checkpoint frequency (steps, 0 = off)",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cuda",
        help="force device; 'cuda' selects GPU-0 if available",
    )

    args = parser.parse_args()

    # unified configuration dict (easy to extend later)
    cfg = {
        "policy_type": "MlpPolicy",
        "total_timesteps": args.total_steps,
        "save_every": args.save_every,
        "device": ("cuda:0" if torch.cuda.is_available() else "cpu")
        if args.device == "cuda"
        else "cpu",
        "env_id_source": "CustomHopper-source-v0",
        "env_id_target": "CustomHopper-target-v0",
        "test_episodes": 50,
        "success_threshold": 1_000,
    }

    print(
        f"⚙️  device = {cfg['device']}   "
        f"steps = {cfg['total_timesteps']:,d}"
    )
    train(cfg)
