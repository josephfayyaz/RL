"""
train_baseline.py
~~~~~~~~~~~~~~~~~
Vanilla REINFORCE “source–only” baseline for the Sim-to-Real Hopper study.

The script is intentionally self-contained so that newcomers can run

    $ python -m src.training.train_baseline

and obtain:
    • training CSV under logs/csv/baseline/
    • WEIGHTS  under models/reinforce/
    • Weights-and-Biases tracking (optional)

Author: MLDL course group, 2025-06-22
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from timeit import default_timer as timer
from typing import Tuple, List

import csv
import numpy as np
import torch
import gymnasium as gym
import wandb

# --------------------------------------------------------------------------- #
# Import project modules (works after `pip install -e .` or PYTHONPATH setup) #
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parents[2]          # repo root
sys.path.append(str(PROJECT_ROOT))                          # for ad-hoc runs

from src.envs.custom_hopper import *                       # registers env IDs
from src.agents.agent_baseline import Agent, Policy
# If you created utils.schedules but do NOT need it here, simply leave it:
# from src.utils.schedules import noop_schedule             # placeholder hook


# ========================================================================== #
#                       1.  Hyper-parameters & paths                         #
# ========================================================================== #
@dataclass(slots=True)
class Config:
    """Centralised experiment knobs."""
    policy_type: str = "MlpPolicy"
    total_timesteps: int = 100_000
    env_id_source: str = "CustomHopper-source-v0"
    env_id_target: str = "CustomHopper-target-v0"
    test_episodes: int = 50
    success_threshold: int = 1_000
    run_name: str = "reinforce_baseline_100k"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # folders (created if absent)
    log_dir: Path = PROJECT_ROOT / "logs" / "csv" / "baseline"
    model_dir: Path = PROJECT_ROOT / "models" / "reinforce"


CFG = Config()                                             # global config


# ========================================================================== #
#                          2.  Utility Functions                             #
# ========================================================================== #
def evaluate_agent_on_env(
    env: gym.Env,
    agent: Agent,
    episodes: int,
    threshold: float,
) -> Tuple[float, float, float, float, List[float]]:
    """Run *episodes* roll-outs and compute summary statistics."""
    returns = []
    for _ in range(episodes):
        state, _ = env.reset()
        done, total_reward = False, 0.0
        while not done:
            action, _ = agent.get_action(state, evaluation=True)
            state, reward, done, _, _ = env.step(action.cpu().numpy())
            total_reward += reward
        returns.append(total_reward)

    returns_np = np.asarray(returns, dtype=np.float32)
    mean_r = float(returns_np.mean())
    std_r = float(returns_np.std())
    p5_r = float(np.percentile(returns_np, 5))
    success_rate = float((returns_np >= threshold).mean())
    return mean_r, std_r, p5_r, success_rate, returns                         # noqa: R503


def maybe_mkdir(path: Path) -> None:
    """Create *path* (including parents) if it does not already exist."""
    path.mkdir(parents=True, exist_ok=True)


# ========================================================================== #
#                         3.  Main Training Routine                          #
# ========================================================================== #
def main() -> None:
    # -------------------- WandB set-up (can be disabled) ------------------ #
    run = wandb.init(
        project="hopper_sim2real",
        name=CFG.run_name,
        config=CFG.__dict__,
        sync_tensorboard=False,
        mode="online",       # set "disabled" for offline tests
    )

    # -------------------- Environment Creation --------------------------- #
    env = gym.make(CFG.env_id_source)
    env_target = gym.make(CFG.env_id_target)

    obs_dim = env.observation_space.shape[-1]
    act_dim = env.action_space.shape[-1]

    policy = Policy(obs_dim, act_dim)
    agent = Agent(policy, device=CFG.device)

    # -------------------- I/O preparation -------------------------------- #
    maybe_mkdir(CFG.log_dir)
    maybe_mkdir(CFG.model_dir)

    csv_path = CFG.log_dir / f"training_{CFG.run_name}.csv"
    csv_file = csv_path.open("w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(
        ["timestep", "mean_reward", "std_reward", "steps_to_1000_return"]
    )

    # -------------------- Training Loop ---------------------------------- #
    total_rewards: List[float] = []
    train_reward = 0.0
    state, _ = env.reset()

    reached_1000 = False
    steps_to_1000: int | None = None
    start_time = timer()

    for t in range(CFG.total_timesteps):
        # ********** Schedule hook (“Block B”) **************************** #
        # For vanilla REINFORCE nothing happens, but you could plug in:
        #   beta = entropy_schedule(t)   or    mass_bounds = curriculum(t)
        # ***************************************************************** #

        action, action_logp = agent.get_action(state)
        prev_state = state
        state, reward, done, _, _ = env.step(action.cpu().numpy())

        agent.store_outcome(prev_state, state, action_logp, reward, done)

        train_reward += reward

        if done:
            agent.update_policy()            # policy gradient step
            total_rewards.append(train_reward)

            if train_reward >= CFG.success_threshold and not reached_1000:
                reached_1000 = True
                steps_to_1000 = t + 1

            mean_reward = float(np.mean(total_rewards))
            std_reward = float(np.std(total_rewards))

            wandb.log(
                {
                    "mean_reward": mean_reward,
                    "std_reward": std_reward,
                    "timestep": t + 1,
                }
            )
            csv_writer.writerow(
                [t + 1, mean_reward, std_reward, steps_to_1000 or ""]
            )

            state, _ = env.reset()
            train_reward = 0.0

    csv_file.close()
    total_train_time = timer() - start_time
    print(f"Training finished in {total_train_time:.1f} s.")

    # ===================================================================== #
    #                               Testing                                 #
    # ===================================================================== #
    def log_eval_results(
        prefix: str, metrics: Tuple[float, float, float, float, List[float]]
    ) -> None:
        mean_r, std_r, p5_r, success, _ = metrics
        wandb.log(
            {
                f"{prefix}_mean_reward": mean_r,
                f"{prefix}_std_reward": std_r,
                f"{prefix}_5th_percentile": p5_r,
                f"{prefix}_success_rate": success,
            }
        )

    log_eval_results("test_source", evaluate_agent_on_env(
        env, agent, CFG.test_episodes, CFG.success_threshold
    ))
    log_eval_results("test_target", evaluate_agent_on_env(
        env_target, agent, CFG.test_episodes, CFG.success_threshold
    ))

    # -------------------- Robustness AUC (optional) ---------------------- #
    levels = [f"CustomHopper-sudr-{i}-v0" for i in range(5)]
    returns_per_level = []
    for level_id in levels:
        try:
            test_env = gym.make(level_id)
        except gym.error.Error:
            print(f"[warn] {level_id} not registered; skipping.")
            continue
        mean_r_lvl, *_ = evaluate_agent_on_env(
            test_env, agent, CFG.test_episodes, CFG.success_threshold
        )
        returns_per_level.append(mean_r_lvl)

    if returns_per_level:
        auc = float(np.trapz(returns_per_level, dx=1))
        wandb.log({"auc_robustness_curve": auc})
        print(f"AUC across {len(returns_per_level)} levels = {auc:.2f}")

    # -------------------- Persist weights & wrap-up ---------------------- #
    model_path = CFG.model_dir / f"{CFG.run_name}.mdl"
    torch.save(agent.policy.state_dict(), model_path)
    print(f"Saved model → {model_path}")
    run.finish()


# ========================================================================== #
#                               Entry point                                  #
# ========================================================================== #
if __name__ == "__main__":
    main()
