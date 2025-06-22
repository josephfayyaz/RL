"""
Train_Reinforce_vanila.py
=========================
Minimal REINFORCE baseline on the CustomHopper environment.

Changes over the original version
---------------------------------
1. Added extensive inline comments / doc-strings.
2. Reorganized imports for clarity.
3. Guarded all file I/O paths with `os.makedirs(..., exist_ok=True)`.
4. Added Block B (post-training utility) that summarises CSV logs.
"""

# ----------------------------- standard imports -----------------------------
import os
import sys
import csv
from timeit import default_timer as timer
from typing import Tuple, List

import numpy as np
import torch
import gym
import wandb

# ---------------------------------------------------------------------------
# make `src/` visible so that `from env.custom_hopper import *` works when the
# script is launched from the project root via `python -m src.training...`
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)

# ----------------------------- local imports -------------------------------
from env.custom_hopper import *                     # custom MuJoCo env
from agents.agent_reinforce_normal import Agent, Policy  # policy + wrapper

# ----------------------------- global settings -----------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ===========================================================================


def evaluate_agent_on_env(
    env: gym.Env,
    agent: Agent,
    episodes: int,
    threshold: float
) -> Tuple[float, float, float, float, List[float]]:
    """
    Run `episodes` roll-outs with the *deterministic* policy and report
    statistics that we need for the paper and CSV logs.

    Returns
    -------
    mean_r : float
        Average episodic return.
    std_r : float
        Standard deviation of returns.
    p5_r : float
        5-th percentile (CVaR proxy).
    success_rate : float
        Fraction of episodes whose return ≥ `threshold`.
    returns : list[float]
        Raw episodic returns for potential offline analysis.
    """
    returns = []

    for _ in range(episodes):
        state = env.reset()
        total_reward = 0.0
        done = False

        # -------- rollout ------------
        while not done:
            action, _ = agent.get_action(state, evaluation=True)
            state, reward, done, _ = env.step(action.detach().cpu().numpy())
            total_reward += reward
        # -----------------------------
        returns.append(total_reward)

    mean_r = float(np.mean(returns))
    std_r = float(np.std(returns))
    p5_r = float(np.percentile(returns, 5))
    success_rate = float(sum(r >= threshold for r in returns) / len(returns))
    return mean_r, std_r, p5_r, success_rate, returns


# ===========================================================================
#                               main routine
# ===========================================================================
def main() -> None:
    """Train a REINFORCE policy on the *source* domain and evaluate."""
    # ------------------ run-specific hyper-parameters ----------------------
    config = {
        "policy_type": "MlpPolicy",
        "total_timesteps": 100_000,               # ← easy to sweep
        "env_id_source": "CustomHopper-udr-v0",
        "env_id_target": "CustomHopper-target-v0",
        "test_episodes": 50,
        "success_threshold": 1_000
    }

    # ------------------ Weights & Biases initialisation -------------------
    run = wandb.init(
        project="reinforce_baseline_100K_UDR_saghal_1",
        entity=None,                    # set if you have a team/org account
        config=config,
        sync_tensorboard=True
    )
    wandb.run.name = "Reinforce_Baseline_Run"
    wandb.run.save()

    # ------------------ create training / test environments ---------------
    env = gym.make(config["env_id_source"])
    env_target = gym.make(config["env_id_target"])

    observation_space_dim = env.observation_space.shape[-1]
    action_space_dim = env.action_space.shape[-1]

    # ------------------ instantiate policy & agent ------------------------
    policy = Policy(observation_space_dim, action_space_dim)
    agent = Agent(policy, device=DEVICE)

    # ------------------ bookkeeping variables -----------------------------
    total_rewards: List[float] = []
    train_reward = 0.0
    state = env.reset()
    reached_1000 = False
    steps_to_1000 = None
    start_time = timer()

    # ------------------ CSV logger set-up ---------------------------------
    log_dir = os.path.join(PROJECT_ROOT, "logs", "baseline")
    os.makedirs(log_dir, exist_ok=True)
    training_csv_path = os.path.join(
        log_dir, "training_baseline_100K_UDR_log.csv"
    )
    with open(training_csv_path, "w", newline="") as training_csv:
        train_writer = csv.writer(training_csv)
        train_writer.writerow(
            ["timestep", "mean_reward", "std_reward", "steps_to_1000_return"]
        )

        # ========================== training loop =========================
        for t in range(config["total_timesteps"]):
            # policy forward pass (stochastic during training)
            action, action_probs = agent.get_action(state)
            prev_state = state
            state, reward, done, _ = env.step(action.detach().cpu().numpy())

            # store transition for REINFORCE update
            agent.store_outcome(prev_state, state, action_probs, reward, done)

            train_reward += reward

            if done:
                agent.update_policy()             # single gradient step
                total_rewards.append(train_reward)

                # measure sample-efficiency proxy
                if train_reward >= config["success_threshold"] and not reached_1000:
                    reached_1000 = True
                    steps_to_1000 = t + 1          # current timestep index

                # -------- scalar logs to CSV and wandb -----------------
                mean_r = float(np.mean(total_rewards))
                std_r = float(np.std(total_rewards))

                wandb.log(
                    {
                        "mean_reward": mean_r,
                        "std_reward": std_r,
                        "timestep": t + 1
                    }
                )
                train_writer.writerow(
                    [t + 1, mean_r, std_r, steps_to_1000 or ""]
                )

                # reset episode
                state = env.reset()
                train_reward = 0.0

    # ------------------ training done – save model & stats ---------------
    total_time = timer() - start_time
    print(f"Training completed in {total_time:.1f} s.")

    model_dir = os.path.join(PROJECT_ROOT, "models", "reinforce")
    os.makedirs(model_dir, exist_ok=True)
    torch.save(
        agent.policy.state_dict(),
        os.path.join(model_dir, "reinforce_baseline_100k.mdl")
    )

    if steps_to_1000:
        wandb.log({"steps_to_1000_return": steps_to_1000})
        print(f"Reached ≥ {config['success_threshold']} return after "
              f"{steps_to_1000} environment steps.")
    else:
        print(f"Return ≥ {config['success_threshold']} was never reached.")

    # ============================== evaluation ============================
    # ---- evaluate on *source* domain ----
    mean_r, std_r, p5_r, success_r, _ = evaluate_agent_on_env(
        env, agent, config["test_episodes"], config["success_threshold"]
    )
    wandb.log(
        {
            "test_source_mean_reward": mean_r,
            "test_source_std_reward": std_r,
            "test_source_5th_percentile": p5_r,
            "test_source_success_rate": success_r
        }
    )

    # ---- evaluate on *target* domain ----
    mean_rt, std_rt, p5_rt, success_rt, _ = evaluate_agent_on_env(
        env_target, agent, config["test_episodes"], config["success_threshold"]
    )
    wandb.log(
        {
            "test_target_mean_reward": mean_rt,
            "test_target_std_reward": std_rt,
            "test_target_5th_percentile": p5_rt,
            "test_target_success_rate": success_rt
        }
    )

    # ---- robustness curve (mass levels) ----
    levels = [f"CustomHopper-sudr-{i}-v0" for i in range(5)]
    returns_per_level: List[float] = []

    for level_id in levels:
        try:
            test_env = gym.make(level_id)
        except gym.error.Error:
            print(f"[warn] {level_id} not registered – skipping.")
            continue

        mean_r_l, *_ = evaluate_agent_on_env(
            test_env, agent, config["test_episodes"], config["success_threshold"]
        )
        returns_per_level.append(mean_r_l)

    if returns_per_level:
        auc = float(np.trapz(returns_per_level, dx=1))
        wandb.log({"AUC_robustness_curve": auc})
        print(f"AUC across levels: {auc:.2f}")

    # ---- save evaluation CSV ----
    eval_csv_path = os.path.join(log_dir, "test_log_baseline.csv")
    with open(eval_csv_path, "w", newline="") as test_log:
        test_writer = csv.writer(test_log)
        test_writer.writerow(
            ["env_type", "mean_reward", "std_reward", "5th_percentile", "success_rate"]
        )
        test_writer.writerow(["source", mean_r, std_r, p5_r, success_r])
        test_writer.writerow(["target", mean_rt, std_rt, p5_rt, success_rt])

    # ======================================================================
    #                            Block B (optional)
    # ======================================================================
    # Post-process *all* CSVs in `logs/baseline/` to build a single summary
    # table.  This is entirely offline and does not affect the training run.
    # ----------------------------------------------------------------------
    summary_csv = os.path.join(log_dir, "summary_baseline.csv")
    csv_header_written = os.path.exists(summary_csv)

    import glob
    import pandas as pd

    dfs = []
    for csv_file in glob.glob(os.path.join(log_dir, "training_baseline_*_log.csv")):
        df = pd.read_csv(csv_file)
        # extract final row (= end-of-run statistics)
        final_row = df.iloc[-1]
        final_row["run_name"] = os.path.basename(csv_file).replace("_log.csv", "")
        dfs.append(final_row)

    if dfs:
        summary_df = pd.concat(dfs, axis=1).T
        summary_df.to_csv(summary_csv, index=False,
                          mode="a" if csv_header_written else "w",
                          header=not csv_header_written)
        print(f"Appended summary row(s) to {summary_csv}")

    # ----------------------------------------------------------------------
    run.finish()


# ---------------------------------------------------------------------------
if __name__ == '__main__':
    main()
