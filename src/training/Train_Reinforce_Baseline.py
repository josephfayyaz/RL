# -------------------- Imports -------------------- #
import sys, os
import csv
import torch
import numpy as np
from timeit import default_timer as timer
import gym

# Include parent directory in path for custom imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.env.custom_hopper import *  # Custom MuJoCo Hopper environments
from src.agents.agent_baseline import Agent, Policy  # REINFORCE agent without baseline

# -------------------- Device Setup -------------------- #
device = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------- Config -------------------- #
SAVE_INTERVAL = 200000
MODEL_SAVE_DIR = "../../models/reinforce_baseline"
TRAIN_LOG_PATH = "../../Logs/baseline/source_training_baseline_5M.csv"
FINAL_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, "model_reinforce_baseline_final_source_5M.mdl")

# -------------------- Evaluation Function -------------------- #
def evaluate_agent_on_env(env, agent, episodes, threshold):
    returns = []
    for _ in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        while not done:
            action, _ = agent.get_action(state, evaluation=True)
            state, reward, done, _ = env.step(action.detach().cpu().numpy())
            total_reward += reward
        returns.append(total_reward)

    mean_r = np.mean(returns)
    std_r = np.std(returns)
    p5_r = np.percentile(returns, 5)
    success_rate = sum(r >= threshold for r in returns) / len(returns)
    return mean_r, std_r, p5_r, success_rate, returns


# -------------------- Main Training Loop -------------------- #
def main():
    config = {
        "policy_type": "MlpPolicy",
        "total_timesteps": 5000000,
        "env_id_source": "CustomHopper-source-v0",
        "env_id_target": "CustomHopper-target-v0",
        "test_episodes": 50,
        "success_threshold": 1000
    }

    # Setup
    os.makedirs("../../Logs/baseline", exist_ok=True)
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

    env = gym.make(config["env_id_source"])
    env_target = gym.make(config["env_id_target"])

    observation_space_dim = env.observation_space.shape[-1]
    action_space_dim = env.action_space.shape[-1]

    policy = Policy(observation_space_dim, action_space_dim)
    agent = Agent(policy, device=device)

    total_rewards = []
    train_reward = 0
    state = env.reset()
    episode_number = 1
    global_timesteps = 0
    reached_1000 = False
    steps_to_1000 = None
    next_save_step = SAVE_INTERVAL  # ← NEW
    start = timer()

    with open(TRAIN_LOG_PATH, "w", newline="") as training_csv:
        train_writer = csv.writer(training_csv)
        train_writer.writerow(["episode", "timestep", "mean_reward", "std_reward", "steps_to_1000_return", "policy_loss", "entropy"])

        while global_timesteps < config["total_timesteps"]:
            action, action_log_prob = agent.get_action(state)
            prev_state = state
            state, reward, done, _ = env.step(action.detach().cpu().numpy())

            agent.store_outcome(prev_state, state, action_log_prob, reward, done)
            train_reward += reward
            global_timesteps += 1

            # Checkpoint at fixed intervals (even during episode)
            if global_timesteps >= next_save_step:
                ckpt_path = os.path.join(MODEL_SAVE_DIR, f"model_reinforce_vanilla_source_step_{global_timesteps}.mdl")
                torch.save(agent.policy.state_dict(), ckpt_path)
                print(f"📦 Checkpoint saved to {ckpt_path}")
                next_save_step += SAVE_INTERVAL

            if done:
                # Estimate entropy from saved log_probs
                entropies = [-log_prob.item() for log_prob in agent.action_log_probs]
                entropy = np.mean(entropies)

                # Policy update
                policy_loss = agent.update_policy() or 0.0

                total_rewards.append(train_reward)

                # Log first time reaching threshold
                if train_reward >= config["success_threshold"] and not reached_1000:
                    reached_1000 = True
                    steps_to_1000 = global_timesteps
                    print(f"🎯 Return ≥ {config['success_threshold']} at timestep {global_timesteps}")

                mean_r = np.mean(total_rewards)
                std_r = np.std(total_rewards)

                train_writer.writerow([
                    episode_number,
                    global_timesteps,
                    mean_r,
                    std_r,
                    steps_to_1000 or "",
                    policy_loss,
                    entropy
                ])

                print(f"[{global_timesteps}] Ep:{episode_number} R:{train_reward:.1f} | Mean:{mean_r:.1f}, Entropy:{entropy:.3f}, Loss:{policy_loss:.3f}")

                state = env.reset()
                train_reward = 0
                episode_number += 1

    # Save final model
    torch.save(agent.policy.state_dict(), FINAL_MODEL_PATH)
    print(f"✅ Final model saved to {FINAL_MODEL_PATH}")

    # End training
    end = timer()
    print(f"⏱ Training completed in {end - start:.2f} seconds")

    # Evaluate on source and target
    mean_s, std_s, p5_s, sr_s, _ = evaluate_agent_on_env(env, agent, config["test_episodes"], config["success_threshold"])

    print(f"📊 Source → Mean: {mean_s:.1f}, STD: {std_s:.1f}, P5: {p5_s:.1f}, Success: {sr_s:.2f}")

# -------------------- Run -------------------- #
if __name__ == '__main__':
    main()
