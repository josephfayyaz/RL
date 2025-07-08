# -------------------- Imports -------------------- #
import sys, os
import csv
import torch
import numpy as np
from timeit import default_timer as timer
import gym

# Include parent directory in path for custom imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from env.custom_hopper import *  # Custom MuJoCo Hopper environments
from agents.agent_baseline import Agent, Policy  # REINFORCE agent without baseline

# -------------------- Device Setup -------------------- #
device = "cuda"

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
    print("=== Starting REINFORCE training ===", flush=True)

    config = {
        "policy_type": "MlpPolicy",
        "total_timesteps": 1000000,
        "env_id_source": "CustomHopper-source-v0",
        "env_id_target": "CustomHopper-source-v0",
        "test_episodes": 50,
        "success_threshold": 1000,
        "seed": 42
    }

    # Create output folders
    os.makedirs("../../Logs/baseline", exist_ok=True)
    os.makedirs("models/model_reinforce_baseline", exist_ok=True)

    print("[INFO] Environments and agent setup...", flush=True)

    # Create environments
    env = gym.make(config["env_id_source"])
    env_target = gym.make(config["env_id_target"])

    observation_space_dim = env.observation_space.shape[-1]
    action_space_dim = env.action_space.shape[-1]

    policy = Policy(observation_space_dim, action_space_dim)
    agent = Agent(policy, device="cuda")

    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])

    # Logging setup
    total_rewards = []
    train_reward = 0
    state = env.reset()
    reached_1000 = False
    steps_to_1000 = None
    global_timesteps = 0
    eval_interval = 100
    start = timer()

    suffix = f"{config['total_timesteps']}_seed{config['seed']}"
    training_csv = open(f"../../Logs/baseline/training_baseline_{suffix}.csv", "w", newline="")
    train_writer = csv.writer(training_csv)
    train_writer.writerow([ "episode" ,"timestep", "mean_reward", "std_reward", "steps_to_1000_return"])

    lc_file = open(f"../../Logs/baseline/learning_curve_baseline_{suffix}.csv", "w", newline="")
    lc_writer = csv.writer(lc_file)
    lc_writer.writerow(["timesteps", "mean_reward"])

    episode_rewards_csv = open(f"../../Logs/baseline/episode_rewards_baseline_{suffix}.csv", "w", newline="")
    episode_writer = csv.writer(episode_rewards_csv)
    episode_writer.writerow(["episode", "reward"])

    episode_curve_file = open(f"../../Logs/baseline/episode_curve_baseline_{suffix}.csv", "w", newline="")
    episode_curve_writer = csv.writer(episode_curve_file)
    episode_curve_writer.writerow(["episode", "timestep", "return"])

    print("[INFO] Starting training loop", flush=True)

    for _ in range(config["total_timesteps"]):
        action, action_probabilities = agent.get_action(state)
        previous_state = state
        state, reward, done, _ = env.step(action.detach().cpu().numpy())
        global_timesteps += 1
        train_reward += reward

        agent.store_outcome(previous_state, state, action_probabilities, reward, done)

        print(f"[Step {global_timesteps}] reward: {reward:.2f}  done: {done}", end="\r", flush=True)

        if done:
            print(f"\n[Episode {len(total_rewards)+1}] Ended | Return: {train_reward:.2f}", flush=True)

            agent.update_policy()
            total_rewards.append(train_reward)
            episode_writer.writerow([len(total_rewards), train_reward])

            episode_curve_writer.writerow([len(total_rewards), global_timesteps, train_reward])

            if train_reward >= config["success_threshold"] and not reached_1000:
                reached_1000 = True
                steps_to_1000 = global_timesteps
                torch.save(agent.policy.state_dict(), f"models/model_reinforce_baseline/best_model_step_{steps_to_1000}.mdl")

            mean_reward = np.mean(total_rewards)
            std_reward = np.std(total_rewards)
            train_writer.writerow([len(total_rewards) , global_timesteps, mean_reward, std_reward, steps_to_1000 or ""])

            state = env.reset()
            train_reward = 0
            if global_timesteps % eval_interval == 0 and len(total_rewards) > 0:
                mean_reward = np.mean(total_rewards)
                lc_writer.writerow([global_timesteps, f"{mean_reward:.6f}"])


    training_csv.close()
    lc_file.close()
    episode_rewards_csv.close()
    episode_curve_file.close()

    print("\n[INFO] Training complete.", flush=True)
    if steps_to_1000:
        print(f"Reached return ≥ {config['success_threshold']} at timestep {steps_to_1000}")
    else:
        print(f"Return ≥ {config['success_threshold']} was never reached.")

    end = timer()
    print(f"Total training time: {end - start:.2f} seconds", flush=True)


# -------------------- Entry Point -------------------- #
if __name__ == '__main__':
    main()