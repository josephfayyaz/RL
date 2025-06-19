import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import csv
import torch
import numpy as np
from timeit import default_timer as timer
from agents.agent_ac import Agent_ac, Policy_ac
import gym
from env.custom_hopper import *




device = "cuda"

if device == 'cuda':
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f'training on {torch.cuda.get_device_name(torch.cuda.current_device()) }' if torch.cuda.is_available() else 'training on cpu')



SAVE_INTERVAL = 10000
MODEL_SAVE_DIR = "../Models/actor_critic/"
LOG_CSV_PATH = "../Logs/actor_critic/training_actor_critic_upgraded_3.csv"
TEST_LOG_PATH = "../Logs/actor_critic/test_log_3.csv"
FINAL_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, "model_actor_critic_3.mdl")


def evaluate_agent_on_env(env, agent, episodes, success_threshold):
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
    success_rate = sum(r >= success_threshold for r in returns) / len(returns)

    return mean_r, std_r, p5_r, success_rate, returns


def main():
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(LOG_CSV_PATH), exist_ok=True)

    config = {
        "policy_type": "MlpPolicy",
        "total_timesteps": 1000000,
        "env_id_source": "CustomHopper-source-v0",
        "env_id_target": "CustomHopper-source-v0",
        "test_episodes": 50,
        "success_threshold": 1000
    }

    env = gym.make(config["env_id_source"])
    env_target = gym.make(config["env_id_target"])

    observation_space_dim = env.observation_space.shape[-1]
    action_space_dim = env.action_space.shape[-1]

    policy = Policy_ac(observation_space_dim, action_space_dim)
    agent = Agent_ac(policy, device)

    total_rewards = []
    train_reward = 0
    state = env.reset()
    total_timesteps = 0
    reached_1000 = False
    steps_to_1000 = None
    start = timer()

    # CSV Logging Setup
    with open(LOG_CSV_PATH, "w", newline="") as training_csv:
        train_writer = csv.writer(training_csv)
        train_writer.writerow(["timestep", "mean_reward", "std_reward", "steps_to_1000_return", "actor_loss", "critic_loss", "entropy"])

        while total_timesteps < config["total_timesteps"]:
            action, action_probabilities = agent.get_action(state)
            prev_state = state
            state, reward, done, _ = env.step(action.detach().cpu().numpy())

            agent.store_outcome(prev_state, state, action_probabilities, reward, done)

            train_reward += reward
            total_timesteps += 1

            if done:
                actor_loss, critic_loss, entropy = agent.update_policy()
                total_rewards.append(train_reward)

                if train_reward >= config["success_threshold"] and not reached_1000:
                    reached_1000 = True
                    steps_to_1000 = total_timesteps
                    print(f"🎉 Episode solved! Return ≥ {config['success_threshold']} at timestep {total_timesteps}")

                mean_reward = np.mean(total_rewards)
                std_reward = np.std(total_rewards)

                train_writer.writerow([
                    total_timesteps,
                    mean_reward,
                    std_reward,
                    steps_to_1000 or "",
                    actor_loss,
                    critic_loss,
                    entropy
                ])

                print(f"Train reward:{train_reward:.1f},[{total_timesteps}] R: {mean_reward:.1f}, Entropy: {entropy:.3f}, Loss(A): {actor_loss:.3f}, Loss(C): {critic_loss:.3f}")

                # Save intermediate model
                if total_timesteps % SAVE_INTERVAL == 0:
                    ckpt_path = os.path.join(MODEL_SAVE_DIR, f"model_actor_critic_step_{total_timesteps}.mdl")
                    torch.save(agent.policy.state_dict(), ckpt_path)
                    print(f"Model checkpoint saved to: {ckpt_path}")

                state = env.reset()
                train_reward = 0

    end = timer()
    print(f"Training completed in {end - start:.2f} seconds")

    # Save final model
    torch.save(agent.policy.state_dict(), FINAL_MODEL_PATH)
    print(f"Final model saved to: {FINAL_MODEL_PATH}")

    # -------------------- Testing on Source --------------------
    mean_r, std_r, p5_r, success_rate, _ = evaluate_agent_on_env(env, agent, config["test_episodes"], config["success_threshold"])
    print(f"Test on source env → Mean: {mean_r:.2f}, STD: {std_r:.2f}, P5: {p5_r:.2f}, Success Rate: {success_rate:.2f}")

    # -------------------- Testing on Target --------------------
    mean_r_t, std_r_t, p5_r_t, success_rate_t, _ = evaluate_agent_on_env(env_target, agent, config["test_episodes"], config["success_threshold"])
    print(f"Test on target env → Mean: {mean_r_t:.2f}, STD: {std_r_t:.2f}, P5: {p5_r_t:.2f}, Success Rate: {success_rate_t:.2f}")

    # -------------------- AUC under robustness curve (source-to-source variation) --------------------
    # returns_per_seed = []
    # print("Evaluating robustness (source to source with different seeds)...")
    # for eval_seed in [0, 11, 42, 101, 256]:
    #     test_env = gym.make(config["env_id_source"])
    #     test_env.seed(eval_seed)
    #     test_env.action_space.seed(eval_seed)
    #     test_env.observation_space.seed(eval_seed)
    #
    #     mean_r_l, _, _, _, _ = evaluate_agent_on_env(test_env, agent, config["test_episodes"], config["success_threshold"])
    #     returns_per_seed.append(mean_r_l)
    #
    # auc = np.trapz(returns_per_seed, dx=1)
    # print(f"AUC across different seeds of source env: {auc:.2f}")

    # Save test results
    with open(TEST_LOG_PATH, "w", newline="") as test_log:
        test_writer = csv.writer(test_log)
        test_writer.writerow(["env_type", "mean_reward", "std_reward", "5th_percentile", "success_rate"])
        test_writer.writerow(["source", mean_r, std_r, p5_r, success_rate])
        test_writer.writerow(["target", mean_r_t, std_r_t, p5_r_t, success_rate_t])


if __name__ == '__main__':
    main()
