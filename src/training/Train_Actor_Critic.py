# -------------------- Imports -------------------- #
import os
import sys
import csv
import torch
import numpy as np
from timeit import default_timer as timer
import gym

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from agents.agent_ac import Agent_ac, Policy_ac
from env.custom_hopper import *

# -------------------- Device Setup -------------------- #
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Training on {torch.cuda.get_device_name(torch.cuda.current_device())}' if torch.cuda.is_available() else 'Training on CPU')

# -------------------- Configuration -------------------- #
CHECKPOINT_PATH = "../../models/actor_critic/checkpoint/checkpoint.pth"
MODEL_SAVE_DIR = "../../models/actor_critic/"
FINAL_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, "model_actor_critic_source_8_5M.mdl")
LOG_CSV_PATH = "../../Logs/actor_critic/training_source_actor_critic_upgraded_8.csv"
TEST_LOG_PATH = "../../Logs/actor_critic/test_source_log_8.csv"
SAVE_INTERVAL = 200000  # Every 500k timesteps

# -------------------- Helper Functions -------------------- #
def save_checkpoint(path, agent, total_timesteps, episode_number, steps_to_1000, total_rewards):
    torch.save({
        'policy_state_dict': agent.policy.state_dict(),
        'optimizer_state_dict': agent.optimizer.state_dict(),
        'total_timesteps': total_timesteps,
        'episode_number': episode_number,
        'steps_to_1000': steps_to_1000,
        'total_rewards': total_rewards
    }, path)
    print(f"Checkpoint saved at timestep {total_timesteps}")

def load_checkpoint(path, agent):
    checkpoint = torch.load(path)
    agent.policy.load_state_dict(checkpoint['policy_state_dict'])
    agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print("Checkpoint loaded!")
    return checkpoint['total_timesteps'], checkpoint['episode_number'], checkpoint['steps_to_1000'], checkpoint['total_rewards']

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
    return (
        np.mean(returns), np.std(returns),
        np.percentile(returns, 5),
        sum(r >= success_threshold for r in returns) / len(returns),
        returns
    )

# -------------------- Main Training Function -------------------- #
def main():
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(LOG_CSV_PATH), exist_ok=True)

    config = {
        "total_timesteps": 5000000,
        "env_id_source": "CustomHopper-source-v0",
        "env_id_target": "CustomHopper-source-v0",
        "test_episodes": 50,
        "success_threshold": 1000
    }

    env = gym.make(config["env_id_source"])
    env_target = gym.make(config["env_id_target"])
    obs_dim, act_dim = env.observation_space.shape[-1], env.action_space.shape[-1]

    policy = Policy_ac(obs_dim, act_dim)
    agent = Agent_ac(policy, device)

    # Resume training if checkpoint exists
    if os.path.exists(CHECKPOINT_PATH):
        total_timesteps, episode_number, steps_to_1000, total_rewards = load_checkpoint(CHECKPOINT_PATH, agent)
        state = env.reset()
        reached_1000 = steps_to_1000 is not None
    else:
        total_timesteps, episode_number = 0, 1
        steps_to_1000, total_rewards = None, []
        state = env.reset()
        reached_1000 = False

    train_reward = 0
    start = timer()

    with open(LOG_CSV_PATH, "a", newline="") as training_csv:
        train_writer = csv.writer(training_csv)
        if total_timesteps == 0:
            train_writer.writerow(["episode", "timestep", "mean_reward", "std_reward", "steps_to_1000_return", "actor_loss", "critic_loss", "entropy"])

        while total_timesteps < config["total_timesteps"]:
            action, probs = agent.get_action(state)
            prev_state = state
            state, reward, done, _ = env.step(action.detach().cpu().numpy())
            agent.store_outcome(prev_state, state, probs, reward, done)
            train_reward += reward
            total_timesteps += 1

            if done:
                actor_loss, critic_loss, entropy = agent.update_policy()
                total_rewards.append(train_reward)

                if train_reward >= config["success_threshold"] and not reached_1000:
                    reached_1000 = True
                    steps_to_1000 = total_timesteps
                    print(f"🎉 Solved! Return ≥ {config['success_threshold']} at timestep {total_timesteps}")

                mean_reward = np.mean(total_rewards)
                std_reward = np.std(total_rewards)

                train_writer.writerow([
                    episode_number, total_timesteps,
                    mean_reward, std_reward,
                    steps_to_1000 or "",
                    actor_loss, critic_loss, entropy
                ])

                print(f"[{total_timesteps}] Ep {episode_number} | R: {train_reward:.1f}, Mean: {mean_reward:.1f}, LossA: {actor_loss:.3f}, LossC: {critic_loss:.3f}, Entropy: {entropy:.3f}")

                # -------------------- Save Checkpoint -------------------- #
                if total_timesteps % SAVE_INTERVAL == 0:
                    save_checkpoint(CHECKPOINT_PATH, agent, total_timesteps, episode_number, steps_to_1000, total_rewards)

                # Reset for next episode
                state = env.reset()
                train_reward = 0
                episode_number += 1

    # Save final model and evaluate
    end = timer()
    torch.save(agent.policy.state_dict(), FINAL_MODEL_PATH)
    print(f"Final model saved at {FINAL_MODEL_PATH} | Training time: {end - start:.1f} sec")

    mean_r, std_r, p5_r, success_rate, _ = evaluate_agent_on_env(env, agent, config["test_episodes"], config["success_threshold"])
    mean_r_t, std_r_t, p5_r_t, success_rate_t, _ = evaluate_agent_on_env(env_target, agent, config["test_episodes"], config["success_threshold"])

    print(f"Eval Source → Mean: {mean_r:.1f}, STD: {std_r:.1f}, P5: {p5_r:.1f}, Success: {success_rate:.2f}")
    print(f"Eval Target → Mean: {mean_r_t:.1f}, STD: {std_r_t:.1f}, P5: {p5_r_t:.1f}, Success: {success_rate_t:.2f}")

    with open(TEST_LOG_PATH, "w", newline="") as test_log:
        test_writer = csv.writer(test_log)
        test_writer.writerow(["env_type", "mean_reward", "std_reward", "5th_percentile", "success_rate"])
        test_writer.writerow(["source", mean_r, std_r, p5_r, success_rate])
        test_writer.writerow(["target", mean_r_t, std_r_t, p5_r_t, success_rate_t])

# -------------------- Run -------------------- #
if __name__ == '__main__':
    main()
