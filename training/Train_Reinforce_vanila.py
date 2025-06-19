import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import csv
import torch
from timeit import default_timer as timer
from env.custom_hopper import *
from agents.agent_reinforce_normal import Agent, Policy
import wandb

device = "cuda"

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

def main():
    config = {
        "policy_type": "MlpPolicy",
        "total_timesteps": 100000,
        "env_id_source": "CustomHopper-udr-v0",
        "env_id_target" : "CustomHopper-target-v0",
        "test_episodes": 50,
        "success_threshold": 1000
    }

    run = wandb.init(
        project="reinforce_baseline_100K_UDR_saghal_1",
        config=config,
        sync_tensorboard=True
    )
    wandb.run.name = "Reinforce_Baseline_Run"
    wandb.run.save()

    env = gym.make(config["env_id_source"])
    env_target = gym.make(config["env_id_target"])

    observation_space_dim = env.observation_space.shape[-1]
    action_space_dim = env.action_space.shape[-1]

    policy = Policy(observation_space_dim, action_space_dim)
    agent = Agent(policy, device= device)

    total_rewards = []
    train_reward = 0
    state = env.reset()
    total_timesteps = 0
    reached_1000 = False
    steps_to_1000 = None
    start = timer()

    # CSV Logging
    training_csv = open("Logs/baseline/training_baseline_100K_UDR_log.csv", "w", newline="")
    train_writer = csv.writer(training_csv)
    train_writer.writerow(["timestep", "mean_reward", "std_reward", "steps_to_1000_return"])

    for total_timesteps in range(config["total_timesteps"]):
        action, action_probabilities = agent.get_action(state)
        previous_state = state
        state, reward, done, _ = env.step(action.detach().cpu().numpy())

        agent.store_outcome(previous_state, state, action_probabilities, reward, done)
        print('Training episode:', total_timesteps)

        train_reward += reward
        total_timesteps += 1

        if done:
            agent.update_policy()
            total_rewards.append(train_reward)

            if train_reward >= config["success_threshold"] and not reached_1000:
                reached_1000 = True
                steps_to_1000 = total_timesteps

            mean_reward = np.mean(total_rewards)
            std_reward = np.std(total_rewards)

            wandb.log({
                "mean_reward": mean_reward,
                "std_reward": std_reward,
                "timestep": total_timesteps + 1
            })
            train_writer.writerow([total_timesteps + 1, mean_reward, std_reward, steps_to_1000 or ""])

            state = env.reset()
            train_reward = 0

    training_csv.close()
    print("Training completed.")
    end = timer()

    if steps_to_1000:
        wandb.log({"steps_to_1000_return": steps_to_1000})
        print(f"Reached return ≥ {config['success_threshold']} at timestep {steps_to_1000}")
    else:
        print(f"Return ≥ {config['success_threshold']} was never reached.")

    # -------------------- Testing Source Env --------------------
    mean_r, std_r, p5_r, success_rate, _ = evaluate_agent_on_env(env, agent, config["test_episodes"], config["success_threshold"])
    wandb.log({
        "test_source_mean_reward": mean_r,
        "test_source_std_reward": std_r,
        "test_source_5th_percentile": p5_r,
        "test_source_success_rate": success_rate
    })

    # -------------------- Testing Target Env --------------------
    mean_rt, std_rt, p5_rt, success_rate_t, _ = evaluate_agent_on_env(env_target, agent, config["test_episodes"], config["success_threshold"])
    wandb.log({
        "test_target_mean_reward": mean_rt,
        "test_target_std_reward": std_rt,
        "test_target_5th_percentile": p5_rt,
        "test_target_success_rate": success_rate_t
    })

    # -------------------- AUC under robustness curve --------------------
    levels = [f"CustomHopper-sudr-{i}-v0" for i in range(5)]
    returns_per_level = []

    for level_id in levels:
        try:
            test_env = gym.make(level_id)
        except:
            print(f"Skipping {level_id}, not registered.")
            continue

        mean_r_l, _, _, _, _ = evaluate_agent_on_env(test_env, agent, config["test_episodes"], config["success_threshold"])
        returns_per_level.append(mean_r_l)

    auc = np.trapz(returns_per_level, dx=1)
    wandb.log({"AUC_robustness_curve": auc})
    print(f"AUC across levels: {auc:.2f}")

    # Save test results
    with open("Logs/baseline/test_log_baseline.csv", "w", newline="") as test_log:
        test_writer = csv.writer(test_log)
        test_writer.writerow(["env_type", "mean_reward", "std_reward", "5th_percentile", "success_rate"])
        test_writer.writerow(["source", mean_r, std_r, p5_r, success_rate])
        test_writer.writerow(["target", mean_rt, std_rt, p5_rt, success_rate_t])

    torch.save(agent.policy.state_dict(), "Models/model_reinforce_baseline/model_reinforce_baseline_2_100K.mdl")
    print(f"Total training time: {end - start:.2f} seconds")

    run.finish()

if __name__ == '__main__':
    main()
