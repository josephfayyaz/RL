import os,sys
import ctypes
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# __file__mujoco_path = "C:/.mujoco/mujoco210/bin"  # manually append library for running on windoes
# os.environ["PATH"] += ";" + mujoco_path
# ctypes.CDLL(os.path.join(mujoco_path, "mujoco210.dll"))
import gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO  # or your RL library of choice
import time
import random, numpy as np
import gym
from datetime import datetime
import csv
from env.custom_hopper import *

# 1. Load your saved model
model = PPO.load("/home/joseph/python-proj/udr_ES/Logs/csv/PPO_ES/PPO_Domain_source_ES_True_seed_14_CustomHopper_source_v0_CustomHopper.zip")
algorithm = "PPO_ES"
number_of_episodes=30
# 2. Wrap the environment to inject ε-bounded noise
class ObsNoiseWrapper(gym.ObservationWrapper):
    def __init__(self, env, eps):
        super().__init__(env)
        self.eps = eps
    def observation(self, obs):
        noise = np.random.uniform(-self.eps, self.eps, size=obs.shape)
        return obs + noise
import csv

# 4b. Save results to CSV

def make_noisy_env(eps):
    base_env = gym.make("CustomHopper-target-v0")
    return ObsNoiseWrapper(base_env, eps)

# 3. Evaluation function under a given ε
def evaluate(model, eps, n_episodes):
    env = make_noisy_env(eps)
    returns = []
    for _ in range(n_episodes):
        obs = env.reset()
        done = False
        total_r = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, done, _ = env.step(action)
            total_r += r
        returns.append(total_r)
    env.close()
    return np.mean(returns)

# 4. Sweep ε and record J(ε)
epsilons = np.linspace(0.0, 0.5, 15)   # e.g. from no noise to ±0.5
mean_returns = []
for eps in epsilons:
    jr = evaluate(model, eps, n_episodes=number_of_episodes)
    mean_returns.append(jr)
    print(f"ε={eps:.2f} → return={jr:.1f}")

csv_path = f"/home/joseph/python-proj/udr_ES/Render/csv_robustness/robustness_results_{algorithm}.csv"
with open(csv_path, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["epsilon", "mean_return"])
    for eps, ret in zip(epsilons, mean_returns):
        writer.writerow([eps, ret])
print(f"Saved robustness results to {csv_path}")

# 5. Plot the robustness curve
plt.figure(figsize=(6,4))
plt.plot(epsilons, mean_returns, marker='o')
plt.xlabel("Perturbation noise size ε")
plt.ylabel("Avg. return J(ε)")
plt.title(f"Robustness Curve for {algorithm} (mean over {number_of_episodes} episodes)")
plt.grid(True)
plt.tight_layout()
plt.savefig(f"/home/joseph/python-proj/udr_ES/Render/plots/noise_robustness_curve_{algorithm}.png", dpi=300)
# plt.show()
