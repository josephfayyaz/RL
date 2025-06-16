"""Sample script for training a control policy on the Hopper environment
   using stable-baselines3 (https://stable-baselines3.readthedocs.io/en/master/)

    Read the stable-baselines3 documentation and implement a training
    pipeline with an RL algorithm of your choice between PPO and SAC.
"""
import os,sys
import ctypes
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
mujoco_path = "C:/.mujoco/mujoco210/bin"  # manually append library for running on windoes
os.environ["PATH"] += ";" + mujoco_path
ctypes.CDLL(os.path.join(mujoco_path, "mujoco210.dll"))
import random, numpy as np
import gym
from datetime import datetime
import csv
from env.custom_hopper import *
import argparse
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
import json
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-episodes', default=2000, type=int, help='Number of training episodes')
    parser.add_argument('--print-every', default=100, type=int, help='Print info every <> episodes')
    parser.add_argument('--device', default='cuda', type=str, help='network device [cpu, cuda]')
    parser.add_argument('--algorithm' , default='PPO' ,type=str, choices=['PPO'], help='Algorithm to use for training')
    parser.add_argument('--CDR', default=False, type=bool, choices=[True,False], help='toggle to use Curriculum Domain Randomization')    
    parser.add_argument('--Entropy_Scheduling', default=False, type=bool, choices=[True,False], help='toggle to use Entropy Scheduling')
    parser.add_argument('--seed', default=42, type=int, help='Random seed for reproducibility') 
    parser.add_argument('--n_envs', default=8, type=int, help='Number of parallel environments for training') 

    return parser.parse_args()


args = parse_args()
# ---------- Config ---------- #

Total_timesteps = args.n_episodes # Total timesteps for training
device = args.device
if device == 'cuda':    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'   
print(f'training on {torch.cuda.get_device_name(torch.cuda.current_device()) }' if torch.cuda.is_available() else 'training on cpu')
   

HP_PATH = "./models/PPO/best_hyperparameters.json"
ENV_ID = 'CustomHopper-cdr-v0' if args.CDR else 'CustomHopper-source-v0'
EVAL_ENV = 'CustomHopper-target-v0'  # Change to your specific environment
SAVE_PATH = "./models/PPO"
LOG_PATH     = './logs/PPO/'
os.makedirs("logs/csv", exist_ok=True)

timestamp    = datetime.now().strftime("%Y%m%d_%H%M%S")
csv_filename = os.path.join(LOG_PATH, f"{timestamp}_ppo_{Total_timesteps}.csv")
USE_CDR = args.CDR  # Set to True if you want to use Curriculum Domain Randomization
USE_entropy_scheduler = args.Entropy_Scheduling  # Set to True if you want to use Entropy Scheduler
SEED = args.seed  # Random seed for reproducibility
random.seed(SEED)
np.random.seed(SEED)




class CSVLoggerCallback(BaseCallback):

    #Logs (episode, reward) to a CSV file at each episode end.
    
    def __init__(self, csv_path: str, verbose: int = 0):
        super().__init__(verbose)
        self.csv_path = csv_path
        self.episode  = 0
        
        # write header
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['episode', 'reward'])
        print(f"CSV logger initialized at {self.csv_path}")

    def _on_step(self) -> bool:
        for info in self.locals.get('infos', []):
            ep_info = info.get('episode')
            if ep_info is not None:
                self.episode += 1
                reward = ep_info['r']
                with open(self.csv_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([self.episode, f"{reward:.6f}"])
        return True

# 1) Curriculum Domain Randomization Wrapper
class CDRWrapper(gym.Wrapper):
    def __init__(self, env, total_timesteps):
        super().__init__(env)
        self.total_timesteps = total_timesteps
        self.elapsed = 0

    def reset(self, **kwargs):
        level = min(1.0, self.elapsed / self.total_timesteps)
        try:
            params = self.env.sample_parameters(level)       # requires modifying your CustomHopper.sample_parameters signature
            self.env.set_parameters(params)
        except TypeError:
            self.env.set_random_parameters()                # fallback
        return super().reset(**kwargs)

    def step(self, action):
        self.elapsed += 1
        return super().step(action)

# ——————————————————————————————————————————


# 2) Entropy‐coefficient scheduler callback
class EntropyScheduler(BaseCallback):
    def __init__(self, start_coef, end_coef, total_timesteps, verbose=0):
        super().__init__(verbose)
        self.start = start_coef
        self.end   = end_coef
        self.total = total_timesteps

    def _on_step(self) -> bool:
        frac = min(1.0, self.num_timesteps / self.total)
        self.model.ent_coef = self.start + frac * (self.end - self.start)
        return True
# ——————————————————————————————————————————
def load_best_hyperparameters(path):
    with open(path, 'r') as f:
        return json.load(f)
def make_env(env_id, seed, rank):
    def _init():
        env = gym.make(env_id)
        env.seed(seed + rank)
        env.action_space.seed(seed + rank)
        env.observation_space.seed(seed + rank)
        if args.CDR:
            env = CDRWrapper(env, total_timesteps=Total_timesteps)
        return Monitor(env)
    return _init
def train_agent(algo, env_id, eval_env_id, USE_entropy_scheduler, total_timesteps, save_path, log_path):
    best_hp = load_best_hyperparameters(HP_PATH)

    env = SubprocVecEnv([make_env(env_id, SEED, i) for i in range(args.n_envs)])
   




    csv_logger = CSVLoggerCallback(csv_path=csv_filename)

    model = PPO('MlpPolicy', env, device=device,
                    learning_rate=best_hp["learning_rate"],
                    n_steps=best_hp["n_steps"],
                    gamma=best_hp["gamma"],
                    batch_size=best_hp["batch_size"],
                    n_epochs=best_hp["n_epochs"],
                    gae_lambda=best_hp["gae_lambda"],
                    seed=SEED,
                    verbose=1,
                    tensorboard_log=log_path,
                    )
   

    checkpoint_callback = CheckpointCallback(save_freq=9000, save_path=save_path,
                                             name_prefix='rl_model')




    eval_env = DummyVecEnv([lambda: Monitor(gym.make(eval_env_id))])
    eval_env.seed(SEED)


    eval_callback = EvalCallback(eval_env, best_model_save_path=save_path,
                                 log_path=log_path, eval_freq=10000,
                                 deterministic=True, render=False)
    # 1) prepare your core callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=1000, save_path=save_path, name_prefix='rl_model'
    )
    eval_callback = EvalCallback(
        eval_env, best_model_save_path=save_path,
        log_path=log_path, eval_freq=5000,
        deterministic=True, render=False
    )

    # 2) assemble into one list
    callbacks = [checkpoint_callback, eval_callback, csv_logger]




    if USE_entropy_scheduler:    
        entropy_cb = EntropyScheduler(
        start_coef=0.01,
        end_coef=1e-4,
        total_timesteps=total_timesteps
        )
        callbacks.append(entropy_cb)
    # 3) pass that list into learn()
    model.learn(total_timesteps=total_timesteps, callback=callbacks)
    model.save(os.path.join(save_path, f"{algo}_CDR_ES_model_{total_timesteps}M"))
    print(f"Model saved to {save_path}")
    
if __name__ == "__main__":

    os.makedirs(SAVE_PATH, exist_ok=True)
    os.makedirs(LOG_PATH, exist_ok=True)

    train_agent(args.algorithm, ENV_ID,EVAL_ENV,args.Entropy_Scheduling, Total_timesteps, SAVE_PATH, LOG_PATH)