"""Sample script for training a control policy on the Hopper environment
   using stable-baselines3 (https://stable-baselines3.readthedocs.io/en/master/)

    Read the stable-baselines3 documentation and implement a training
    pipeline with an RL algorithm of your choice between PPO and SAC.
"""
import os,sys
import ctypes
import random, numpy as np
import gym
from datetime import datetime
import csv
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from env.custom_hopper import *
import argparse
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback , BaseCallback
from stable_baselines3.common.monitor import Monitor
import json
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
import multiprocessing
import shutil
from stable_baselines3.common.evaluation import evaluate_policy





def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-episodes', default=5000000, type=int, help='Number of training episodes')
    parser.add_argument('--print-every', default=100, type=int, help='Print info every <> episodes')
    parser.add_argument('--device', default='cuda', type=str, help='network device [cpu, cuda]')
    parser.add_argument('--algorithm' , default='PPO' ,type=str, choices=['PPO'], help='Algorithm to use for training')
    parser.add_argument("--Domain",default="cdr",choices=["source","cdr","udr"],type=str,help="Which mass‐randomization regime to use")
    parser.add_argument('--Entropy_Scheduling', default=False , type=bool, choices=[True,False], help='toggle to use Entropy Scheduling')
    parser.add_argument('--seed', default=[0,14,42], type=int,nargs="+", help='Random seed for reproducibility')
    parser.add_argument('--n_envs', default=8, type=int, help='Number of parallel environments for training')

    return parser.parse_args()


args = parse_args()
# ---------- Config ---------- #

Total_timesteps = args.n_episodes # Total timesteps for training
device = args.device
if device == 'cuda':
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f'training on {torch.cuda.get_device_name(torch.cuda.current_device()) }' if torch.cuda.is_available() else 'training on cpu')


HP_PATH = "../Models/PPO/best_hyperparameters.json"
ENV_ID = f'CustomHopper-{args.Domain}-v0'
EVAL_ENV = 'CustomHopper-target-v0'  # Change to your specific environment
SAVE_PATH = "../Models/PPO"
LOG_PATH     = '../Logs/PPO/'
# os.makedirs("logs/csv", exist_ok=True)

timestamp    = datetime.now().strftime("%Y%m%d_%H%M%S")
# Set to True if you want to use Curriculum Domain Randomization
USE_entropy_scheduler = args.Entropy_Scheduling  # Set to True if you want to use Entropy Scheduler
seeds = args.seed  # Random seed for reproducibility



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

# ——————————————————————————————————————————
def load_best_hyperparameters(path):
    with open(path, 'r') as f:
        return json.load(f)
def make_env(env_id, seed, rank):
    def _init():
        env = CustomHopper(
            domain=args.Domain,
            total_timesteps=Total_timesteps
        )
        env.seed(seed + rank)
        env.action_space.seed(seed + rank)
        env.observation_space.seed(seed + rank)
        return Monitor(env)
    return _init

# ---------- average return every N timesteps ( learning curves ) ---------- #
class LearningCurveCallback(BaseCallback):
    """
    Logs (timesteps, mean_reward) to a CSV file every `eval_interval` steps.
    """

    def __init__(self, eval_env, csv_path, eval_interval=10000, n_eval_episodes=5, verbose=0):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.csv_path = csv_path
        self.eval_interval = eval_interval
        self.n_eval_episodes = n_eval_episodes

        # write header
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timesteps', 'mean_reward'])

        if self.verbose > 0:
            print(f"EvalLogger initialized at {self.csv_path}")

    def _on_step(self):
        if self.num_timesteps % self.eval_interval == 0:
            mean_reward, _ = evaluate_policy(
                self.model, self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                deterministic=True
            )

            with open(self.csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([self.num_timesteps, f"{mean_reward:.6f}"])

            if self.verbose > 0:
                print(f"[EvalLogger] Step {self.num_timesteps} → MeanReward: {mean_reward:.2f}")
        return True


class SaveAllBestCallback(EvalCallback):
    def __init__(
        self,
        eval_env,
        best_model_save_path: str,
        log_path: str,
        eval_freq: int,
        prefix: str = "best_model",     # your custom prefix
        n_eval_episodes: int = 5,
        deterministic: bool = True,
        render: bool = False,
        verbose: int = 0,
    ):
        # Call base class WITHOUT name_prefix
        super().__init__(
            eval_env=eval_env,
            best_model_save_path=best_model_save_path,
            log_path=log_path,
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            deterministic=deterministic,
            render=render,
            verbose=verbose
        )
        self.prefix = prefix

    def _on_step(self) -> bool:
        prev_best = getattr(self, "best_mean_reward", float("-inf"))
        cont = super()._on_step()
        if hasattr(self, "best_mean_reward") and self.best_mean_reward > prev_best:
            # copy best_model.zip → prefix_<timesteps>_steps.zip
            src = os.path.join(self.best_model_save_path, "best_model.zip")
            dst = os.path.join(
                self.best_model_save_path,
                f"{self.prefix}_{self.num_timesteps}_steps.zip"
            )
            shutil.copyfile(src, dst)
        return cont


def train_agent(algo, env_id, eval_env_id, USE_entropy_scheduler, total_timesteps, save_path, log_path,seed,csv_filename):

    best_hp = load_best_hyperparameters(HP_PATH)

    env = SubprocVecEnv([make_env(env_id, seed, i) for i in range(args.n_envs)])  #8 envs

    csv_logger = CSVLoggerCallback(csv_path=csv_filename)

    model = PPO('MlpPolicy', env, device=device,
                    learning_rate=best_hp["learning_rate"],
                    n_steps=best_hp["n_steps"],
                    gamma=best_hp["gamma"],
                    batch_size=best_hp["batch_size"],
                    n_epochs=best_hp["n_epochs"],
                    gae_lambda=best_hp["gae_lambda"],
                    seed=seed,
                    verbose=1,
                    tensorboard_log=log_path,
                    )

    eval_env = DummyVecEnv([lambda: Monitor(gym.make(eval_env_id))])
    eval_env.seed(seed)

    checkpoint_callback = CheckpointCallback(
        save_freq= 10**9 // args.n_envs,
        save_path=save_path,
        name_prefix='rl_model'
        )

    learning_curve_cb = LearningCurveCallback(
        eval_env=eval_env,
        csv_path=f"../Logs/learning_curve_{algo}_Domain _{args.Domain}_ES_{args.Entropy_Scheduling}_seed_{seed}_5M.csv",
        eval_interval=5000,
        n_eval_episodes=5,
        verbose=1
        )

    eval_callback = SaveAllBestCallback(
        eval_env=eval_env,
        best_model_save_path=SAVE_PATH,
        log_path=LOG_PATH,
        eval_freq= 10**9 // args.n_envs,   # or whatever you choose
        prefix=f"EVAL_BEST_{algo}_Domain _{args.Domain}_ES_{args.Entropy_Scheduling}_seed_{seed}_5M",
        n_eval_episodes=10,
        deterministic=True,
        render=False,
        verbose=1
    )


    # 2) assemble into one list
    callbacks = [checkpoint_callback, eval_callback, csv_logger,learning_curve_cb]

    if USE_entropy_scheduler:
        callbacks.append(
            EntropyScheduler(
                start_coef=0.01,
                end_coef=1e-4,
                total_timesteps=Total_timesteps
            )
        )
    # 3) pass that list into learn()
    model.learn(total_timesteps=total_timesteps, callback=callbacks)
    modelpath=os.path.join(save_path, f"{algo}_Domain_{args.Domain}_ES_{args.Entropy_Scheduling}_seed_{seed}_({env_id}_{eval_env_id})_{Total_timesteps}")
    model.save(modelpath)
    print(f"Model saved to {modelpath}")
def main():
    os.makedirs(SAVE_PATH, exist_ok=True)
    os.makedirs(LOG_PATH, exist_ok=True)

    for run_seed in seeds:
        print(f"=== Running experiment with seed={run_seed} ===")
    # set SEED for this iteration
        random.seed(run_seed)
        np.random.seed(run_seed)
        torch.manual_seed(run_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(run_seed)
    # regenerate timestamp or include seed

        envname=args.Domain
        evalenvname="source"

        csv_filename = os.path.join(LOG_PATH, f"{args.algorithm}_Domain_{args.Domain}_ES_{args.Entropy_Scheduling}_seed_{run_seed}_{Total_timesteps}({envname},{evalenvname}).csv")

        train_agent(args.algorithm, ENV_ID,EVAL_ENV,args.Entropy_Scheduling, Total_timesteps, SAVE_PATH, LOG_PATH,run_seed,csv_filename)
if __name__ == "__main__":

    multiprocessing.freeze_support()
    multiprocessing.set_start_method('spawn', force=True)
    main()

