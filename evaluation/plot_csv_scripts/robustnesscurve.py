import os, sys, glob
import numpy as np
import matplotlib.pyplot as plt
import gym
from stable_baselines3 import PPO

# ---------- USER CONFIG -------------------------------------------------
LOG_ROOT        = "/Users/youseffayyaz/Documents/GitHub/RL/Logs/PPO_episode_rewards/"   # <— change once
ALG_NAME        = "PPO_CDR_ES"                     # sub-folder name
SEEDS           = [0, 14, 42]                      # which seeds exist
N_EPISODES      = 30
EPS_GRID        = np.linspace(0.0, 0.5, 15)
SAVE_DIR        = "/Users/youseffayyaz/Documents/GitHub/RL/render/plots"
os.makedirs(SAVE_DIR, exist_ok=True)
# IEEE-safe colours
COL_ALGO        = {"PPO_CDR_ES": "#4E79A7", "PPO_UDR": "#F28E2B",
                   "PPO_CDR": "#59A14F"}
# ------------------------------------------------------------------------

class ObsNoiseWrapper(gym.ObservationWrapper):
    def __init__(self, env, eps):
        super().__init__(env); self.eps = eps
    def observation(self, obs):
        return obs + self.np_random.uniform(-self.eps, self.eps, size=obs.shape)

def make_noisy_env(eps):
    base = gym.make("CustomHopper-target-v0")
    return ObsNoiseWrapper(base, eps)

def evaluate(model, eps, n_epi):
    env = make_noisy_env(eps); rets = []
    for _ in range(n_epi):
        done, obs, tot = False, env.reset(), 0.0
        while not done:
            act, _ = model.predict(obs, deterministic=True)
            obs, r, done, _ = env.step(act); tot += r
        rets.append(tot)
    env.close()
    return np.mean(rets)

# ---------- load all seeds ----------------------------------------------
seed_curves = []
for seed in SEEDS:
    mpath = glob.glob(os.path.join(LOG_ROOT, ALG_NAME,
                                   f"*seed_{seed}*.zip"))
    if not mpath:
        print(f"[warn] model for seed {seed} not found"); continue
    model = PPO.load(mpath[0])
    seed_ret = [evaluate(model, eps, N_EPISODES) for eps in EPS_GRID]
    seed_curves.append(seed_ret)

seed_curves = np.array(seed_curves)        # shape = (n_seed, n_eps)

# ---------- save CSV -----------------------------------------------------
csv_out = os.path.join(
            SAVE_DIR, f"robustness_results_{ALG_NAME}.csv")
np.savetxt(csv_out, np.c_[EPS_GRID, seed_curves.T],
           delimiter=",",
           header="eps," + ",".join(f"seed{ s}" for s in SEEDS),
           comments='')
print(f"saved CSV -> {csv_out}")

# ---------- averaged plot (main paper) -----------------------------------
plt.figure(figsize=(6,4))
mean, std = seed_curves.mean(0), seed_curves.std(0)
plt.plot(EPS_GRID, mean, lw=2, color=COL_ALGO.get(ALG_NAME, "#333"))
plt.fill_between(EPS_GRID, mean-std, mean+std, alpha=0.25,
                 color=COL_ALGO.get(ALG_NAME, "#333"))
plt.xlabel(r"Perturbation noise size $\varepsilon$")
plt.ylabel(r"Avg. return $J(\varepsilon)$")
plt.title(f"Robustness Curve — {ALG_NAME} (n={len(SEEDS)})")
plt.tight_layout()
plt.grid(alpha=.4)
plt.savefig(os.path.join(SAVE_DIR, f"robustness_{ALG_NAME}.png"),
            dpi=300)

# ---------- per-seed appendix plot ---------------------------------------
plt.figure(figsize=(6,4))
for s, curve in zip(SEEDS, seed_curves):
    plt.plot(EPS_GRID, curve, label=f"seed {s}", lw=1.3)
plt.legend(); plt.title(f"Robustness curves by seed — {ALG_NAME}")
plt.xlabel(r"$\varepsilon$"); plt.ylabel("Return"); plt.grid(alpha=.4)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR,
            f"robustness_{ALG_NAME}_by_seed.png"), dpi=300)
