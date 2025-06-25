import re, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path

BASE   = Path("/content/")
T, R   = "timesteps", "mean_reward"

COLORS = {
    "source→source": "#1b9e77",
    "target→target": "#7570b3",
    "source→target": "#d95f02",
}

# --- helper ---------------------------------------------------------------
seed_re = re.compile(r"_seed_(\d+)", re.IGNORECASE)
cross_p = re.compile(r"learning_curve_ppo_seed_\d+\.csv", re.IGNORECASE)
def cfg_of(fname: str) -> str:
    f = fname.lower().replace(" ", "")
    if "srctrain" in f and "tgteval" in f:  return "source→target"
    if "_5m" in f:                          return "source→target"
    if cross_p.match(f):                    return "source→target"
    if "domain_source" in f:                return "source→source"
    if "domain_target" in f:                return "target→target"
    raise ValueError(f"unhandled name {fname}")

def load(fp: Path) -> pd.DataFrame:
    cfg  = cfg_of(fp.name)
    seed = int(seed_re.search(fp.name).group(1))
    df   = pd.read_csv(fp).rename(columns={T: "t", R: "ret"})
    if cfg == "source→target":
        df = df[df["t"] <= 1_000_000]
    df["cfg"], df["seed"] = cfg, seed
    return df[["cfg", "seed", "t", "ret"]]

# --- load & aggregate ------------------------------------------------------
fps = (list(BASE.glob("learning_curve_PPO*.csv")) +
       list(BASE.glob("ppo_srcTrain_tgtEval_seed_*.csv")))
curves = pd.concat([load(fp) for fp in fps], ignore_index=True)

agg = (curves.groupby(["cfg", "t"])
              .agg(mean=("ret", "mean"), std=("ret", "std"))
              .reset_index()
              .sort_values("t"))          # global sort once

# --- plot ------------------------------------------------------------------
plt.figure(figsize=(8, 5))
for cfg, style in [("source→source", "-"), ("target→target", "-"), ("source→target", "--")]:
    sub = agg[agg["cfg"] == cfg]
    plt.plot(sub["t"], sub["mean"], style, color=COLORS[cfg], linewidth=2,
             label=cfg)
    plt.fill_between(sub["t"],
                     sub["mean"] - sub["std"].fillna(0),
                     sub["mean"] + sub["std"].fillna(0),
                     color=COLORS[cfg], alpha=0.2)

plt.title("PPO (vanilla) – learning curves (seeds 0, 14, 42, first 1 M steps)")
plt.xlabel("Environment steps");  plt.ylabel("Episodic reward")
plt.legend(frameon=False);  plt.tight_layout();  plt.show()
