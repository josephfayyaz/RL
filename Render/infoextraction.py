# compute_metrics_multiple_seeds.py

import os, glob, re
import numpy as np
import pandas as pd
from scipy.integrate import trapz
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration ---
data_dir         = r"D:\rl\RL-master\Logs\csv"
success_threshold= 1000
seeds            = [0, 14, 42]
metrics          = [
    'Mean Return',
    '5th %-ile Return',
    'Success Rate',
    'Steps→1 000',
    'Robustness AUC'
]
savepath         = r".\show\plots\\"
os.makedirs(savepath, exist_ok=True)
# --- Plot style configuration (edit here) ---
sns.set_theme(style="whitegrid", palette=["#4477AA","#4988B0","#2DC6DA"])
FONT_TITLE  = 16
FONT_LABEL  = 14
FONT_TICKS  = 12
BAR_WIDTH   = 0.6
FIGSIZE_GRID    = (18, 10)
FIGSIZE_SINGLE  = (8, 5)
DPI              = 300

# --- Helper to auto-detect columns ---
def detect_columns(df, for_time=False):
    cols = df.columns.tolist()
    if for_time:
        xcol = next((c for c in cols if re.search(r'timestep|step', c, re.I)), None)
    else:
        xcol = next((c for c in cols if re.search(r'episode|epis|iter|^index$', c, re.I)), None)
    ycol = next((c for c in cols if re.search(r'return|reward', c, re.I)), None)
    return (xcol or cols[0], ycol or cols[1 if len(cols)>1 else 0])

# --- Gather metrics for each (algo,seed) ---
all_rows = []
for algo in sorted(os.listdir(data_dir)):
    path = os.path.join(data_dir, algo)
    if not os.path.isdir(path): continue
    files = glob.glob(os.path.join(path, "*.csv"))
    for seed in seeds:
        tfile = next((f for f in files if f"seed_{seed}" in f and "learning_curve" in f), None)
        efile = next((f for f in files if f"seed_{seed}" in f and "1000000" not in f), None)
        if not tfile or not efile:
            continue

        df_t = pd.read_csv(tfile)
        tc, rc = detect_columns(df_t, for_time=True)
        ts, mr = df_t[tc].values, df_t[rc].values
        auc = trapz(mr, ts)
        step_thr = np.nan
        if np.any(mr >= success_threshold):
            step_thr = ts[mr >= success_threshold][0]

        df_e = pd.read_csv(efile)
        ec, rr = detect_columns(df_e, for_time=False)
        eps, rets = df_e[ec].values, df_e[rr].values
        mean_r = rets.mean()
        p5_r   = np.percentile(rets, 5)
        sr     = (rets >= success_threshold).mean()

        all_rows.append({
            'Algorithm': algo,
            'Seed':       seed,
            'Mean Return':      mean_r,
            '5th %-ile Return': p5_r,
            'Success Rate':     sr,
            'Steps→1 000':      step_thr,
            'Robustness AUC':   auc
        })

# --- Build DataFrame ---
df_raw     = pd.DataFrame(all_rows).set_index(['Algorithm','Seed'])
metric_dfs = {m: df_raw[m].unstack('Seed') for m in metrics}

# --- 1) Combined 2x3 grid using seaborn ---
fig, axes = plt.subplots(2, 3, figsize=FIGSIZE_GRID)
axes = axes.flatten()
x = np.arange(len(metric_dfs[metrics[0]].index))
algos = metric_dfs[metrics[0]].index

for ax, metric in zip(axes, metrics):
    mdf = metric_dfs[metric]
    sns.barplot(
        data=mdf.reset_index().melt(id_vars='Algorithm', var_name='Seed', value_name=metric),
        x='Algorithm',
        y=metric,
        hue='Seed',
        ax=ax,
        palette=None,
        dodge=True
    )
    ax.set_title(metric, fontsize=FONT_TITLE)
    ax.set_xlabel('')
    ax.set_ylabel(metric, fontsize=FONT_LABEL)
    ax.tick_params(axis='x', rotation=30, labelsize=FONT_TICKS)
    ax.tick_params(axis='y', labelsize=FONT_TICKS)
    # annotate bars

axes[0].legend(title='Seed', fontsize=FONT_TICKS, title_fontsize=FONT_LABEL, loc='upper right')
plt.tight_layout(pad=3)
plt.savefig(savepath + 'all_metrics_grid_seaborn.png', dpi=DPI)


# --- 2) Individual metric plots ---
for metric in metrics:
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
    mdf = metric_dfs[metric]
    sns.barplot(
        data=mdf.reset_index().melt(id_vars='Algorithm', var_name='Seed', value_name=metric),
        x='Algorithm',
        y=metric,
        hue='Seed',
        ax=ax,
        palette=None,
        dodge=True
    )
    ax.set_title(metric, fontsize=FONT_TITLE)
    ax.set_xlabel('')
    ax.set_ylabel(metric, fontsize=FONT_LABEL)
    ax.tick_params(axis='x', rotation=30, labelsize=FONT_TICKS)
    ax.tick_params(axis='y', labelsize=FONT_TICKS)
    ax.legend(title='Seed', fontsize=FONT_TICKS, title_fontsize=FONT_LABEL)
    plt.tight_layout(pad=3)
    plt.savefig(f"{savepath}{metric.replace(' ','_')}_seaborn.png", dpi=DPI)
    

# --- 3) Average Across Seeds ---
# Compute and plot average metric per algorithm
avg_tables = {m: table.mean(axis=1) for m, table in metric_dfs.items()}
for metric, series in avg_tables.items():
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
    sns.barplot(
        x=series.index,
        y=series.values,
        ax=ax,
        palette=["#4477AA"]
    )
    ax.set_title(f"Average {metric} (across seeds)", fontsize=FONT_TITLE)
    ax.set_ylabel(metric, fontsize=FONT_LABEL)
    ax.set_xlabel('Algorithm', fontsize=FONT_LABEL)
    ax.tick_params(axis='x', rotation=30, labelsize=FONT_TICKS)
    ax.tick_params(axis='y', labelsize=FONT_TICKS)
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', padding=2, fontsize=FONT_TICKS)

    plt.tight_layout()
    avg_file = f"avg_{metric.replace(' ', '_')}.png"
    plt.savefig(os.path.join(savepath, avg_file), dpi=DPI)
    plt.close()



# --- 4) Plot robustness curves (return‐threshold vs success rate) ---
import numpy as np

# First, re‐load all episode returns into a dict:
# returns_data[algorithm][seed] = 1D numpy array of episode returns
returns_data = {}
for algo in sorted(os.listdir(data_dir)):
    algo_path = os.path.join(data_dir, algo)
    if not os.path.isdir(algo_path):
        continue
    returns_data[algo] = {}
    files = glob.glob(os.path.join(algo_path, "*.csv"))
    for seed in seeds:
        epi_file = next((f for f in files
                         if f"seed_{seed}" in f and "1000000" not in f), None)
        if epi_file is None:
            continue
        df_ep = pd.read_csv(epi_file)
        _, ret_col = detect_columns(df_ep, for_time=False)
        returns_data[algo][seed] = df_ep[ret_col].values

# Build a common grid of thresholds
all_returns = np.hstack([rets for algo in returns_data for rets in returns_data[algo].values()])
thr_min, thr_max = np.min(all_returns), np.max(all_returns)
thresholds = np.linspace(thr_min, thr_max, 200)

# 4a) One curve per algorithm (averaging seeds)
for algo, seed_dict in returns_data.items():
    # compute mean success‐rate across seeds
    succ_rates = []
    for τ in thresholds:
        # for each seed, compute fraction of episodes ≥ τ, then average over seeds
        per_seed = [np.mean(returns_data[algo][s] >= τ) for s in seed_dict]
        succ_rates.append(np.mean(per_seed))

    plt.figure(figsize=(6,4))
    plt.plot(thresholds, succ_rates, lw=2)
    plt.title(f"Robustness Curve (averaged) — {algo}", fontsize=FONT_TITLE)
    plt.xlabel("Return threshold", fontsize=FONT_LABEL)
    plt.ylabel("fraction of runs exceeding the threshold", fontsize=FONT_LABEL)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(savepath, f"robustness_curve_{algo}.png"), dpi=DPI)
    plt.close()

# 4b) (Optionally) One curve per seed, per algorithm
for algo, seed_dict in returns_data.items():
    plt.figure(figsize=(6,4))
    for seed in seed_dict:
        succ_rates = [np.mean(returns_data[algo][seed] >= τ) for τ in thresholds]
        plt.plot(thresholds, succ_rates, label=f"seed {seed}", lw=1.5)
    plt.title(f"Robustness Curves — {algo}", fontsize=FONT_TITLE)
    plt.xlabel("Return threshold", fontsize=FONT_LABEL)
    plt.ylabel("fraction of runs exceeding the threshold", fontsize=FONT_LABEL)
    plt.legend(fontsize=FONT_TICKS)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(savepath, f"robustness_curves_{algo}_by_seed.png"), dpi=DPI)
    plt.close()