#!/usr/bin/env python3
"""
infoextraction.py   — extracts the five key metrics from all per-seed CSVs
and generates publication-ready bar grids (IEEE colour-blind palette).
"""

import os
import glob
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# =============== configuration =================================================
SUCCESS_THRESHOLD = 1_000          # ≥1000  == task success
METRIC_LABELS = [
    "Mean Return",
    "5th %-ile Return",
    "Success Rate",
    "Steps→Success",               # clearer than “Steps→1 000”
    "Robustness AUC"
]

IEEE_COLOURS = sns.color_palette("colorblind", 6)
SEED_PALETTE = sns.color_palette("colorblind")   # auto-adjusts to n seeds
sns.set_theme(style="whitegrid")

# =============== helper functions ==============================================
def load_metric_csv(csv_file: Path) -> pd.DataFrame:
    """Each learning-curve CSV has columns:
       timestep, mean_return, cvar5, success_rate, robustness_auc
    """
    df = pd.read_csv(csv_file)
    return df

def steps_to_success(df: pd.DataFrame) -> int:
    """Return the first timestep where moving-average mean_return ≥ threshold"""
    mv = df["mean_return"].rolling(window=5, min_periods=1).mean()
    reached = df.loc[mv >= SUCCESS_THRESHOLD, "timestep"]
    return int(reached.iloc[0]) if not reached.empty else np.nan

# =============== main workflow =================================================
def main(log_root: str, out_dir: str):
    log_root = Path(log_root)
    out_dir  = Path(out_dir);  out_dir.mkdir(parents=True, exist_ok=True)

    # algorithm folders = immediate children
    alg_dirs = [d for d in log_root.iterdir() if d.is_dir()]
    summary_rows = []

    for alg in sorted(alg_dirs):
        csvs = sorted(alg.glob("learning_curve_*seed_*.csv"))
        if not csvs:
            print(f"[warn] no CSVs in {alg}")
            continue

        per_seed_vals = []

        for csv in csvs:
            seed = int(csv.stem.split("seed_")[-1])
            df   = load_metric_csv(csv)
            per_seed_vals.append({
                "Algorithm"   : alg.name,
                "Seed"        : seed,
                "Mean Return" : df["mean_return"].iloc[-1],
                "5th %-ile Return": df["cvar5"].iloc[-1],
                "Success Rate": df["success_rate"].iloc[-1],
                "Steps→Success": steps_to_success(df),
                "Robustness AUC": df["robustness_auc"].iloc[-1]
            })

        # ---------- per-seed bar-plot (appendix) ------------------------------
        seed_df = pd.DataFrame(per_seed_vals).melt(
            id_vars=["Seed"], var_name="Metric", value_name="Value",
            value_vars=METRIC_LABELS)
        g = sns.catplot(
            data=seed_df, x="Metric", y="Value", hue="Seed",
            kind="bar", palette=SEED_PALETTE, height=4, aspect=1.8)
        g.set_xticklabels(rotation=30, ha="right")
        g.despine(left=True)
        g.fig.suptitle(f"{alg.name}: per-seed metrics", y=1.02, fontsize=10)
        g.savefig(out_dir / f"{alg.name}_per_seed.png", dpi=300)
        plt.close(g.fig)

        # ---------- average & std  -------------------------------------------
        df_alg = pd.DataFrame(per_seed_vals).drop(columns=["Seed"])
        avg = df_alg.groupby("Algorithm").mean().reset_index()
        std = df_alg.groupby("Algorithm").std().reset_index()
        avg["std"] = std[METRIC_LABELS].values.tolist()
        summary_rows.append(avg)

    # ---------- summary table -------------------------------------------------
    summary = pd.concat(summary_rows, ignore_index=True)
    summary.to_csv(out_dir / "metric_summary.csv", index=False)

    # ---------- grid figure ---------------------------------------------------
    melt = summary.melt(id_vars=["Algorithm"], var_name="Metric",
                        value_name="Value")
    g = sns.catplot(
        data=melt, x="Algorithm", y="Value", hue="Algorithm",
        col="Metric", col_wrap=3, sharey=False, height=3.0,
        kind="bar", palette=IEEE_COLOURS, legend=False)

    for ax in g.axes.flatten():
        ax.set_xlabel("")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=25, ha="right")

    g.fig.tight_layout()
    g.fig.subplots_adjust(top=0.90)
    g.fig.suptitle("Five key metrics (mean over seeds ± sd)")
    g.savefig(out_dir / "five_metric_grid.png", dpi=400)
    plt.close(g.fig)

    print(f"✓  figures + CSV saved to {out_dir}")

# ==============================================================================
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--log_root", type=str, required=True,
                   help="root folder that contains one sub-folder per algorithm")
    p.add_argument("--out_dir",  type=str, default="./figures",
                   help="where to save plots / tables")
    args = p.parse_args()
    main(args.log_root, args.out_dir)
