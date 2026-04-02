import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import argparse, os
from glob import glob

def smooth_rewards(x, y, window=300, sigma=50, num_points=1000):
    sorted_idx = np.argsort(x)
    x, y = x[sorted_idx], y[sorted_idx]
    y_roll = pd.Series(y).rolling(window=window, min_periods=1).mean().values
    y_gauss = gaussian_filter1d(y_roll, sigma=sigma)
    x_uniform = np.linspace(x.min(), x.max(), num_points)
    y_uniform = np.interp(x_uniform, x, y_gauss)
    return x_uniform, y_uniform

def average_seeds(files, window=300, sigma=50, num_points=1000):
    all_curves = []
    for f in files:
        df = pd.read_csv(f)
        if 'timesteps' not in df or 'mean_reward' not in df:
            print(f"⚠️ Skipping bad file: {f}")
            continue
        x, y = smooth_rewards(df['timesteps'].values,
                              df['mean_reward'].values,
                              window, sigma, num_points)
        all_curves.append((x, y))

    if not all_curves:
        return None, None

    # Assume all x are identical due to uniform interpolation
    x_base = all_curves[0][0]
    y_all = np.array([y for _, y in all_curves])
    y_mean = y_all.mean(axis=0)
    return x_base, y_mean

def plot(curves, labels, title, output, y_limit=None):
    plt.figure(figsize=(8, 6), dpi=300)
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 12,
        "axes.labelweight": "bold"
    })

    palette = ['#1b9e77', '#d95f02', '#7570b3', '#e7298a', '#66a61e', '#e6ab02']
    dash_styles = ['-', '--', '-.', ':', (0, (3, 5, 1, 5)), (0, (5, 10))]

    for i, ((x, y), label) in enumerate(zip(curves, labels)):
        plt.plot(x, y,
                 color=palette[i % len(palette)],
                 linestyle=dash_styles[i % len(dash_styles)],
                 linewidth=2.2,
                 label=label)

    plt.suptitle("Learning Curve", fontsize=16, weight='bold', y=1)
    plt.title(title, fontsize=14, weight='bold')
    plt.xlabel("Timesteps", fontsize=12)
    plt.ylabel("Mean Reward", fontsize=12)
    if y_limit:
        plt.ylim(0, y_limit)
    plt.grid(linestyle='--', linewidth=0.5, alpha=0.7)
    plt.legend(fontsize=10, loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False)
    plt.tight_layout()
    plt.savefig(output + '.png', bbox_inches='tight')
    plt.savefig(output + '.pdf', bbox_inches='tight')
    print(f"✅ Saved outputs: {output}.png & {output}.pdf")
    plt.close()

def main(args):
    # Map of original label → variant name used in filename
    label_to_variant = {
        label: label.replace("PPO_", "") if label != "PPO" else "PPO"
        for label in args.labels
    }

    label_to_files = {}
    for label, variant in label_to_variant.items():
        files = []
        for seed in args.seeds:
            if variant == "PPO":
                pattern = f"{args.dir}/**/learning_curve_PPO_seed_{seed}_5M*.csv"
            else:
                pattern = f"{args.dir}/**/learning_curve_PPO_{variant}_seed_{seed}_5M*.csv"
            matched = glob(pattern, recursive=True)
            if matched:
                files.extend(matched)
        if not files:
            print(f"❌ No CSVs found for config: {label}")
        else:
            label_to_files[label] = files

    # Average and smooth
    curves = []
    valid_labels = []
    for label, files in label_to_files.items():
        x, y = average_seeds(files)
        if x is not None:
            curves.append((x, y))
            valid_labels.append(label)

    if not curves:
        print("🚫 No valid data to plot.")
        return

    auc_records = []
    for (x, y), label in zip(curves, valid_labels):
        auc = np.trapz(y, x)
        auc_records.append({"Algorithm": label, "AUC": auc})

    auc_df = pd.DataFrame(auc_records).sort_values(by="AUC", ascending=False)
    print("\n📊 Area Under Learning Curve (AULC):")
    print(auc_df.to_string(index=False))

    output_dir = os.path.join("Logs", "PPO_robustness")
    os.makedirs(output_dir, exist_ok=True)
    auc_csv_path = os.path.join(output_dir, "auc_scores.csv")
    auc_df.to_csv(auc_csv_path, index=False)
    print(f"✅ AUC scores saved to: {auc_csv_path}")

    plot(curves, valid_labels, args.title, args.output, args.y_limit)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dir", required=True, help="Directory containing CSVs")
    p.add_argument("--labels", nargs='+', required=True,
                   help="Legend labels: PPO_CDR_ES PPO_CDR PPO_ES PPO etc.")
    p.add_argument("--seeds", nargs='+', required=True,
                   help="Seed values to average over")
    p.add_argument("--title", default="Smoothed PPO Comparison")
    p.add_argument("--output", default="ppo_mean_plot")
    p.add_argument("--y_limit", type=int)
    main(p.parse_args())
