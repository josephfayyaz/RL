import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse, os

def average_seed_curves(files):
    all_curves = []
    for f in files:
        df = pd.read_csv(f)
        if 'epsilon' not in df or 'mean_return' not in df:
            print(f"⚠️ Skipping bad file: {f}")
            continue
        all_curves.append(df.set_index('epsilon')['mean_return'])

    if not all_curves:
        return None, None

    combined = pd.concat(all_curves, axis=1)
    mean_curve = combined.mean(axis=1)
    return mean_curve.index.values, mean_curve.values

def plot_robustness(curves, labels, title, output_path, y_limit=None):
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
                 linewidth=2.2, label=label)

    plt.title(title, fontsize=14, weight='bold')
    plt.xlabel("Perturbation noise size ε", fontsize=12)
    plt.ylabel("Mean Return", fontsize=12)
    if y_limit:
        plt.ylim(0, y_limit)
    plt.grid(linestyle='--', linewidth=0.5, alpha=0.7)
    plt.legend(fontsize=10, loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False)
    plt.tight_layout()

    os.makedirs(output_path, exist_ok=True)
    out_base = os.path.join(output_path, "robustness_comparison")
    plt.savefig(out_base + '.png', bbox_inches='tight')
    plt.savefig(out_base + '.pdf', bbox_inches='tight')
    print(f"✅ Saved: {out_base}.pdf")
    print(f"✅ Saved: {out_base}.png")
    plt.close()

def main(args):
    curves = []
    valid_labels = []
    auc_records = []

    for label in args.labels:
        files = []
        for seed in args.seeds:
            filename = f"robustness_results_{label}_seed_{seed}.csv"
            filepath = os.path.join(args.dirrobus, filename)
            if os.path.exists(filepath):
                files.append(filepath)
            else:
                print(f"❌ Missing: {filepath}")
        x, y = average_seed_curves(files)
        if x is not None:
            curves.append((x, y))
            valid_labels.append(label)

            ### ✅ Compute AUC using trapezoidal rule
            auc = np.trapz(y, x)
            auc_records.append({"Algorithm": label, "AUC": auc})

    if not curves:
        print("🚫 No valid data to plot.")
        return

    ### ✅ Create and show AUC DataFrame
    auc_df = pd.DataFrame(auc_records).sort_values(by="AUC", ascending=False)
    print("\n📊 Area Under Robustness Curve:")
    print(auc_df.to_string(index=False))

    # Optionally save it
    auc_df.to_csv(os.path.join(r".\Logs\PPO_robustness", "auc_scores.csv"), index=False)
    print(r"✅ Saved to: .\Logs\PPO_robustness\auc_scores.csv")
    plot_robustness(curves, valid_labels, args.title, args.output, args.y_limit)
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dirrobus", type=str,
                   default=r"D:\rl\RL-Final\Logs\PPO_robustness",
                   help="Directory containing CSVs")
    p.add_argument("--labels", nargs='+',
                   default=["PPO_ES", "PPO", "PPO_cdr_ES", "PPO_cdr", "PPO_UDR_ES", "PPO_UDR"],
                   help="List of algorithm labels")
    p.add_argument("--seeds", nargs='+', type=int,
                   default=[0, 14, 42],
                   help="Seed values to average over")
    p.add_argument("--title", type=str,
                   default="Robustness Curve",
                   help="Title of the plot")
    p.add_argument("--output", type=str,
                   default="./render/plots",
                   help="Directory to save plots")
    p.add_argument("--y_limit", type=int,
                   default=None,
                   help="Optional y-axis limit")
    main(p.parse_args())
