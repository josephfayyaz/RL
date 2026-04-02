import argparse
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from project_paths import FIGURES_DIR, LOGS_DIR


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", default=str(LOGS_DIR / "PPO_robustness" / "auc_scores.csv"))
    parser.add_argument("--output", default=str(FIGURES_DIR / "robustness_auc_comparison.png"))
    return parser.parse_args()


def main():
    args = parse_args()
    df = pd.read_csv(args.csv_path)

    plt.figure(figsize=(6, 4))
    bars = plt.bar(df['Algorithm'], df['AUC'])
    plt.xlabel('Method')
    plt.ylabel('Robustness AUC')
    plt.title('Robustness AUC Comparison Across Methods')
    plt.ylim(0, df['AUC'].max() * 1.1)
    plt.xticks(rotation=30, ha='right')

    for bar, val in zip(bars, df['AUC']):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.02 * df['AUC'].max(),
            f'{val:.1f}',
            ha='center', va='bottom', fontsize=9
        )

    plt.tight_layout()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    plt.savefig(args.output, dpi=300)
    print(f"Saved AUC bar chart to {args.output}")


if __name__ == "__main__":
    main()
