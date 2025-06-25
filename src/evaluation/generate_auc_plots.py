import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file with correct path
csv_path = '/content/auc_scores.csv'
df = pd.read_csv(csv_path)

# Create the bar chart
plt.figure(figsize=(6, 4))
bars = plt.bar(df['Algorithm'], df['AUC'])

# Labels and title
plt.xlabel('Method')
plt.ylabel('Robustness AUC')
plt.title('Robustness AUC Comparison Across Methods')
plt.ylim(0, df['AUC'].max() * 1.1)
plt.xticks(rotation=30, ha='right')

# Annotate bars with values
for bar, val in zip(bars, df['AUC']):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        val + 0.02 * df['AUC'].max(),
        f'{val:.1f}',
        ha='center', va='bottom', fontsize=9
    )

plt.tight_layout()
plt.show()
