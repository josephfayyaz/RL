import pandas as pd
import numpy as np
from numpy import trapz

# 1) Load your CSV
df = pd.read_csv(r'D:\rl\RL-yosef33333\custom-files\pposajjad\logs\20250614_231226_ppo.csv')   # adjust path if needed

# 2) Mean episodic return
mean_return = df['reward'].mean()

# 3) Fifth‐percentile return
fifth_percentile = np.percentile(df['reward'], 5)
# 4) Success rate
#    define a success threshold—e.g. “success” if reward ≥ 1 000
SUCCESS_THRESH = 1000
success_rate = (df['reward'] >= SUCCESS_THRESH).mean()

# 5) Env‐steps to reach a normalized return of 1 000
#    find the first episode where the reward ≥ 1 000
hit = df.index[df['reward'] >= SUCCESS_THRESH]
if len(hit) > 0:
    ep_idx = hit[0] + 1                       # episodes are 1-indexed
    # assume each episode is exactly T steps long (e.g. 1 000 steps)
    EP_LENGTH = 1000                         
    steps_to_1000 = ep_idx * EP_LENGTH
else:
    steps_to_1000 = None                     # never reached

print(f"Mean return:            {mean_return:.2f}")
print(f"5th percentile return: {fifth_percentile:.2f}")
print(f"Success rate:           {success_rate*100:.1f}%")
print(f"Steps to 1 000 return:  {steps_to_1000}")

# 6) (Bonus) AUC of the robustness curve
# If you’ve evaluated at multiple CDR levels (say levels=[0,0.25,0.5,0.75,1.0])
# and recorded mean return per level in a list `returns_per_level`,
# you can compute:
levels = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
returns_per_level = np.array([800, 900, 950, 980, 1000])  # example
robust_auc = trapz(returns_per_level, levels)
print(f"Robustness‐curve AUC:   {robust_auc:.1f}")


import matplotlib.pyplot as plt

plt.plot(levels, returns_per_level, marker='o')
plt.xlabel('CDR Level')
plt.ylabel('Mean Return')
plt.title('Robustness Curve')
plt.grid(True)
plt.show()

import pandas as pd
import matplotlib.pyplot as plt

# 1. Load the CSV file
csv_path = r'D:\rl\RL-yosef33333\custom-files\pposajjad\logs\20250614_231226_ppo.csv'  # Change path as needed
df = pd.read_csv(csv_path)

# 2. Basic learning curve: Episode return vs Episode
plt.figure(figsize=(10, 5))
plt.plot(df['episode'], df['reward'], label='Episode Return', alpha=0.6)

# 3. (Optional) Moving average for smoothing
window = 50  # e.g., 50 episodes
df['smoothed'] = df['reward'].rolling(window, min_periods=1).mean()
plt.plot(df['episode'], df['smoothed'], label=f'{window}-Episode Moving Avg', linewidth=2)

# 4. Plot formatting
plt.title('Learning Curve')
plt.xlabel('Episode')
plt.ylabel('Episode Return')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()