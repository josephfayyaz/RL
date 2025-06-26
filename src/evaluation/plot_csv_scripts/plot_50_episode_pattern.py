import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV files
df_source = pd.read_csv("/home/joseph/python-proj/udr_ES/Logs/PPO_eval/PPO_Vanilla_source_source_seed_0_50_episodes.csv")
df_target = pd.read_csv("/home/joseph/python-proj/udr_ES/Logs/PPO_eval/PPO_Vanilla_source_target_seed_0_50_episodes.csv")

# Apply moving average smoothing with window size 5
df_source['smoothed_reward'] = df_source['reward'].rolling(window=5).mean()
df_target['smoothed_reward'] = df_target['reward'].rolling(window=5).mean()

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(df_source['episode'], df_source['smoothed_reward'], label='source to source Environment',
         color='blue', linestyle='-', linewidth=2)
plt.plot(df_target['episode'], df_target['smoothed_reward'], label='source to target Environment',
         color='green', linestyle='-', linewidth=2)

plt.title('PPO_Vanilla Reward Comparison: Source vs Target Environment')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save the figure to PNG
plt.savefig("ppo_vanilla_reward_comparison.png", format='png', dpi=300)

# Show the plot
# plt.show()

