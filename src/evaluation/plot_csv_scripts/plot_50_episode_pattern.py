import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV files
df_vanilla_source_source = pd.read_csv("/home/joseph/python-proj/udr_ES/Logs/PPO_eval/PPO_Vanilla_source_source_seed_0_50_episodes.csv")
df_vanilla_source_target = pd.read_csv("/home/joseph/python-proj/udr_ES/Logs/PPO_eval/PPO_Vanilla_source_target_seed_0_50_episodes.csv")
df_vanilla_target_target = pd.read_csv("/home/joseph/python-proj/udr_ES/Logs/PPO_eval/PPO_Vanilla_target_target_seed_0_50_episodes.csv")
df_udr_source_source = pd.read_csv("/home/joseph/python-proj/udr_ES/Logs/PPO_eval/PPO_UDR_source_source_seed_0_eval_50_episodes.csv")
df_udr_source_target = pd.read_csv("/home/joseph/python-proj/udr_ES/Logs/PPO_eval/PPO_UDR_source_target_seed_0_50_episodes.csv")

# Apply moving average smoothing with window size 5
df_vanilla_source_source['smoothed_reward'] = df_vanilla_source_source['reward'].rolling(window=5).mean()
df_vanilla_source_target['smoothed_reward'] = df_vanilla_source_target['reward'].rolling(window=5).mean()
df_vanilla_target_target['smoothed_reward'] = df_vanilla_target_target['reward'].rolling(window=5).mean()
df_udr_source_source['smoothed_reward'] = df_udr_source_source['reward'].rolling(window=5).mean()
df_udr_source_target['smoothed_reward'] = df_udr_source_target['reward'].rolling(window=5).mean()

# Plotting
plt.figure(figsize=(12, 6))

plt.plot(df_vanilla_source_source['episode'], df_vanilla_source_source['smoothed_reward'],
         label='Vanilla source to source Environment', color='#4169E1', linestyle='-', linewidth=2)
plt.plot(df_vanilla_source_target['episode'], df_vanilla_source_target['smoothed_reward'],
         label='Vanilla source to target Environment', color='#DAA520', linestyle='-', linewidth=2)
plt.plot(df_vanilla_target_target['episode'], df_vanilla_target_target['smoothed_reward'],
         label='Vanilla target to target Environment', color='#2E8B57', linestyle='-', linewidth=2)
plt.plot(df_udr_source_source['episode'], df_udr_source_source['smoothed_reward'],
         label='UDR source to source Environment', color='#DC143C', linestyle='-', linewidth=2)
plt.plot(df_udr_source_target['episode'], df_udr_source_target['smoothed_reward'],
         label='UDR source to target Environment', color='#708090', linestyle='-', linewidth=2)

plt.title('PPO Vanilla and PPO UDR Reward Comparison: Different Environments')
plt.xlabel('Episode')
plt.ylabel('Reward')

# Legend fully outside to the right
plt.legend(loc='center left', bbox_to_anchor=(1.05, 0.5), frameon=True)

# Adjust layout so legend fits fully
plt.subplots_adjust(right=0.75)
plt.grid(True)

# Save the figure
plt.savefig("ppo_reward_comparison.png", format='png', dpi=300, bbox_inches='tight')
