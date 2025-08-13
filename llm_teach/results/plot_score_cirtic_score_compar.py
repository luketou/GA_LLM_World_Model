
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import argparse

# Set up argument parser
parser = argparse.ArgumentParser(description='Compare true scores and proxy margins from CSV files.')
parser.add_argument('--truth_csv', type=str, required=True, help='Path to the ground truth CSV file containing true scores.')
parser.add_argument('--proxy_csv', type=str, required=True, help='Path to the proxy margin CSV file containing critic scores.')
parser.add_argument('--max_generation', type=int, help='Maximum generation to include in the plot.')

args = parser.parse_args()

# Load the datasets
df_truth = pd.read_csv(args.truth_csv)
df_proxy = pd.read_csv(args.proxy_csv)

# Prepare for merging
df_truth.rename(columns={'true_score': 'score'}, inplace=True)
df_proxy.rename(columns={'proxy_margin': 'critic_score'}, inplace=True)

# Merge the two dataframes
df_merged = pd.merge(df_truth, df_proxy, on=['generation', 'smiles'])

# Filter by generation if specified
if args.max_generation:
    df_merged = df_merged[df_merged['generation'] <= args.max_generation]

if df_merged.empty:
    print("No data to plot after filtering. Please check your input files and max_generation value.")
else:
    # Normalize the scores
    scaler = MinMaxScaler()
    scores_normalized = scaler.fit_transform(df_merged[['score', 'critic_score']])
    df_merged[['score_norm', 'critic_score_norm']] = scores_normalized

    # Group by generation and calculate the mean of the normalized scores
    df_agg = df_merged.groupby('generation')[['score_norm', 'critic_score_norm']].mean().reset_index()

    # Plotting
    plt.figure(figsize=(12, 7))
    plt.plot(df_agg['generation'], df_agg['score_norm'], label='Normalized True Score')
    plt.plot(df_agg['generation'], df_agg['critic_score_norm'], label='Normalized Critic Score')

    plt.xlabel('Generation')
    plt.ylabel('Average Normalized Value')
    plt.title('Comparison of Normalized True Score and Critic Score per Generation')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save the figure
    output_filename = "score_critic_comparison.png"
    plt.savefig(output_filename)
    print(f"Plot successfully saved to {output_filename}")
