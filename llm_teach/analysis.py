import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- File paths: update if necessary ---
file_fg = '/home/luketou/GA_LLM_World_Model/llm_teach/results/results_qwen_3_235b_a22b/amlodipine_reflection_fg.csv'
file_sm = '/home/luketou/GA_LLM_World_Model/llm_teach/results/results_qwen_3_235b_a22b/amlodipine_nofg.csv'

# 1. Load data
df_fg = pd.read_csv(file_fg)
df_sm = pd.read_csv(file_sm)

# 2. Compute per-generation statistics
metrics_fg = df_fg.groupby('generation')['score'] \
    .agg(fg_max='max', fg_avg='mean', fg_min='min') \
    .reset_index()
metrics_sm = df_sm.groupby('generation')['score'] \
    .agg(sm_max='max', sm_avg='mean', sm_min='min') \
    .reset_index()

# 3. Merge datasets
df_m = pd.merge(metrics_fg, metrics_sm, on='generation')

# 4. Calculate metrics
results = []
for prefix, label in [('fg', 'With Function Group'), ('sm', 'Only SMILES')]:
    gens = df_m['generation'].values
    max_s = df_m[f'{prefix}_max'].values
    avg_s = df_m[f'{prefix}_avg'].values
    min_s = df_m[f'{prefix}_min'].values

    auc_max = np.trapz(max_s, gens)
    auc_avg = np.trapz(avg_s, gens)
    auc_min = np.trapz(min_s, gens)

    thresholds = {}
    for t in [0.2, 0.3, 0.4]:
        idx = np.where(avg_s >= t)[0]
        thresholds[f'gen_to_{int(t*100)}%'] = int(gens[idx[0]]) if idx.size else np.nan

    last10 = avg_s[-10:]
    final_mean = last10.mean()
    final_std = last10.std()

    results.append({
        'Method': label,
        'AUC Max': auc_max,
        'AUC Avg': auc_avg,
        'AUC Min': auc_min,
        **thresholds,
        'Final Avg Mean': final_mean,
        'Final Avg Std': final_std
    })

results_df = pd.DataFrame(results)

# 5. Output results table
print(results_df.to_string(index=False))

# 6. Plot Average Score vs Generation
plt.figure(figsize=(8, 4))
plt.plot(df_m['generation'], df_m['fg_avg'], label='With Function Group Avg')
plt.plot(df_m['generation'], df_m['sm_avg'], label='Only SMILES Avg')
plt.xlabel('Generation')
plt.ylabel('Average Score')
plt.title('Average Score vs Generation')
plt.legend()
plt.tight_layout()
plt.show()

# 7. Plot Difference Curve (FG Avg - SM Avg)
delta = df_m['fg_avg'] - df_m['sm_avg']
plt.figure(figsize=(8, 4))
plt.plot(df_m['generation'], delta, label='Difference (FG Avg - SM Avg)')
plt.axhline(0, color='gray', linestyle='--')
plt.xlabel('Generation')
plt.ylabel('Score Difference')
plt.title('Difference in Average Score vs Generation')
plt.legend()
plt.tight_layout()
plt.show()
