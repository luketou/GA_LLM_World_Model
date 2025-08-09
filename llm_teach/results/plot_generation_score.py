'''
python plot_generation_score.py --graphga_dir data/offspring --cerebras_dir results_llm_select results/results_qwen_3_235b_a22b --task amlodipine
'''
import os
import pandas as pd
import matplotlib.pyplot as plt

def find_score_column(df):
    # 自動偵測 'score' 欄位（忽略大小寫、空白）
    for col in df.columns:
        if col.strip().lower() == 'score':
            return col
    raise KeyError("CSV 檔案未找到 'score' 欄位，請確認欄位名稱是否正確。")

def extract_scores(results_dir, task, max_generations=None):
    csv_path = os.path.join(results_dir, f"{task}.csv")
    df = pd.read_csv(csv_path)
    score_col = find_score_column(df)
    generations = sorted(df['generation'].unique())
    if max_generations is not None:
        generations = generations[:max_generations]
    scores_per_gen = [df[df['generation'] == gen][score_col].tolist() for gen in generations]
    max_scores = [max(scores) for scores in scores_per_gen]
    avg_scores = [sum(scores)/len(scores) for scores in scores_per_gen]
    min_scores = [min(scores) for scores in scores_per_gen]
    return generations, max_scores, avg_scores, min_scores, df

def extract_scores_with_fallback(graphga_dir, cerebras_dir, task, max_generations=None):
    _, _, _, _, df_graphga = extract_scores(graphga_dir, task)
    csv_path_cerebras = os.path.join(cerebras_dir, f"{task}.csv")
    df_cerebras = pd.read_csv(csv_path_cerebras)
    score_col_cerebras = find_score_column(df_cerebras)
    score_col_graphga = find_score_column(df_graphga)
    generations = sorted(df_cerebras['generation'].unique())
    if max_generations is not None:
        generations = generations[:max_generations]
    scores_per_gen = []
    for gen in generations:
        gen_df = df_cerebras[df_cerebras['generation'] == gen]
        scores = []
        for idx, row in gen_df.iterrows():
            score = row[score_col_cerebras]
            smiles = row['smiles']
            # score 為 nan 或缺失時啟動備用計畫
            if pd.isna(score):
                # 從 graphga 補充
                match = df_graphga[df_graphga['smiles'] == smiles]
                if not match.empty:
                    score = match.iloc[0][score_col_graphga]
                else:
                    continue  # 找不到就忽略
            scores.append(score)
        if scores:
            scores_per_gen.append(scores)
        else:
            scores_per_gen.append([float('nan')])  # 若整個 generation 都沒分數
    max_scores = [max(scores) if scores and not pd.isna(scores[0]) else float('nan') for scores in scores_per_gen]
    avg_scores = [sum(scores)/len(scores) if scores and not pd.isna(scores[0]) else float('nan') for scores in scores_per_gen]
    min_scores = [min(scores) if scores and not pd.isna(scores[0]) else float('nan') for scores in scores_per_gen]
    return generations, max_scores, avg_scores, min_scores

def plot_comparison_lines(graphga_dir, cerebras_dir, task, max_generations):
    gens_g, max_g, avg_g, min_g, _ = extract_scores(graphga_dir, task, max_generations)
    gens_c, max_c, avg_c, min_c = extract_scores_with_fallback(graphga_dir, cerebras_dir, task, max_generations)

    # Normalize GraphGA scores
    all_g_scores = max_g + avg_g + min_g
    g_min = min(all_g_scores)
    g_max = max(all_g_scores)
    def normalize_g(x):
        return [(v-g_min)/(g_max-g_min) if g_max>g_min else 0 for v in x]
    max_g_n = normalize_g(max_g)
    avg_g_n = normalize_g(avg_g)
    min_g_n = normalize_g(min_g)

    # Normalize Cerebras scores
    all_c_scores = max_c + avg_c + min_c
    c_min = min([v for v in all_c_scores if not pd.isna(v)])
    c_max = max([v for v in all_c_scores if not pd.isna(v)])
    def normalize_c(x):
        return [(v-c_min)/(c_max-c_min) if c_max>c_min and not pd.isna(v) else 0 for v in x]
    max_c_n = normalize_c(max_c)
    avg_c_n = normalize_c(avg_c)
    min_c_n = normalize_c(min_c)

    plt.figure(figsize=(12, 7))
    plt.plot(gens_g, max_g_n, label='GraphGA Max', color='blue')
    plt.plot(gens_g, avg_g_n, label='GraphGA Avg', color='cyan')
    plt.plot(gens_g, min_g_n, label='GraphGA Min', color='navy')

    plt.plot(gens_c, max_c_n, label='LLM Max', color='red')
    plt.plot(gens_c, avg_c_n, label='LLM Avg', color='orange')
    plt.plot(gens_c, min_c_n, label='LLM Min', color='darkred')

    plt.xlabel('Generation')
    plt.ylabel('Normalized Score')
    plt.title(f'50 Generations Score Comparison for {task} (Normalized) and reflection with RL')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{task}_generation.png")
    plt.show()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Compare generation scores from GraphGA and Cerebras CSV files.')
    parser.add_argument('--graphga_dir', type=str, help='Directory for GraphGA result CSV files.',default='data/offspring')
    parser.add_argument('--cerebras_dir', type=str,  help='Directory for Cerebras result CSV files.',default='results/muvera_rl_critic_results')
    parser.add_argument('--task', type=str,  help='Task name to plot scores for.',default='amlodipine')
    parser.add_argument('--max_generations', type=int, default=50, help='Maximum number of generations to plot.')
    # example : python plot_generation_score.py --graphga_dir results_graphga --cerebras_dir results_llm_select results_cerebras --task fexofenadine
    args = parser.parse_args()
    plot_comparison_lines(args.graphga_dir, args.cerebras_dir, args.task, args.max_generations)
    print(f"Comparison plot saved as {args.task}_generation_comparison.png")