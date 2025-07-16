# ...existing code...
import os
import csv
import argparse
from typing import List, Dict
import matplotlib.pyplot as plt
from llm.generator import generate_llm_selection  # 假設此函式負責 LLM 推理

GRAPHGA_CSV = os.path.join('results', 'graphga.csv')
LLM_CSV = os.path.join('results', 'llm.csv')


def read_graphga_csv(path: str) -> Dict[int, List[Dict]]:
    """讀取 graphga.csv，回傳 generation 對應分子列表"""
    generations = {}
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            gen = int(row['generation'])
            if gen not in generations:
                generations[gen] = []
            generations[gen].append({
                'smiles': row['smiles'],
                'score': float(row['score'])
            })
    return generations


def write_llm_csv(path: str, llm_results: Dict[int, List[Dict]]):
    """將 LLM 選出的分子寫入 llm.csv"""
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['generation', 'smiles', 'oracle'])
        writer.writeheader()
        for gen, molecules in llm_results.items():
            for mol in molecules:
                writer.writerow({
                    'generation': gen,
                    'smiles': mol['smiles'],
                    'oracle': mol['score']
                })


def plot_scores(generations: Dict[int, List[Dict]], llm_results: Dict[int, List[Dict]]):
    """繪製 max/avg 曲線，並標示 LLM 選出的分子分數"""
    gens = sorted(generations.keys())
    max_scores = [max([m['score'] for m in generations[g]]) for g in gens]
    avg_scores = [sum([m['score'] for m in generations[g]]) / len(generations[g]) for g in gens]
    llm_max = [max([m['score'] for m in llm_results.get(g, [])]) if g in llm_results else None for g in gens]
    llm_avg = [sum([m['score'] for m in llm_results.get(g, [])]) / len(llm_results.get(g, [])) if g in llm_results and llm_results[g] else None for g in gens]

    plt.plot(gens, max_scores, label='GraphGA Max')
    plt.plot(gens, avg_scores, label='GraphGA Avg')
    plt.plot(gens, llm_max, 'o-', label='LLM Max')
    plt.plot(gens, llm_avg, 'o-', label='LLM Avg')
    plt.xlabel('Generation')
    plt.ylabel('Score')
    plt.legend()
    plt.title('GraphGA & LLM Molecule Selection')
    plt.show()


def main(task: str, description: str):
    generations = read_graphga_csv(GRAPHGA_CSV)
    llm_results = {}
    for gen, molecules in generations.items():
        smiles_list = [m['smiles'] for m in molecules]
        # 呼叫 LLM，選出 10 個分子（不洩漏分數）
        selected = generate_llm_selection(smiles_list, task, description, top_k=10)
        # 取得 oracle 分數（從原始 molecules 查找）
        selected_mols = [next(m for m in molecules if m['smiles'] == s) for s in selected]
        llm_results[gen] = selected_mols
    write_llm_csv(LLM_CSV, llm_results)
    plot_scores(generations, llm_results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', required=True, help='任務名稱')
    parser.add_argument('--description', required=True, help='任務描述')
    args = parser.parse_args()
    main(args.task, args.description)
# ...existing code...
