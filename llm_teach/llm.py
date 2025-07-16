# llm.py
# 用於控制 LLM 分子篩選，符合 README 的要求
import csv
from typing import List, Dict

# 假設有一個 LLM API 或本地推理函式
# 這裡用 mock_llm_select 代表 LLM 選擇分子的邏輯

def mock_llm_select(smiles_list: List[str], task: str, description: str, top_k: int = 10) -> List[str]:
    """
    模擬 LLM 根據 SMILES、任務、描述選出 top_k 分子
    實際應用時請替換為 LLM 推理 API
    """
    # 這裡僅隨機選擇，實際應用請用 LLM
    import random
    return random.sample(smiles_list, min(top_k, len(smiles_list)))


def read_graphga_csv(path: str) -> Dict[int, List[Dict]]:
    """
    讀取 graphga.csv，回傳 generation 對應分子列表
    """
    generations = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            gen = int(row["generation"])
            if gen not in generations:
                generations[gen] = []
            generations[gen].append({
                "smiles": row["smiles"],
                "score": float(row["score"])
            })
    return generations


def write_llm_csv(path: str, llm_results: List[Dict]):
    """
    寫入 LLM 選出的分子到 llm.csv
    """
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["generation", "smiles", "oracle"])
        writer.writeheader()
        for row in llm_results:
            writer.writerow(row)


def llm_select_main(graphga_path: str, llm_path: str, task: str, description: str):
    """
    主流程：讀取 graphga.csv，呼叫 LLM 選分子，寫入 llm.csv
    """
    generations = read_graphga_csv(graphga_path)
    llm_results = []
    for gen, molecules in generations.items():
        smiles_list = [m["smiles"] for m in molecules]
        selected_smiles = mock_llm_select(smiles_list, task, description)
        # oracle 分數不可洩漏給 LLM，但 llm.csv 需記錄
        for s in selected_smiles:
            score = next((m["score"] for m in molecules if m["smiles"] == s), None)
            llm_results.append({
                "generation": gen,
                "smiles": s,
                "oracle": score
            })
    write_llm_csv(llm_path, llm_results)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="LLM 分子篩選流程")
    parser.add_argument("--graphga", type=str, default="results/graphga.csv", help="GraphGA 結果檔案路徑")
    parser.add_argument("--llm", type=str, default="results/llm.csv", help="LLM 選擇結果檔案路徑")
    parser.add_argument("--task", type=str, required=True, help="任務名稱")
    parser.add_argument("--desc", type=str, required=True, help="任務描述")
    args = parser.parse_args()
    llm_select_main(args.graphga, args.llm, args.task, args.desc)
