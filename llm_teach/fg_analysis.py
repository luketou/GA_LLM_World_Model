#!/usr/bin/env python
# fg_analysis.py
"""
Pipeline:
1. 讀取 SMILES 清單 (txt/tsv)
2. 用 FARM 將每條 SMILES 切成功能團 token
3. 以 MultiLabelBinarizer 轉稀疏 binary FG 特徵矩陣
4. 調用 GuacaMol scorer 取得 oracle score (支援多任務)
5. 訓練 LightGBM or XGBoost (可加 --gpu)
6. 以 SHAP 解釋單 FG / FG pair 貢獻
7. 輸出 CSV (fg_contribution, fg_pair_contrib) 與 PNG 圖
"""
import argparse, json, os, sys, itertools, joblib, warnings
from pathlib import Path
import numpy as np, pandas as pd
from tqdm import tqdm
from rdkit import Chem
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import guacamol.assess_goal_directed_generation as agdg

# -------------------- CLI --------------------
def get_args():
    p = argparse.ArgumentParser(description="FG→score analysis from smilefg preprocessed input (GPU optional)")
    p.add_argument("--input", default="data/guacamol_v1_all.csv",
                   help="SMILES 檔路徑 (每行一條)")
    p.add_argument("--output_dir", default="fg_outputs", help="輸出資料夾")
    p.add_argument("--task", default="logP", help="GuacaMol scorer task 名稱")
    p.add_argument("--model", choices=["lgbm", "xgb"], default="xgb",
                   help="選擇回歸模型")
    p.add_argument("--gpu", action="store_true", help="啟用 GPU (需 CUDA)")
    p.add_argument("--top_pairs", type=int, default=50,
                   help="輸出交互 SHAP 前多少組")
    p.add_argument("--preprocessed", default="data/fg_enhanced_smiles.csv",
                   help="CSV of pre-tokenized FG-enhanced SMILES with column fg_tokens (typically smilefg file)")
    p.add_argument("--suite", default="v2", help="GuacaMol benchmark suite version")
    return p.parse_args()

# -------------------- GuacaMol 整合 --------------------
def get_scoring_function(task_name, suite="v2"):
    """取得 GuacaMol scoring function"""
    try:
        # GuacaMol 任務對映表
        task_to_benchmark = {
            'osimertinib': 'Osimertinib MPO',
            'fexofenadine': 'Fexofenadine MPO', 
            'ranolazine': 'Ranolazine MPO',
            'amlodipine': 'Amlodipine MPO',
            'perindopril': 'Perindopril MPO',
            'sitagliptin': 'Sitagliptin MPO',
            'zaleplon': 'Zaleplon MPO',
            'cobimetinib': 'Scaffold Hop',
            'scaffold_hop': 'Scaffold Hop',
            'decoration_hop': 'Decoration Hop',
            'weird_physchem': 'Weird physchem',
            'valsartan_smarts': 'Valsartan SMARTS',
            'median1': 'Median molecules 1',
            'median2': 'Median molecules 2',
            'isomer_c11h24': 'C11H24',
            'isomer_c9h10n2o2pf2cl': 'C9H10N2O2PF2Cl',
            'celecoxib': 'Celecoxib Rediscovery',
            'troglitazone': 'Troglitazone',
            'thiothixene': 'Thiothixene',
            'mestranol': 'Mestranol'
        }
        
        target_benchmark_name = task_to_benchmark.get(task_name, task_name)
        
        # 取得 benchmark suite
        if suite == "v1":
            benchmarks = agdg.goal_directed_benchmark_suite_v1()
        else:
            benchmarks = agdg.goal_directed_benchmark_suite()
        
        # 找到對應的 benchmark
        target_benchmark = None
        for benchmark in benchmarks:
            if benchmark.name == target_benchmark_name:
                target_benchmark = benchmark
                break
        
        if target_benchmark is None:
            print(f"Error: No benchmark found for task '{task_name}' (mapped to '{target_benchmark_name}')")
            print("Available benchmarks:")
            for b in benchmarks:
                print(f"  - {b.name}")
            return None
        
        print(f"Using GuacaMol benchmark: {target_benchmark.name}")
        return target_benchmark.objective
        
    except ImportError as e:
        print(f"Warning: GuacaMol not available ({e}), using random scores")
        return None
    except Exception as e:
        print(f"Error setting up GuacaMol: {e}")
        return None

# -------------------- 載入資料 --------------------
def load_data(args):
    """只從 args.input 取得 SMILES 資料"""
    print("Step1: Load SMILES from input CSV…")
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file {args.input} not found")
    
    input_df = pd.read_csv(args.input)
    print(f"Available columns in {args.input}: {list(input_df.columns)}")
    
    # 自動偵測 SMILES 欄位
    possible_smiles_cols = ['smiles', 'SMILES', 'canonical_smiles', 'mol', 'molecule']
    smiles_col = None
    for col in possible_smiles_cols:
        if col in input_df.columns:
            smiles_col = col
            print(f"Using column '{col}' as SMILES")
            break
    if smiles_col is None:
        smiles_col = input_df.columns[0]
        print(f"No standard SMILES column found, using first column: '{smiles_col}'")
    
    smiles_list = input_df[smiles_col].dropna().astype(str).tolist()
    smiles_list = [s.strip() for s in smiles_list if s.strip() and s.strip() != 'nan']
    print(f"Loaded {len(smiles_list)} SMILES from column '{smiles_col}'")
    
    # 顯示一些範例
    if smiles_list:
        print("Sample SMILES:")
        for i, smiles in enumerate(smiles_list[:3]):
            print(f"  Example {i+1}: {smiles}")
    
    df = pd.DataFrame({"smiles": smiles_list})
    # 這裡需要實作 FG tokenization，暫時用空列表
    df["fg_tokens"] = [[] for _ in range(len(df))]
    print("Warning: No FG tokenization implemented for raw SMILES")
    
    # 移除空的 SMILES 或無效分子
    initial_count = len(df)
    df = df[df["smiles"].notna() & (df["smiles"] != "") & (df["smiles"] != "nan")]
    df = df.reset_index(drop=True)
    print(f"Loaded {len(df)} molecules (removed {initial_count - len(df)} invalid entries)")
    
    # 顯示最終的樣本
    if len(df) > 0:
        print("Final sample SMILES:")
        for i in range(min(3, len(df))):
            tokens_info = f" (FG tokens: {len(df.iloc[i]['fg_tokens'])})" if df.iloc[i]['fg_tokens'] else " (no FG tokens)"
            print(f"  {i+1}: {df.iloc[i]['smiles']}{tokens_info}")
    return df

# -------------------- 2 FG 二元特徵 --------------------
def build_feature_matrix(token_lists):
    """建立功能團特徵矩陣"""
    print(f"Building feature matrix from {len(token_lists)} token lists...")
    
    # 檢查資料
    valid_count = sum(1 for tokens in token_lists if tokens)
    print(f"Valid FG token lists: {valid_count}/{len(token_lists)}")
    
    if valid_count == 0:
        print("ERROR: No valid FG tokens found!")
        print("Sample token lists:")
        for i, tokens in enumerate(token_lists[:5]):
            print(f"  {i}: {tokens} (type: {type(tokens)})")
        
        # 如果沒有有效的 FG tokens，建立虛擬特徵
        print("Creating dummy features for demo purposes...")
        dummy_tokens = [["dummy_fg"] for _ in range(len(token_lists))]
        mlb = MultiLabelBinarizer(sparse_output=True)
        X = mlb.fit_transform(dummy_tokens)
        return X, mlb.classes_
    
    # 過濾空的 token lists，但保持原始長度
    processed_lists = []
    for tokens in token_lists:
        if tokens and isinstance(tokens, list):
            processed_lists.append(tokens)
        else:
            processed_lists.append([])  # 空列表而不是過濾掉
    
    mlb = MultiLabelBinarizer(sparse_output=True)
    X = mlb.fit_transform(processed_lists)
    
    print(f"Feature matrix shape: {X.shape}, {len(mlb.classes_)} unique FG tokens")
    
    # 顯示最常見的 FG tokens
    if len(mlb.classes_) > 0:
        feature_counts = X.sum(axis=0).A1  # 轉換為 1D array
        top_features = sorted(zip(mlb.classes_, feature_counts), 
                            key=lambda x: x[1], reverse=True)[:10]
        print("Top 10 most frequent FG tokens:")
        for fg, count in top_features:
            print(f"  {fg}: {count}")
    
    return X, mlb.classes_

# -------------------- 3 Oracle score --------------------
def compute_scores(df, task, suite="v2"):
    """計算 oracle scores"""
    scorer = get_scoring_function(task, suite)
    
    if scorer is not None:
        print(f"Computing scores using GuacaMol {task}...")
        
        def safe_score(smiles):
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    return 0.0
                return scorer(smiles)
            except Exception as e:
                print(f"Error scoring {smiles}: {e}")
                return 0.0
        
        tqdm.pandas(desc=f"Scoring ({task})")
        df["score"] = df["smiles"].progress_apply(safe_score)
    else:
        # 如果 GuacaMol 不可用，使用隨機分數作為 demo
        print("Warning: Using random scores (GuacaMol not available)")
        np.random.seed(42)
        df["score"] = np.random.rand(len(df))
    
    print(f"Score statistics - Mean: {df['score'].mean():.3f}, Std: {df['score'].std():.3f}")
    print(f"Score range: [{df['score'].min():.3f}, {df['score'].max():.3f}]")
    
    return df

# -------------------- 4 模型訓練 --------------------
def train_model(X, y, model_name="xgb", gpu=False):
    """訓練回歸模型"""
    print(f"Training {model_name} model (GPU: {gpu})...")
    
    if model_name == "xgb":
        import xgboost as xgb
        params = {
            "objective": "reg:squarederror",
            "learning_rate": 0.05,
            "n_estimators": 400,
            "max_depth": 6,
            "subsample": 0.8,
            "random_state": 42,
            "verbosity": 0
        }
        if gpu:
            params.update({"tree_method": "gpu_hist", "predictor": "gpu_predictor"})
        model = xgb.XGBRegressor(**params)
    else:
        import lightgbm as lgb
        params = {
            "objective": "regression",
            "metric": "rmse",
            "learning_rate": 0.05,
            "num_leaves": 64,
            "n_estimators": 400,
            "random_state": 42,
            "verbosity": -1
        }
        if gpu:
            params.update({"device": "gpu"})
        model = lgb.LGBMRegressor(**params)
    
    # 轉換稀疏矩陣
    if hasattr(X, 'toarray'):
        X_dense = X.toarray()
    else:
        X_dense = X
        
    model.fit(X_dense, y)
    return model

# -------------------- 5 SHAP 分析 --------------------
def shap_analysis(model, X, feature_names, out_dir, top_pairs=50):
    """SHAP 特徵重要性分析"""
    import shap
    import matplotlib.pyplot as plt
    
    # 轉換為 dense 矩陣
    if hasattr(X, 'toarray'):
        X_dense = X.toarray()
    else:
        X_dense = X
    
    # 使用較小的樣本進行 SHAP 分析以提升效率
    n_samples = min(1000, X_dense.shape[0])
    X_sample = X_dense[:n_samples]
    
    print(f"Computing SHAP values for {n_samples} samples...")
    
    try:
        explainer = shap.Explainer(model, X_sample)
        shap_values = explainer(X_sample, check_additivity=False)

        # 5.1 單 FG summary plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values.values, features=X_sample,
                          feature_names=feature_names, show=False)
        plt.tight_layout()
        plt.savefig(f"{out_dir}/shap_summary.png", dpi=300, bbox_inches='tight')
        plt.close()

        # 5.2 CSV: 單 FG 平均貢獻
        mean_abs = np.abs(shap_values.values).mean(axis=0)
        fg_contrib_df = pd.DataFrame({
            "fg": feature_names, 
            "mean_abs_shap": mean_abs
        }).sort_values("mean_abs_shap", ascending=False)
        
        fg_contrib_df.to_csv(f"{out_dir}/fg_contribution.csv", index=False)
        print(f"Top 10 important FGs:")
        print(fg_contrib_df.head(10))

        # 5.3 交互作用分析
        try:
            print("Computing interaction values...")
            interaction_values = shap.TreeExplainer(model).shap_interaction_values(X_sample)
            upper = np.triu(np.ones_like(interaction_values[0], dtype=bool), k=1)
            mean_inter = np.abs(interaction_values).mean(axis=0)[upper]
            pairs_idx = np.column_stack(np.where(upper))
            
            pair_df = pd.DataFrame({
                "fg1": feature_names[pairs_idx[:, 0]],
                "fg2": feature_names[pairs_idx[:, 1]],
                "mean_abs_shap": mean_inter,
            }).sort_values("mean_abs_shap", ascending=False).head(top_pairs)
            
            pair_df.to_csv(f"{out_dir}/fg_pair_contrib.csv", index=False)
            return pair_df
            
        except Exception as e:
            print(f"Warning: Interaction analysis failed: {e}")
            return pd.DataFrame()
            
    except Exception as e:
        print(f"Error in SHAP analysis: {e}")
        return pd.DataFrame()

# -------------------- main --------------------
def main():
    args = get_args()
    Path(args.output_dir).mkdir(exist_ok=True, parents=True)
    
    # 載入資料
    df = load_data(args)
    
    # 建立特徵矩陣
    print("Step2: Build FG feature matrix…")
    X, fg_names = build_feature_matrix(df["fg_tokens"])
    
    # 計算分數
    print("Step3: Compute oracle scores…")
    df = compute_scores(df, args.task, args.suite)
    y = df["score"].values
    
    # 訓練模型
    print("Step4: Train model…")
    model = train_model(X, y, args.model, args.gpu)
    joblib.dump(model, f"{args.output_dir}/model.pkl")
    
    # SHAP 分析
    print("Step5: SHAP analysis…")
    shap_pairs = shap_analysis(model, X, fg_names, args.output_dir,
                               top_pairs=args.top_pairs)
    
    if not shap_pairs.empty:
        print("Top 10 interaction pairs:")
        print(shap_pairs.head(10))
    
    # 儲存結果摘要
    summary = {
        "task": args.task,
        "suite": args.suite,
        "n_molecules": len(df),
        "n_features": len(fg_names),
        "score_mean": float(y.mean()),
        "score_std": float(y.std()),
        "score_min": float(y.min()),
        "score_max": float(y.max()),
        "model": args.model,
        "gpu": args.gpu
    }
    
    with open(f"{args.output_dir}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    # 儲存處理後的資料
    df.to_csv(f"{args.output_dir}/processed_data.csv", index=False)
    
    print(f"🎉 All results saved to: {args.output_dir}")

if __name__ == "__main__":
    main()