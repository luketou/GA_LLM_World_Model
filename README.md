# GA_LLM_World_Model - 分子優化系統

基於遺傳算法(GA)和大語言模型(LLM)的分子世界模型，用於藥物分子設計和優化。

### v1.2.2 (最新版本)
- 🚀 **MCTS 搜索策略增強**: 實現從根節點開始的完整 MCTS 選擇階段 (UCT`select_node_for_expansion`)解決過早終止和無法回溯探索的問題。
- 🔧 **MCTS 數據一致性修復**: 修正 MCTS 節點統計數據重複更新的 Bug，確保 UCT 計算的準確性。
- 💡 **LLM 引導動作選擇啟用**: 成功啟用 LLM 軌跡感知動作選擇功能，LLM 現在會被調用以輔助動作決策 (儘管 LLM 回應格式仍需優化)。
- 🧩 **動作系統重構**: 所有動作設定已統一到單一 `rdkit_action_rulebook.json`，並由 `actions/action.py` 提供唯一介面。

### 核心改進
1. **正確的分數基準**: 所有關鍵決策(早停、最佳選擇)現在基於 Oracle 的原始評分
2. **分數透明度**: 輸出同時顯示 Oracle 分數和累積分數，提高可調試性
3. **更穩健的評估**: 避免因累積分數偏差導致的錯誤決策
4. **動作設定單一來源**: 所有動作規則集中於 `rdkit_action_rulebook.json`，維護更簡單

## 🎯 項目概述

這是一個先進的分子優化系統，結合了：
- **蒙地卡羅樹搜索 (MCTS)**: 智能探索分子空間
- **大語言模型 (LLM)**: 生成化學合理的分子變體
- **Oracle 評估**: 使用 GuacaMol 基準進行分子評分
- **軌跡感知優化**: LLM 根據歷史成功模式選擇動作
- **統一動作規則**: 所有動作設定集中於 `rdkit_action_rulebook.json`，由 `actions/action.py` 單一管理

## 🚀 快速開始

### 環境設置
```bash
# 創建虛擬環境
conda create -n ga_llm python=3.10
conda activate ga_llm

# 安裝依賴
pip install -r requirements.txt

# 或使用 conda 環境文件
conda env create -f environment.yml
```

### 配置設置
1. 複製並編輯配置文件：
```bash
cp config/settings.yml.example config/settings.yml
```

2. 設置 API 金鑰：
```yaml
llm:
  provider: "github"  # 或 "cerebras"
  github_api_key: "your_github_token"
  # 或
  api_key: "your_cerebras_key"
```

### 運行優化
```bash
# 使用默認配置
python main.py

# 指定 LLM 提供商
python main.py --provider github
python main.py --provider cerebras
```

## 📊 系統架構

### 核心組件
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   LLM Generator │    │  MCTS Engine    │    │ Oracle Evaluator│
│                 │    │                 │    │                 │
│ • GitHub Models │    │ • Node管理      │    │ • GuacaMol      │
│ • Cerebras API  │    │ • UCT選擇       │    │ • 分子評分      │
│ • SMILES生成    │    │ • 樹擴展        │    │ • 早停檢查      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │ Workflow Graph  │
                    │                 │
                    │ • LangGraph     │
                    │ • 狀態管理      │
                    │ • 流程控制      │
                    └─────────────────┘
```

### 動作系統（Action Engine）
- **單一來源**：所有動作規則集中於 `actions/rdkit_action_rulebook.json`，每個 action 皆為自足物件，包含執行所需所有資訊。
- **唯一介面**：所有動作產生、混合、執行皆透過 `actions/action.py`，不再分 coarse/fine。
- **維護簡單**：新增/修改動作只需編輯 `rdkit_action_rulebook.json`，不需更動多個檔案。

#### 如何新增/修改動作？
1. 編輯 `actions/rdkit_action_rulebook.json`，新增一個 action 物件：
```json
{
  "add_carboxyl_group": {
    "type": "add_functional_group",
    "description": "Adds a carboxyl group (-COOH) to the molecule.",
    "params": {
      "group_smiles": "C(=O)O"
    }
  }
}
```
2. 透過 `actions/action.py` 介面存取：
```python
from actions import action

# 取得所有可用動作（給 LLM）
llm_actions = action.prepare_actions_for_llm()
for act in llm_actions:
    print(act['name'], act['description'])

# 執行某個動作
result = action.execute_action("Nc1ccccc1", "add_carboxyl_group")
print(result)
```

## 🎛️ 配置選項

### LLM 配置
```yaml
llm:
  provider: "github"           # "github" 或 "cerebras"
  model_name: "qwen-3-32b"    # 模型名稱
  temperature: 0.2             # 創造性控制
  max_completion_tokens: 4000  # 最大輸出長度
  max_smiles_length: 100       # SMILES 長度限制
```

### 工作流程配置
```yaml
workflow:
  early_stop_threshold: 0.8    # 早停分數閾值
  max_smiles_length: 100       # 分子複雜度限制
  recursion_limit: 1000        # 遞歸深度限制
  max_iterations: 1000         # 最大迭代次數
```

### MCTS 配置
```yaml
mcts:
  c_uct: 1.414                 # UCT 探索常數
  prune_interval: 50           # 樹修剪間隔
  max_tree_size: 1000          # 最大樹大小
  keep_top_k: 100              # 修剪時保留的節點數
```

## 📈 性能優化

### 分數追蹤改進
- **Oracle 分數**: 直接來自 GuacaMol 評估的原始分數
- **累積分數**: 用於 UCT 計算的歷史累積
- **優勢分數**: 相對於批次基線的相對優勢

### 早停機制
```python
# 現在正確基於 Oracle 分數
if engine.best.oracle_score >= early_stop_threshold:
    return {"best": engine.best, "reason": f"Early stop - high score ({engine.best.oracle_score:.4f})"}
```

### 最佳分子選擇
```python
# 使用正確的 Oracle 分數比較
if not engine.best or score > getattr(engine.best, 'oracle_score', 0.0):
    engine.best = node
```

## 🔍 輸出解讀

### 最終結果
```
=== FINAL RESULT ===
BEST SMILES: c1ccc(Cl)cc1
BEST SCORE:  0.8245        # 主要顯示分數
ORACLE SCORE: 0.8245       # Oracle 原始評分
TOTAL SCORE:  2.4735       # 累積分數(如果不同)
VISITS:      15             # 訪問次數
ORACLE CALLS REMAINING: 234 # 剩餘評估次數
```

### 調試信息
- `oracle_score`: GuacaMol 的直接評分，用於最終決策
- `total_score`: MCTS 中的累積分數，用於樹搜索
- `avg_score`: 平均分數 (total_score / visits)

## 🛠️ 開發和調試

### 動作系統（Action Engine）重點
- **單一來源**：所有動作規則集中於 `actions/rdkit_action_rulebook.json`。
- **唯一介面**：所有動作產生、混合、執行皆透過 `actions/action.py`。
- **如何擴充**：只需編輯 JSON 檔案並（如有新類型）擴充 action.py 的 dispatcher。

#### 動作系統簡易範例
```python
from actions import action

# 取得所有可用動作
llm_actions = action.prepare_actions_for_llm()
for act in llm_actions:
    print(act['name'], act['description'])

# 執行某個動作
result = action.execute_action("Nc1ccccc1", "swap_to_furan")
print(result)
```

### 日誌級別
```python
# 在 main.py 中設置
logging.basicConfig(level=logging.DEBUG)  # 詳細調試
logging.basicConfig(level=logging.INFO)   # 一般信息
```

### 關鍵調試點
1. **分數一致性**: 檢查 `oracle_score` vs `total_score`
2. **早停觸發**: 監控早停條件的觸發
3. **LLM 生成**: 驗證 SMILES 生成質量
4. **樹擴展**: 確認子節點正確創建
5. **動作規則維護**: 所有動作只需維護一份 JSON

### 測試工具
```bash
# 測試 SMILES token 系統
python test_smiles_token_system.py

# 測試 LLM 引導選擇
python test_llm_guided_selection.py

# 檢查配置
python check_config.py
```

## 📋 常見問題

### Q: 如何新增/修改動作？
A: 只需編輯 `actions/rdkit_action_rulebook.json`，不需更動多個檔案。

### Q: 動作系統是否還有 coarse/fine 之分？
A: 沒有，所有動作都統一管理於單一 JSON 檔與 action.py。

### Q: 如何在程式中取得所有動作？
A: 使用 `action.prepare_actions_for_llm()`。

### Q: 如何執行一個動作？
A: 使用 `action.execute_action(parent_smiles, action_name)`。

