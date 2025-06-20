🗂 Molecular-Agent v2.0

## 🔧 最新更新 (v2.0)
- ✅ **重大架構重構**：MCTS 模組化設計，職責分離
- ✅ **新增模組**：`SearchStrategies`, `TreeAnalytics`, `TreeManipulator`
- ✅ **工作流程擴展**：新增 `Expand` 節點，優化節點擴展邏輯
- ✅ **合規性強化**：嚴格的 Oracle 評估前 RDKit 限制
- ✅ **容錯機制**：多層後備策略確保系統穩定運行

## 🏗️ **系統架構 (v2.0)**

### **核心模組分層**
```
📦 GA_LLM_World_Model/
├── 🧠 mcts/                    # 蒙特卡羅樹搜索核心
│   ├── node.py                 # 節點數據結構（純數據）
│   ├── uct.py                  # UCT 選擇策略
│   ├── search_strategies.py    # 多種搜索策略
│   ├── tree_analytics.py       # 樹分析與統計
│   ├── tree_manipulator.py     # 樹操作與修剪
│   ├── progressive_widening.py # 漸進拓寬策略
│   └── mcts_engine.py          # 主引擎協調器
├── 🎬 graph/                   # LangGraph 工作流
│   └── workflow_graph.py       # 7節點異步工作流
├── 🧪 actions/                 # 動作庫
│   ├── fine_actions.py         # 精細動作（20種）
│   └── coarse_actions.py       # 粗粒度動作（44種）
├── 🤖 llm/                     # LLM 集成
│   └── generator.py            # Cerebras Qwen-32B
├── 📊 oracle/                  # 評分引擎
│   └── guacamol_client.py      # GuacaMol 異步客戶端
└── 💾 kg/                      # 知識圖譜
    └── kg_store.py             # Neo4j/MockKG 雙模式
```

### **🔒 Oracle 合規性約束**
**嚴格限制**：在 Oracle 評估前，**僅允許**分子量計算
- ✅ **允許**：`rdMolDescriptors.CalcExactMolWt()` 
- ❌ **禁止**：環數計算、拓撲指數、任何其他 RDKit 屬性
- 🧠 **替代方案**：LLM 驅動的字符串分析和模式識別

## 🌐 Workflow (v2.0)

### **7節點 LangGraph 工作流** (graph/workflow_graph.py)

| Node | 功能描述 | LLM 介入 | MCTS 介入 | 狀態更新 |
|------|----------|----------|-----------|----------|
| **Generate** | 🎯 **智慧動作選擇**：MCTS 歷史分析 + 深度適應策略 | ❌ | ✅ | `actions` |
| **LLM** | 🔬 **分子生成**：基於動作生成新 SMILES | ✅ | ❌ | `batch_smiles` |
| **Oracle** | 📊 **異步評分**：GuacaMol 適應度 + 長度過濾 | ❌ | ❌ | `scores` |
| **Adv** | 📈 **優勢計算**：Baseline 標準化 | ❌ | ❌ | `advantages` |
| **UpdateStores** | 💾 **數據同步**：KG + MCTS 統計更新 | ❌ | ✅ | - |
| **Expand** | 🌳 **節點擴展**：MCTS 樹結構擴展 | ❌ | ✅ | - |
| **Decide** | 🎲 **決策路由**：UCT 選擇 + 終止判斷 | ❌ | ✅ | `parent_smiles`, `depth` |

### **MCTS 模組化架構**

#### **📊 TreeAnalytics** - 樹分析器
```python
TreeAnalytics.get_tree_statistics(root_node)     # 完整樹統計
TreeAnalytics.get_subtree_best_node(root_node)   # 最佳節點搜索
TreeAnalytics.calculate_tree_depth(root_node)    # 深度計算
TreeAnalytics.get_path_statistics(root_node)     # 路徑分析
```

#### **🔍 SearchStrategies** - 搜索策略集合
```python
SearchStrategies.select_best_child_by_score()    # 分數導向選擇
SearchStrategies.select_most_visited_child()     # 訪問頻次選擇  
SearchStrategies.select_balanced_child()         # 平衡探索-利用
SearchStrategies.select_least_visited_child()    # 探索導向選擇
```

#### **🛠️ TreeManipulator** - 樹操作器
```python
TreeManipulator.prune_children(node, keep_k)           # 子節點修剪
TreeManipulator.prune_tree_recursive(root, keep_k)     # 遞歸樹修剪
TreeManipulator.remove_low_performing_subtrees()       # 低性能清理
TreeManipulator.balance_tree(root, max_variance)       # 樹平衡優化
```

#### **🎯 UCTSelector** - UCT 策略 (嚴格合規)
```python
UCTSelector.select_best_child(parent)            # 標準 UCT 選擇
UCTSelector.calculate_uct_score(child, parent)   # UCT 分數計算
# 多樣性獎勵：僅使用分子量 + LLM 字符串分析
```

### **🔄 工作流程路由邏輯**
```python
Generate → LLM → Oracle → Adv → UpdateStores → Expand → Decide
                                                              ↓
                                 ← ← ← ← ← ← Generate ← ← ← ← ←
                                        (繼續探索)
                                                              ↓
                                                            END
                                                      (終止條件)
```

### **⛔ 終止條件 (優先級排序)**
1. **Oracle 預算耗盡** (`oracle.calls_left <= 0`) - 主要終止條件
2. **高分早停** (`score >= early_stop_threshold`) - 成功終止
3. **最大深度** (`depth >= max_depth`) - 探索限制
4. **分子複雜度** (`len(smiles) > max_smiles_length`) - 安全限制
5. **無可探索節點** - 自然終止

## 🧬 **動作選擇智能化 (v2.0)**

### **深度適應策略** (`propose_mixed_actions`)
```python
def propose_mixed_actions(parent_smiles: str, depth: int, k_init: int):
    # 基於 MCTS 搜索深度動態調整策略
    if depth == 0:        # 根節點：廣度探索
        coarse_ratio = 0.7    # 70% 粗粒度骨架變換
    elif depth <= 2:      # 中層：平衡策略  
        coarse_ratio = 0.5    # 50% 平衡探索
    else:                 # 深層：精細調整
        coarse_ratio = 0.3    # 30% 官能基修飾
```

### **🛡️ 六層容錯機制**
1. **Action 模組故障** → `_get_fallback_actions()` 基礎化學操作
2. **MCTS 策略故障** → `SearchStrategies` 多策略後備
3. **UCT 選擇故障** → 隨機選擇保證運行
4. **TreeAnalytics 故障** → 基礎統計計算
5. **LLM 生成故障** → 空結果優雅處理
6. **Oracle 評分故障** → 默認分數繼續流程

## ⚙️ **配置參數 (v2.0)**
```yaml
# MCTS 模組化參數
mcts:
  c_uct: 1.414                    # UCT 探索常數
  progressive_widening:
    alpha: 0.5                    # 拓寬參數 alpha  
    beta: 2.0                     # 拓寬參數 beta
  tree_analytics:
    max_depth_variance: 2         # 樹平衡參數
  search_strategies:
    exploration_weight: 0.5       # 探索-利用平衡

# 工作流程控制 (v2.0)
workflow:
  max_iterations: 1000            # 最大迭代次數
  recursion_limit: 200            # LangGraph 遞歸限制
  early_stop_threshold: 0.8       # 早停分數閾值  
  max_smiles_length: 100          # SMILES 長度限制
  batch_size: 30                  # 批次大小

# Oracle 合規性
oracle:
  strict_compliance: true         # 啟用嚴格合規模式
  allowed_rdkit_functions:        # 允許的 RDKit 函數白名單
    - "CalcExactMolWt"           # 僅分子量計算
```

## 🚀 **使用方式 (v2.0)**
```bash
# 標準運行（使用所有新架構特性）
python main.py

# 模組化測試
python -c "from mcts import TreeAnalytics, SearchStrategies; print('架構測試通過')"

# 合規性驗證
python -c "from mcts.uct import UCTSelector; print('Oracle 合規性已驗證')"

# 工作流程調試
LANGSMITH_TRACING=true python main.py
```

### **📊 輸出增強**
- `score_log.csv` - Oracle 評分記錄（含合規性標記）
- `log/main.log` - 系統運行日誌（模組化日誌）
- `mcts_statistics.json` - MCTS 樹統計信息（新增）
- `.lg_ckpt.db` - LangGraph 檢查點

## 🎯 **v2.0 架構優勢**

### **1. 模組化設計**
- **單一職責**：每個模組專注核心功能
- **可擴展性**：新策略可無縫添加
- **可測試性**：模組間依賴清晰

### **2. 容錯能力**
- **多層後備**：系統故障時自動降級
- **優雅降級**：功能受限但保持運行
- **錯誤隔離**：單模組故障不影響整體

### **3. 合規性保證**
- **嚴格約束**：Oracle 評估前 RDKit 限制
- **LLM 替代**：智能字符串分析
- **審計追蹤**：完整的合規性日誌

### **4. 性能優化**  
- **智能修剪**：動態樹結構優化
- **自適應策略**：基於性能調整探索
- **並發安全**：異步工作流支持




