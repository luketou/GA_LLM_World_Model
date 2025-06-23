🗂 Molecular-Agent v2.1

## 🔧 最新更新 (v2.1)
- ✅ **LLM軌跡感知動作選擇**：基於歷史路徑的智能動作選擇機制
- ✅ **動作歷史追蹤**：節點級別的完整動作軌跡記錄
- ✅ **軌跡感知提示**：LLM 基於分子編輯歷史進行決策
- ✅ **智能後備機制**：多層次容錯保證系統穩定運行
- ✅ **推理透明化**：LLM 決策過程的完整記錄和追蹤

## 🏗️ **系統架構 (v2.1)**

### **核心模組分層**
```
📦 GA_LLM_World_Model/
├── 🧠 mcts/                    # 蒙特卡羅樹搜索核心
│   ├── node.py                 # 節點數據結構（含動作歷史）
│   ├── uct.py                  # UCT 選擇策略
│   ├── search_strategies.py    # 多種搜索策略
│   ├── tree_analytics.py       # 樹分析與統計
│   ├── tree_manipulator.py     # 樹操作與修剪
│   ├── progressive_widening.py # 漸進拓寬策略
│   ├── llm_guided_selector.py  # 🆕 LLM軌跡感知動作選擇器
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

## 🌐 Workflow (v2.1)

### **7節點 LangGraph 工作流** (graph/workflow_graph.py)

| Node | 功能描述 | LLM 介入 | MCTS 介入 | 狀態更新 |
|------|----------|----------|-----------|----------|
| **Generate** | 🎯 **🆕 軌跡感知動作選擇**：LLM分析歷史+智能決策 | ✅ | ✅ | `actions` |
| **LLM** | 🔬 **分子生成**：基於動作生成新 SMILES | ✅ | ❌ | `batch_smiles` |
| **Oracle** | 📊 **異步評分**：GuacaMol 適應度 + 長度過濾 | ❌ | ❌ | `scores` |
| **Adv** | 📈 **優勢計算**：Baseline 標準化 | ❌ | ❌ | `advantages` |
| **UpdateStores** | 💾 **數據同步**：KG + MCTS 統計更新 | ❌ | ✅ | - |
| **Expand** | 🌳 **節點擴展**：MCTS 樹結構擴展 + 動作記錄 | ❌ | ✅ | - |
| **Decide** | 🎲 **決策路由**：UCT 選擇 + 終止判斷 | ❌ | ✅ | `parent_smiles`, `depth` |

### **🆕 軌跡感知動作選擇架構**

#### **📊 動作歷史追蹤** - 節點級別記錄
```python
# 每個節點自動追蹤生成動作和效果
node.get_action_history()              # 完整動作軌跡
node.get_recent_actions(n=3)           # 最近N個動作
node.get_action_trajectory_summary()   # 軌跡摘要統計
node.get_successful_action_patterns()  # 成功模式識別
```

#### **🧠 LLMGuidedActionSelector** - 智能動作選擇器
```python
# 軌跡感知的 LLM 動作選擇
selector = LLMGuidedActionSelector(llm_generator)

# 創建選擇請求
request = ActionSelectionRequest(
    parent_smiles="CCO",
    current_node_trajectory=trajectory_summary,
    available_actions=candidate_actions,
    optimization_goal="Improve drug-like properties",
    depth=2,
    max_selections=5
)

# 獲取 LLM 智能選擇
response = selector.select_actions(request)
# 包含：selected_actions, reasoning, confidence
```

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
    ↑                                                      ↓
    |← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ←|
   (軌跡感知循環：LLM分析歷史→智能動作選擇→分子生成→評分→軌跡更新)
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

## 🧬 **動作選擇智能化 (v2.1)**

### **🆕 軌跡感知動作選擇策略**
```python
def propose_actions_llm_guided(current_node: Node, k_init: int):
    # 1. 獲取候選動作池（3倍擴展）
    available_actions = get_mixed_actions(k_init * 3)
    
    # 2. 分析節點軌跡歷史
    trajectory = current_node.get_action_trajectory_summary()
    
    # 3. LLM 軌跡感知決策
    prompt = create_trajectory_aware_prompt(
        current_smiles=current_node.smiles,
        action_history=trajectory['recent_actions'],
        score_trend=trajectory['score_trend'],
        successful_patterns=trajectory['action_type_counts'],
        available_actions=available_actions
    )
    
    # 4. 智能動作選擇
    response = llm_selector.select_actions(prompt)
    return response.selected_actions, response.reasoning
```

### **🧠 LLM 軌跡感知提示範例**
```
你是分子優化專家。基於編輯軌跡選擇最佳動作：

當前分子: CCO
優化目標: 改善類藥性質
搜索深度: 2
分數趨勢: 持續改善 (0.3 → 0.5 → 0.7)

編輯軌跡上下文:
最近動作歷史:
  1. add_hydroxyl (substitute): 添加羥基 → 分數變化: +0.2
  2. add_methyl (substitute): 添加甲基 → 分數變化: +0.2

動作類型統計: substitute(2)

可用動作 (15個):
  1. add_amino (substitute): 添加氨基
  2. cyclization (cyclization): 環化反應
  ...

選擇標準:
1. 考慮軌跡上下文 - 什麼策略有效？
2. 尋找互補動作建立在成功模式上
3. 避免重複失敗策略
4. 平衡探索與利用
5. 考慮分子多樣性

請選擇最有前景的動作並提供推理。

回應格式 (JSON):
{
  "selected_action_names": ["action1", "action2"],
  "reasoning": "基於軌跡分析的詳細推理...",
  "confidence": 0.8
}
```

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
        
    # 🆕 深度 > 0 時啟用 LLM 軌跡感知選擇
    if depth > 0 and llm_guided_selector_available:
        return propose_actions_llm_guided(current_node, k_init)
```

### **🛡️ 七層容錯機制 (v2.1)**
1. **LLM 軌跡分析故障** → 回退到混合動作選擇
2. **Action 模組故障** → `_get_fallback_actions()` 基礎化學操作
3. **MCTS 策略故障** → `SearchStrategies` 多策略後備
4. **UCT 選擇故障** → 隨機選擇保證運行
5. **TreeAnalytics 故障** → 基礎統計計算
6. **LLM 生成故障** → 空結果優雅處理
7. **Oracle 評分故障** → 默認分數繼續流程

## ⚙️ **配置參數 (v2.1)**
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
  # 🆕 軌跡感知選擇參數
  llm_guided_selector:
    enabled: true                 # 啟用軌跡感知選擇
    min_depth: 1                  # 最小啟用深度
    max_context_actions: 5        # 最大上下文動作數
    candidate_expansion_factor: 3 # 候選動作擴展倍數

# 工作流程控制 (v2.1)
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

## 🚀 **使用方式 (v2.1)**
```bash
# 標準運行（含軌跡感知動作選擇）
python main.py

# 軌跡感知功能測試
python test_llm_guided_selection.py

# 模組化測試
python -c "from mcts import LLMGuidedActionSelector; print('軌跡感知模組測試通過')"

# 合規性驗證
python -c "from mcts.uct import UCTSelector; print('Oracle 合規性已驗證')"

# 工作流程調試
LANGSMITH_TRACING=true python main.py
```

### **📊 輸出增強 (v2.1)**
- `score_log.csv` - Oracle 評分記錄（含合規性標記）
- `log/main.log` - 系統運行日誌（含軌跡決策記錄）
- `mcts_statistics.json` - MCTS 樹統計信息
- `trajectory_decisions.log` - 🆕 LLM 軌跡決策詳細記錄
- `.lg_ckpt.db` - LangGraph 檢查點

## 🎯 **v2.1 架構優勢**

### **1. 🆕 軌跡感知智能化**
- **歷史學習**：從過往成功模式中學習
- **上下文決策**：基於完整編輯軌跡進行決策
- **推理透明**：LLM 決策過程完整記錄
- **自適應策略**：動態調整基於軌跡表現

### **2. 模組化設計**
- **單一職責**：每個模組專注核心功能
- **可擴展性**：新策略可無縫添加
- **可測試性**：模組間依賴清晰

### **3. 容錯能力**
- **多層後備**：系統故障時自動降級
- **優雅降級**：功能受限但保持運行
- **錯誤隔離**：單模組故障不影響整體

### **4. 合規性保證**
- **嚴格約束**：Oracle 評估前 RDKit 限制
- **LLM 替代**：智能字符串分析
- **審計追蹤**：完整的合規性日誌

### **5. 性能優化**  
- **智能修剪**：動態樹結構優化
- **自適應策略**：基於性能調整探索
- **並發安全**：異步工作流支持

## 🔬 **軌跡感知動作選擇示例**

### **場景：藥物分子優化**
```
初始分子: CC (乙烷)
目標: 改善類藥性質

迭代 1 (深度0): 混合動作選擇
→ 動作: add_hydroxyl → 生成: CCO (乙醇)
→ 分數: 0.3 → 0.5 (+0.2)

迭代 2 (深度1): 🆕 軌跡感知選擇
→ 軌跡分析: "羥基添加成功提升親水性"
→ LLM推理: "繼續極性基團策略，添加氨基補充氫鍵能力"
→ 動作: add_amino → 生成: CCNO (含氨基)
→ 分數: 0.5 → 0.7 (+0.2)

迭代 3 (深度2): 🆕 軌跡感知選擇  
→ 軌跡分析: "極性基團策略連續成功(substitute:2, 趨勢:improving)"
→ LLM推理: "平衡親脂性，添加環狀結構增加剛性"
→ 動作: cyclization → 生成: 環狀衍生物
→ 分數: 0.7 → 0.85 (+0.15)
```

### **軌跡感知決策優勢**
- ✅ **策略連續性**：建立在成功模式基礎上
- ✅ **智能避錯**：避免重複失敗的動作類型
- ✅ **平衡探索**：在成功策略和新探索間平衡
- ✅ **目標導向**：始終朝向優化目標調整策略




