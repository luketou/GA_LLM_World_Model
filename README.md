🗂 Molecular-Agent v1.0

🌐 Workflow

1️⃣ 啟動階段 (main.py)
    •	讀取 settings.yaml
    •	初始化：
        ○	GuacaMolOracle (分子評分引擎)
        ○	KGStore (知識圖譜存儲，Neo4j/MockKG)
        ○	LLMGenerator (Cerebras LLM，模型：llama-3.3-70b)
        ○	MCTSEngine (蒙特卡羅樹搜索引擎)
    •	Prescan：
        ○	對 smi_file 全列表評分 → 取最低分 SMILES 作為 seed
        ○	初始化 AgentState(seed) → 啟動 graph_app

⸻

2️⃣ LangGraph 流程 (graph/workflow_graph.py)

| Node | 功能 | LLM 介入 |
|------|------|----------|
| **Generate** | 🧠 **LLM 智慧決策**：分析當前分支歷史，從 Coarse/Fine Action 庫中選擇 K 個最佳動作 | ✅ Action 選擇 |
| **LLM** | 🔬 **分子生成**：LLMGenerator.generate_batch() → 依選定動作產生子 SMILES | ✅ 分子合成 |
| **Oracle (async)** | 📊 **分子評分**：GuacaMolOracle.score_async() → 回傳適應度分數 | ❌ |
| **Adv** | 📈 **學習更新**：計算 baseline + advantages，更新 KG + MCTS 統計 | ❌ |
| **Decide** | 🌳 **樹搜索**：UCT 選子節點；達終態或配額耗盡時結束 | ❌ |

⸻

3️⃣ LLM 導引的動作選擇 (LLM-Guided Action Selection)
取代原有的規則式/隨機抽樣，新的 `Generate` 節點採用 **雙重 LLM 智慧決策**：

### 🔍 歷史分析與動作選擇流程
1. **歷史回溯** (`kg.get_branch_history()`)
   - 從知識圖譜中提取當前分子節點回溯至根節點的完整路徑
   - 包含：父代分子 SMILES + 執行的 Action + 獲得的適應度分數

2. **情境化提示** (`llm_gen.select_actions()`)
   ```
   You are an expert medicinal chemist...
   Path history: [Root] → [Action1] → [Molecule1, Score1] → ...
   Current molecule: {parent_smiles} (Score: {current_score})
   Available actions: [44 Coarse + 20 Fine actions]
   Select top K most promising actions...
   ```

3. **LLM 決策** (Cerebras llama-3.3-70b)
   - 扮演專家化學家，分析歷史趨勢
   - 從 64 個候選動作中選出最具潛力的 K 個
   - 回傳 JSON 格式：`["action_name_1", "action_name_2", ...]`

4. **分子執行** (`llm_gen.generate_batch()`)
   - 將選定的 Action 傳遞給分子生成模組
   - LLM 根據動作指令合成新的 SMILES 結構

### 🔄 容錯機制
- **根節點處理**：ROOT 節點直接使用 coarse action 隨機抽樣
- **歷史不足時**：
  - 歷史長度 < 2：自動退回規則式抽樣
  - 第一層節點：使用規則式方法建立初始歷史
  - depth < 5: Coarse actions, depth ≥ 5: Fine actions
- **LLM 失敗時**：Exception handling → 規則式備援
- **空結果處理**：確保始終返回有效的動作列表
- **循環檢測**：防止歷史回溯時出現無限循環

⸻

4️⃣ 資料儲存與追蹤
    •	**Oracle 評分日誌**：score_log.csv (限額 500 次調用)
    •	**知識圖譜**：分子節點/Action 邊/統計數據 → Neo4j KG (或 MockKG)
    •	**工作流檢查點**：LangGraph checkpoint → .lg_ckpt.db
    •	**全程追蹤**：LangSmith 節點級 tracing (project: "world model agent")

⸻

### 🧪 Action Library (總計 64 種動作)

#### 🔧 Coarse Actions (44 種)
**單環芳香族系統 (12種)**
- 基本芳環：苯環、吡啶、嘧啶、吡嗪、噠嗪
- 五員雜環：噻吩、呋喃、吡咯、咪唑、噻唑、噁唑、吡唑

**稠環芳香族系統 (10種)**
- 萘系：萘、喹啉、異喹啉、喹唑啉、喹噁啉
- 雜稠環：苯並噻吩、苯並呋喃、吲哚、苯並咪唑、嘌呤

**飽和環系統 (8種)**
- 烷環：環己烷、環戊烷
- 含氮飽和環：哌啶、哌嗪、嗎啉、吡咯烷
- 含氧飽和環：四氫呋喃、四氫吡喃

**藥物常見骨架 (5種)**
- 特殊結構：聯苯、嘧啶酮、三嗪、三唑、四唑

**官能基添加 (3種)**
- 基本基團：羥基 (-OH)、氨基 (-NH2)、羧基 (-COOH)

**結構調整 (6種)**
- 分子修飾：減少分子量、雜原子交換、成環反應、開環反應、鏈延長、鏈縮短

#### 🎯 Fine Actions (20 種)
**小分子基團 (6種)**
- 烷基：甲基、乙基、丙基、異丙基、丁基、叔丁基

**官能基 (10種)**
- 含氧：羥基、羰基、酯基、醚鍵、醯胺基
- 含氮：氨基、腈基、硝基
- 含硫：磺醯基

**鹵素取代 (4種)**
- 鹵素：氟 (-F)、氯 (-Cl)、溴 (-Br)、碘 (-I)

⸻

### ⚙️ 核心參數 (config/settings.yml)
```yaml
# MCTS 搜索參數
K_init: 30          # 每節點生成分子數量
max_depth: 50       # 搜索樹最大深度
c_uct: 1.4         # UCT 探索常數

# Oracle 限制
oracle_limit: 500   # 最大評分次數
oracle_rate: 600_per_hour

# LLM 配置
llm:
  provider: cerebras
  model_name: llama-3.3-70b
  temperature: 0.2
  max_completion_tokens: 2048

# 工作流程
workflow:
  max_iterations: 1000
  early_stop_threshold: 0.9
  max_smiles_length: 100
```

⸻

### 🎯 核心創新點
1. **雙重 LLM 架構**：Action 選擇 + 分子生成，兩階段智慧決策
2. **歷史感知探索**：從過往成功/失敗路徑中學習最佳策略
3. **動態動作庫**：64 種精心設計的化學轉換，涵蓋粗粒度到微調
4. **容錯備援機制**：LLM 失敗時自動切換至規則式方法
5. **全程可追蹤**：LangSmith + Neo4j + Checkpoint 多層次記錄




