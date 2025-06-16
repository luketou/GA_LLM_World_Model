🗂 Molecular-Agent v1.0

🌐 Workflow

1️⃣ 啟動階段 (main.py)
	•	讀取 settings.yaml
	•	初始化：
	•	GuacaMolOracle
	•	KGStore
	•	MCTSEngine
	•	Prescan：
	•	對 smi_file 全列表評分 → 取最低分 SMILES 作為 seed
	•	初始化 AgentState(seed) → 啟動 graph_app

⸻

2️⃣ LangGraph 流程 (graph/workflow_graph.py)

Node	功能
Generate	MCTSEngine.propose_actions() → 產生 K 個 coarse/fine 動作
LLM	LLMGenerator.generate_batch() → 依動作產生子 SMILES
Oracle (async)	GuacaMolOracle.score_async() → 回傳分數
Adv	計算 baseline + advantages，更新 KG + MCTS
Decide	UCT 選子節點；達終態或配額耗盡時結束


⸻

3️⃣ 資料儲存與追蹤
	•	Oracle 成績寫入 score_log.csv（限額 500）
	•	分子 / Action / 統計寫入 Neo4j KG
	•	Checkpoint 寫入 .lg_ckpt.db
	•	LangSmith 全節點 tracing

⸻

### Action Types
#### Coarse Actions
- 單環芳香族系統（12種）
- 基本芳環：苯環、吡啶、嘧啶、吡嗪、噠嗪
- 五員雜環：噻吩、呋喃、吡咯、咪唑、噻唑、惡唑、吡唑
- 稠環芳香族系統（10種）
- 萘系：萘、喹啉、異喹啉、喹唑啉、喹噁啉
- 雜稠環：苯並噻吩、苯並呋喃、吲哚、苯並咪唑、嘌呤
- 飽和環系統（8種）
- 烷環：環己烷、環戊烷
- 含氮飽和環：哌啶、哌嗪、嗎啉、吡咯烷
- 含氧飽和環：四氫呋喃、四氫吡喃
- 藥物常見骨架（3種）
- 特殊結構：聯苯、嘧啶酮、三嗪、三唑、四唑
- 保留的操作（9種）
- 官能基添加：OH, NH2, COOH（3種）
- 結構調整：減少分子量、雜原子交換、成環/開環（6種）



📝 TODO List

優先	檔案	範圍 / 行號	待辦事項
🔴	oracle/guacamol_client.py	末尾	➊ TOTAL_LIMIT 達成時應優化錯誤處理訊息 / soft stop 機制➋ rate_cap 改讀 settings.yaml
🔴	llm/generator.py	generate_batch	➌ Prompt 還是 template 塊拼接，建議加 few-shot 或 chain-of-thought 提升生成品質
🔴	mcts/mcts_engine.py	propose_actions	➍ Fine-grained expand() 應加入智能篩選（如結構相似度、高分片段）
🔴	grpo/trainer.py	step()	➎ old_logp／new_logp 尚未實作（需 LLM logits 或對應 index 建立）
🟠	actions/fine_actions.py	全檔	➏ 需補上 *_groups.json，否則 fine-expand 出現空列表
🟠	kg/kg_store.py	create_action	➐ 建議 Action edge 加 epoch 欄位，方便日後可視化 / 時序查詢
🟠	graph/workflow_graph.py	decide()	➑ 回溯策略未實作，若連續 N 次 Advantage < 0，應回溯至祖先
🟢	utils/smiles_tools.py	新增	➒ 實作 valid_smiles()、canonicalize() 供 LLM 產出後篩選用
🟢	kg/kg_pruner.py	全檔	➓ 定期 pruning Job 建議提供 CLI python -m kg.kg_pruner
🟢	requirements.txt	末尾	⓫ 可加 py-cpuinfo 或系統檢測套件，利於記錄 run info


⸻

🚦 總結優先順序建議

優先順序	目標
必做	➊➋➌➍➎ → 確保流程能長跑到 500 次且結果穩定
建議做	➏➐➑ → 增強 KG / Fine-grained 操作 / 回溯策略
可優化	➒➓⓫ → 提升工程品質與操作便利性


