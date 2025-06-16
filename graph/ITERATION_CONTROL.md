## 迭代控制機制說明

### 控制迭代的關鍵程式碼片段

#### 1. 主要控制器：`decide()` 函數 (graph/workflow_graph.py)

這是控制整個搜索過程的核心函數，決定何時停止迭代：

**終止條件優先級：**

1. **Oracle 預算用完** (最高優先級)
   ```python
   if oracle.calls_left <= 0:
       print("[Debug] Terminating: Oracle budget exhausted")
       state.result = {"best": engine.best, "reason": "Oracle budget exhausted"}
       return state
   ```

2. **早停條件：找到高分分子**
   ```python
   early_stop_threshold = cfg.get("workflow", {}).get("early_stop_threshold", 0.8)
   if (hasattr(engine, 'best') and engine.best and 
       hasattr(engine.best, 'total_score') and 
       engine.best.total_score >= early_stop_threshold):
       print(f"[Debug] Early stopping: Found high-score molecule")
       state.result = {"best": engine.best, "reason": f"Early stop - high score"}
       return state
   ```

3. **達到最大深度**
   ```python
   if state.depth >= cfg["max_depth"]:
       print("[Debug] Terminating: Max depth reached")
       state.result = {"best": engine.best, "reason": "Max depth reached"}
       return state
   ```

4. **分子複雜度限制**
   ```python
   if len(state.parent_smiles) > 200:  # SMILES 長度限制
       print("[Debug] Terminating: Molecule too complex")
       state.result = {"best": engine.best, "reason": "Molecule too complex"}
       return state
   ```

5. **無更多節點可探索**
   ```python
   nxt = engine.select_child(state.parent_smiles)
   if not nxt:
       print("[Debug] Terminating: No more children to explore")
       state.result = {"best": engine.best, "reason": "No more children to explore"}
       return state
   ```

#### 2. 工作流程執行器：`run_workflow()` 函數

控制整個工作流程的執行：

**主要檢查：**
```python
# 主要檢查：Oracle 預算
if oracle.calls_left <= 0:
    logger.info(f"Oracle budget exhausted after {iteration} iterations")
    break
    
# 次要檢查：防止無限循環的安全機制
if iteration > max_iterations:
    logger.warning(f"Reached maximum iterations ({max_iterations}), stopping workflow")
    break
```

#### 3. 配置文件控制 (config/settings.yml)

```yaml
# Oracle 設定
oracle_limit: 500                        # 主要控制：Oracle 預算

# 工作流程設定
workflow:
  recursion_limit: 200                   # LangGraph 遞迴限制
  max_iterations: 1000                   # 安全機制：最大迭代次數
  early_stop_threshold: 0.8              # 早停閾值：找到此分數以上的分子就停止

# MCTS 參數
max_depth: 5                             # 搜索深度限制
```

### 修改後的行為

1. **主要終止條件**：Oracle 預算用完 (500 次調用)
2. **早停機制**：找到分數 ≥ 0.8 的分子立即停止
3. **安全機制**：
   - 最大迭代次數 (1000) - 防止無限循環
   - 最大深度 (5) - 避免過深搜索
   - 分子複雜度限制 (SMILES 長度 > 200) - 避免過於複雜的分子

4. **移除的限制**：
   - 硬編碼的 20 次迭代限制 (已移除)

### 如何調整

1. **增加 Oracle 預算**：修改 `settings.yml` 中的 `oracle_limit`
2. **調整早停閾值**：修改 `settings.yml` 中的 `early_stop_threshold`
3. **調整安全限制**：修改 `settings.yml` 中的 `max_iterations`
4. **調整搜索深度**：修改 `settings.yml` 中的 `max_depth`

### 運行時監控

- 查看 `[Debug] Oracle calls left: X` 來監控剩餘預算
- 查看 `[Debug] Iteration X` 來監控迭代進度
- 查看終止原因在日誌中會明確顯示

現在程式會持續運行直到 Oracle 預算用完，而不是被任意的迭代限制所束縛。
