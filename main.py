"""
Main entry point for GA_LLM_World_Model
主要流程啟動腳本：
1. 讀取 settings.yaml
2. 初始化組件 (GuacaMolOracle, KGStore, MCTSEngine, LLMGenerator)
3. 呼叫 oracle.prescan_lowest() → 更新 seed_smiles
4. 建立初始 AgentState
5. 進入 graph_app.astream() 迴圈
6. 輸出最佳結果並終止
"""
import asyncio
import yaml
import pathlib
import logging
from graph.workflow_graph import graph_app, AgentState, oracle, cfg

# 設置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def prescan():
    """
    Prescan SMILES 檔案，找出最低分分子作為起始種子
    """
    try:
        seed, score = oracle.prescan_lowest(cfg["smi_file"])
        logger.info(f"[Prescan] Lowest-score seed: {seed} (score: {score:.4f})")
        return seed
    except Exception as e:
        logger.error(f"[Prescan] Error: {e}")
        # 使用預設種子
        default_seed = cfg.get("START_SMILES", "C1CCCCC1")
        logger.warning(f"[Prescan] Using default seed: {default_seed}")
        return default_seed


async def run():
    """
    主要執行流程
    """
    logger.info("=== GA LLM World Model Started ===")
    logger.info(f"Task: {cfg['TASK_NAME']}")
    logger.info(f"Max depth: {cfg['max_depth']}")
    logger.info(f"Oracle limit: {cfg['oracle_limit']}")
    
    # 1. Prescan 獲取起始分子
    seed = prescan()
    
    # 2. 創建初始狀態
    init_state = AgentState(parent_smiles=seed, depth=0)
    logger.info(f"[Workflow] Starting with seed: {seed}")
    
    # 3. 進入 LangGraph 工作流程
    best_result = None
    iteration = 0
    
    try:
        async for state in graph_app.astream(init_state):
            iteration += 1
            logger.info(f"[Workflow] Iteration {iteration}")
            
            # 檢查是否有結果
            if hasattr(state, 'result') and state.result:
                best_result = state.result.get("best")
                if best_result:
                    logger.info(f"[Result] Found best molecule: {best_result.smiles}")
                    logger.info(f"[Result] Score: {best_result.mean_score:.4f}")
                    logger.info(f"[Result] Visits: {best_result.visits}")
                    break
            
            # 檢查 Oracle 配額
            if oracle.calls_left <= 0:
                logger.warning("[Workflow] Oracle call limit exhausted")
                break
                
            # 防止無限循環
            if iteration > 1000:
                logger.warning("[Workflow] Maximum iterations reached")
                break
                
    except KeyboardInterrupt:
        logger.info("[Workflow] Interrupted by user")
    except Exception as e:
        logger.error(f"[Workflow] Error: {e}")
        
    # 4. 輸出最終結果
    if best_result:
        print(f"\n=== FINAL RESULT ===")
        print(f"BEST SMILES: {best_result.smiles}")
        print(f"BEST SCORE:  {best_result.mean_score:.4f}")
        print(f"VISITS:      {best_result.visits}")
        print(f"ITERATIONS:  {iteration}")
        print(f"ORACLE CALLS REMAINING: {oracle.calls_left}")
    else:
        print(f"\n=== NO RESULT FOUND ===")
        print(f"ITERATIONS:  {iteration}")
        print(f"ORACLE CALLS REMAINING: {oracle.calls_left}")
    
    # 5. 清理資源
    oracle.close()
    logger.info("=== GA LLM World Model Finished ===")


if __name__ == "__main__":
    asyncio.run(run())