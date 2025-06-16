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
    
    try:
        # 使用新的異步運行函數
        from graph.workflow_graph import run_workflow
        result = await run_workflow(init_state)
        
        if result and isinstance(result, dict):
            # 從結果中提取最終狀態和結果
            final_state = None
            final_result = None
            
            for node_name, node_state in result.items():
                if hasattr(node_state, 'result') and node_state.result:
                    final_result = node_state.result
                    final_state = node_state
                    break
                elif isinstance(node_state, AgentState):
                    final_state = node_state
            
            if final_result:
                best_result = final_result.get("best")
                reason = final_result.get("reason", "Unknown")
                logger.info(f"[Workflow] Completed: {reason}")
                
                if best_result and hasattr(best_result, 'smiles'):
                    logger.info(f"[Workflow] Best SMILES: {best_result.smiles}")
                    logger.info(f"[Workflow] Best score: {getattr(best_result, 'total_score', 'N/A')}")
                    logger.info(f"[Workflow] Depth: {getattr(best_result, 'depth', 'N/A')}")
                    logger.info(f"[Workflow] Visits: {getattr(best_result, 'visits', 'N/A')}")
                else:
                    logger.warning("[Workflow] Best result found but no valid molecule")
            else:
                logger.warning("[Workflow] No final result found")
                
            # 輸出一些統計信息
            if final_state:
                logger.info(f"[Workflow] Final depth reached: {getattr(final_state, 'depth', 'Unknown')}")
                logger.info(f"[Workflow] Final parent: {getattr(final_state, 'parent_smiles', 'Unknown')}")
        else:
            logger.warning("[Workflow] No result returned")
        
    except KeyboardInterrupt:
        logger.info("[Workflow] Interrupted by user")
    except Exception as e:
        logger.error(f"[Workflow] Error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        
    # 4. 輸出最終結果
    if best_result:
        print(f"\n=== FINAL RESULT ===")
        print(f"BEST SMILES: {best_result.smiles}")
        print(f"BEST SCORE:  {best_result.mean_score:.4f}")
        print(f"VISITS:      {best_result.visits}")
        print(f"ORACLE CALLS REMAINING: {oracle.calls_left}")
    else:
        print(f"\n=== NO RESULT FOUND ===")
        print(f"ORACLE CALLS REMAINING: {oracle.calls_left}")
    
    # 5. 清理資源
    oracle.close()
    logger.info("=== GA LLM World Model Finished ===")


if __name__ == "__main__":
    asyncio.run(run())