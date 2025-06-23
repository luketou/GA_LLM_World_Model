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
import os
import sys
import argparse

# Ensure project root is in Python path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from langsmith import traceable

# 早期初始化 LangSmith（在導入其他模組之前）
def init_langsmith():
    """初始化 LangSmith 追蹤，直接從 settings.yml 讀取配置"""
    import re
    
    # 自定義 YAML 載入器來處理環境變數
    def env_var_constructor(loader, node):
        """處理 !ENV 標籤，讀取環境變數"""
        value = loader.construct_scalar(node)
        # 支援 ${VAR_NAME} 格式
        pattern = re.compile(r'\$\{([^}]+)\}')
        match = pattern.search(value)
        if match:
            env_var = match.group(1)
            return os.environ.get(env_var, "")
        return value

    # 註冊自定義構造器
    yaml.SafeLoader.add_constructor('!ENV', env_var_constructor)
    
    config_path = pathlib.Path("config/settings.yml")
    if config_path.exists():
        cfg = yaml.safe_load(config_path.read_text())
        langsmith_config = cfg.get("langsmith", {})
        
        if langsmith_config.get("enabled", False):
            os.environ["LANGSMITH_TRACING"] = str(langsmith_config.get("tracing", True)).lower()
            os.environ["LANGSMITH_ENDPOINT"] = langsmith_config.get("endpoint", "https://api.smith.langchain.com")
            os.environ["LANGSMITH_API_KEY"] = langsmith_config.get("api_key", "")
            os.environ["LANGSMITH_PROJECT"] = langsmith_config.get("project", "world model agent")
            print(f"LangSmith tracing initialized for project: {langsmith_config.get('project', 'world model agent')}")

def parse_args():
    """解析命令行參數"""
    parser = argparse.ArgumentParser(description='GA LLM World Model - Molecular Optimization')
    parser.add_argument('--provider', 
                       choices=['github', 'cerebras'], 
                       default=None,
                       help='LLM provider to use (github or cerebras). Overrides settings.yml')
    return parser.parse_args()

def setup_environment(args):
    """設置環境變數和配置"""
    # 初始化 LangSmith
    init_langsmith()
    
    # 讀取配置檔案
    import re
    def env_var_constructor(loader, node):
        value = loader.construct_scalar(node)
        pattern = re.compile(r'\$\{([^}]+)\}')
        match = pattern.search(value)
        if match:
            env_var = match.group(1)
            return os.environ.get(env_var, "")
        return value

    yaml.SafeLoader.add_constructor('!ENV', env_var_constructor)
    
    config_path = pathlib.Path("config/settings.yml")
    if config_path.exists():
        cfg = yaml.safe_load(config_path.read_text())
        
        # 如果命令行指定了 provider，覆蓋配置檔案設定
        if args.provider:
            cfg["llm"]["provider"] = args.provider
            print(f"Using LLM provider: {args.provider} (from command line)")
        else:
            print(f"Using LLM provider: {cfg['llm'].get('provider', 'cerebras')} (from settings.yml)")
        
        # 根據選擇的 provider 設置相應的 API 金鑰
        llm_config = cfg.get("llm", {})
        provider = llm_config.get("provider", "cerebras")
        
        if provider == "github":
            # 設置 GitHub API 金鑰
            github_api_key = llm_config.get("github_api_key")
            if github_api_key and not os.environ.get("GITHUB_TOKEN"):
                os.environ["GITHUB_TOKEN"] = github_api_key
                print("GitHub API key loaded from settings.yml")
        elif provider == "cerebras":
            # 設置 Cerebras API 金鑰
            cerebras_api_key = llm_config.get("api_key")
            if cerebras_api_key and not os.environ.get("CEREBRAS_API_KEY"):
                os.environ["CEREBRAS_API_KEY"] = cerebras_api_key
                print("Cerebras API key loaded from settings.yml")
        
        return cfg
    else:
        raise FileNotFoundError("config/settings.yml not found")

# 設置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def prescan(cfg, oracle):
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


@traceable(
    name="WorldModelAgent::run",
    metadata={"task": "molecular_optimization", "agent_type": "ga_llm_world_model"}
)
async def run():
    """
    主要執行流程
    """
    # 解析命令行參數
    args = parse_args()
    
    # 設置環境和配置
    cfg = setup_environment(args)
    
    # 延遲導入以確保環境變數已設置
    from graph.workflow_graph import create_workflow_components, AgentState
    
    logger.info("=== GA LLM World Model Started ===")
    logger.info(f"Task: {cfg['TASK_NAME']}")
    logger.info(f"Max depth: {cfg['max_depth']}")
    logger.info(f"Oracle limit: {cfg['oracle_limit']}")
    logger.info(f"LLM Provider: {cfg['llm'].get('provider', 'cerebras')}")
    
    # 創建工作流程組件
    oracle, engine, graph_app = create_workflow_components(cfg)
    
    # 1. Prescan 獲取起始分子
    seed = prescan(cfg, oracle)
    
    # 2. 創建初始狀態
    init_state = AgentState(parent_smiles=seed, depth=0)
    logger.info(f"[Workflow] Starting with seed: {seed}")
    
    # 為 LangSmith 記錄輸入信息
    langsmith_inputs = {
        "task_name": cfg['TASK_NAME'],
        "seed_smiles": seed,
        "max_depth": cfg['max_depth'],
        "oracle_limit": cfg['oracle_limit'],
        "llm_provider": cfg['llm'].get('provider', 'cerebras')
    }
    
    # 3. 進入 LangGraph 工作流程
    best_result_node = None
    
    try:
        # 添加超時機制
        async def run_with_timeout():
            # 使用新的異步運行函數
            from graph.workflow_graph import run_workflow
            return await run_workflow(init_state)
        
        # 設置 30 分鐘超時
        workflow_output = await asyncio.wait_for(run_with_timeout(), timeout=1800)
        
        if workflow_output and isinstance(workflow_output, dict):
            # 從結果中提取最終狀態和結果
            final_state = None
            final_result = None
            
            for node_name, node_state in workflow_output.items():
                # 檢查當前節點狀態是否為字典，並且是否包含 'result' 鍵
                if isinstance(node_state, dict) and 'result' in node_state:
                    potential_result = node_state['result']
                    # 檢查 'result' 鍵的值是否為字典，並且是否包含 'best' 鍵
                    if isinstance(potential_result, dict) and 'best' in potential_result:
                        final_result = potential_result
                        final_state = node_state
                        break
                elif hasattr(node_state, 'result') and node_state.result:
                    final_result = node_state.result
                    final_state = node_state
                    break
                elif isinstance(node_state, AgentState):
                    final_state = node_state
            
            if final_result:
                best_result_node = final_result.get("best")
                reason = final_result.get("reason", "Unknown")
                logger.info(f"[Workflow] Completed: {reason}")
                
                if best_result_node and hasattr(best_result_node, 'smiles'):
                    logger.info(f"[Workflow] Best SMILES: {best_result_node.smiles}")
                    logger.info(f"[Workflow] Best score: {getattr(best_result_node, 'total_score', 'N/A')}")
                    logger.info(f"[Workflow] Depth: {getattr(best_result_node, 'depth', 'N/A')}")
                    logger.info(f"[Workflow] Visits: {getattr(best_result_node, 'visits', 'N/A')}")
                else:
                    logger.warning("[Workflow] Best result found but no valid molecule")
            else:
                logger.warning("[Workflow] No final result found")
                
            # 輸出一些統計信息
            if final_state:
                if isinstance(final_state, dict):
                    logger.info(f"[Workflow] Final depth reached: {final_state.get('depth', 'Unknown')}")
                    logger.info(f"[Workflow] Final parent: {final_state.get('parent_smiles', 'Unknown')}")
                else:
                    logger.info(f"[Workflow] Final depth reached: {getattr(final_state, 'depth', 'Unknown')}")
                    logger.info(f"[Workflow] Final parent: {getattr(final_state, 'parent_smiles', 'Unknown')}")
        else:
            logger.warning("[Workflow] No result returned")
        
    except asyncio.TimeoutError:
        logger.error("[Workflow] Timeout: Workflow took too long (30 minutes)")
        # 嘗試獲取當前最佳結果
        try:
            best_result_node = engine.best if hasattr(engine, 'best') and engine.best else None
            if best_result_node:
                logger.info(f"[Workflow] Using best result from timeout: {best_result_node.smiles}")
        except Exception as e:
            logger.warning(f"Could not retrieve best result from engine: {e}")
            
    except KeyboardInterrupt:
        logger.info("[Workflow] Interrupted by user")
        # 嘗試獲取當前最佳結果
        try:
            best_result_node = engine.best if hasattr(engine, 'best') and engine.best else None
            if best_result_node:
                logger.info(f"[Workflow] Using best result from interruption: {best_result_node.smiles}")
        except Exception as e:
            logger.warning(f"Could not retrieve best result from engine: {e}")
            
    except Exception as e:
        logger.error(f"[Workflow] Error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        # 嘗試獲取當前最佳結果
        try:
            best_result_node = engine.best if hasattr(engine, 'best') and engine.best else None
            if best_result_node:
                logger.info(f"[Workflow] Using best result from error: {best_result_node.smiles}")
        except Exception as e2:
            logger.warning(f"Could not retrieve best result from engine: {e2}")
        
    # 4. 輸出最終結果
    if best_result_node and hasattr(best_result_node, 'smiles'):
        score_to_print = getattr(best_result_node, 'total_score', 0.0)
        if not isinstance(score_to_print, (int, float)):
            score_to_print = 0.0
            
        # 為 LangSmith 記錄輸出信息
        langsmith_outputs = {
            "success": True,
            "best_smiles": best_result_node.smiles,
            "best_score": score_to_print,
            "final_depth": getattr(best_result_node, 'depth', 'N/A'),
            "visits": getattr(best_result_node, 'visits', 'N/A'),
            "oracle_calls_remaining": oracle.calls_left,
            "llm_provider": cfg['llm'].get('provider', 'cerebras')
        }
        
        print(f"\n=== FINAL RESULT ===")
        print(f"BEST SMILES: {best_result_node.smiles}")
        print(f"BEST SCORE:  {score_to_print:.4f}")
        print(f"VISITS:      {getattr(best_result_node, 'visits', 'N/A')}")
        print(f"ORACLE CALLS REMAINING: {oracle.calls_left}")
        print(f"LLM PROVIDER: {cfg['llm'].get('provider', 'cerebras')}")
    else:
        # 為 LangSmith 記錄失敗信息
        langsmith_outputs = {
            "success": False,
            "best_smiles": None,
            "best_score": 0.0,
            "oracle_calls_remaining": oracle.calls_left,
            "reason": "No valid result found",
            "llm_provider": cfg['llm'].get('provider', 'cerebras')
        }
        
        print(f"\n=== NO RESULT FOUND ===")
        print(f"ORACLE CALLS REMAINING: {oracle.calls_left}")
        print(f"LLM PROVIDER: {cfg['llm'].get('provider', 'cerebras')}")
    
    # 5. 清理資源
    oracle.close()
    logger.info("=== GA LLM World Model Finished ===")


if __name__ == "__main__":
    asyncio.run(run())