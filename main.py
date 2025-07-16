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

# Set up logging FIRST to ensure all modules can use it
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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

def prescan(cfg, oracle):
    """Return a list of seed molecules for the initial population."""
    try:
        seeds = oracle.prescan_lowest_n(cfg["smi_file"], n=100)
        logger.info(f"[Prescan] Selected {len(seeds)} seed molecules")
        return seeds
    except Exception as e:
        logger.error(f"[Prescan] Error: {e}")
        default_seed = cfg.get("START_SMILES", "C1CCCCC1")
        logger.warning(f"[Prescan] Using default seed: {default_seed}")
        return [default_seed]


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
    
    # 1. Prescan 獲取起始分子列表
    seeds = prescan(cfg, oracle)
    overall_best = None

    for idx, seed in enumerate(seeds):
        if oracle.calls_left <= 0:
            break

        engine.best = None
        init_state = AgentState(parent_smiles=seed, depth=0)
        logger.info(f"[Workflow {idx+1}/{len(seeds)}] Starting seed: {seed}")

        async def run_with_timeout():
            from graph.workflow_graph import run_workflow
            return await run_workflow(init_state)

        try:
            await asyncio.wait_for(run_with_timeout(), timeout=1800)
        except Exception as e:
            logger.error(f"[Workflow {idx+1}] Error: {e}")
            continue

        if engine.best and (
            not overall_best or
            getattr(engine.best, 'oracle_score', 0.0) > getattr(overall_best, 'oracle_score', 0.0)
        ):
            overall_best = engine.best

    if overall_best and hasattr(overall_best, 'smiles'):
        score = getattr(overall_best, 'oracle_score', getattr(overall_best, 'total_score', 0.0))
        print("\n=== FINAL RESULT ===")
        print(f"BEST SMILES: {overall_best.smiles}")
        print(f"BEST SCORE: {score:.4f}")
    else:
        print("\n=== NO RESULT FOUND ===")

    oracle.close()
    logger.info("=== GA LLM World Model Finished ===")


if __name__ == "__main__":
    asyncio.run(run())