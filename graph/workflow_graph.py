"""

"""
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langsmith import traceable
from dataclasses import dataclass, field
from typing import List, Dict, Any
import logging
import asyncio

from configparser import ConfigParser
import yaml, pathlib, os, re

logger = logging.getLogger(__name__)

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

# Global placeholders for components that will be initialized later.
# This prevents eager initialization at module import time.
cfg = None
kg = None
oracle = None
llm_gen = None
engine = None
MAX_SMILES_LENGTH = 100  # Default, will be updated by create_workflow_components

graph_app = None # Placeholder for the compiled graph

from oracle.guacamol_client import GuacaMolOracle
from llm.generator import LLMGenerator
from mcts.mcts_engine import MCTSEngine
from kg.kg_store import KGConfig, create_kg_store

# ---------- shared state ----------
@dataclass
class AgentState:
    parent_smiles: str
    depth: int
    actions: List[Dict[str, Any]] = field(default_factory=list)
    batch_smiles: List[str] = field(default_factory=list)
    scores: List[float] = field(default_factory=list)
    advantages: List[float] = field(default_factory=list)
    result: Dict[str, Any] = field(default_factory=dict)

# ---------- init objects ----------
# ---------- nodes ----------
@traceable
def generate_actions(state: AgentState):
    # 確保引擎有正確的根節點
    if state.parent_smiles != "ROOT" and state.parent_smiles not in engine.nodes:
        # 創建根節點（種子分子）
        root_node = engine._get_or_create_node(state.parent_smiles, 0)
        engine.root = root_node
        logger.debug(f"Created root node: {state.parent_smiles}")
    
    state.actions = engine.propose_actions(state.parent_smiles,
                                           state.depth,
                                           cfg["K_init"])
    logger.debug(f"Generated {len(state.actions)} actions for {state.parent_smiles}")
    return state

@traceable
def llm_generate(state: AgentState):
    logger.debug(f"LLM generating for {state.parent_smiles} with {len(state.actions)} actions")
    state.batch_smiles = llm_gen.generate_batch(state.parent_smiles,
                                                state.actions)
    logger.debug(f"LLM generated {len(state.batch_smiles)} SMILES")
    return state

@traceable
async def oracle_score(state: AgentState):
    """Oracle 評分節點（異步）"""
    logger.debug(f"Oracle scoring {len(state.batch_smiles)} SMILES")
    
    if not state.batch_smiles:
        logger.warning("No SMILES to score")
        state.scores = []
        logger.debug("No SMILES to score, continuing with empty scores")
        return state
    
    # 過濾掉過長的 SMILES
    filtered_smiles = []
    filtered_indices = []
    
    for i, smiles in enumerate(state.batch_smiles):
        if len(smiles) <= MAX_SMILES_LENGTH:
            filtered_smiles.append(smiles)
            filtered_indices.append(i)
        else:
            logger.debug(f"Filtering out long SMILES (length: {len(smiles)}): {smiles[:50]}...")
    
    if not filtered_smiles:
        logger.warning("All SMILES were filtered out due to length constraints")
        state.scores = []
        logger.debug("All SMILES filtered out, continuing with empty scores")
        return state
    
    # 如果有SMILES被過濾掉，需要調整相關列表
    if len(filtered_smiles) < len(state.batch_smiles):
        print(f"[Debug] Filtered {len(state.batch_smiles) - len(filtered_smiles)} SMILES due to length constraints")
        # 更新 batch_smiles 和 actions 列表
        state.batch_smiles = filtered_smiles
        if state.actions:
            state.actions = [state.actions[i] for i in filtered_indices]
    
    # Deduplicate SMILES before scoring
    unique_smiles = list(set(state.batch_smiles))
    if len(unique_smiles) < len(state.batch_smiles):
        logger.info(f"Removed {len(state.batch_smiles) - len(unique_smiles)} duplicate SMILES before scoring.")
    
    if not unique_smiles:
        logger.warning("No unique SMILES to score.")
        state.scores = []
        logger.debug("No unique SMILES to score, continuing with empty scores")
        return state
    
    try:
        scores = await oracle.score_async(unique_smiles)
        
        # Map scores back to original SMILES order
        score_map = {smile: score for smile, score in zip(unique_smiles, scores)}
        state.scores = [score_map[smile] for smile in state.batch_smiles]
        
        logger.debug(f"Oracle returned scores: {state.scores}")
        logger.debug(f"Oracle calls remaining: {oracle.calls_left}")
    except Exception as e:
        logger.error(f"Oracle scoring failed: {e}")
        # 提供默認分數以避免工作流停止
        state.scores = [0.0] * len(state.batch_smiles)
        logger.debug(f"Using default scores: {state.scores}")
    
    logger.debug("Oracle scoring complete, proceeding to next node")
    return state

@traceable
def compute_adv(state: AgentState):
    logger.debug(f"Computing advantages for {len(state.scores)} scores")
    import numpy as np
    baseline = float(np.mean(state.scores)) if state.scores else 0.0
    state.advantages = [s-baseline for s in state.scores]
    logger.debug(f"Baseline: {baseline:.6f}, advantages computed")
    # engine.update_batch moved to update_stores node
    return state

@traceable
def expand_node(state: AgentState):
    """
    擴展 MCTS 樹：創建新的子節點結構，並記錄生成動作
    職責：純粹的樹結構擴展，包含動作歷史追蹤
    """
    logger.debug(f"Expanding tree from parent {state.parent_smiles} with {len(state.batch_smiles)} potential children")
    
    # 確保父節點存在
    if not engine.has_node(state.parent_smiles):
        parent_node = engine._get_or_create_node(state.parent_smiles, state.depth)
        logger.debug(f"Created missing parent node: {state.parent_smiles}")
    else:
        parent_node = engine.get_node(state.parent_smiles)
    
    # 創建子節點結構（包含動作歷史）
    new_children_count = 0
    for i, child_smiles in enumerate(state.batch_smiles):
        if child_smiles not in parent_node.children:
            # 創建子節點，並記錄生成動作
            generating_action = state.actions[i] if i < len(state.actions) else None
            
            # 使用增強的 add_child 方法
            child_node = parent_node.add_child(
                child_smiles, 
                state.depth + 1, 
                generating_action=generating_action
            )
            
            # 確保節點也在引擎中註冊
            engine.nodes[child_smiles] = child_node
            new_children_count += 1
            
            logger.debug(f"Created child {child_smiles[:30]}... with action: {generating_action.get('name', 'Unknown') if generating_action else 'None'}")
    
    logger.debug(f"Expansion complete: {new_children_count} new children added, parent now has {len(parent_node.children)} total children")
    return state

@traceable
def update_stores(state: AgentState):
    """
    更新存儲：專注於反向傳播和知識圖譜更新
    職責：分數傳播 + 數據持久化
    """
    logger.debug(f"UpdateStores: Backpropagating scores for {len(state.batch_smiles)} molecules")
    
    # 數據完整性檢查
    if not all([hasattr(state, attr) and getattr(state, attr) for attr in ['batch_smiles', 'scores', 'advantages']]):
        logger.warning("Skipping stores update due to missing state data")
        return state
        
    if not (len(state.batch_smiles) == len(state.scores) == len(state.advantages)):
        print(f"[Warning] Inconsistent data lengths, trimming to minimum")
        min_len = min(len(state.batch_smiles), len(state.scores), len(state.advantages))
        state.batch_smiles = state.batch_smiles[:min_len]
        state.scores = state.scores[:min_len]
        state.advantages = state.advantages[:min_len]
        if state.actions:
            state.actions = state.actions[:min_len]

    parent_node = engine.get_node(state.parent_smiles)
    if not parent_node:
        logger.error("Parent node not found during UpdateStores")
        return state

    # 1. 更新子節點分數和統計
    for i, (child_smiles, score, advantage) in enumerate(zip(state.batch_smiles, state.scores, state.advantages)):
        child_node = parent_node.children.get(child_smiles)
        if child_node:
            # 更新節點分數和統計
            child_node.update(score)
            child_node.advantage = advantage
            
            # 更新全局最佳 - 使用 oracle_score 而非 avg_score
            if not engine.best or score > getattr(engine.best, 'oracle_score', 0.0):
                engine.best = child_node
                logger.debug(f"New best: {child_smiles[:30]}... oracle_score={score:.4f}")

    # 2. 執行 MCTS 反向傳播
    engine.backpropagate(state.batch_smiles, state.scores)
    
    # 3. 知識圖譜更新
    try:
        for i, (child_smiles, score, advantage) in enumerate(zip(state.batch_smiles, state.scores, state.advantages)):
            # 分子記錄
            kg.create_molecule(
                smiles=child_smiles,
                score=score,
                advantage=advantage,
            )
            
            # 動作記錄
            if i < len(state.actions):
                kg.create_action(
                    parent_smiles=state.parent_smiles,
                    child_smiles=child_smiles,
                    action_type=state.actions[i].get("type", "unknown"),
                    action_params=str(state.actions[i].get("params", {})),
                    score_delta=score
                )
    except Exception as e:
        logger.debug(f"KG update error: {e}")

    logger.debug("UpdateStores complete: backpropagation finished, KG updated")
    return state

@traceable
def decide(state: AgentState):
    logger.debug(f"Decide: current parent={state.parent_smiles}, depth={state.depth}")
    logger.debug(f"Oracle calls left: {oracle.calls_left}")
    
    # 動態樹修剪觰發器
    if not hasattr(engine, 'iteration_count'):
        engine.iteration_count = 0
    engine.iteration_count += 1
    
    # 每 N 次迭代或樹大小超過閾值時進行修剪
    prune_interval = cfg.get("mcts", {}).get("prune_interval", 50)  # 每50次迭代修剪一次
    max_tree_size = cfg.get("mcts", {}).get("max_tree_size", 1000)  # 樹節點數超過1000時修剪
    
    should_prune = (
        engine.iteration_count % prune_interval == 0 or
        len(engine.nodes) > max_tree_size
    )
    
    if should_prune and hasattr(engine, 'tree_manipulator'):
        logger.debug(f"Triggering tree pruning: iteration={engine.iteration_count}, tree_size={len(engine.nodes)}")
        try:
            # 保留前 k 個最佳子樹
            keep_top_k = cfg.get("mcts", {}).get("keep_top_k", 100)
            pruned_count = engine.tree_manipulator.prune_tree_recursive(engine.root, keep_top_k)
            logger.debug(f"Pruning complete: removed {pruned_count} nodes, remaining: {len(engine.nodes)}")
        except Exception as e:
            logger.debug(f"Tree pruning failed: {e}")

    # 主要終止條件：Oracle 預算用完
    if oracle.calls_left <= 0:
        logger.info("Terminating: Oracle budget exhausted")
        state.result = {"best": engine.best, "reason": "Oracle budget exhausted"}
        return state
    
    # 早停條件：找到高分分子
    early_stop_threshold = cfg.get("workflow", {}).get("early_stop_threshold", 0.8)
    if (hasattr(engine, 'best') and engine.best and 
        hasattr(engine.best, 'oracle_score') and 
        engine.best.oracle_score >= early_stop_threshold):
        logger.info(f"Early stopping: Found high-score molecule (oracle_score: {engine.best.oracle_score:.4f} >= {early_stop_threshold})")
        state.result = {"best": engine.best, "reason": f"Early stop - high score ({engine.best.oracle_score:.4f})"}
        return state
    
    # 次要終止條件：達到最大深度
    if state.depth >= cfg["max_depth"]:
        logger.info("Terminating: Max depth reached")
        state.result = {"best": engine.best, "reason": "Max depth reached"}
        return state
    
    # 安全性檢查：分子複雜度限制
    max_smiles_length = cfg.get("workflow", {}).get("max_smiles_length", 100)
    if len(state.parent_smiles) > max_smiles_length:
        logger.info(f"Terminating: Molecule too complex (length: {len(state.parent_smiles)} > {max_smiles_length})")
        state.result = {"best": engine.best, "reason": f"Molecule too complex (length: {len(state.parent_smiles)})"}
        return state
    
    # 新增：檢查是否所有子節點都相同（LLM 生成失敗）
    parent_node = engine.get_node(state.parent_smiles)
    if parent_node and parent_node.children:
        unique_children = set(parent_node.children.keys())
        if len(unique_children) == 1 and state.parent_smiles in unique_children:
            logger.info("Terminating: LLM failed to generate diverse molecules")
            state.result = {"best": engine.best, "reason": "LLM generation failure - all molecules identical"}
            return state
    
    # 新增：迭代限制作為安全網
    # This is now handled in run_workflow to avoid statefulness issues in the node.
    
    # 安全網：防止無限循環
    # This is also handled in run_workflow.
    
    # 嘗試選擇下一個節點
    nxt = engine.select_node_for_expansion()
    if not nxt:
        logger.info("Terminating: Could not select a node for expansion from the tree.")
        state.result = {"best": engine.best, "reason": "MCTS selection failed to find a node to expand."}
        return state
    
    # 檢查下一個節點是否達到深度限制
    if nxt.depth >= cfg["max_depth"]:
        logger.info("Terminating: Next node would exceed max depth")
        state.result = {"best": engine.best, "reason": "Next node would exceed max depth"}
        return state
    
    # 檢查是否選擇了相同的節點（避免死循環）
    if nxt.smiles == state.parent_smiles:
        logger.warning("Selected same molecule, incrementing depth")
        state.depth += 1
        if state.depth >= cfg["max_depth"]:
            logger.info("Terminating: Forced depth increment exceeded max depth")
            state.result = {"best": engine.best, "reason": "Forced termination due to identical molecule selection"}
            return state
    else:
        # 更新狀態繼續探索
        logger.debug(f"Continuing: next parent={nxt.smiles[:50]}..., next depth={nxt.depth}")
        state.parent_smiles = nxt.smiles
        state.depth = nxt.depth
    
    # 清空之前的批次數據
    state.actions = []
    state.batch_smiles = []
    state.scores = []
    state.advantages = []
    return state

def should_continue(state: AgentState):
    """
    判斷是否繼續工作流程
    """
    try:
        logger.debug("should_continue called, checking state...")
        
        # 檢查是否有結果（終止條件）
        if hasattr(state, 'result') and state.result:
            logger.debug(f"Found result: {state.result}")
            logger.info("Workflow completed with result")
            return "end"
        
        # 檢查是否有錯誤
        if hasattr(state, 'result') and state.result and state.result.get("error"):
            logger.error(f"Workflow error: {state.result['error']}")
            return "end"
        
        # 檢查 Oracle 調用次數
        if oracle.calls_left <= 0:
            logger.debug(f"Oracle calls exhausted: {oracle.calls_left}")
            logger.info("Oracle calls exhausted")
            return "end"
        
        # 正常流程：繼續生成
        print(f"[Debug] Continuing workflow to Generate node")
        logger.debug("Continuing workflow to Generate node")
        return "Generate"
        
    except Exception as e:
        logger.error(f"Error in should_continue: {e}")
        logger.debug(f"Error in should_continue: {e}")
        return "end"

# ---------- LangGraph ----------
sg = StateGraph(AgentState)
sg.add_node("Generate", generate_actions)
sg.add_node("LLM", llm_generate)
sg.add_node("Oracle", oracle_score)  
sg.add_node("Adv", compute_adv) 
sg.add_node("Expand", expand_node) 
sg.add_node("UpdateStores", update_stores) 
sg.add_node("Decide", decide) 

sg.set_entry_point("Generate")
sg.add_edge("Generate", "LLM")
sg.add_edge("LLM", "Oracle")
sg.add_edge("Oracle", "Adv")
sg.add_edge("Adv", "Expand")        # Adv -> Expand (新增)
sg.add_edge("Expand", "UpdateStores") # Expand -> UpdateStores (新增)
sg.add_edge("UpdateStores", "Decide")  # UpdateStores -> Decide (保持)
sg.add_conditional_edges(
    "Decide", 
    should_continue, 
    {
        "Generate": "Generate",
        "end": END  # 修復：正確映射 "end" 到 END
    }
)

# 直接編譯，不使用 checkpoint，並設置遞歸限制
# graph_app is compiled within create_workflow_components to use the correct configuration.

# 設置執行配置
async def run_workflow(initial_state):
    """運行工作流程，以 Oracle 預算為主要終止條件（異步版本）"""
    # 從配置文件讀取相關參數
    recursion_limit = cfg.get("workflow", {}).get("recursion_limit", 200)
    max_iterations = cfg.get("workflow", {}).get("max_iterations", 1000)
    
    config = {"recursion_limit": recursion_limit}
    iteration = 0
    last_chunk = None
    final_result = None
    
    logger.info(f"Starting workflow with recursion_limit={recursion_limit}, max_iterations={max_iterations}")
    logger.info(f"Oracle budget: {oracle.calls_left} calls remaining")
    
    try:
        async for chunk in graph_app.astream(initial_state, config=config):
            iteration += 1
            
            # 主要檢查：Oracle 預算
            if oracle.calls_left <= 0:
                logger.info(f"Oracle budget exhausted after {iteration} iterations")
                break
                
            # 次要檢查：防止無限循環的安全機制
            if iteration > max_iterations:
                logger.warning(f"Reached maximum iterations ({max_iterations}), stopping workflow")
                break
                
            logger.debug(f"Iteration {iteration}: {list(chunk.keys())}")
            
            # 檢查每個節點的狀態
            for node_name, state in chunk.items():
                logger.debug(f"Node {node_name} completed")
                
                # 檢查是否有終止結果
                if hasattr(state, 'result') and state.result:
                    logger.debug(f"Found termination result in {node_name}: {state.result}")
                    final_result = state.result
                    return chunk  # 立即返回包含結果的 chunk
            
            last_chunk = chunk
            
            # 檢查是否到達 END 狀態
            if END in chunk:
                logger.debug("Reached END state")
                return chunk
            
            # 添加小延遲以確保異步操作完成
            await asyncio.sleep(0.1)
                
        # 工作流程結束，檢查結束原因
        if oracle.calls_left <= 0:
            logger.info(f"Workflow completed: Oracle budget exhausted after {iteration} iterations")
        elif iteration > max_iterations:
            logger.warning(f"Workflow completed: Maximum iterations ({max_iterations}) reached")
        else:
            logger.info(f"Workflow completed normally after {iteration} iterations")
            
        # 如果正常結束但沒有明確的結果，嘗試獲取最佳結果
        if last_chunk and not final_result:
            for node_name, state in last_chunk.items():
                if hasattr(state, 'parent_smiles'):
                    best_node = engine.best if hasattr(engine, 'best') else None
                    
                    # 根據結束原因設定不同的訊息
                    if oracle.calls_left <= 0:
                        reason = "Oracle budget exhausted"
                    elif iteration > max_iterations:
                        reason = f"Maximum iterations reached ({max_iterations})"
                    else:
                        reason = "Workflow completed without explicit termination"
                    
                    fallback_result = {
                        "best": best_node,
                        "reason": reason
                    }
                    state.result = fallback_result
                    logger.debug(f"Fallback result: {fallback_result}")
                    break
                    
        return last_chunk
    except Exception as e:
        logger.error(f"Workflow failed at iteration {iteration}: {e}")
        logger.debug(f"Workflow exception: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

# 確保 engine 可以被外部導入
__all__ = ['graph_app', 'AgentState', 'oracle', 'cfg', 'engine', 'run_workflow', 'create_workflow_components']

def create_workflow_components(config):
    """
    創建工作流程組件的工廠函數，供 main.py 使用。
    此函數會初始化所有全局組件並編譯 LangGraph。
    
    Args:
        config: 配置字典
        
    Returns:
        tuple: (oracle, engine, graph_app)
    """
    global cfg, kg, oracle, llm_gen, engine, MAX_SMILES_LENGTH, graph_app
    cfg = config
    
    # 設定 LangSmith 環境變數（如果尚未設定）
    if cfg.get("langsmith", {}).get("enabled", False) and not os.environ.get("LANGSMITH_API_KEY"):
        langsmith_config = cfg["langsmith"]
        os.environ["LANGSMITH_TRACING"] = str(langsmith_config.get("tracing", True)).lower()
        os.environ["LANGSMITH_ENDPOINT"] = langsmith_config.get("endpoint", "https://api.smith.langchain.com")
        os.environ["LANGSMITH_API_KEY"] = langsmith_config.get("api_key", "")
        os.environ["LANGSMITH_PROJECT"] = langsmith_config.get("project", "world model agent")
        logger.info(f"LangSmith tracing enabled for project: {langsmith_config.get('project', 'world model agent')}")

    MAX_SMILES_LENGTH = cfg.get("workflow", {}).get("max_smiles_length", 100)

    # 重新初始化組件以使用新配置
    kg = create_kg_store(KGConfig(**cfg["kg"]))
    oracle = GuacaMolOracle(cfg["TASK_NAME"])
    
    # 重新初始化 LLM Generator
    llm_config = cfg.get("llm", {}).copy()
    provider = llm_config.get("provider", "cerebras")
    
    if provider == "github":
        model_name = llm_config.get("github_model_name", "openai/gpt-4.1")
        api_key = llm_config.get("github_api_key")
    else:
        model_name = llm_config.get("model_name", "qwen-3-32b")
        api_key = llm_config.get("api_key")
    
    llm_gen = LLMGenerator(
        provider=provider,
        model_name=model_name,
        temperature=llm_config.get("temperature", 0.2),
        max_completion_tokens=llm_config.get("max_completion_tokens", 8192),
        max_smiles_length=llm_config.get("max_smiles_length", cfg.get("workflow", {}).get("max_smiles_length", 100)),
        top_p=llm_config.get("top_p", 1.0),
        stream=llm_config.get("stream", False),
        api_key=api_key
    )
    
    engine = MCTSEngine(kg, cfg["max_depth"], llm_gen=llm_gen)
    
    # 編譯 LangGraph，確保它使用最新初始化的全局組件
    graph_app = sg.compile()
    
    return oracle, engine, graph_app
