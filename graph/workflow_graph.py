"""

"""
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langsmith import traceable
from dataclasses import dataclass, field
from typing import List, Dict, Any
import logging

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

# 載入配置
cfg = yaml.safe_load(pathlib.Path("config/settings.yml").read_text())

# 從配置獲取 SMILES 長度限制
MAX_SMILES_LENGTH = cfg.get("workflow", {}).get("max_smiles_length", 100)

# 設定 LangSmith 環境變數（如果尚未設定）
if cfg.get("langsmith", {}).get("enabled", False) and not os.environ.get("LANGSMITH_API_KEY"):
    langsmith_config = cfg["langsmith"]
    os.environ["LANGSMITH_TRACING"] = str(langsmith_config.get("tracing", True)).lower()
    os.environ["LANGSMITH_ENDPOINT"] = langsmith_config.get("endpoint", "https://api.smith.langchain.com")
    os.environ["LANGSMITH_API_KEY"] = langsmith_config.get("api_key", "")
    os.environ["LANGSMITH_PROJECT"] = langsmith_config.get("project", "world model agent")
    logger.info(f"LangSmith tracing enabled for project: {langsmith_config.get('project', 'world model agent')}")

# 設定 Cerebras API 金鑰（如果尚未設定）
llm_config = cfg.get("llm", {})
if llm_config.get("api_key") and not os.environ.get("CEREBRAS_API_KEY"):
    os.environ["CEREBRAS_API_KEY"] = llm_config.get("api_key")
    logger.info("Cerebras API key loaded from settings.yml")

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
kg = create_kg_store(KGConfig(**cfg["kg"]))
oracle = GuacaMolOracle(cfg["TASK_NAME"])

# 初始化 LLM Generator，API 金鑰已在上面設定到環境變數
llm_config = cfg.get("llm", {}).copy()
llm_config["max_smiles_length"] = MAX_SMILES_LENGTH  # 添加 SMILES 長度限制
llm_gen = LLMGenerator(**llm_config)
engine = MCTSEngine(kg, cfg["max_depth"])

# ---------- nodes ----------
@traceable
def generate_actions(state: AgentState):
    # 確保引擎有正確的根節點
    if state.parent_smiles != "ROOT" and state.parent_smiles not in engine.nodes:
        # 創建根節點（種子分子）
        root_node = engine._get_or_create_node(state.parent_smiles, 0)
        engine.root = root_node
        print(f"[Debug] Created root node: {state.parent_smiles}")
    
    state.actions = engine.propose_actions(state.parent_smiles,
                                           state.depth,
                                           cfg["K_init"])
    print(f"[Debug] Generated {len(state.actions)} actions for {state.parent_smiles}")
    return state

@traceable
def llm_generate(state: AgentState):
    print(f"[Debug] LLM generating for {state.parent_smiles} with {len(state.actions)} actions")
    state.batch_smiles = llm_gen.generate_batch(state.parent_smiles,
                                                state.actions)
    print(f"[Debug] LLM generated {len(state.batch_smiles)} SMILES")
    return state

@traceable
async def oracle_score(state: AgentState):
    """Oracle 評分節點（異步）"""
    print(f"[Debug] Oracle scoring {len(state.batch_smiles)} SMILES")
    
    if not state.batch_smiles:
        logger.warning("No SMILES to score")
        state.scores = []
        return state
    
    # 過濾掉過長的 SMILES
    filtered_smiles = []
    filtered_indices = []
    
    for i, smiles in enumerate(state.batch_smiles):
        if len(smiles) <= MAX_SMILES_LENGTH:
            filtered_smiles.append(smiles)
            filtered_indices.append(i)
        else:
            print(f"[Debug] Filtering out long SMILES (length: {len(smiles)}): {smiles[:50]}...")
    
    if not filtered_smiles:
        logger.warning("All SMILES were filtered out due to length constraints")
        state.scores = []
        return state
    
    # 如果有SMILES被過濾掉，需要調整相關列表
    if len(filtered_smiles) < len(state.batch_smiles):
        print(f"[Debug] Filtered {len(state.batch_smiles) - len(filtered_smiles)} SMILES due to length constraints")
        # 更新 batch_smiles 和 actions 列表
        state.batch_smiles = filtered_smiles
        if state.actions:
            state.actions = [state.actions[i] for i in filtered_indices]
    
    try:
        scores = await oracle.score_async(filtered_smiles)
        state.scores = scores
        print(f"[Debug] Oracle returned scores: {scores}")
    except Exception as e:
        logger.error(f"Oracle scoring failed: {e}")
        # 提供默認分數以避免工作流停止
        state.scores = [0.0] * len(filtered_smiles)
        print(f"[Debug] Using default scores: {state.scores}")
    
    return state

@traceable
def compute_adv(state: AgentState):
    import numpy as np
    baseline = float(np.mean(state.scores)) if state.scores else 0.0
    state.advantages = [s-baseline for s in state.scores]
    # engine.update_batch moved to update_stores node
    return state

@traceable # Added traceable decorator
def update_stores(state: AgentState):
    # Update Knowledge Graph
    # Ensure actions, batch_smiles, scores, advantages are all populated and have same length
    if not all([hasattr(state, attr) and state.actions and state.batch_smiles and state.scores and state.advantages and
                len(state.actions) == len(state.batch_smiles) == len(state.scores) == len(state.advantages)
                for attr in ['actions', 'batch_smiles', 'scores', 'advantages']]):
        print("Warning: Skipping KG/MCTS update due to inconsistent state data.")
        # Potentially raise an error or handle more gracefully
        return state

    for i in range(len(state.batch_smiles)):
        action_taken = state.actions[i]
        generated_smiles = state.batch_smiles[i]
        score = state.scores[i]
        advantage = state.advantages[i]

        # Log molecule in Knowledge Graph
        # Assuming kg.create_molecule and kg.create_action signatures
        # These methods would be defined in your kg_store.py
        kg.create_molecule(
            smiles=generated_smiles,
            score=score,
            advantage=advantage,
            # parent_smiles=state.parent_smiles, # If needed by create_molecule
            # action_details=action_taken # If needed by create_molecule
        )
        
        # Log action and its result in Knowledge Graph
        kg.create_action(
            parent_smiles=state.parent_smiles,
            child_smiles=generated_smiles,
            action_type=action_taken.get("type", "unknown"),
            action_params=str(action_taken.get("params", {})),
            score_delta=score
        )

    # Update MCTS Engine
    engine.update_batch(
        state.parent_smiles,
        state.batch_smiles,
        state.scores,
        state.advantages
    )
    return state

@traceable # Added traceable decorator
def decide(state: AgentState):
    print(f"[Debug] Decide: current parent={state.parent_smiles}, depth={state.depth}")
    print(f"[Debug] Oracle calls left: {oracle.calls_left}")
    
    # 主要終止條件：Oracle 預算用完
    if oracle.calls_left <= 0:
        print("[Debug] Terminating: Oracle budget exhausted")
        state.result = {"best": engine.best, "reason": "Oracle budget exhausted"}
        return state
    
    # 早停條件：找到高分分子
    early_stop_threshold = cfg.get("workflow", {}).get("early_stop_threshold", 0.8)
    if (hasattr(engine, 'best') and engine.best and 
        hasattr(engine.best, 'total_score') and 
        engine.best.total_score >= early_stop_threshold):
        print(f"[Debug] Early stopping: Found high-score molecule (score: {engine.best.total_score:.4f} >= {early_stop_threshold})")
        state.result = {"best": engine.best, "reason": f"Early stop - high score ({engine.best.total_score:.4f})"}
        return state
    
    # 次要終止條件：達到最大深度
    if state.depth >= cfg["max_depth"]:
        print("[Debug] Terminating: Max depth reached")
        state.result = {"best": engine.best, "reason": "Max depth reached"}
        return state
    
    # 安全性檢查：分子複雜度限制
    max_smiles_length = cfg.get("workflow", {}).get("max_smiles_length", 100)
    if len(state.parent_smiles) > max_smiles_length:
        print(f"[Debug] Terminating: Molecule too complex (length: {len(state.parent_smiles)} > {max_smiles_length})")
        state.result = {"best": engine.best, "reason": f"Molecule too complex (length: {len(state.parent_smiles)})"}
        return state
    
    # 移除硬編碼的迭代限制，讓程式持續運行直到 Oracle 預算用完
    # 只保留基本的迭代計數用於調試
    if hasattr(engine, 'iteration_count'):
        engine.iteration_count += 1
    else:
        engine.iteration_count = 1
    
    # 嘗試選擇下一個節點
    nxt = engine.select_child(state.parent_smiles)
    if not nxt:
        print("[Debug] Terminating: No more children to explore")
        state.result = {"best": engine.best, "reason": "No more children to explore"}
        return state
    
    # 檢查下一個節點是否達到深度限制
    if nxt.depth >= cfg["max_depth"]:
        print("[Debug] Terminating: Next node would exceed max depth")
        state.result = {"best": engine.best, "reason": "Next node would exceed max depth"}
        return state
    
    # 更新狀態繼續探索
    print(f"[Debug] Continuing: next parent={nxt.smiles}, next depth={nxt.depth}")
    state.parent_smiles = nxt.smiles
    state.depth = nxt.depth
    # 清空之前的批次數據
    state.actions = []
    state.batch_smiles = []
    state.scores = []
    state.advantages = []
    return state

def should_continue(state: AgentState):
    """決定是否繼續工作流程"""
    if state.result:  # 如果已經有結果，則結束
        return END
    return "Generate"

# ---------- LangGraph ----------
sg = StateGraph(AgentState)
sg.add_node("Generate", generate_actions)
sg.add_node("LLM", llm_generate)
sg.add_node("Oracle", oracle_score)  # 修正：使用 add_node 而不是 add_async_node
sg.add_node("Adv", compute_adv)
sg.add_node("UpdateStores", update_stores) # Added new node
sg.add_node("Decide", decide)

sg.set_entry_point("Generate")
sg.add_edge("Generate", "LLM")
sg.add_edge("LLM", "Oracle")
sg.add_edge("Oracle", "Adv")
sg.add_edge("Adv", "UpdateStores") # Edge to new node
sg.add_edge("UpdateStores", "Decide") # Edge from new node
sg.add_conditional_edges("Decide", should_continue, {"Generate": "Generate", END: END})

# 直接編譯，不使用 checkpoint，並設置遞歸限制
graph_app = sg.compile()

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
                
            print(f"[Debug] Iteration {iteration}: {chunk}")
            last_chunk = chunk
            
            # 檢查是否有終止結果
            for node_name, state in chunk.items():
                if hasattr(state, 'result') and state.result:
                    print(f"[Debug] Found termination result in {node_name}: {state.result}")
                    final_result = state.result
                    return chunk  # 立即返回包含結果的 chunk
            
            # 檢查是否到達 END 狀態
            if END in chunk:
                print(f"[Debug] Reached END state")
                return chunk
                
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
                    print(f"[Debug] Fallback result: {fallback_result}")
                    break
                    
        return last_chunk
    except Exception as e:
        logger.error(f"Workflow failed at iteration {iteration}: {e}")
        return None