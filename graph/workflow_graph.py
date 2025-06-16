"""
LangGraph workflow orchestrator with LangSmith tracing.
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

from oracle.guacamol_client import GuacaMolOracle
from llm.generator import LLMGenerator
from mcts.mcts_engine import MCTSEngine
from kg.kg_store import KGConfig, create_kg_store

cfg = yaml.safe_load(pathlib.Path("config/settings.yml").read_text())

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

# 準備 LLM 配置，處理 API 金鑰
llm_config = cfg.get("llm", {}).copy()
if "api_key" in llm_config and llm_config["api_key"] == "!ENV ${CEREBRAS_API_KEY}":
    import os
    llm_config["api_key"] = os.environ.get("CEREBRAS_API_KEY")

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
    
    try:
        scores = await oracle.score_async(state.batch_smiles)
        state.scores = scores
        print(f"[Debug] Oracle returned scores: {scores}")
    except Exception as e:
        logger.error(f"Oracle scoring failed: {e}")
        # 提供默認分數以避免工作流停止
        state.scores = [0.0] * len(state.batch_smiles)
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
    
    # 檢查終止條件
    if oracle.calls_left <= 0:
        print("[Debug] Terminating: Oracle calls exhausted")
        state.result = {"best": engine.best, "reason": "Oracle calls exhausted"}
        return state
    
    if state.depth >= cfg["max_depth"]:
        print("[Debug] Terminating: Max depth reached")
        state.result = {"best": engine.best, "reason": "Max depth reached"}
        return state
    
    # 添加額外的終止條件：如果分子變得過於複雜
    if len(state.parent_smiles) > 200:  # SMILES 長度限制
        print("[Debug] Terminating: Molecule too complex")
        state.result = {"best": engine.best, "reason": "Molecule too complex"}
        return state
    
    # 添加迭代次數限制
    if hasattr(engine, 'iteration_count'):
        engine.iteration_count += 1
    else:
        engine.iteration_count = 1
    
    if engine.iteration_count > 20:  # 限制迭代次數
        print("[Debug] Terminating: Too many iterations")
        state.result = {"best": engine.best, "reason": "Too many iterations"}
        return state
    
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
    """運行工作流程，帶有遞歸限制和調試信息（異步版本）"""
    config = {"recursion_limit": 50}  # 增加遞歸限制
    iteration = 0
    last_chunk = None
    final_result = None
    
    try:
        async for chunk in graph_app.astream(initial_state, config=config):
            iteration += 1
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
            
            # 防止無限循環的安全檢查
            if iteration > 50:  # 增加最大迭代次數
                print("[Warning] Too many iterations, stopping")
                # 嘗試從最後的狀態獲取最佳結果
                if last_chunk:
                    for node_name, state in last_chunk.items():
                        if hasattr(state, 'parent_smiles'):
                            best_node = engine.best if hasattr(engine, 'best') else None
                            forced_result = {
                                "best": best_node,
                                "reason": "Forced termination due to iteration limit"
                            }
                            # 添加結果到狀態
                            state.result = forced_result
                            print(f"[Debug] Forced termination result: {forced_result}")
                            return last_chunk
                break
                
        # 如果正常結束但沒有明確的結果，嘗試獲取最佳結果
        if last_chunk and not final_result:
            for node_name, state in last_chunk.items():
                if hasattr(state, 'parent_smiles'):
                    best_node = engine.best if hasattr(engine, 'best') else None
                    fallback_result = {
                        "best": best_node,
                        "reason": "Workflow completed without explicit termination"
                    }
                    state.result = fallback_result
                    print(f"[Debug] Fallback result: {fallback_result}")
                    break
                    
        return last_chunk
    except Exception as e:
        print(f"[Error] Workflow failed at iteration {iteration}: {e}")
        return None