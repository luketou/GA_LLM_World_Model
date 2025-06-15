"""
LangGraph workflow orchestrator with LangSmith tracing.
"""
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langsmith import traceable
from dataclasses import dataclass, field
from typing import List, Dict, Any

from configparser import ConfigParser
import yaml, pathlib

from oracle.guacamol_client import GuacaMolOracle
from llm.generator import LLMGenerator
from mcts.mcts_engine import MCTSEngine
from kg.kg_store import KGStore, KGConfig

cfg = yaml.safe_load(pathlib.Path("config/settings.yaml").read_text())

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
kg = KGStore(KGConfig(**cfg["kg"]))
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
    state.actions = engine.propose_actions(state.parent_smiles,
                                           state.depth,
                                           cfg["K_init"])
    return state

@traceable
def llm_generate(state: AgentState):
    state.batch_smiles = llm_gen.generate_batch(state.parent_smiles,
                                                state.actions)
    return state

@traceable
async def oracle_score(state: AgentState):
    state.scores = await oracle.score_async(state.batch_smiles,
                                            epoch=engine.epoch)
    # Assuming GuacaMolOracle handles CSV logging internally as per flowchart
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
    nxt = engine.select_child(state.parent_smiles)
    if not nxt or nxt.depth >= cfg["max_depth"] \
       or oracle.calls_left <= 0:
        state.result = {"best": engine.best}
        return END, state
    state.parent_smiles = nxt.smiles
    state.depth = nxt.depth
    return "Generate", state

# ---------- LangGraph ----------
sg = StateGraph(AgentState)
sg.add_node("Generate", generate_actions)
sg.add_node("LLM", llm_generate)
sg.add_async_node("Oracle", oracle_score)
sg.add_node("Adv", compute_adv)
sg.add_node("UpdateStores", update_stores) # Added new node
sg.add_node("Decide", decide)

sg.set_entry_point("Generate")
sg.add_edge("Generate", "LLM")
sg.add_edge("LLM", "Oracle")
sg.add_edge("Oracle", "Adv")
sg.add_edge("Adv", "UpdateStores") # Edge to new node
sg.add_edge("UpdateStores", "Decide") # Edge from new node
sg.add_edge("Decide", "Generate")

sg.set_checkpoint(SqliteSaver(".lg_ckpt.db"))
graph_app = sg.compile()