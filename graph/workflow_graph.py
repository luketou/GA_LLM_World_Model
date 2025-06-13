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
llm_gen = LLMGenerator()
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
    return state

@traceable
def compute_adv(state: AgentState):
    import numpy as np
    baseline = float(np.mean(state.scores))
    state.advantages = [s-baseline for s in state.scores]
    engine.update_batch(state.parent_smiles,
                        state.batch_smiles,
                        state.scores,
                        state.advantages)
    return state

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
sg.add_node("Decide", decide)

sg.set_entry_point("Generate")
sg.add_edge("Generate", "LLM")
sg.add_edge("LLM", "Oracle")
sg.add_edge("Oracle", "Adv")
sg.add_edge("Adv", "Decide")
sg.add_edge("Decide", "Generate")

sg.set_checkpoint(SqliteSaver(".lg_ckpt.db"))
graph_app = sg.compile()