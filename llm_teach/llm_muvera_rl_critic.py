#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM-based molecule selection with MUVERA-inspired RAG, a fine-tuning RL Critic,
and a LangGraph-orchestrated workflow.

This script implements an advanced molecule selection pipeline:
1.  **LangGraph Orchestration**: The entire process is structured as a stateful graph,
    making the complex workflow manageable and explicit.
2.  **Multi-Vector Retrieval (MUVERA-inspired)**: Each molecule is represented by two
    distinct embeddings (SMILES and Functional Group), allowing the RAG system to
    retrieve more nuanced and relevant historical data.
3.  **Reinforcement Learning (RL) Critic**: A PyTorch-based neural network acts as a
    value function, predicting the potential score of candidate molecules.
4.  **Dynamic Critic Fine-Tuning**: After each generation, the main LLM reflects on the
    critic's performance against the true oracle scores. This reflection generates
    high-quality training data used to fine-tune the critic, enabling it to learn
    and adapt from generation to generation.
5.  **API Key Rotation**: Inherits the robust API key rotation mechanism to ensure
    stable communication with the Cerebras API.
"""

import argparse
import csv
import os
import json
import re
import random
import time
import threading
from collections import defaultdict
from typing import List, Dict, Tuple, TypedDict, Annotated
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- Dependency Imports ---
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sentence_transformers import SentenceTransformer
import faiss
from langgraph.graph import StateGraph, END
from cerebras.cloud.sdk import Cerebras, RateLimitError
from transformers import AutoModelForCausalLM, AutoTokenizer
import deepspeed

# --- Environment Setup ---
os.environ.setdefault("OMP_NUM_THREADS", "8")

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# DeepSpeed distributed vars (set by deepspeed launcher)
LOCAL_RANK = int(os.getenv("LOCAL_RANK", "0"))
WORLD_SIZE = int(os.getenv("WORLD_SIZE", "1"))

# --- Defaults (formerly CLI flags) ---
DEFAULT_CRITIC_LR = 1e-4
DEFAULT_VALUE_LR = 1e-4
DEFAULT_IQL_TAU = 0.005
DEFAULT_IQL_GAMMA = 0.99
DEFAULT_IQL_EXPECTILE = 0.8
DEFAULT_DS_ZERO_STAGE = 0
DEFAULT_CB_FORMAT = 'json'                 # 'json' or 'stream'
DEFAULT_CB_MAX_COMPLETION_TOKENS = 2000
DEFAULT_CB_REASONING_EFFORT = 'Medium'     # low|medium|high|none (case-insensitive)
DEFAULT_CB_SEL_MAX_COMPLETION_TOKENS = None
DEFAULT_CB_REF_MAX_COMPLETION_TOKENS = None

# --- Global Components ---
# Use a single, globally available embedding model
EMBEDDING_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
EMBEDDING_DIM = EMBEDDING_MODEL.get_sentence_embedding_dimension()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Data Loading Utilities ---

def load_task_csv(csv_path: str) -> Dict[int, List[Tuple[str, float]]]:
    generation_data = defaultdict(list)
    with open(csv_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            generation = int(row['generation'])
            smiles = row['smiles']
            score = float(row['score'])
            generation_data[generation].append((smiles, score))
    return generation_data

def load_fg_csv(csv_path: str) -> Dict[int, List[str]]:
    fg_data = defaultdict(list)
    if not os.path.exists(csv_path):
        return fg_data
    with open(csv_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                gen = int(row['generation'])
                fg = row['smiles_fg']
                fg_data[gen].append(fg)
            except (KeyError, ValueError):
                continue
    return fg_data

def load_completed_generations(output_path: str) -> set:
    completed = set()
    if not os.path.exists(output_path):
        return completed
    with open(output_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                completed.add(int(row['generation']))
            except (KeyError, ValueError):
                continue
    return completed

csv_write_lock = threading.Lock()
def append_generation_to_csv(generation: int, molecules: list, output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    try:
        with csv_write_lock:
            file_exists = os.path.exists(output_path)
            with open(output_path, 'a', newline='') as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(['generation', 'smiles', 'score', 'critic_score'])
                for smiles, _, score, critic_score in molecules:
                    writer.writerow([generation, smiles, score, critic_score])
                # 強制立即可見
                f.flush()
                os.fsync(f.fileno())
        logger.info(f"CSV updated: {output_path} (gen={generation}, rows={len(molecules)})")
    except Exception as e:
        logger.exception(f"Failed to append to CSV {output_path}: {e}")

# --- Core Components: MUVERA Retriever and RL Critic ---

class MuveraRetriever:
    """A MUVERA-inspired retriever with separate indexes for SMILES and FGs."""
    def __init__(self, dim):
        self.dim = dim
        # Index for SMILES embeddings
        self.smiles_index = faiss.IndexFlatL2(dim)
        self.smiles_map = []  # List to map index to (smiles, fg, score)
        # Index for Functional Group (FG) embeddings
        self.fg_index = faiss.IndexFlatL2(dim)
        self.fg_map = []

    def add(self, molecules: List[Tuple[str, str, float]]):
        """Encodes and adds molecules to both indexes."""
        if not molecules:
            return

        smiles_list = [mol[0] for mol in molecules]
        fg_list = [mol[1] for mol in molecules]

        smiles_embeddings = EMBEDDING_MODEL.encode(smiles_list, convert_to_numpy=True)
        fg_embeddings = EMBEDDING_MODEL.encode(fg_list, convert_to_numpy=True)

        self.smiles_index.add(smiles_embeddings)
        self.smiles_map.extend(molecules)
        self.fg_index.add(fg_embeddings)
        self.fg_map.extend(molecules)
        logger.info(f"Retriever: Added {len(molecules)} molecules to SMILES and FG indexes.")

    def retrieve(self, query_mol: Tuple[str, str], k: int = 3) -> List[Tuple[str, str, float]]:
        """Retrieves k most similar molecules from both indexes and combines them."""
        if self.smiles_index.ntotal == 0:
            return []

        query_smiles_emb = EMBEDDING_MODEL.encode([query_mol[0]], convert_to_numpy=True)
        query_fg_emb = EMBEDDING_MODEL.encode([query_mol[1]], convert_to_numpy=True)

        # Search SMILES index
        _, I_smiles = self.smiles_index.search(query_smiles_emb, k)
        # Search FG index
        _, I_fg = self.fg_index.search(query_fg_emb, k)

        retrieved = {}
        for i in I_smiles[0]:
            if i != -1:
                mol = self.smiles_map[i]
                retrieved[mol[0]] = mol # Use SMILES as key to avoid duplicates
        for i in I_fg[0]:
            if i != -1:
                mol = self.fg_map[i]
                retrieved[mol[0]] = mol

        return list(retrieved.values())

class ValueFunction(nn.Module):
    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.network(x)

class Critic(nn.Module):
    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        self.network1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )
        self.network2 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.network1(x), self.network2(x)

class ImplicitQLearning:
    def __init__(self, value_fn, critic, value_optimizer, critic_optimizer, tau, gamma, expectile, lambda_rank=0.3):
        self.value_fn = value_fn
        self.critic = critic
        self.value_optimizer = value_optimizer
        self.critic_optimizer = critic_optimizer
        self.tau = tau
        self.gamma = gamma
        self.expectile = expectile
        self.lambda_rank = lambda_rank
        self.target_critic = Critic(critic.network1[0].in_features).to(DEVICE)
        self.target_critic.load_state_dict(critic.state_dict())

    def update(self, embeddings, rewards):
        # embeddings: [B, D], rewards: [B, 1]
        huber = nn.SmoothL1Loss()

        with torch.no_grad():
            next_v = self.value_fn(embeddings)
            q1_t, q2_t = self.critic(embeddings)

        # ----- Value function update (expectile + ranking) -----
        v = self.value_fn(embeddings)
        q1, q2 = self.critic(embeddings)
        adv = torch.min(q1, q2) - v
        expectile_mask = (self.expectile * (adv > 0).float() - (1 - self.expectile) * (adv < 0).float())
        value_loss = (expectile_mask * adv.pow(2)).mean()

        # Pairwise ranking loss to enforce monotonicity w.r.t. rewards
        # Build pair indices i,j where rewards[i] != rewards[j]
        B = rewards.shape[0]
        if B >= 2:
            # Sample up to 1024 pairs for efficiency
            idx_i = torch.randint(0, B, (min(1024, B * (B - 1) // 2),), device=embeddings.device)
            idx_j = torch.randint(0, B, (idx_i.numel(),), device=embeddings.device)
            mask = idx_i != idx_j
            idx_i, idx_j = idx_i[mask], idx_j[mask]
            ri, rj = rewards[idx_i], rewards[idx_j]
            vi, vj = v[idx_i], v[idx_j]
            # We want sign(ri - rj) == sign(vi - vj)
            # Logistic ranking loss: log(1 + exp(-sign * (vi - vj)))
            sign = torch.sign(ri - rj)
            # Drop equal pairs
            valid = sign != 0
            if valid.any():
                sign = sign[valid]
                vi = vi[valid]
                vj = vj[valid]
                rank_loss = torch.log1p(torch.exp(-(sign * (vi - vj)))).mean()
            else:
                rank_loss = torch.zeros((), device=embeddings.device)
        else:
            rank_loss = torch.zeros((), device=embeddings.device)

        total_value_loss = value_loss + self.lambda_rank * rank_loss

        self.value_optimizer.zero_grad()
        total_value_loss.backward(retain_graph=True)
        self.value_optimizer.step()

        # ----- Critic update with Huber (SmoothL1) targets -----
        with torch.no_grad():
            target_q = rewards + self.gamma * next_v

        q1, q2 = self.critic(embeddings)
        critic_loss = huber(q1, target_q) + huber(q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ----- Target critic Polyak update -----
        for param, target_param in zip(self.critic.parameters(), self.target_critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        # Return scalar floats for logging
        return (total_value_loss.detach().item(), critic_loss.detach().item())

    def get_embedding(self, molecule: Tuple[str, str]):
        """Generate a combined embedding for a molecule."""
        smiles_emb = EMBEDDING_MODEL.encode(molecule[0], convert_to_tensor=True)
        fg_emb = EMBEDDING_MODEL.encode(molecule[1], convert_to_tensor=True)
        return torch.cat([smiles_emb, fg_emb]).to(DEVICE)

# --- LangGraph State Definition ---

class GraphState(TypedDict):
    task: str
    generation: int
    candidate_molecules: List[Tuple[str, str, float]]
    retrieved_docs: Dict[str, List[Tuple[str, str, float]]]
    critic_scores: List[float]
    llm_selection_indices: List[int]
    llm_reasoning: str
    reflection: str
    training_data: List[Dict]
    api_key_iterator: object
    output_path: str
    llm_instance: object # Can be a vLLM instance or None for Cerebras
    
    # Components that persist across the graph
    retriever: MuveraRetriever
    iql: ImplicitQLearning

# --- LLM Interaction and Prompting ---

def get_next_api_key(state: GraphState) -> str:
    return next(state['api_key_iterator'])

def call_llm(
    llm_instance,
    prompt: str,
    model_name: str,
    max_tokens: int,
    temp: float,
    llm_option: str,
    api_key: str = None,
    response_format: dict | None = None,
    stream: bool | None = None,
    stop: list[str] | None = None,
    max_completion_tokens: int | None = None,
    reasoning_effort: str | None = None,
) -> str:
    """
    A robust wrapper for Cerebras API calls with retry and auto-continuation logic.
    """
    if llm_option == 'vllm':
        # Transformers + DeepSpeed path (llm_instance is a dict with model/tokenizer)
        tokenizer = llm_instance['tokenizer']
        model = llm_instance['model']
        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        inputs = tokenizer(prompt, return_tensors='pt')
        if torch.cuda.is_available():
            inputs = {k: v.to(f"cuda:{LOCAL_RANK}") for k, v in inputs.items()}
        model.eval()
        with torch.no_grad():
            gen_ids = model.generate(
                **inputs,
                max_new_tokens=min(max_tokens, 2048),
                do_sample=(temp is not None and temp > 0),
                temperature=(temp if temp is not None else 1.0),
                top_p=1.0,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
        generated = gen_ids[0][inputs['input_ids'].shape[1]:]
        return tokenizer.decode(generated, skip_special_tokens=True)

    # Cerebras logic
    if llm_instance is None:
        if api_key is None:
            raise ValueError("API key must be provided for Cerebras client.")
        from cerebras.cloud.sdk import Cerebras
        client = Cerebras(api_key=api_key)
    else:
        client = llm_instance
    reff = str(reasoning_effort).lower() if reasoning_effort is not None else None

    # default: 若有 response_format -> 非串流；否則串流
    if stream is None:
        stream = False if response_format else True
    
    if not stream:
        completion = client.chat.completions.create(
            messages = [
                {"role": "system", "content": "You are an expert medicinal chemist."},
                {"role": "user", "content": prompt}
            ],
            model=model_name,
            stream=False,
            max_completion_tokens=(max_completion_tokens or max_tokens),
            temperature=temp,
            top_p=1,
            response_format=response_format,
            reasoning_effort=reff,
        )

        msg = completion.choices[0].message
        content = getattr(msg, 'content', None) or ""
        parsed = getattr(msg, 'parsed', None)
        if response_format and isinstance(response_format, dict) and parsed is not None:
            try:
                return json.dumps(parsed)
            except Exception:
                pass
        return content

    full_response = ""
    completion_stream = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are an expert medicinal chemist."},
            {"role": "user", "content": prompt}
        ],
        model=model_name,
        stream=True,
        max_completion_tokens=(max_completion_tokens or max_tokens),
        temperature=temp,
        top_p=1,
        stop=stop,
        reasoning_effort=reff
    )
    for chunk in completion_stream:
        delta = getattr(chunk.choices[0], 'delta', None)
        if delta and getattr(delta, 'content', None):
            full_response += delta.content
    return full_response


def create_selection_prompt(task_desc: str, generation: int, molecules: List[Tuple[str, str, float]],
                            retrieved_docs: Dict[str, list], critic_scores: List[float]) -> str:
    
    mol_texts = []
    for i, (smiles, fg, score) in enumerate(molecules):
        docs = retrieved_docs.get(smiles, [])
        doc_text = "\n".join([f"  - Similar Past Example: {d_smiles} (FG: {d_fg}) -> Scored {d_score:.3f}" for d_smiles, d_fg, d_score in docs])
        mol_texts.append(
            f"{i+1}. SMILES: {smiles}\n"
            f"   Functional Groups: {fg}\n"
            f"   RL Critic Score: {critic_scores[i]:.3f}\n"
            f"   Historical Context:\n{doc_text if docs else '  - No relevant history found.'}"
        )
    molecules_text = "\n\n".join(mol_texts)

    return f"""You are a world-class medicinal chemist tasked with selecting promising drug candidates.

**Task Description**: {task_desc}
**Current Generation**: {generation}

You are provided with {len(molecules)} candidate molecules. For each, you have its structure (SMILES), key functional groups (FGs), a predicted score from an adaptive RL Critic, and historical data of similar molecules from past generations.

**Your Goal**: Select the 10 most promising molecules for the next round of optimization.

**Decision Criteria**:
1.  **Task Potential**: How well does the molecule fit the task description?
2.  **RL Critic Score**: This score (0-1) is a prediction of success. Higher is generally better, but treat it as a strong suggestion, not an absolute truth.
3.  **Historical Context**: Learn from past successes and failures. If similar molecules performed well, it's a good sign.
4.  **Chemical Diversity**: Do not just pick the top 10 critic scores. Ensure a diverse set of structures to explore the chemical space effectively. Avoid selecting molecules that are too similar to each other.

**Candidate Molecules**:
{molecules_text}

**Instructions**:
First, provide a brief reasoning for your selections, explaining your strategy.
Then, on a new line, provide ONLY a valid JSON array of the 10 selected molecule indices (1-based).

Example:
<json>
[1, 5, 12, 25, 33, 45, 51, 67, 88, 99]
</json>
"""

def create_reflection_prompt(task_desc: str, generation: int, selection_reasoning: str,
                             selected_molecules: List[Tuple[str, str, float, float]]) -> str:
    
    selected_info = "\n".join([
        f"- Mol {i+1} (SMILES: {smiles}): Critic Score={critic_score:.3f}, Actual Oracle Score={oracle_score:.3f}, Delta={oracle_score - critic_score:+.3f}"
        for i, (smiles, _, oracle_score, critic_score) in enumerate(selected_molecules)
    ])

    return f"""You are an expert chemist training a junior RL Critic model. Your goal is to teach it to better predict molecule scores.

**Task Description**: {task_desc}
**Generation**: {generation}

In the last round, an LLM agent made selections based on the RL Critic's predictions. Here is the agent's reasoning:
**Agent's Reasoning**:
{selection_reasoning}

Now, let's review the performance. Here are the molecules the agent selected, with the Critic's prediction vs. the true Oracle Score:
**Performance Review**:
{selected_info}

**Your Task**:
Based on this review, provide targeted feedback to improve the RL Critic for the next generation.
1.  **Analyze**: Identify where the Critic was accurate and where it was wrong. Did it overestimate or underestimate certain types of molecules or functional groups?
2.  **Hypothesize**: What chemical features or patterns might explain the difference between the predicted and actual scores?
3.  **Generate Training Rules**: Formulate clear, actionable rules for the Critic. For each selected molecule, create a JSON object with the SMILES, the true oracle_score, and a "feedback" string explaining what the critic should learn.

**Output Format**:
Provide your analysis and then, on a new line, output ONLY a valid JSON array of training rule objects.

Example:
<json>
[
    {{"smiles": "...", "oracle_score": 0.85, "feedback": "Good prediction. The combination of a pyrimidine ring and a sulfonamide group was correctly identified as high-potential."}},
    {{"smiles": "...", "oracle_score": 0.21, "feedback": "Overestimation. The critic failed to penalize the sterically hindered ester group, which reduces binding affinity."}}
]
</json>
"""

def parse_llm_selection(response: str, max_molecules: int) -> Tuple[str, List[int]]:
    reasoning = response.split('<json>')[0].strip()
    
    json_part_match = re.search(r'(\[.*?\])', response, re.DOTALL)
    if not json_part_match:
        # Fallback: find any list of numbers
        numbers = [int(n) for n in re.findall(r'\d+', response)]
        return reasoning, [n - 1 for n in numbers if 1 <= n <= max_molecules][:10]

    try:
        indices = json.loads(json_part_match.group(1))
        return reasoning, [idx - 1 for idx in indices if isinstance(idx, int) and 1 <= idx <= max_molecules][:10]
    except json.JSONDecodeError:
        numbers = [int(n) for n in re.findall(r'\d+', json_part_match.group(1))]
        return reasoning, [n - 1 for n in numbers if 1 <= n <= max_molecules][:10]

def parse_reflection_output(response: str) -> List[Dict]:
    if not response:
        return []
    text = response.strip()
    # 速通：整段就是 JSON array
    if text.startswith('['):
        try:
            return json.loads(text)
        except Exception:
            pass
    # 一般情形：從混合文字中找第一個陣列
    json_part_match = re.search(r'(\[.*?\])', text, re.DOTALL)
    if not json_part_match:
        return []
    try:
        return json.loads(json_part_match.group(1))
    except json.JSONDecodeError:
        return []

# --- LangGraph Nodes ---

def multi_vector_retrieval(state: GraphState) -> GraphState:
    logger.info(f"Generation {state['generation']}: Running Multi-Vector Retrieval.")
    retriever = state['retriever']
    retrieved_docs = defaultdict(list)
    for smiles, fg, _ in state['candidate_molecules']:
        retrieved_docs[smiles] = retriever.retrieve((smiles, fg), k=3)
    return {**state, "retrieved_docs": retrieved_docs}

def rl_critic_scoring(state: GraphState) -> GraphState:
    logger.info(f"Generation {state['generation']}: Scoring molecules with IQL Value Function.")
    iql = state['iql']
    iql.value_fn.eval()
    scores = []
    with torch.no_grad():
        for smiles, fg, _ in state['candidate_molecules']:
            embedding = iql.get_embedding((smiles, fg))
            raw = iql.value_fn(embedding.unsqueeze(0))
            score = torch.sigmoid(raw).item()
            scores.append(score)
    return {**state, "critic_scores": scores}

def llm_molecule_selection(state: GraphState, args) -> GraphState:
    logger.info(f"Generation {state['generation']}: LLM is selecting molecules.")
    task_desc = "..." # Placeholder for actual task description logic
    prompt = create_selection_prompt(
        task_desc, state['generation'], state['candidate_molecules'],
        state['retrieved_docs'], state['critic_scores']
    )
    
    logger.info(f"Full Prompt for a task:\n{prompt}")
    api_key = get_next_api_key(state)
    response = call_llm(state['llm_instance'], prompt, args.model, args.max_tokens, args.temperature, args.llm_option, api_key)
    logger.info(f"Full Response for a task:\n{response}")
    
    reasoning, indices = parse_llm_selection(response, len(state['candidate_molecules']))
    
    logger.info(f"LLM selected indices: {indices}")
    return {**state, "llm_selection_indices": indices, "llm_reasoning": reasoning}

def llm_reflection_and_training_data_generation(state: GraphState, args) -> GraphState:
    logger.info(f"Generation {state['generation']}: LLM is reflecting and generating training data.")
    
    # Prepare data for reflection
    selected_molecules_with_scores = []
    for idx in state['llm_selection_indices']:
        smiles, fg, oracle_score = state['candidate_molecules'][idx]
        critic_score = state['critic_scores'][idx]
        selected_molecules_with_scores.append((smiles, fg, oracle_score, critic_score))

    task_desc = "..." # Placeholder
    prompt = create_reflection_prompt(
        task_desc, state['generation'], state['llm_reasoning'], selected_molecules_with_scores
    )

    # JSON schema 模式：只要 JSON，避免模型輸出分析文字而被丟棄
    if getattr(args, 'cb_format', None) == 'json':
        prompt += (
            "\nReturn ONLY a JSON array per the schema. Do not include any analysis or prose. "
            "Prefer including the EXACT 1-based 'index' from the Performance Review list. "
            "If you include 'smiles', it MUST be copied verbatim from the list; do not invent or alter strings."
        )

    logger.info(f"Full Prompt for a task:\n{prompt}")
    api_key = get_next_api_key(state)

    if args.llm_option == 'cerebras':
        if getattr(args, 'cb_format', None) == 'json':
            enum_smiles = [sm for sm, _fg, _os, _cs in selected_molecules_with_scores]
            n_sel = len(enum_smiles)

            item_schema = {
                "type": "object",
                "properties": {
                    # 1-based index as shown in the Performance Review list above
                    "index": {"type": "integer", "minimum": 1, "maximum": n_sel},
                    # exact SMILES string (verbatim). Model should pick from this enum.
                    "smiles": {"type": "string", "enum": enum_smiles},
                    "oracle_score": {"type": "number"},
                    "feedback": {"type": "string"}
                },
                # Require oracle_score + feedback; index is now optional, we infer it from smiles if missing
                "required": ["oracle_score", "feedback"],
                "additionalProperties": False
            }
            arr_schema = {"type": "array", "items": item_schema}
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": "critic_feedback",
                    # Cerebras GPT‑OSS rejects strict=True; keep False
                    "strict": False,
                    "schema": arr_schema
                }
            }
            response = call_llm(
                state['llm_instance'], prompt, args.model, 1500, 0.3,
                args.llm_option, api_key,
                response_format=response_format,
                stream=False,
                stop=None,
                max_completion_tokens=getattr(args, 'cb_max_completion_tokens', None),
                reasoning_effort=getattr(args, 'cb_reasoning_effort', None),
            )
        else:
            response = call_llm(
                state['llm_instance'], prompt, args.model, 1500, 0.3,
                args.llm_option, api_key,
                response_format=None,
                stream=True,
                stop=["</json>"],
                max_completion_tokens=getattr(args, 'cb_max_completion_tokens', None),
                reasoning_effort=getattr(args, 'cb_reasoning_effort', None),
            )
    else:
        response = call_llm(state['llm_instance'], prompt, args.model, 1500, 0.3, args.llm_option, api_key)

    logger.info(f"Full Response for a task:\n{response}")
    logger.info(f"Reflection response length: {len(response)} bytes")
    
    training_data = parse_reflection_output(response)
    # Normalize: more permissive logic, infer index from smiles if needed
    normalized = []
    matched_by_smiles = 0
    for rec in (training_data or []):
        if not isinstance(rec, dict):
            continue
        idx = rec.get("index")
        score = rec.get("oracle_score")
        fb = rec.get("feedback")

        # Basic field checks
        if not isinstance(score, (int, float)):
            continue
        if not isinstance(fb, str) or not fb.strip():
            continue

        # Resolve index: prefer provided index; otherwise infer from exact SMILES match within the selected set
        if not (isinstance(idx, int) and 1 <= idx <= len(selected_molecules_with_scores)):
            idx = None
            rec_smiles = rec.get("smiles")
            if isinstance(rec_smiles, str) and rec_smiles:
                for j, (sm, _fg, _os, _cs) in enumerate(selected_molecules_with_scores, start=1):
                    if sm == rec_smiles:
                        idx = j
                        matched_by_smiles += 1
                        break
        # If still no valid index, drop this record
        if not (isinstance(idx, int) and 1 <= idx <= len(selected_molecules_with_scores)):
            continue

        # Map to canonical SMILES for that index
        sel_smiles, sel_fg, _os, _cs = selected_molecules_with_scores[idx - 1]
        normalized.append({
            "index": idx,
            "smiles": sel_smiles,
            "oracle_score": float(score),
            "feedback": fb.strip()
        })

    # Compute dropped count safely
    orig_len = len(training_data) if isinstance(training_data, list) else 0
    dropped = orig_len - len(normalized)
    logger.info(f"Generated {len(normalized)} valid training samples for the critic. Dropped {dropped} invalid/mismatched items. Inferred index by SMILES for {matched_by_smiles} items.")

    # Replace training_data in state with normalized list
    training_data = normalized

    # Also add all selected molecules to the retriever for the next generation
    state['retriever'].add([(sm, fg, score) for sm, fg, score, _ in selected_molecules_with_scores])

    return {**state, "reflection": response.split('<json>')[0], "training_data": training_data}

def update_rl_critic(state: GraphState) -> GraphState:
    logger.info(f"Generation {state['generation']}: Fine-tuning the IQL models.")
    iql = state['iql']
    training_data = state['training_data']

    # 先把本世代的選擇結果持久化，避免早退就沒寫檔
    final_selection = []
    for idx in state['llm_selection_indices']:
        smiles, fg, oracle_score = state['candidate_molecules'][idx]
        critic_score = state['critic_scores'][idx]
        final_selection.append((smiles, fg, oracle_score, critic_score))
    append_generation_to_csv(state['generation'], final_selection, state['output_path'])

    if not training_data:
        logger.warning("No training data generated for IQL. Skipping update.")
        return state

    iql.value_fn.train()
    iql.critic.train()

    embeddings = []
    rewards = []
    rejects = 0

    # Build 1-based index -> (smiles, fg) map for this generation
    idx_map = {i + 1: (sm, fg) for i, (sm, fg, _os) in enumerate(state['candidate_molecules'])}

    for sample in training_data:
        # Prefer index; fall back to smiles
        sel = None
        if isinstance(sample.get("index"), int) and sample["index"] in idx_map:
            sel = idx_map[sample["index"]]
        elif isinstance(sample.get("smiles"), str):
            # Exact match only — we now canonicalize upstream, but keep this for safety
            for i, (sm, fg, _os) in enumerate(state['candidate_molecules']):
                if sm == sample["smiles"]:
                    sel = (sm, fg)
                    break

        if sel is None:
            rejects += 1
            continue

        oracle_score = sample.get("oracle_score")
        if not isinstance(oracle_score, (int, float)):
            rejects += 1
            continue

        emb = iql.get_embedding(sel)
        embeddings.append(emb)
        rewards.append(float(oracle_score))

    if rejects:
        logger.info(f"IQL trainer rejected {rejects} malformed/unknown samples.")

    if not embeddings:
        logger.warning("No valid training samples found. Skipping update.")
        return state

    embeddings = torch.stack(embeddings)
    rewards = torch.tensor(rewards, device=DEVICE).unsqueeze(1)

    value_loss, critic_loss = iql.update(embeddings, rewards)

    logger.info(f"IQL fine-tuning complete. Value Loss: {value_loss:.4f}, Critic Loss: {critic_loss:.4f}")

    return state

# --- Main Application Logic ---

def main():
    parser = argparse.ArgumentParser(description="LLM-based molecule selection with MUVERA, RL Critic, and LangGraph.")
    # Add arguments from the reference script
    parser.add_argument('--llm_option', type=str, default='cerebras', choices=['vllm', 'cerebras'], help='Choose the LLM to use: vllm or cerebras')
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.9, help='GPU memory utilization for VLLM.')
    parser.add_argument('--tensor_parallel_size', type=int, default=4,
                        help='Number of GPUs for tensor-parallel inference (DeepSpeed). Overridden by WORLD_SIZE when launched with deepspeed.')
    parser.add_argument('--hf_model', type=str, default=None,
                        help='Hugging Face model id for Transformers+DeepSpeed inference (used when --llm_option vllm).')
    parser.add_argument('--tasks', nargs='+', required=True, help='List of task names to process.')
    parser.add_argument('--input_dir', type=str, default='data/offspring', help='Input directory for task CSVs.')
    parser.add_argument('--output_dir', type=str, default='results/muvera_rl_critic', help='Output directory for results.')
    parser.add_argument('--fg_dir', type=str, default='data/functiongroup_offspring', help='Directory for functional group CSVs.')
    parser.add_argument('--model', type=str, default='gpt-oss-120b', help='Cerebras model name.')
    parser.add_argument('--temperature', type=float, default=0.7, help='LLM sampling temperature.')
    parser.add_argument('--max_tokens', type=int, default=30000, help='Max tokens for LLM response.')
    parser.add_argument('--max_model_len', type=int, default=6096, help='Maximum model context length.')
    parser.add_argument('--max_generations', type=int, default=50, help='Max generations to process.')
    parser.add_argument('--cerebras_api_keys', nargs=5, required=True, help='Five Cerebras API keys to rotate.')
    args = parser.parse_args()

    # Backward-compat: attach defaults for removed CLI flags
    args.critic_lr = DEFAULT_CRITIC_LR
    args.value_lr = DEFAULT_VALUE_LR
    args.iql_tau = DEFAULT_IQL_TAU
    args.iql_gamma = DEFAULT_IQL_GAMMA
    args.iql_expectile = DEFAULT_IQL_EXPECTILE
    args.ds_zero_stage = DEFAULT_DS_ZERO_STAGE
    args.cb_format = DEFAULT_CB_FORMAT
    args.cb_max_completion_tokens = DEFAULT_CB_MAX_COMPLETION_TOKENS
    args.cb_reasoning_effort = DEFAULT_CB_REASONING_EFFORT
    args.cb_sel_max_completion_tokens = DEFAULT_CB_SEL_MAX_COMPLETION_TOKENS
    args.cb_ref_max_completion_tokens = DEFAULT_CB_REF_MAX_COMPLETION_TOKENS

    # --- Initialize LLM --- 
    if args.llm_option == 'vllm':
        if not args.hf_model:
            raise RuntimeError("When --llm_option vllm is selected, you must provide --hf_model with a valid Hugging Face model id.")
        tokenizer = AutoTokenizer.from_pretrained(args.hf_model, use_fast=True)
        _dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            args.hf_model,
            torch_dtype=_dtype,
            low_cpu_mem_usage=True,
        )
        tp_size = int(os.getenv('WORLD_SIZE', str(args.tensor_parallel_size)))
        zero_cfg = {"stage": args.ds_zero_stage} if args.ds_zero_stage and args.ds_zero_stage > 0 else None
        ds_engine = deepspeed.init_inference(
            model=model,
            tensor_parallel={"tp_size": tp_size},
            dtype=_dtype,
            replace_with_kernel_inject=True,
            **({"zero": zero_cfg} if zero_cfg else {})
        )
        model = ds_engine.module
        llm_instance = {"model": model, "tokenizer": tokenizer}
    else:
        # The API key iterator will be used to create Cerebras clients on the fly
        llm_instance = None # Cerebras client is created within the call_llm function

    # --- Initialize Components ---
    retriever = MuveraRetriever(dim=EMBEDDING_DIM)
    value_fn = ValueFunction(input_dim=EMBEDDING_DIM * 2).to(DEVICE)
    critic = Critic(input_dim=EMBEDDING_DIM * 2).to(DEVICE)
    value_optimizer = optim.Adam(value_fn.parameters(), lr=args.value_lr)
    critic_optimizer = optim.Adam(critic.parameters(), lr=args.critic_lr)

    iql = ImplicitQLearning(
        value_fn=value_fn,
        critic=critic,
        value_optimizer=value_optimizer,
        critic_optimizer=critic_optimizer,
        tau=args.iql_tau,
        gamma=args.iql_gamma,
        expectile=args.iql_expectile,
        lambda_rank=0.3
    )

    # --- Build LangGraph ---
    workflow = StateGraph(GraphState)
    workflow.add_node("multi_vector_retrieval", multi_vector_retrieval)
    workflow.add_node("rl_critic_scoring", rl_critic_scoring)
    workflow.add_node("llm_molecule_selection", lambda state: llm_molecule_selection(state, args))
    workflow.add_node("llm_reflection_and_training_data_generation", lambda state: llm_reflection_and_training_data_generation(state, args))
    workflow.add_node("update_rl_critic", update_rl_critic)

    workflow.set_entry_point("multi_vector_retrieval")
    workflow.add_edge("multi_vector_retrieval", "rl_critic_scoring")
    workflow.add_edge("rl_critic_scoring", "llm_molecule_selection")
    workflow.add_edge("llm_molecule_selection", "llm_reflection_and_training_data_generation")
    workflow.add_edge("llm_reflection_and_training_data_generation", "update_rl_critic")
    workflow.add_edge("update_rl_critic", END)
    
    app = workflow.compile()

    # --- Main Execution Loop ---
    for task in args.tasks:
        logger.info(f"--- Starting Task: {task} ---")
        input_path = os.path.join(args.input_dir, f"{task}.csv")
        fg_path = os.path.join(args.fg_dir, f"{task}.csv")
        output_path = os.path.join(args.output_dir, f"{task}.csv")

        if not os.path.exists(input_path):
            logger.error(f"Input file not found: {input_path}, skipping task.")
            continue

        generation_data = load_task_csv(input_path)
        fg_data = load_fg_csv(fg_path)
        completed_generations = load_completed_generations(output_path)

        # Merge data
        merged_data = defaultdict(list)
        for gen, mols in generation_data.items():
            fg_list = fg_data.get(gen, [])
            for i, (smiles, score) in enumerate(mols):
                fg = fg_list[i] if i < len(fg_list) else ""
                merged_data[gen].append((smiles, fg, score))

        # Process generations
        generations_to_process = sorted([
            g for g in merged_data.keys() 
            if g not in completed_generations and g < args.max_generations
        ])

        for gen in generations_to_process:
            logger.info(f"--- Processing Generation {gen} for Task {task} ---")
            
            # Create a fresh iterator for API keys for each generation run
            from itertools import cycle
            api_key_iterator = cycle(args.cerebras_api_keys)

            initial_state = GraphState(
                task=task,
                generation=gen,
                candidate_molecules=merged_data[gen],
                retriever=retriever,
                iql=iql,
                api_key_iterator=api_key_iterator,
                output_path=output_path,
                llm_instance=llm_instance,
                # The rest are populated by the graph
                retrieved_docs={},
                critic_scores=[],
                llm_selection_indices=[],
                llm_reasoning="",
                reflection="",
                training_data=[],
            )
            
            # Run the graph
            app.invoke(initial_state)
            
            # The graph now handles saving, so we just log completion
            logger.info(f"--- Finished Generation {gen} for Task {task} ---")

if __name__ == "__main__":
    main()

'''
python llm_muvera_rl_critic.py --tasks amlodipine --input_dir data/offspring --output_dir results/muvera_rl_critic_results --fg_dir data/functiongroup_offspring \
  --model gpt-oss-120b --cerebras_api_keys "csk-yc5xd56kcxwc9x5y5rfc6mw95mfknd892mjjkhdyj39y898h" "csk-8f3ct6y23mw2fw3fdmyx8t4tx8rw8p85tdx9nrm8mv2t9m26" "csk-38kjr3mnp22x9w2m2wejdej5m55cpkwhmh3p6p6f3wt2wxw2" "csk-22c9dktpnx6yc94rpv4h8xp952nvcnypnmn36nrk3ym953yf" "csk-n58d54hd8njxfnd225fkvhyj8wd28v2t656eexecxt3mh6t4" \
  --max_generations 50 --llm_option cerebras --max_model_len 60000 \
  2>&1 | tee -a log/job_muvera_rl_critic.log
'''