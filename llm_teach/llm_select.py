#!/usr/bin/env python3
"""
LLM-based molecule selection script.
Reads {task}.csv files and uses LLM to select the 10 most promising molecules per generation.
"""

import argparse
import csv
import os
from collections import defaultdict
from typing import List, Dict, Tuple

# 強制 VLLM_USE_V1=0，V100 只能用 v0 engine
os.environ["VLLM_USE_V1"] = "0"
os.environ.setdefault("VLLM_ATTENTION_BACKEND", "XFORMERS")  # Flash Attention for V100
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7")  # Use all 8 GPUs

import json
import re
import random
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from transformers import AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    AutoTokenizer = None

import logging
try:
    from logdecorator import log_on_start, log_on_end
except ImportError:
    def log_on_start(*args, **kwargs):
        def decorator(fn):
            return fn
        return decorator
    def log_on_end(*args, **kwargs):
        def decorator(fn):
            return fn
        return decorator

# Detect if bitsandbytes is available for 8-bit quantization
try:
    import bitsandbytes
    BNB_AVAILABLE = True
except ImportError:
    BNB_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Global perf env for V100 optimisation ---
os.environ.setdefault("OMP_NUM_THREADS", "8")  # CPU tokeniser並行


# Utility: split a list into chunks of size n
def chunk_list(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

# --- Embedding/FAISS imports for RAG ---
from sentence_transformers import SentenceTransformer
import faiss

# --- LLM generation logging wrapper ---
def llm_generate_and_log(llm, prompt, sampling_params, generation, batch):
    print('[Prompt]', prompt.replace('\n', ' '))
    outputs = llm.generate([prompt], sampling_params)
    response = outputs[0].outputs[0].text
    print('[Response]', response.replace('\n', ' '))
    return response

# --- RateLimitError import and time ---
import time
try:
    from cerebras.cloud.sdk import RateLimitError  # Newer SDKs
except Exception:
    try:
        from cerebras.cloud.sdk.exceptions import RateLimitError  # Older SDKs
    except Exception:
        RateLimitError = Exception  # Fallback if not found

# Attempt to import vllm and Cerebras
try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    LLM, SamplingParams = None, None

try:
    from cerebras.cloud.sdk import Cerebras
    CEREBRAS_AVAILABLE = True
except ImportError:
    CEREBRAS_AVAILABLE = False
    Cerebras = None

def build_molecule_index(all_molecules):
    # all_molecules: list of SMILES strings
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(all_molecules, convert_to_numpy=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return model, index

def retrieve_all_molecules(model, index, molecules, k=None):
    """
    Return up to k molecules while preserving original order.
    The previous implementation attempted a global FAISS lookup that
    produced indices outside the local 'molecules' list, causing IndexError.
    For the current use‑case (feeding all 100 candidates to the LLM), we
    don't need cross‑generation retrieval, so we keep the list as‑is.

    Args:
        model   : unused (kept for API compatibility)
        index   : unused
        molecules: list of (smiles, score)
        k       : int | None – desired number of molecules (default: all)

    Returns:
        List[(smiles, score)]  – at most k items, original order.
    """
    if k is None or k >= len(molecules):
        return molecules
    return molecules[:k]

def load_task_csv(csv_path: str) -> Dict[int, List[Tuple[str, float]]]:
    """
    Load a task CSV file and organize by generation.
    
    Args:
        csv_path: Path to the CSV fi
        
    Returns:
        Dictionrs AutoTokenizer
        Dictionary mapping generation number to list of (smiles, score) tuples
    """
    generation_data = defaultdict(list)
    
    with open(csv_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            generation = int(row['generation'])
            smiles = row['smiles']
            score = float(row['score'])
            generation_data[generation].append((smiles, score))
    
    return generation_data

def create_selection_prompt(task: str, generation: int, molecules: List[Tuple[str, float]]) -> str:
    """
    Create a prompt for the LLM to select promising molecules.
    
    Args:
        task: The task name (e.g., 'celecoxib', 'osimertinib')
        generation: The generation number
        molecules: List of (smiles, score) tuples
        
    Returns:
        The prompt string
    """
    # Create a numbered list of molecules (no scores in the prompt)
    molecule_list = []
    for i, (smiles, score) in enumerate(molecules, 1):
        molecule_list.append(f"{i}. SMILES: {smiles}")
    molecules_text = "\n".join(molecule_list)
    
    # GuacaMol task descriptions for context
    task_descriptions = {
  'celecoxib': 
    "Rediscovery of Celecoxib, a COX-2 inhibitor anti-inflammatory drug. "
    "Task: generate the exact target molecule. "
    "Scoring: top-1 Tanimoto similarity to Celecoxib computed on ECFC4 fingerprints (sim(Celecoxib, ECFC4)). "
    "Benchmark score: score of the single best molecule (s₁).",
    #  [oai_citation:0‡arXiv](https://arxiv.org/pdf/1811.09621)

  'troglitazone': 
    "Rediscovery of Troglitazone, an antidiabetic drug containing chromane and thiazolidinedione rings. "
    "Task: regenerate the exact target molecule. "
    "Scoring: top-1 Tanimoto similarity to Troglitazone on ECFC4 fingerprints (sim(Troglitazone, ECFC4)). "
    "Benchmark score: s₁.",
    #  [oai_citation:1‡arXiv](https://arxiv.org/pdf/1811.09621)

  'thiothixene': 
    "Rediscovery of Thiothixene, a typical thioxanthene-class antipsychotic. "
    "Task: reproduce the exact Thixothixene molecule. "
    "Scoring: top-1 Tanimoto similarity to Thiothixene on ECFC4 fingerprints (sim(Thiothixene, ECFC4)). "
    "Benchmark score: s₁.",
    #  [oai_citation:2‡arXiv](https://arxiv.org/pdf/1811.09621)

  'aripiprazole': 
    "Design molecules similar to Aripiprazole, an antipsychotic. "
    "Scoring: top-1, top-10, top-100 Tanimoto similarity to Aripiprazole on ECFC4 fingerprints, with a Thresholded(0.75) modifier. "
    "Benchmark score: average of s₁, mean(s₂–s₁₀), mean(s₁₁–s₁₀₀).",
    #  [oai_citation:3‡arXiv](https://arxiv.org/pdf/1811.09621)

  'albuterol': 
    "Design molecules similar to Albuterol, a β₂-adrenergic agonist. "
    "Scoring: top-1, top-10, top-100 Tanimoto similarity to Albuterol on FCFC4 fingerprints, Thresholded(0.75). "
    "Benchmark score: average of s₁, mean(s₂–s₁₀), mean(s₁₁–s₁₀₀).",
    #  [oai_citation:4‡arXiv](https://arxiv.org/pdf/1811.09621)

  'mestranol': 
    "Design molecules similar to Mestranol, an estrogenic hormone. "
    "Scoring: top-1, top-10, top-100 Tanimoto similarity to Mestranol on AtomPair fingerprints, Thresholded(0.75). "
    "Benchmark score: average of s₁, mean(s₂–s₁₀), mean(s₁₁–s₁₀₀).",
    #  [oai_citation:5‡arXiv](https://arxiv.org/pdf/1811.09621)

  'C11H24': 
    "Isomer enumeration for molecular formula C₁₁H₂₄. "
    "Task: generate all 159 possible isomers (ignoring stereochemistry). "
    "Scoring: top-159 score isomer(C11H24) computed as the geometric mean of Gaussian modifiers on counts of C, H, and total atoms. "
    "Benchmark score: mean over the 159 highest-scoring molecules.",
   #  [oai_citation:6‡arXiv](https://arxiv.org/pdf/1811.09621)

  'C9H10N2O2PF2Cl': 
    "Isomer enumeration for molecular formula C₉H₁₀N₂O₂PF₂Cl. "
    "Task: generate all 250 possible isomers. "
    "Scoring: top-250 score isomer(C9H10N2O2PF2Cl) using Gaussian modifiers on elemental counts. "
    "Benchmark score: mean over the 250 highest-scoring molecules.",
    #  [oai_citation:7‡arXiv](https://arxiv.org/pdf/1811.09621)

  'median_molecules_1': 
    "Generate median molecules between camphor and menthol. "
    "Scoring: top-1, top-10, top-100 geometric mean of sim(camphor, ECFC4) and sim(menthol, ECFC4) (no modifiers). "
    "Benchmark score: average of s₁, mean(s₂–s₁₀), mean(s₁₁–s₁₀₀).",
    #  [oai_citation:8‡arXiv](https://arxiv.org/pdf/1811.09621)

  'median_molecules_2': 
    "Generate median molecules between tadalafil and sildenafil. "
    "Scoring: top-1, top-10, top-100 geometric mean of sim(tadalafil, ECFC6) and sim(sildenafil, ECFC6). "
    "Benchmark score: average of s₁, mean(s₂–s₁₀), mean(s₁₁–s₁₀₀).",
    #  [oai_citation:9‡arXiv](https://arxiv.org/pdf/1811.09621)

  'osimertinib': 
    "Design molecules similar to Osimertinib, an EGFR inhibitor for cancer treatment. "
    "Scoring: top-1, top-10, top-100 geometric mean of four contributions: "
    "sim(osimertinib, FCFC4) Thresholded(0.8); "
    "sim(osimertinib, ECFC6) MinGaussian(0.85,2); "
    "TPSA MaxGaussian(100,2); "
    "logP MinGaussian(1,2). "
    "Benchmark score: average of s₁, mean(s₂–s₁₀), mean(s₁₁–s₁₀₀).",
    #  [oai_citation:10‡arXiv](https://arxiv.org/pdf/1811.09621)

  'fexofenadine': 
    "Design molecules similar to Fexofenadine, an antihistamine. "
    "Scoring: top-1, top-10, top-100 geometric mean of sim(fexofenadine, AP) Thresholded(0.8); "
    "TPSA MaxGaussian(90,2); "
    "logP MinGaussian(4,2). "
    "Benchmark score: average of s₁, mean(s₂–s₁₀), mean(s₁₁–s₁₀₀).",
    #  [oai_citation:11‡arXiv](https://arxiv.org/pdf/1811.09621)

  'ranolazine': 
    "Design molecules similar to Ranolazine, an anti-anginal medication. "
    "Scoring: top-1, top-10, top-100 geometric mean of sim(ranolazine, AP) Thresholded(0.7); "
    "logP MaxGaussian(7,1); "
    "TPSA MaxGaussian(95,20); "
    "number_of_fluorine_atoms Gaussian(1,1). "
    "Benchmark score: average of s₁, mean(s₂–s₁₀), mean(s₁₁–s₁₀₀).",
    #  [oai_citation:12‡arXiv](https://arxiv.org/pdf/1811.09621)

  'perindopril': 
    "Design molecules similar to Perindopril, an ACE inhibitor. "
    "Scoring: top-1, top-10, top-100 geometric mean of sim(perindopril, ECFC4) and "
    "number_of_aromatic_rings Gaussian(2,0.5). "
    "Benchmark score: average of s₁, mean(s₂–s₁₀), mean(s₁₁–s₁₀₀).",
    #  [oai_citation:13‡arXiv](https://arxiv.org/pdf/1811.09621)

  'amlodipine': 
    "Design molecules similar to Amlodipine, a calcium channel blocker. "
    "Scoring: top-1, top-10, top-100 geometric mean of sim(amlodipine, ECFC4) and "
    "number_of_rings Gaussian(3,0.5). "
    "Benchmark score: average of s₁, mean(s₂–s₁₀), mean(s₁₁–s₁₀₀).",
    #  [oai_citation:14‡arXiv](https://arxiv.org/pdf/1811.09621)

  'sitagliptin': 
    "Design molecules similar to Sitagliptin, a DPP-4 inhibitor for diabetes. "
    "Scoring: top-1, top-10, top-100 geometric mean of sim(sitagliptin, ECFC4) Gaussian(0,0.1); "
    "logP Gaussian(2.0165,0.2); "
    "TPSA Gaussian(77.04,5); "
    "isomer(C16H15F6N5O) (no modifier). "
    "Benchmark score: average of s₁, mean(s₂–s₁₀), mean(s₁₁–s₁₀₀).",
    #  [oai_citation:15‡arXiv](https://arxiv.org/pdf/1811.09621)

  'zaleplon': 
    "Design molecules similar to Zaleplon, a sedative-hypnotic. "
    "Scoring: top-1, top-10, top-100 geometric mean of sim(zaleplon, ECFC4) and "
    "isomer(C19H17N3O2) (no modifier). "
    "Benchmark score: average of s₁, mean(s₂–s₁₀), mean(s₁₁–s₁₀₀).",
    #  [oai_citation:16‡arXiv](https://arxiv.org/pdf/1811.09621)

  'qed': 
    "Optimize for drug-likeness using the Quantitative Estimate of Drug-likeness (QED) score. "
    "Scoring function: QED calculated by RDKit (range [0,1]). "
    "Benchmark uses top-1, top-10, top-100 QED values and computes S = ⅓(s₁ + 1/10∑₁¹⁰ sᵢ + 1/100∑₁¹⁰⁰ sᵢ).",
    #  [oai_citation:17‡arXiv](https://arxiv.org/pdf/1811.09621)

  'cns_mpo': 
    "Optimize for central nervous system (CNS) drug-likeness using Pfizer’s CNS MPO score. "
    "Scoring: sum of desirability functions (Gaussian-style) over six properties—ClogP, ClogD, MW, TPSA, HBD, pKₐ—each mapped to [0,1], yielding a total score in [0,6]. "
    "Benchmark uses top-1, top-10, top-100 CNS MPO values and the same S-formula as QED; a high-quality CNS drug typically scores ≥4.",
    #  [oai_citation:18‡arXiv](https://arxiv.org/pdf/1811.09621) [oai_citation:19‡PubMed](https://pubmed.ncbi.nlm.nih.gov/22778837/?utm_source=chatgpt.com)

  'scaffold_hop': 
    "Perform scaffold hopping while maintaining activity. "
    "Scoring: arithmetic mean of contributions—SMARTS(s2) absent; SMARTS(s6) present; sim(s5,PHCO) Thresholded(0.75). "
    "Benchmark uses top-1, top-10, top-100 and computes S = ⅓(s₁ + …).",
    #  [oai_citation:20‡arXiv](https://arxiv.org/pdf/1811.09621)

  'decoration_hop': 
    "Perform decoration hopping while maintaining core structure. "
    "Scoring: arithmetic mean of contributions—SMARTS(s2) present; SMARTS(s3) absent; SMARTS(s4) absent; sim(s5,PHCO) Thresholded(0.85). "
    "Benchmark uses top-1, top-10, top-100 and computes S = ⅓(s₁ + …)."
    #  [oai_citation:21‡arXiv](https://arxiv.org/pdf/1811.09621)
    }
    
    task_desc = task_descriptions.get(task, f'Optimize molecules for {task}')
    
    prompt = f"""You are a medicinal chemistry expert evaluating molecules for drug discovery.

Task: {task_desc}
Generation: {generation}

Below are {len(molecules)} molecules from a genetic algorithm optimization, each with their current fitness score.
Your job is to select the 10 most promising molecules based on:
1. Their potential for the given task
2. Chemical feasibility and drug-likeness
3. Structural diversity to maintain genetic diversity
4. Current fitness scores (higher is better)

Molecules:
{molecules_text}

Please select exactly 10 molecules that you believe are most promising for further optimization.
Consider both high-scoring molecules and those with interesting structural features that could lead to better derivatives.

After your reasoning, output ONLY a valid JSON array of exactly 10 integers on a new line. 
Do not output any other text, explanation, or formatting. 
If you output anything else, your answer will be ignored.

Example:
<json>
[1,5,8,12,15,23,27,31,38,42]
</json>
"""
    
    return prompt

def parse_llm_selection(response: str, max_molecules: int) -> List[int]:
    """
    Parse the LLM response to extract selected molecule indices.
    Args:
        response: LLM response string
        max_molecules: Maximum valid molecule index
    Returns:
        List of selected indices (0-based)
    """
    import json, re

    # 先找出第一個合法的 JSON array
    array_match = re.search(r'\[[^\[\]]{0,1000}\]', response)
    if array_match:
        try:
            indices = json.loads(array_match.group(0))
            selected_indices = [idx - 1 for idx in indices if isinstance(idx, int) and 1 <= idx <= max_molecules]
            if len(selected_indices) == 10:
                return selected_indices
        except Exception:
            pass
    # fallback: extract所有數字
    numbers = re.findall(r'\d+', response)
    selected = []
    for num_str in numbers:
        idx = int(num_str)
        if 1 <= idx <= max_molecules:
            selected.append(idx - 1)
        if len(selected) >= 10:
            break
    return selected[:10]

def select_molecules_with_llm(
    llm,
    task,
    generation_data,
    sampling_params,
    max_candidates,
    batch_size,
    model_len_limit,
):
    """
    Use LLM to select promising molecules for each generation.
    
    Args:
        llm: VLLM model
        task: Task name
        generation_data: Dict of generation to molecules
        sampling_params: Sampling parameters for LLM
        max_model_len: Maximum model context length
    Returns:
        Dict of generation to selected molecules
    """
    # Build global RAG index once
    for task_key in generation_data:
        all_smiles = [sm for sm, score in sum(generation_data.values(), [])]
        break  # Only need one pass, all generations pooled
    model, index = build_molecule_index(all_smiles)
    selected_data = {}
    for generation, molecules in generation_data.items():
        # Instead of slicing, retrieve full 100 candidates in a global order
        candidate_molecules = retrieve_all_molecules(model, index, molecules, k=len(molecules))
        prompt = create_selection_prompt(task, generation, candidate_molecules)
        response = llm_generate_and_log(llm, prompt, sampling_params, generation, candidate_molecules)
        indices = parse_llm_selection(response, len(candidate_molecules))
        selected_pool = [candidate_molecules[idx] for idx in indices if 0 <= idx < len(candidate_molecules)]
        selected_data[generation] = selected_pool[:10]
    return selected_data


def select_llm(args):
    """Instantiate and return the requested LLM backend."""
    # ---- vLLM backend ----
    if args.llm_option == "vllm":
        if not VLLM_AVAILABLE:
            raise RuntimeError("vLLM is not installed in this environment.")
        logger.info("Using vLLM (v0 engine, V100 compatible)…")
        llm = LLM(
            model=args.model,
            gpu_memory_utilization=args.gpu_memory_utilization,
            tensor_parallel_size=16,
            download_dir="/work/luketou123/llm_model",
            trust_remote_code=True,
            dtype="float16",
            max_model_len=args.max_model_len,
        )
        return llm

    # ---- Cerebras backend ----
    if args.llm_option == "cerebras":
        if not CEREBRAS_AVAILABLE:
            raise RuntimeError("Cerebras SDK is not installed in this environment.")
        logger.info("Using Cerebras Cloud LLM backend…")
        client = Cerebras(
            api_key=args.cerebras_api_key or os.environ.get("CEREBRAS_API_KEY", "")
        )
        return client

    # ---- Fallback ----
    raise ValueError(f"Unsupported llm_option: {args.llm_option}")


# --- process_one_generation definition ---

def process_one_generation(args_tuple):
    """
    Helper for ThreadPoolExecutor: select molecules for one generation using RAG.
    Args (packed tuple):
        llm: vLLM/Cerebras client
        task: task name (str)
        generation: generation index (int)
        molecules: list of (smiles, score) tuples
        sampling_params: VLLM SamplingParams or None
        cfg: tuple(max_candidates, batch_size, max_len)
        model: SentenceTransformer instance
        index: FAISS index
        tokenizer: HuggingFace tokenizer for token length check
        max_model_len: int for context limit
    Returns:
        (generation, selected_molecules) with up to 10 (smiles, score).
    """
    llm, task, generation, molecules, sampling_params, cfg, model, index, tokenizer, max_model_len = args_tuple

    # Automatically reduce candidate count to fit context
    for k in range(len(molecules), 0, -5):
        candidate_molecules = retrieve_all_molecules(model, index, molecules, k=k)
        prompt = create_selection_prompt(task, generation, candidate_molecules)
        if len(tokenizer.encode(prompt)) <= max_model_len:
            break
    else:
        # Fall back to a single candidate if nothing fits
        candidate_molecules = retrieve_all_molecules(model, index, molecules, k=1)
        prompt = create_selection_prompt(task, generation, candidate_molecules)
        tokens = tokenizer.encode(prompt)[:max_model_len]
        prompt = tokenizer.decode(tokens, skip_special_tokens=True)

    # Invoke LLM and parse response
    response = llm_generate_and_log(llm, prompt, sampling_params, generation, candidate_molecules)
    idxs = parse_llm_selection(response, len(candidate_molecules))
    selected = [candidate_molecules[i] for i in idxs if 0 <= i < len(candidate_molecules)]
    return generation, selected[:10]


# --- New: Load completed generations from output CSV ---
def load_completed_generations(output_path: str) -> set:
    """
    Return a set of generations already written to output CSV.
    """
    completed = set()
    if not os.path.exists(output_path):
        return completed
    with open(output_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                completed.add(int(row['generation']))
            except Exception:
                continue
    return completed

# --- New: Append one generation to CSV (thread-safe) ---
import threading
csv_write_lock = threading.Lock()
def append_generation_to_csv(generation: int, molecules: List[Tuple[str, float]], output_path: str):
    """
    Append selected molecules for one generation to CSV (thread-safe).
    """
    with csv_write_lock:
        file_exists = os.path.exists(output_path)
        with open(output_path, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['generation', 'smiles', 'score'])
            for smiles, score in molecules:
                writer.writerow([generation, smiles, score])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=False,
                        help='Task name (e.g., celecoxib, osimertinib)')
    parser.add_argument('--llm_option', type=str, required=True, choices=['vllm', 'cerebras'],
                        help='Choose the LLM to use: vllm or cerebras')
    parser.add_argument('--input_dir', type=str, default='results_graphga',
                        help='Input directory containing {task}.csv')
    parser.add_argument('--output_dir', type=str, default='results_llm_select',
                        help='Output directory for selected molecules')
    parser.add_argument('--model', type=str, default='deepseek-ai/DeepSeek-R1-Distill-Llama-70B',
                        help='LLM model to use')
    parser.add_argument('--temperature', type=float, default=0.7,
                        help='Temperature for LLM sampling')
    parser.add_argument('--max_tokens', type=int, default=8000,
                        help='Maximum tokens for LLM response')
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.9,
                        help='GPU memory utilization for VLLM')
    parser.add_argument('--max_model_len', type=int, default=4096,
                        help='Maximum model context length')
    parser.add_argument('--max_candidates', type=int, default=100,
                        help='Maximum number of molecules to include in each prompt before LLM selection (no score pre‑filter)')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='Number of molecules to include per LLM prompt batch')
    parser.add_argument('--enable_speculative', action='store_true',
                        help='Enable vLLM speculative decoding (requires vLLM>=0.4.2)')
    parser.add_argument('--draft_model', type=str, default='Qwen/Qwen3-1.7B',
                        help='Draft model ID for speculative decoding')
    parser.add_argument('--concurrency', type=int, default=1,
                        help='Number of concurrent generations (threaded)')
    parser.add_argument('--retry_attempts', type=int, default=10,
                        help='Number of times to retry a Cerebras request if rate‑limited')
    parser.add_argument('--retry_base_wait', type=int, default=1,
                        help='Base seconds for exponential back‑off when hitting rate‑limit')
    parser.add_argument('--cerebras_api_key', type=str, default="csk-38kjr3mnp22x9w2m2wejdej5m55cpkwhmh3p6p6f3wt2wxw2",
                        help='API key for Cerebras Cloud SDK')
    parser.add_argument('--tasks', nargs='+',
                        help='List of task names to process, e.g. celecoxib fexofenadine')
    parser.add_argument('--pipeline_parallel_size', type=int, default=1,
                        help='Number of nodes for pipeline parallelism (multi-node distributed)')
    parser.add_argument('--distributed_executor_backend', type=str, default=None,
                        help='Distributed backend for vLLM ("ray" for multi-node, "mp" for single node)')
    args = parser.parse_args()

    # Determine tasks to run
    if args.tasks:
        task_list = args.tasks
    elif args.task:
        task_list = [args.task]
    else:
        raise ValueError('Either --task or --tasks must be provided')

    # Initialize LLM and sampling params once
    if args.llm_option == 'vllm':
        llm_kwargs = {
            'model': args.model,
            'gpu_memory_utilization': args.gpu_memory_utilization,
            'tensor_parallel_size': 8,  # 建議直接用 8，或用 os.environ 自動偵測
            'download_dir': "/work/luketou123/llm_model",
            'trust_remote_code': True,
            'dtype': "float16",
            'max_model_len': args.max_model_len,
        }
        if args.pipeline_parallel_size > 1:
            llm_kwargs['pipeline_parallel_size'] = args.pipeline_parallel_size
        if args.distributed_executor_backend:
            llm_kwargs['distributed_executor_backend'] = args.distributed_executor_backend
        llm = LLM(**llm_kwargs)
        sampling_params = SamplingParams(
            temperature=args.temperature,
            max_tokens=args.max_tokens
        )
    else:
        llm = select_llm(args)
        sampling_params = None


    for task in task_list:
        input_path = os.path.join(args.input_dir, f"{task}.csv")
        output_path = os.path.join(args.output_dir, f"llm_{task}.csv")

        if not os.path.exists(input_path):
            print(f"Input file {input_path} does not exist, skipping {task}.")
            continue

        print(f"Loading molecules from {input_path}")
        generation_data = load_task_csv(input_path)
        print(f"Loaded {sum(len(mols) for mols in generation_data.values())} molecules from {len(generation_data)} generations.")

        # --- Load completed generations for resume ---
        completed_generations = load_completed_generations(output_path)
        print(f"Already completed generations: {sorted(completed_generations)}")

        # --- Only process unfinished generations ---
        unfinished = {gen: mols for gen, mols in generation_data.items() if gen not in completed_generations}
        print(f"Unfinished generations: {sorted(unfinished.keys())}")

        if not unfinished:
            print(f"All generations already completed for {task}.")
            continue

        print(f"Selecting molecules with {args.llm_option} for task {task}...")
        if args.llm_option == 'vllm':
            all_smiles = [sm for sm, score in sum(generation_data.values(), [])]
            model, index = build_molecule_index(all_smiles)
            cfg = (args.max_candidates, args.batch_size, args.max_model_len)
            if TRANSFORMERS_AVAILABLE:
                tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
            else:
                raise RuntimeError("transformers is required for tokenizer length check.")
            max_model_len = args.max_model_len
            with ThreadPoolExecutor(max_workers=args.concurrency) as exe:
                futures = {exe.submit(process_one_generation, (llm, task, gen, mols, sampling_params, cfg, model, index, tokenizer, max_model_len)): gen for gen, mols in unfinished.items()}
                for fut in as_completed(futures):
                    gen, sel = fut.result()
                    append_generation_to_csv(gen, sel, output_path)
        else:  # cerebras
            client = llm
            for generation, molecules in unfinished.items():
                candidate_molecules = molecules[:args.max_candidates] if args.max_candidates > 0 else molecules
                prompt = create_selection_prompt(task, generation, candidate_molecules)
                print('[Prompt]', prompt.replace('\n', ' '))
                success = False
                for attempt in range(1, args.retry_attempts + 1):
                    try:
                        stream = client.chat.completions.create(
                            messages=[
                                {"role": "system", "content": "You are a medicinal chemistry expert evaluating molecules for drug discovery."},
                                {"role": "user", "content": prompt}
                            ],
                            model=args.model,
                            stream=True,
                            max_completion_tokens=args.max_tokens,
                            temperature=args.temperature,
                            top_p=1
                        )
                        success = True
                        break
                    except RateLimitError:
                        wait_time = args.retry_base_wait * (2 ** (attempt - 1))
                        print(f"[RateLimit] attempt {attempt}/{args.retry_attempts}. Sleeping {wait_time}s …", flush=True)
                        time.sleep(wait_time)
                if not success:
                    continue
                response = "".join(chunk.choices[0].delta.content or "" for chunk in stream)
                print('[Response]', response.replace('\n', ' '))
                indices = parse_llm_selection(response, len(candidate_molecules))
                selected_pool = [candidate_molecules[idx] for idx in indices if 0 <= idx < len(candidate_molecules)]
                append_generation_to_csv(generation, selected_pool[:10], output_path)

        # 統計總數
        completed_generations = load_completed_generations(output_path)
        total_selected = 0
        with open(output_path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                total_selected += 1
        print(f"Total molecules selected: {total_selected}")
        print(f"Average per generation: {total_selected / max(1, len(completed_generations)):.2f}")

if __name__ == "__main__":
    main()
