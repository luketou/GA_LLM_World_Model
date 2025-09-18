#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, json, argparse, random, math, pickle, time
import platform
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd

# ---------- Optional FAISS ----------
try:
    import faiss
    HAS_FAISS = True
except Exception:
    HAS_FAISS = False

# ---------- Torch / HF ----------
# Preemptively disable CUDA on systems with old glibc (<2.27) to avoid CUDA preload at import
def _glibc_version_tuple():
    name, ver = platform.libc_ver()
    if not ver:
        return (0, 0)
    parts = ver.split(".")
    try:
        major = int(parts[0]) if len(parts) > 0 else 0
        minor = int(parts[1]) if len(parts) > 1 else 0
        return (major, minor)
    except Exception:
        return (0, 0)

if _glibc_version_tuple() < (2, 27) or os.environ.get("PYTORCH_FORCE_CPU", ""):
    os.environ.setdefault("PYTORCH_NO_CUDA", "1")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

# Robust import: fall back to CPU-only if CUDA deps are missing
try:
    import torch
except Exception as _e:  # GLIBC/libcublas/libcurand errors on older systems
    import importlib as _importlib
    os.environ.setdefault("PYTORCH_NO_CUDA", "1")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    sys.modules.pop("torch", None)
    torch = _importlib.import_module("torch")
from transformers import AutoTokenizer, AutoModel

# ---------- RDKit ----------
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdFingerprintGenerator
from rdkit import DataStructs

# ---------- LLM backends (optional) ----------
try:
    from vllm import LLM, SamplingParams
    HAS_VLLM = True
except Exception:
    HAS_VLLM = False

try:
    from cerebras.cloud.sdk import Cerebras
    from cerebras.cloud.sdk import RateLimitError
    HAS_CEREBRAS = True
except Exception:
    HAS_CEREBRAS = False

# ------------------- Data schema -------------------
@dataclass
class Rec:
    generation: int
    smiles: str
    score: float

def read_task_csv(path: str, max_generation: Optional[int] = None) -> Dict[int, List[Rec]]:
    df = pd.read_csv(path)
    need = {"generation","smiles","score"}
    assert need.issubset(df.columns), f"CSV must have columns {need}"
    out: Dict[int, List[Rec]] = {}
    for _, r in df.iterrows():
        try:
            g = int(r["generation"]); s = str(r["smiles"]); sc = float(r["score"])
            if max_generation is not None and g > int(max_generation):
                continue
            out.setdefault(g, []).append(Rec(generation=g, smiles=s, score=sc))
        except Exception:
            continue
    return out

# ------------------- SMILES utils -------------------
def randomized_smiles_list(smi: str, n: int = 4, include_canonical: bool = True) -> List[str]:
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return [smi]
    out = []
    if include_canonical:
        out.append(Chem.MolToSmiles(m, canonical=True))
    for _ in range(n):
        out.append(Chem.MolToSmiles(m, canonical=False, doRandom=True, isomericSmiles=True))
    # unique keep order
    seen=set(); uniq=[]
    for x in out:
        if x not in seen:
            seen.add(x); uniq.append(x)
    return uniq

# ------------------- Encoder -------------------
class ChemEncoder:
    def __init__(self, model_name: str, device: str = None, fp_bits: int = 2048, fp_radius: int = 2,
                 random_smiles_n: int = 4, dense_pool: str = "mean"):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device).eval()
        self.fp_bits = int(fp_bits)
        self.fp_radius = int(fp_radius)
        self.random_smiles_n = int(random_smiles_n)
        assert dense_pool in ("mean","cls")
        self.dense_pool = dense_pool
        try:
            self.fp_gen = rdFingerprintGenerator.GetMorganGenerator(radius=self.fp_radius, fpSize=self.fp_bits)
            self._use_fp_generator = True
        except Exception:
            self.fp_gen = None
            self._use_fp_generator = False
        # cache: smi -> {"dense":np.float32[D], "fp":bool[B], "on":int}
        self.cache: Dict[str, Dict[str, object]] = {}

    @torch.no_grad()
    def _encode_dense_once(self, smiles_batch: List[str]) -> np.ndarray:
        enc = self.tokenizer(smiles_batch, padding=True, truncation=True, return_tensors="pt", max_length=256)
        enc = {k:v.to(self.device) for k,v in enc.items()}
        out = self.model(**enc)
        if self.dense_pool == "cls":
            vec = out.last_hidden_state[:,0,:]
        else:
            token = out.last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).float()
            vec = (token * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        vec = torch.nn.functional.normalize(vec, dim=-1)
        return vec.detach().cpu().numpy()

    def _ecfp(self, smi: str) -> Tuple[np.ndarray, int]:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            arr = np.zeros((self.fp_bits,), dtype=np.uint8)
            return arr.astype(bool), 0
        if getattr(self, '_use_fp_generator', False) and self.fp_gen is not None:
            fp = self.fp_gen.GetFingerprint(m)
        else:
            fp = AllChem.GetMorganFingerprintAsBitVect(m, self.fp_radius, nBits=self.fp_bits)
        arr = np.zeros((self.fp_bits,), dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        on = int(arr.sum())
        return arr.astype(bool), on

    def embed_smiles(self, smi: str, use_cache: bool = True) -> Dict[str, object]:
        if use_cache and smi in self.cache:
            return self.cache[smi]
        variants = randomized_smiles_list(smi, n=self.random_smiles_n, include_canonical=True)
        batches = [variants[i:i+32] for i in range(0, len(variants), 32)]
        vecs = [self._encode_dense_once(b) for b in batches]
        v = np.concatenate(vecs, axis=0)  # [V,D]
        dense = v.mean(axis=0).astype(np.float32)
        fp, on = self._ecfp(smi)
        out = {"dense": dense, "fp": fp, "on": on}
        if use_cache:
            self.cache[smi] = out
        return out

# ------------------- Similarities -------------------
def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a,b) / (np.linalg.norm(a)*np.linalg.norm(b) + 1e-12))

def tanimoto_bool(q: np.ndarray, X: np.ndarray, q_on: int, X_on: np.ndarray) -> np.ndarray:
    inter = (X & q).sum(axis=1).astype(np.float32)
    union = (X_on + q_on - inter).astype(np.float32) + 1e-12
    return inter / union

# ------------------- Memory / RAG store -------------------
class MemoryStore:
    """
    持有歷史（< 當代）的分子，用於鄰居檢索與證據生成。
    - dense: FAISS (或 numpy)
    - sparse: 在證據時計算 Tanimoto
    """
    def __init__(self, use_faiss: bool = True):
        self.use_faiss = (use_faiss and HAS_FAISS)
        self.smis: List[str] = []
        self.gens: List[int] = []
        self.scores: List[float] = []
        self.dense: Optional[np.ndarray] = None  # [N,D] float32 normalized
        self.fp: Optional[np.ndarray] = None     # [N,B] bool
        self.on: Optional[np.ndarray] = None     # [N] int
        self.index = None

    def _rebuild_index(self):
        if self.dense is None or self.dense.shape[0] == 0:
            self.index = None
            return
        if self.use_faiss:
            idx = faiss.IndexFlatIP(self.dense.shape[1])
            idx.add(self.dense.astype(np.float32))
            self.index = idx
        else:
            self.index = None  # numpy fallback

    def add_records(self, encoder: ChemEncoder, recs: List[Rec]):
        if not recs: return
        vecs = []; fps = []; ons = []
        for r in recs:
            rep = encoder.embed_smiles(r.smiles, use_cache=True)
            vecs.append(rep["dense"][None,:])
            fps.append(rep["fp"][None,:])
            ons.append(rep["on"])
            self.smis.append(r.smiles)
            self.gens.append(r.generation)
            self.scores.append(r.score)
        V = np.concatenate(vecs, axis=0).astype(np.float32)
        F = np.concatenate(fps, axis=0).astype(bool)
        O = np.asarray(ons, dtype=np.int32)
        if self.dense is None:
            self.dense = V
            self.fp = F
            self.on = O
        else:
            self.dense = np.concatenate([self.dense, V], axis=0)
            self.fp = np.concatenate([self.fp, F], axis=0)
            self.on = np.concatenate([self.on, O], axis=0)
        self._rebuild_index()

    def knn_dense(self, qvec: np.ndarray, topk: int = 50, forbid_gen: Optional[int]=None) -> List[int]:
        if self.dense is None or self.dense.shape[0] == 0:
            return []
        if self.index is not None:
            # FAISS inner product == cosine (vectors normalized)
            D,I = self.index.search(qvec.reshape(1,-1).astype(np.float32), min(topk*3, self.dense.shape[0]))
            idxs = I[0].tolist()
        else:
            sims = self.dense @ qvec.reshape(-1,1)
            idxs = np.argsort(-sims.squeeze(1))[:min(topk*3, len(self.smis))].tolist()
        # 過濾本代
        if forbid_gen is not None:
            idxs = [i for i in idxs if self.gens[i] < forbid_gen]
        # 去重保留前 topk
        seen=set(); out=[]
        for i in idxs:
            if i not in seen:
                seen.add(i); out.append(i)
            if len(out) >= topk: break
        return out

# ------------------- Fusion calibration -------------------
def calibrate_alpha_beta(
    gens: Dict[int, List[Rec]],
    encoder: ChemEncoder,
    start_gen: int,
    end_gen: int,
    folds: int = 5,
    proto_topN: int = 20,
    proto_window: int = 5,
    topK: int = 30
) -> Tuple[float,float]:
    # 為簡潔，我們用「與上一代原型的 dense/sparse 相似度」去回歸當代的 min-max 分數
    # 冷啟：若資料不足則回傳 (0.6, 0.4)
    g_list = [g for g in sorted(gens.keys()) if start_gen <= g <= end_gen and len(gens[g])>0]
    if len(g_list) == 0:
        return 0.6, 0.4

    # 構造原型的工具（與主流程一致）
    def build_protos(g:int)->List[str]:
        protos=[]
        if g>0:
            prev = sorted(gens.get(g-1, []), key=lambda x: x.score, reverse=True)
            protos += [r.smiles for r in prev[:min(proto_topN, len(prev))]]
        if proto_window>1:
            remain = max(0, proto_topN - len(protos))
            per = max(1, remain // max(1, proto_window-1))
            for w in range(2, proto_window+1):
                gg = g - w
                if gg < 0: break
                pool = sorted(gens.get(gg, []), key=lambda x: x.score, reverse=True)
                protos += [r.smiles for r in pool[:min(per, len(pool))]]
        # unique
        seen=set(); u=[]
        for s in protos:
            if s not in seen:
                seen.add(s); u.append(s)
        return u

    # ridge on 2 features
    lam = 1e-3
    folds = max(2, int(folds))
    splits = [[] for _ in range(folds)]
    for i,g in enumerate(g_list):
        splits[i%folds].append(g)

    ws=[]
    for k in range(folds):
        val=set(splits[k]); tr=[g for g in g_list if g not in val]
        X=[]; y=[]
        for g in tr:
            protos = build_protos(g)
            # 聚合原型向量
            if protos:
                vps = [encoder.embed_smiles(s, use_cache=True)["dense"][None,:] for s in protos]
                vp = np.mean(np.concatenate(vps, axis=0), axis=0)
                vp = vp / (np.linalg.norm(vp)+1e-12)
                pfps = [encoder.embed_smiles(s, use_cache=True) for s in protos]
                P = np.concatenate([x["fp"][None,:] for x in pfps], axis=0)
                Pon = np.asarray([x["on"] for x in pfps], dtype=np.int32)
            else:
                vp = None; P=None; Pon=None

            # 候選
            recs = gens[g]
            scs = [r.score for r in recs]
            mn, mx = min(scs), max(scs)
            denom = (mx-mn) if (mx>mn) else 1.0
            for r in recs[:topK]:
                rep = encoder.embed_smiles(r.smiles, use_cache=True)
                d = cosine(rep["dense"], vp) if vp is not None else 0.0
                s = 0.0
                if P is not None:
                    s = float(tanimoto_bool(rep["fp"], P, rep["on"], np.asarray([x["on"] for x in pfps])) .max())
                X.append([d,s]); y.append((r.score-mn)/denom)
        if not X: continue
        X=np.asarray(X, dtype=np.float64); y=np.asarray(y, dtype=np.float64)
        XT=X.T; A=XT@X + lam*np.eye(2); b=XT@y
        w = np.linalg.solve(A,b)
        ws.append(w)
    if not ws:
        return 0.6, 0.4
    w = np.mean(np.stack(ws, axis=0), axis=0)
    w = np.maximum(w, 0.0)
    if w.sum() < 1e-9:
        return 0.6, 0.4
    return float(w[0]/w.sum()), float(w[1]/w.sum())

# ------------------- LLM prompt & parsing -------------------
def make_llm_prompt(g:int,
                    task:str,
                    candidates: List[Dict],
                    alpha: float,
                    beta: float,
                    few_shot: Optional[List[str]] = None) -> str:
    """
    candidates: list of dict per candidate:
      { 'idx':1-based, 'smiles', 'dense_proto', 'sparse_proto', 'hybrid',
        'neighbors':[{'smiles','score','dense_sim','tanimoto'} * up to 5] }
    """
    head = (
f"You are a medicinal chemistry expert selecting molecules for task: {task}.\n"
f"You are given 30 candidate SMILES of generation {g}. Each candidate includes:\n"
"- dense_proto: cosine similarity to prototype aggregate (ChemBERTa)\n"
"- sparse_proto: ECFP Tanimoto to prototypes\n"
"- hybrid = {alpha:.2f}*dense_proto + {beta:.2f}*sparse_proto\n"
"- up to 5 historical nearest neighbors with their oracle scores\n"
"Only use past generations' scores. Do NOT assume any score for current generation.\n"
"Select EXACTLY 10 indices (1..30) that are promising for further optimization, balancing score potential and structural diversity.\n"
"Return ONLY a JSON array of 10 integers (no other text).\n"
    )
    fs = ""
    if few_shot:
        fs = "Examples of good rationales (do not output rationale now):\n" + "\n".join(few_shot) + "\n"

    lines = []
    for c in candidates:
        nbr = "; ".join([f"{n['smiles']}|score={n['score']:.3f}|d={n['dense_sim']:.3f}|t={n['tanimoto']:.3f}"
                         for n in c.get("neighbors",[])])
        lines.append(
            f"{c['idx']:>2d}. SMILES={c['smiles']} | dense_proto={c['dense_proto']:.3f} | "
            f"sparse_proto={c['sparse_proto']:.3f} | hybrid={c['hybrid']:.3f} | neighbors=[{nbr}]"
        )
    body = "\n".join(lines)
    tail = "\nOutput format example:\n[1,5,8,12,15,19,21,23,27,30]\n"
    return head + fs + body + tail

def parse_llm_selection(text: str, max_idx: int) -> List[int]:
    import re, json
    m = re.search(r'\[(.*?)\]', text, re.S)
    if m:
        try:
            arr = json.loads("[" + m.group(1) + "]")
            out = [int(x) for x in arr if isinstance(x,int) and 1 <= x <= max_idx]
            if len(out) == 10:
                return out
        except Exception:
            pass
    # fallback: greedy digits
    nums = re.findall(r'\d+', text)
    out=[]
    for s in nums:
        v=int(s)
        if 1<=v<=max_idx and v not in out:
            out.append(v)
        if len(out)>=10: break
    return out[:10]

# ------------------- LLM wrappers -------------------
class LLMWrapper:
    def __init__(self, backend: str, model: str, temperature: float, max_tokens: int,
                 cerebras_api_keys: Optional[List[str]] = None,
                 vllm_kwargs: Optional[dict] = None):
        self.backend = backend
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.keys = cerebras_api_keys or []
        self.key_idx = 0
        self.client = None
        self.vllm_kwargs = vllm_kwargs or {}
        if backend == "vllm":
            assert HAS_VLLM, "vLLM not available"
            self.client = LLM(model=model, **self.vllm_kwargs)
            self.sampling = SamplingParams(temperature=temperature, max_tokens=max_tokens)
        elif backend == "cerebras":
            assert HAS_CEREBRAS, "Cerebras SDK not available"
            if not self.keys:
                raise RuntimeError("cerebras backend requires --cerebras-api-keys")
            self.client = Cerebras(api_key=self.keys[self.key_idx])
        else:
            raise ValueError("backend must be 'vllm' or 'cerebras'")

    def _rotate_key(self):
        if self.backend != "cerebras": return
        self.key_idx = (self.key_idx + 1) % len(self.keys)
        self.client = Cerebras(api_key=self.keys[self.key_idx])

    def generate(self, prompt: str) -> str:
        if self.backend == "vllm":
            out = self.client.generate([prompt], self.sampling)
            return out[0].outputs[0].text
        else:
            # cerebras streaming with retry
            attempts = 0
            while True:
                try:
                    stream = self.client.chat.completions.create(
                        messages=[{"role":"system", "content":"You are a precise model selection agent. Output only JSON array of 10 integers."},
                                  {"role":"user", "content":prompt}],
                        model=self.model,
                        stream=True,
                        temperature=self.temperature,
                        max_completion_tokens=self.max_tokens,
                        top_p=1
                    )
                    txt = "".join(chunk.choices[0].delta.content or "" for chunk in stream)
                    return txt
                except RateLimitError:
                    attempts += 1
                    time.sleep(1.0)
                    self._rotate_key()
                except Exception as e:
                    raise e

# ------------------- Evidence builder per generation -------------------
def build_prototypes(g:int, gens:Dict[int,List[Rec]], proto_topN:int, proto_window:int) -> List[str]:
    protos=[]
    if g>0:
        prev = sorted(gens.get(g-1,[]), key=lambda x: x.score, reverse=True)
        protos += [r.smiles for r in prev[:min(proto_topN, len(prev))]]
    if proto_window>1:
        remain = max(0, proto_topN - len(protos))
        per = max(1, remain // max(1, proto_window-1))
        for w in range(2, proto_window+1):
            gg = g - w
            if gg < 0: break
            pool = sorted(gens.get(gg,[]), key=lambda x: x.score, reverse=True)
            protos += [r.smiles for r in pool[:min(per, len(pool))]]
    # unique
    seen=set(); u=[]
    for s in protos:
        if s not in seen:
            seen.add(s); u.append(s)
    return u

def build_evidence_for_generation(
    g: int,
    task: str,
    gens: Dict[int,List[Rec]],
    encoder: ChemEncoder,
    memory: MemoryStore,
    alpha: float,
    beta: float,
    topK_emit: int = 30,
    neighbors_k: int = 5,
) -> Tuple[List[Dict], Dict]:
    """
    回傳：
      candidates_for_llm: list of dict (長度<=30)
      raw_evidence: 任務內部紀錄（可另存 jsonl）
    """
    recs = gens.get(g, [])
    # 保守處理：實務上每代就是 30
    recs = recs[:topK_emit]
    protos = build_prototypes(g, gens, proto_topN=20, proto_window=5)

    # 聚合原型向量
    if protos:
        vps = [encoder.embed_smiles(s, use_cache=True)["dense"][None,:] for s in protos]
        vp = np.mean(np.concatenate(vps, axis=0), axis=0)
        vp = vp / (np.linalg.norm(vp)+1e-12)
        pfps = [encoder.embed_smiles(s, use_cache=True) for s in protos]
        P = np.concatenate([x["fp"][None,:] for x in pfps], axis=0)
        Pon = np.asarray([x["on"] for x in pfps], dtype=np.int32)
    else:
        vp = None; P=None; Pon=None

    cands=[]
    ev_all={}
    for idx1, r in enumerate(recs, start=1):
        rep = encoder.embed_smiles(r.smiles, use_cache=True)
        dsim = cosine(rep["dense"], vp) if vp is not None else 0.0
        tsim = 0.0
        if P is not None:
            tsim = float(tanimoto_bool(rep["fp"], P, rep["on"], Pon).max())
        hybrid = alpha*dsim + beta*tsim

        # neighbors from history (< g)
        nbr_idx = memory.knn_dense(rep["dense"], topk=max(50, neighbors_k*5), forbid_gen=g)
        # re-rank by hybrid to reduce false positives
        neighbors=[]
        for j in nbr_idx:
            td = float(np.dot(rep["dense"], memory.dense[j]))
            tt = float(tanimoto_bool(rep["fp"], memory.fp[j:j+1], rep["on"], memory.on[j:j+1])[0])
            score_j = float(memory.scores[j])
            hybrid_j = alpha*td + beta*tt
            neighbors.append((hybrid_j, j, td, tt, score_j))
        neighbors.sort(key=lambda x: -x[0])
        neighbors = neighbors[:neighbors_k]
        nb_pack = [{"smiles": memory.smis[j], "score": score_j, "dense_sim": td, "tanimoto": tt}
                   for (_, j, td, tt, score_j) in neighbors]

        cands.append({
            "idx": idx1,
            "smiles": r.smiles,
            "dense_proto": float(dsim),
            "sparse_proto": float(tsim),
            "hybrid": float(hybrid),
            "neighbors": nb_pack
        })
        ev_all[r.smiles] = {"dense_proto": float(dsim), "sparse_proto": float(tsim), "neighbors": nb_pack}

    return cands, ev_all

# ------------------- Main pipeline -------------------
def run(args):
    # 讀入 CSV
    gens = read_task_csv(args.csv, max_generation=args.max_generation)
    print(f"[Input] generations={len(gens)}, total={sum(len(v) for v in gens.values())}")

    score_lookup = {}
    for _g, _recs in gens.items():
        for _r in _recs:
            score_lookup[(_g, _r.smiles)] = _r.score

    # Encoder
    enc = ChemEncoder(
        model_name=args.model_name,
        device=args.device,
        fp_bits=args.ecfp_bits,
        fp_radius=args.ecfp_radius,
        random_smiles_n=args.random_smiles_n,
        dense_pool=args.dense_pool
    )

    # 載入/保存 embedding 快取
    os.makedirs(args.outdir, exist_ok=True)
    cache_path = os.path.join(args.outdir, "emb_cache.pkl")
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            enc.cache = pickle.load(f)
        print(f"[Cache] loaded {len(enc.cache)} entries")
    else:
        print("[Cache] fresh")

    # 記憶庫（初始化為空；稍後逐代追加歷史）
    memory = MemoryStore(use_faiss=(args.dense_index=="faiss"))

    # 0–30 代校準 α/β（可選）
    if args.calibrate:
        alpha, beta = calibrate_alpha_beta(
            gens, enc, start_gen=0, end_gen=min(30, max(gens.keys())), folds=args.folds,
            proto_topN=args.proto_topN, proto_window=args.proto_window, topK=args.topK
        )
        print(f"[Calibrate] alpha={alpha:.3f}, beta={beta:.3f}")
    else:
        alpha, beta = args.alpha, args.beta
        print(f"[Config] alpha={alpha}, beta={beta}")
    with open(os.path.join(args.outdir, "fusion_weights.json"), "w") as f:
        json.dump({"alpha":alpha, "beta":beta}, f, indent=2)

    # LLM 後端
    llm_kwargs = {}
    if args.llm_backend == "vllm":
        assert HAS_VLLM, "vLLM not available"
        llm_kwargs = dict(
            tensor_parallel_size=args.tensor_parallel_size,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            dtype="float16",
            trust_remote_code=True,
        )
    llm = LLMWrapper(
        backend=args.llm_backend,
        model=args.llm_model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        cerebras_api_keys=args.cerebras_api_keys,
        vllm_kwargs=llm_kwargs
    )

    # 輸出檔
    sel_path = os.path.join(args.outdir, "selected_top10.csv")
    with open(sel_path, "w", newline="") as f:
        pd.DataFrame(columns=["generation","smiles","rank_final","score"]).to_csv(f, index=False)

    evidence_path = os.path.join(args.outdir, "retrieved_evidence.jsonl")
    ev_f = open(evidence_path, "w", encoding="utf-8")

    # 逐代處理（關鍵：LLM 做 30→10）
    for g in sorted(gens.keys()):
        # 構建當代 30 的證據（只使用 <g 的歷史）
        cands, ev = build_evidence_for_generation(
            g, args.task, gens, enc, memory, alpha, beta,
            topK_emit=args.topK, neighbors_k=args.neighbors_k
        )

        # 寫入 jsonl（LLM 亦可直接用此 jsonl 作為輸入）
        ev_line = {"generation": g, "candidates": cands}
        ev_f.write(json.dumps(ev_line, ensure_ascii=False) + "\n")

        # 構建 prompt → LLM
        prompt = make_llm_prompt(g, args.task, cands, alpha, beta, few_shot=None)
        print(f"[Prompt Gen {g}] >>>")
        print(prompt)
        raw = llm.generate(prompt)
        print(f"[LLM Output Gen {g}] <<<")
        print(raw)
        idx10 = parse_llm_selection(raw, max_idx=len(cands))
        if len(idx10) != 10:
            # 嚴格兜底：按 hybrid 排序取前 10
            idx10 = [c["idx"] for c in sorted(cands, key=lambda x:-x["hybrid"])[:10]]

        # 寫 selected_top10
        rows = []
        for r, i in enumerate(idx10):
            smi = cands[i-1]["smiles"]
            rows.append({
                "generation": g,
                "smiles": smi,
                "rank_final": r + 1,
                "score": score_lookup.get((g, smi))
            })
        pd.DataFrame(rows).to_csv(sel_path, mode="a", header=False, index=False)

        # ---- 線上更新記憶庫：把「當代全部 30」加入（含真分數），供下一代檢索 ----
        memory.add_records(enc, gens[g])

        print(f"[Gen {g}] selected {len(idx10)}; memory size={len(memory.smis)}")

    ev_f.close()

    # 保存快取
    with open(cache_path, "wb") as f:
        pickle.dump(enc.cache, f)
    print(f"[Output] wrote {sel_path}")
    print(f"[Output] wrote {evidence_path}")
    print("[Done]")

# ------------------- CLI -------------------
def build_argparser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=str, default="amlodipine")
    ap.add_argument("--csv", type=str, help="path to CSV: generation,smiles,score", default="data/offspring/amlodipine.csv")
    ap.add_argument("--outdir", type=str, default="results/ChemBERTa_RAG")
    ap.add_argument("--max-generation", dest="max_generation", type=int, default=None,
                    help="Only load/process generations <= this value")
    # encoder / fingerprints
    ap.add_argument("--model-name", type=str, default="seyonec/ChemBERTa-zinc-base-v1")
    # Default to CPU to avoid touching torch at arg-parse time on old systems
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--dense-pool", type=str, default="mean", choices=["mean","cls"])
    ap.add_argument("--random-smiles-n", type=int, default=4)
    ap.add_argument("--ecfp-bits", type=int, default=2048)
    ap.add_argument("--ecfp-radius", type=int, default=2)
    # memory / index
    ap.add_argument("--dense-index", type=str, default=("faiss" if HAS_FAISS else "none"), choices=["faiss","none"])
    # retrieval knobs
    ap.add_argument("--topK", type=int, default=30, help="number of candidates emitted to LLM per generation (your case=30)")
    ap.add_argument("--neighbors-k", type=int, default=5, help="# historical neighbors per candidate")
    ap.add_argument("--proto-topN", type=int, default=20)
    ap.add_argument("--proto-window", type=int, default=5)
    # fusion
    ap.add_argument("--alpha", type=float, default=0.6)
    ap.add_argument("--beta", type=float, default=0.4)
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--folds", type=int, default=5)
    # LLM
    ap.add_argument("--llm-backend", type=str, choices=["cerebras","vllm"], default="cerebras")
    ap.add_argument("--llm-model", type=str, default="gpt-oss-120b")
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--max-tokens", type=int, default=45536)
    ap.add_argument("--cerebras-api-keys", nargs="*", default=[])
    # vllm-only
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    return ap

if __name__ == "__main__":
    args = build_argparser().parse_args()
    run(args)
