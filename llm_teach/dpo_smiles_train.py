#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pure-DPO pipeline for SMILES with per-generation blind Top-10 selection.
- Train DPO on generations < warmup_gens (default 30) using true scores to build preference pairs.
- For generations >= warmup_gens: BEFORE reading scores, select Top-10 by DPO margin
    r_hat(y) := log pi_theta(y) - log pi_ref(y)   (length-normalized; optional length penalty)
  After selection, reveal scores only for the selected generation to "grade" and then
  continue DPO updates using that generation's (winner, loser) pairs.
- Only legality check is allowed (optional RDKit MolFromSmiles). No property computation.

References: DPO original (Rafailov et al. 2023); Molecular DPO adaptation (2025).
"""

import os, math, csv, json, argparse, random
import os as _os
from dataclasses import dataclass
from typing import List, Tuple, Dict
from collections import deque

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.cuda.amp import autocast, GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP

def unwrap_model(m): 
    return m.module if isinstance(m, DDP) else m

# ---- Optional RDKit for legality (MolFromSmiles only) ----
HAS_RDKIT = False
try:
    from rdkit import Chem
    HAS_RDKIT = True
except Exception:
    HAS_RDKIT = False

# ---- Optional Optuna for HPO ----
HAS_OPTUNA = False
try:
    import optuna
    from optuna.pruners import MedianPruner
    HAS_OPTUNA = True
except Exception:
    HAS_OPTUNA = False

# ---- Rank correlation helpers (Spearman/Kendall) without SciPy ----
def _rankdata(x):
    """Average ranks for ties; returns 1-indexed average ranks as floats."""
    x = np.asarray(x)
    order = np.argsort(x)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(x) + 1)
    i = 0
    while i < len(x):
        j = i
        # group equal values
        while j + 1 < len(x) and x[order[j + 1]] == x[order[i]]:
            j += 1
        if j > i:
            avg = (i + j + 2) / 2.0  # 1-indexed average
            for k in range(i, j + 1):
                ranks[order[k]] = avg
        i = j + 1
    return ranks

def spearman_rho(a, b):
    """Spearman correlation via Pearson on ranks; robust to ties."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size != b.size or a.size == 0:
        return float('nan')
    ra = _rankdata(a)
    rb = _rankdata(b)
    ra = (ra - ra.mean()) / (ra.std() + 1e-12)
    rb = (rb - rb.mean()) / (rb.std() + 1e-12)
    return float(np.clip((ra * rb).mean(), -1.0, 1.0))

def kendall_tau(a, b):
    """Kendall tau (tau-b-lite): ignores pairs tied in either ranking.
       O(n^2) which is fine for Top-K (<=10)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n = a.size
    if n != b.size or n < 2:
        return float('nan')
    ra = _rankdata(a)
    rb = _rankdata(b)
    conc = 0
    disc = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            s1 = np.sign(ra[j] - ra[i])
            s2 = np.sign(rb[j] - rb[i])
            v = s1 * s2
            if v > 0: conc += 1
            elif v < 0: disc += 1
    denom = conc + disc
    if denom == 0:
        return 0.0
    return float((conc - disc) / denom)

# ---------------- Data structures ----------------
@dataclass
class Rec:
    gen: int
    s: str
    score: float

# --------------- Tokenizer (char-level SMILES) ---------------
SPECIAL = ["<pad>","<bos>","<eos>"]
class CharSmilesTok:
    def __init__(self, smiles_list: List[str], extra_tokens: List[str]=None):
        charset = set()
        for smi in smiles_list:
            for ch in smi.strip():
                charset.add(ch)
        self.two = {"Cl","Br"}  # keep as single tokens if present
        toks = SPECIAL + sorted(list(charset))
        if extra_tokens:
            for t in extra_tokens:
                if t not in toks: toks.append(t)
        self.itos = toks
        self.stoi = {t:i for i,t in enumerate(self.itos)}
        self.pad_id = self.stoi["<pad>"]; self.bos_id = self.stoi["<bos>"]; self.eos_id = self.stoi["<eos>"]
    def encode(self, smi: str, add_bos=True, add_eos=True):
        ids = [self.bos_id] if add_bos else []
        i=0
        while i < len(smi):
            if i+1 < len(smi) and smi[i:i+2] in self.two and smi[i:i+2] in self.stoi:
                ids.append(self.stoi[smi[i:i+2]]); i+=2
            else:
                ch = smi[i]
                if ch in self.stoi: ids.append(self.stoi[ch])
                i+=1
        if add_eos: ids.append(self.eos_id)
        return ids
    def decode(self, ids: List[int]):
        toks = [self.itos[i] for i in ids if i not in (self.pad_id,self.bos_id,self.eos_id)]
        return "".join(toks)
    @property
    def vocab_size(self): return len(self.itos)

# --------------- Lightweight GRU LM ----------------
class GRULM(nn.Module):
    def __init__(self, V, d=256, h=512, L=2, drop=0.1):
        super().__init__()
        self.embed = nn.Embedding(V, d)
        self.gru = nn.GRU(d, h, num_layers=L, batch_first=True, dropout=drop)
        self.head = nn.Linear(h, V)
    def forward(self, ids, h0=None):
        x = self.embed(ids)
        y, h = self.gru(x, h0)
        return self.head(y), h
    def seq_logp(self, ids, mask):
        # teacher-forced total log-likelihood per sequence (length-normalized)
        logits,_ = self.forward(ids[:,:-1])
        tgt = ids[:,1:]; m = mask[:,1:].float()
        logprobs = F.log_softmax(logits, dim=-1).gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        L = m.sum(dim=1).clamp(min=1.0)
        return (logprobs * m).sum(dim=1) / L

# --------------- DDP helpers ---------------
def ddp_is_available():
    return torch.cuda.is_available() and dist.is_available()

def ddp_world_size():
    return dist.get_world_size() if (ddp_is_available() and dist.is_initialized()) else 1

def ddp_rank():
    return dist.get_rank() if (ddp_is_available() and dist.is_initialized()) else 0

def ddp_local_rank():
    # torchrun sets LOCAL_RANK env var
    return int(_os.environ.get("LOCAL_RANK", 0))

def is_main_process():
    return ddp_rank() == 0

def init_distributed(backend: str = "nccl"):
    if dist.is_initialized():
        return
    # torchrun will set RANK, WORLD_SIZE, LOCAL_RANK envs
    if "RANK" in _os.environ and "WORLD_SIZE" in _os.environ:
        dist.init_process_group(backend=backend, init_method="env://")
        # map device best-effort
        try:
            vis = _os.environ.get("CUDA_VISIBLE_DEVICES", "")
            if vis:
                lst = [x for x in vis.split(",") if x.strip() != ""]
                nvis = len(lst)
            else:
                nvis = torch.cuda.device_count()
            lr = ddp_local_rank()
            if nvis > 0 and lr < nvis:
                torch.cuda.set_device(lr)
        except Exception:
            pass

# --------------- Utilities ----------------
def set_seed(seed:int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def read_csv(path:str)->List[Rec]:
    df = pd.read_csv(path)
    need = {"generation","smiles","score"}
    assert need.issubset(set(df.columns)), f"CSV must have columns {need}"
    out=[]
    for _,r in df.iterrows():
        try:
            g = int(r["generation"]); s = str(r["smiles"]); sc = float(r["score"])
            out.append(Rec(g,s,sc))
        except Exception: 
            pass
    return out

# verify smiles legality (RDKit MolFromSmiles)
def legal(smi:str)->bool:
    if not HAS_RDKIT: return True
    try:
        return Chem.MolFromSmiles(smi) is not None
    except Exception:
        return False

def batchify(tokenizer:CharSmilesTok, seqs:List[str], device):
    ids = [tokenizer.encode(s) for s in seqs]
    M = max(len(x) for x in ids)
    arr = np.full((len(ids),M), tokenizer.pad_id, dtype=np.int64)
    msk = np.zeros((len(ids),M), dtype=np.int64)
    for i,seq in enumerate(ids):
        arr[i,:len(seq)] = seq
        msk[i,:len(seq)] = 1
    ids = torch.as_tensor(arr, device=device)
    msk = torch.as_tensor(msk, device=device)
    return ids, msk

# DPO loss (Rafailov et al. 2023)
def dpo_loss(policy:GRULM, ref:GRULM, tok:CharSmilesTok, winners:List[str], losers:List[str], device, beta:float=0.1):
    w_ids, w_m = batchify(tok, winners, device)
    l_ids, l_m = batchify(tok, losers,  device)
    with torch.no_grad():
        ref.eval()
        lp_ref_w = ref.seq_logp(w_ids, w_m)
        lp_ref_l = ref.seq_logp(l_ids, l_m)
    policy.train()
    with autocast(enabled=torch.cuda.is_available()):
        lp_pol_w = policy.seq_logp(w_ids, w_m)
        lp_pol_l = policy.seq_logp(l_ids, l_m)
    margin = (lp_pol_w - lp_ref_w) - (lp_pol_l - lp_ref_l)  # [B]
    return -F.logsigmoid(beta * margin).mean()

# Build preference pairs from scored data with minimum score gap Δs
def build_pairs_scored(recs:List[Rec], delta_min:float, max_pairs:int, seed:int=0)->List[Tuple[str,str]]:
    rng = random.Random(seed)
    recs = [r for r in recs if legal(r.s)]
    recs = sorted(recs, key=lambda x: x.score, reverse=True)
    if len(recs) < 2: return []
    topk = max(1, len(recs)//10)
    T = recs[:topk]; R = recs[topk:]
    pairs=[]
    for _ in range(max_pairs*3):
        if not R: break
        w = rng.choice(T); l = rng.choice(R)
        if (w.score - l.score) >= delta_min and w.s != l.s:
            pairs.append((w.s,l.s))
        if len(pairs) >= max_pairs: break
    # dedup
    seen=set(); uniq=[]
    for p in pairs:
        if p not in seen: seen.add(p); uniq.append(p)
    return uniq

# Advanced pair builder with stratified/hard/mix modes
def build_pairs_scored_adv(recs:List[Rec], delta_min:float, max_pairs:int, mode:str="mix", mix_alpha:float=0.7, seed:int=0)->List[Tuple[str,str]]:
    rng = random.Random(seed)
    recs = [r for r in recs if legal(r.s)]
    recs = sorted(recs, key=lambda x: x.score, reverse=True)
    n = len(recs)
    if n < 2:
        return []
    # wider top bin for small generations
    topk = max(5, min(50, int(0.3*n)))
    T = recs[:topk]
    R = recs[topk:]
    R1 = recs[topk:min(2*topk, n)] if topk < n else []  # near-boundary negatives

    pairs=[]; tries=0; budget=max_pairs*5
    while len(pairs) < max_pairs and tries < budget:
        tries += 1
        strat = (mode == "stratified") or (mode == "mix" and rng.random() < mix_alpha)
        if strat or not R1:
            if not R: break
            w = rng.choice(T); l = rng.choice(R)
        else:
            if not R1: break
            w = rng.choice(T); l = rng.choice(R1)
        if w.s != l.s and (w.score - l.score) >= delta_min:
            pairs.append((w.s, l.s))
    # dedup
    uniq=[]; seen=set()
    for p in pairs:
        if p not in seen:
            seen.add(p); uniq.append(p)
    return uniq

# DPO "proxy reward" for blind selection with optional length penalty
@torch.no_grad()
def dpo_proxy_score(policy:GRULM, ref:GRULM, tok:CharSmilesTok, seqs:List[str], device, len_penalty:float=0.0)->np.ndarray:
    ids, m = batchify(tok, seqs, device)
    policy.eval(); ref.eval()
    lp_pol = policy.seq_logp(ids, m)   # length-normalized
    lp_ref = ref.seq_logp(ids, m)
    margin = (lp_pol - lp_ref)
    if len_penalty and len_penalty != 0.0:
        L = m.sum(dim=1).float()
        margin = margin - float(len_penalty) * L
    return margin.detach().cpu().numpy()

# --------------- Sharding utilities ---------------
def shard_list(xs, world_size: int, rank: int):
    if world_size <= 1:
        return xs
    return xs[rank::world_size]

# --------------- LR scheduler helper ---------------
def build_cosine_warmup_scheduler(optimizer, total_steps:int, warmup_ratio:float):
    total_steps = max(1, int(total_steps))
    warm = max(1, int(total_steps * max(0.0, min(1.0, warmup_ratio))))
    def lr_lambda(step_idx: int):
        t = step_idx + 1
        if t <= warm:
            return t / float(warm)
        # cosine from warm..total
        progress = (t - warm) / float(max(1, total_steps - warm))
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, progress))))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# ---------------- Core training (single run) ----------------
def run_training(args):
    os.makedirs(args.out, exist_ok=True)
    ckdir = os.path.join(args.out, "checkpoints"); os.makedirs(ckdir, exist_ok=True)
    logf = None
    gt_f, pm_f, gt_writer, pm_writer = None, None, None, None
    met_f, met_writer = None, None
    if is_main_process():
        if not getattr(args, "nosave", False):
            logf = open(os.path.join(args.out, "logs.txt"), "w", buffering=1)
        else:
            logf = None
        if not getattr(args, "nosave", False):
            ground_truth_path = os.path.join(args.out, "ground_truth.csv")
            proxy_margins_path = os.path.join(args.out, "proxy_margins.csv")
            metrics_path = os.path.join(args.out, "metrics.csv")
            gt_f = open(ground_truth_path, "w", newline="")
            pm_f = open(proxy_margins_path, "w", newline="")
            met_f = open(metrics_path, "w", newline="")
            gt_writer = csv.writer(gt_f)
            pm_writer = csv.writer(pm_f)
            met_writer = csv.writer(met_f)
            gt_writer.writerow(["generation", "smiles", "true_score"])
            pm_writer.writerow(["generation", "smiles", "proxy_margin"])
            met_writer.writerow(["generation", "k", "spearman", "kendall"])  # correlations within blind Top-K
    def log(msg):
        if is_main_process():
            print(msg)
            if logf is not None:
                print(msg, file=logf)

    set_seed(args.seed)
    data = read_csv(args.csv)
    all_smiles = [r.s for r in data]
    tok = CharSmilesTok(all_smiles, extra_tokens=list({"Cl","Br"}))

    # --- DDP init ---
    if args.dist:
        init_distributed(backend="nccl")
        local_rank = ddp_local_rank()
        # Map local_rank -> visible device index robustly
        if torch.cuda.is_available():
            vis = _os.environ.get("CUDA_VISIBLE_DEVICES", "")
            if vis:
                lst = [x for x in vis.split(",") if x.strip() != ""]
                nvis = len(lst)
            else:
                nvis = torch.cuda.device_count()
            if nvis == 0:
                device = "cpu"
            else:
                if local_rank >= nvis:
                    if is_main_process():
                        print(f"[WARN] LOCAL_RANK={local_rank} but only {nvis} visible CUDA devices; falling back to cuda:0")
                    local_rank = 0
                device = f"cuda:{local_rank}"
        else:
            device = "cpu"
    else:
        device = args.device

    if str(device).startswith("cuda"):
        parts = str(device).split(":")
        if len(parts) > 1 and parts[1].isdigit():
            idx = int(parts[1])
        else:
            idx = 0
            device = f"cuda:{idx}"
        count = torch.cuda.device_count()
        if idx >= count and count > 0:
            if is_main_process():
                print(f"[WARN] Requested cuda:{idx} but only {count} visible; falling back to cuda:0")
            idx = 0
            device = "cuda:0"
        torch.cuda.set_device(idx)

    # init models (place on local device before wrapping)
    V = tok.vocab_size
    policy = GRULM(V, d=args.emb, h=args.hid, L=args.layers, drop=args.dropout).to(device)
    ref    = GRULM(V, d=args.emb, h=args.hid, L=args.layers, drop=args.dropout).to(device)
    ref.load_state_dict(policy.state_dict())
    for p in ref.parameters(): p.requires_grad_(False)

    if args.dist:
        dev_id = int(str(device).split(":")[-1]) if str(device).startswith("cuda") else None
        policy = DDP(policy, device_ids=[dev_id] if dev_id is not None else None,
                     output_device=dev_id if dev_id is not None else None,
                     find_unused_parameters=False)

    opt = torch.optim.AdamW(unwrap_model(policy).parameters(), lr=args.lr, betas=(0.9,0.95), weight_decay=0.01)
    scaler = GradScaler(enabled=(args.amp and torch.cuda.is_available()))

    # LR scheduler (estimate total steps: warmup + online)
    mb = max(1, args.batch_size)
    # very rough estimate; ok for shaping schedule
    est_warm_steps = (args.pairs_per_epoch // mb) * max(1, args.epochs)
    gens = sorted(set(r.gen for r in data))
    n_online = sum(1 for g in gens if g >= args.warmup_gens)
    est_online_steps = n_online * max(1, args.online_steps)
    scheduler = build_cosine_warmup_scheduler(opt, est_online_steps + est_warm_steps, args.warmup_ratio) if args.cosine_schedule else None
    global_step = 0

    # split by generation with HARD limits per user spec:
    # - warmup (< warmup_gens): each generation may contribute at most 30 molecules (by true score desc)
    # - online (>= warmup_gens): each generation can reveal at most Top-K=10 after blind selection; only those enter training
    gens = sorted(set(r.gen for r in data))

    # Build warmup pool by capping EACH generation to 30 items (highest scores) to avoid leakage and maintain fairness
    warm: List[Rec] = []
    for g in gens:
        if g < args.warmup_gens:
            g_recs = [r for r in data if r.gen == g]
            if len(g_recs) > 30:
                g_recs = sorted(g_recs, key=lambda x: x.score, reverse=True)[:30]
            warm.extend(g_recs)

    # Online candidates are kept full for blind selection; training will later restrict to Top-K only
    online_by_gen: Dict[int, List[Rec]] = {}
    for g in gens:
        if g >= args.warmup_gens:
            online_by_gen[g] = [r for r in data if r.gen == g]

    # ---------- Stage A: initial DPO on warmup generations (with scores) ----------
    def delta_for_epoch(ep:int):
        t = 0 if args.epochs<=1 else ep/(args.epochs-1)
        return args.delta_start + (args.delta_end-args.delta_start)*t

    if warm:
        log(f"[Warmup] generations < {args.warmup_gens}, samples={len(warm)}")
        for ep in range(1, args.epochs+1):
            delta = delta_for_epoch(ep-1)
            pairs = build_pairs_scored_adv(warm, delta_min=delta, max_pairs=args.pairs_per_epoch,
                                           mode=args.pairs_mode, mix_alpha=args.mix_alpha, seed=args.seed+ep)
            pairs = shard_list(pairs, ddp_world_size(), ddp_rank())
            random.shuffle(pairs)
            losses=[]
            for i in range(0, len(pairs), mb):
                batch = pairs[i:i+mb]
                wins = [w for (w,l) in batch]
                loses= [l for (w,l) in batch]
                loss = dpo_loss(policy, ref, tok, wins, loses, device, beta=args.beta)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(unwrap_model(policy).parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                if scheduler is not None:
                    scheduler.step(); global_step += 1
                losses.append(loss.item())
            avg = float(np.mean(losses)) if losses else float("nan")
            if is_main_process() and not getattr(args, "nosave", False):
                to_save = unwrap_model(policy)
                torch.save({"model":to_save.state_dict(), "vocab":tok.itos}, os.path.join(ckdir, f"policy_warm_ep{ep}.pt"))
            # refresh reference halfway（穩定化）
            if ep == (args.epochs//2):
                src = unwrap_model(policy)
                ref.load_state_dict(src.state_dict())
                for p in ref.parameters(): p.requires_grad_(False)
            log(f"[Warmup][ep {ep:02d}] Δs_min={delta:.3f} pairs={len(pairs)} loss={avg:.4f}")
    else:
        log("[Warmup] no data (check warmup_gens).")

    # ---------- Stage B: online loop over subsequent generations ----------
    replay = deque(maxlen=max(0, args.replay_k)) if args.replay_k > 0 else None

    # accumulators for HPO metric
    sum_spear = 0.0; sum_kend = 0.0; corr_count = 0
    sum_online_loss = 0.0; online_loss_count = 0

    for g in sorted(online_by_gen.keys()):
        pool = [r for r in online_by_gen[g] if legal(r.s)]
        cand = [r.s for r in pool]
        if not cand:
            log(f"[Gen {g}] no legal candidates.")
            continue

        # (Blind) compute DPO proxy scores BEFORE reading any scores
        scores_proxy = dpo_proxy_score(policy, ref, tok, cand, device, len_penalty=args.proxy_len_penalty)
        order = np.argsort(-scores_proxy)  # descending
        # Top-M prefilter for blind stage (by proxy), then take Top-K
        poolN = min(max(1, args.topM), len(order)) if args.topM and args.topM > 0 else len(order)
        orderM = order[:poolN]
        top_idx = orderM[:min(args.topk, len(orderM))]
        top_smiles = [cand[i] for i in top_idx]

        if is_main_process():
            log(f"[Gen {g}] blind-selected top{len(top_smiles)}" + ("" if getattr(args,"nosave",False) else " and wrote proxy margins"))

        # ---- Rank-correlation within Top-K (only using revealed Top-K true scores) ----
        smi2score = {r.s: r.score for r in pool}
        true_scores_topk = []
        proxy_scores_topk = []
        for s in top_smiles:
            ts = smi2score.get(s, None)
            if ts is not None:
                true_scores_topk.append(ts)
                proxy_scores_topk.append(float(scores_proxy[cand.index(s)]))
        spearman_val = float('nan')
        kendall_val = float('nan')
        if len(true_scores_topk) >= 2:
            spearman_val = spearman_rho(true_scores_topk, proxy_scores_topk)
            kendall_val  = kendall_tau(true_scores_topk,  proxy_scores_topk)
            sum_spear += float(spearman_val); sum_kend += float(kendall_val); corr_count += 1

        if is_main_process() and not getattr(args, "nosave", False):
            # write rows now that we know top_idx
            ground_truth_path = os.path.join(args.out, "ground_truth.csv")
            proxy_margins_path = os.path.join(args.out, "proxy_margins.csv")
            metrics_path = os.path.join(args.out, "metrics.csv")
            with open(proxy_margins_path, "a", newline="") as _fpm:
                _wpm = csv.writer(_fpm)
                for i in top_idx:
                    _wpm.writerow([g, cand[i], float(scores_proxy[i])])
            with open(ground_truth_path, "a", newline="") as _fgt:
                _wgt = csv.writer(_fgt)
                for s in top_smiles:
                    _wgt.writerow([g, s, smi2score.get(s, "NA")])
            with open(metrics_path, "a", newline="") as _fm:
                _wm = csv.writer(_fm); _wm.writerow([g, len(top_smiles), f"{spearman_val:.4f}", f"{kendall_val:.4f}"])

        # Build pairs **only** from the selected Top-K (hard cap by user rule)
        selected_set = set(top_smiles)
        selected_recs = [r for r in pool if r.s in selected_set]
        pairs_g = build_pairs_scored_adv(selected_recs, delta_min=args.delta_end,
                                         max_pairs=min(args.pairs_per_epoch, 2000),
                                         mode=args.pairs_mode, mix_alpha=args.mix_alpha, seed=args.seed+g)
        pairs_g = shard_list(pairs_g, ddp_world_size(), ddp_rank())
        random.shuffle(pairs_g)

        # ---- Fixed number of optimizer steps + optional replay ----
        train_pairs = list(pairs_g)
        if replay is not None:
            for buf in replay:
                train_pairs.extend(buf)
        random.shuffle(train_pairs)

        ep_losses=[]
        if train_pairs:
            steps = 0
            idx = 0
            while steps < max(1, args.online_steps):
                if idx >= len(train_pairs):
                    random.shuffle(train_pairs); idx = 0
                b = train_pairs[idx:idx+mb]
                if not b:
                    break
                idx += mb; steps += 1
                wins=[w for (w,l) in b]; loses=[l for (w,l) in b]
                loss = dpo_loss(policy, ref, tok, wins, loses, device, beta=args.beta)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(unwrap_model(policy).parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                if scheduler is not None:
                    scheduler.step(); global_step += 1
                ep_losses.append(loss.item())
        if replay is not None and pairs_g:
            replay.append(list(pairs_g))

        if is_main_process() and not getattr(args, "nosave", False):
            to_save = unwrap_model(policy)
            torch.save({"model":to_save.state_dict(), "vocab":tok.itos}, os.path.join(ckdir, f"policy_gen{g}.pt"))

        # Optional: refresh ref every few generations (configurable)
        if args.ref_refresh and args.ref_refresh > 0:
            if (g - args.warmup_gens) % args.ref_refresh == 0:
                src = unwrap_model(policy)
                ref.load_state_dict(src.state_dict())
                for p in ref.parameters(): p.requires_grad_(False)

        avg_loss = float(np.mean(ep_losses)) if ep_losses else float("nan")
        if not math.isnan(avg_loss):
            sum_online_loss += avg_loss; online_loss_count += 1
        log(f"[Gen {g}] trained on {len(pairs_g)} pairs from Top-{min(args.topk, len(top_smiles))}; loss≈{avg_loss:.4f}")

    if args.dist and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

    log("Done.")
    if is_main_process():
        if logf is not None: logf.close()
        if gt_f is not None: gt_f.close()
        if pm_f is not None: pm_f.close()
        if met_f is not None: met_f.close()

    # return metrics for HPO
    mean_spear = (sum_spear / max(1, corr_count))
    mean_kend  = (sum_kend  / max(1, corr_count))
    mean_online = (sum_online_loss / max(1, online_loss_count))
    return {
        "mean_spearman_topk": mean_spear,
        "mean_kendall_topk": mean_kend,
        "mean_online_loss": mean_online,
    }

def build_argparser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--warmup_gens", type=int, default=30, help="generations < this go to initial DPO training with full scores")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--pairs_per_epoch", type=int, default=4000)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--delta_start", type=float, default=0.20)
    ap.add_argument("--delta_end", type=float, default=0.02)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--emb", type=int, default=256)
    ap.add_argument("--hid", type=int, default=512)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=1e-4, help="learning rate for AdamW")

    # blind selection & pairing knobs
    ap.add_argument("--topk", type=int, default=10, help="blind selection per generation")
    ap.add_argument("--topM", type=int, default=100, help="prefilter pool size before taking top-k")
    ap.add_argument("--proxy_len_penalty", type=float, default=0.0, help="subtract lambda*length from proxy margin during blind selection")
    ap.add_argument("--pairs_mode", type=str, default="mix", choices=["stratified","hard","mix"], help="pairing strategy for scored data")
    ap.add_argument("--mix_alpha", type=float, default=0.7, help="if pairs_mode=mix, probability of stratified vs hard (alpha=stratified)")

    # online training control
    ap.add_argument("--online_steps", type=int, default=300, help="optimizer steps per online generation (replaces fixed mini-epochs)")
    ap.add_argument("--replay_k", type=int, default=5, help="use preference pairs from last K generations as replay (0 disables)")
    ap.add_argument("--ref_refresh", type=int, default=10, help="refresh reference model every N online generations (0: never)")

    # LR schedule
    ap.add_argument("--cosine_schedule", action="store_true", help="enable cosine LR with warmup")
    ap.add_argument("--warmup_ratio", type=float, default=0.05, help="warmup ratio for cosine scheduler")

    # dist/amp
    ap.add_argument("--dist", action="store_true", help="enable DistributedDataParallel via torchrun")
    ap.add_argument("--amp", action="store_true", help="enable CUDA AMP mixed precision")

    # Optuna flags
    ap.add_argument("--optuna", action="store_true", help="use Optuna to tune hyperparameters")
    ap.add_argument("--trials", type=int, default=20, help="number of Optuna trials")
    ap.add_argument("--study_name", type=str, default="dpo_hpo", help="Optuna study name")
    ap.add_argument("--storage", type=str, default="", help="Optuna storage URL (e.g., sqlite:///optuna.db)")
    ap.add_argument("--direction", type=str, default="maximize", choices=["maximize","minimize"], help="optimize mean_spearman_topk")
    ap.add_argument("--nosave", action="store_true", help="do not write logs/checkpoints (useful in HPO)")
    ap.add_argument("--optuna_seed", type=int, default=42, help="seed for Optuna sampler")
    return ap

# ---------------- Optuna objective ----------------
def optuna_objective(trial, base_args):
    # clone args
    a = argparse.Namespace(**vars(base_args))

    # sample hyperparams（涵蓋你現有可調的重要參數）
    a.lr = trial.suggest_float("lr", 5e-5, 5e-3, log=True)
    a.beta = trial.suggest_float("beta", 0.05, 0.6)
    a.delta_start = trial.suggest_float("delta_start", 0.01, 0.10)
    a.delta_end   = trial.suggest_float("delta_end", 0.0, 0.05)
    a.pairs_per_epoch = trial.suggest_int("pairs_per_epoch", 500, 6000, step=100)
    a.mix_alpha = trial.suggest_float("mix_alpha", 0.3, 0.9)
    a.pairs_mode = trial.suggest_categorical("pairs_mode", ["stratified","hard","mix"])
    a.online_steps = trial.suggest_int("online_steps", 50, 600, step=25)
    a.replay_k = trial.suggest_int("replay_k", 0, 30)
    a.ref_refresh = trial.suggest_int("ref_refresh", 2, 10)
    a.proxy_len_penalty = trial.suggest_float("proxy_len_penalty", 0.0, 0.01)
    a.topM = trial.suggest_int("topM", 30)

    # HPO 過程精簡輸出、不落地檔案
    a.nosave = True
    # 保持你的硬規則
    a.topk = base_args.topk
    a.warmup_gens = base_args.warmup_gens
    a.device = base_args.device
    a.seed = base_args.seed + trial.number  # diversify
    # 隔離trial輸出目錄（nosave=True時不會真的寫內容）
    a.out = os.path.join(base_args.out, f".optuna_trial_{trial.number}")

    # run and get metric
    result = run_training(a)
    # 主目標：讓盲選 Top-K 的 Spearman 平均越好越好
    obj = result.get("mean_spearman_topk", float("nan"))
    # 同步記錄次要指標
    trial.set_user_attr("mean_kendall_topk", result.get("mean_kendall_topk", float("nan")))
    trial.set_user_attr("mean_online_loss", result.get("mean_online_loss", float("nan")))
    return obj

# ---------------- Entrypoint ----------------
def main():
    ap = build_argparser()
    args = ap.parse_args()

    if args.optuna:
        if not HAS_OPTUNA:
            raise RuntimeError("optuna is not installed. Please `pip install optuna` on your environment.")
        sampler = optuna.samplers.TPESampler(seed=args.optuna_seed)
        pruner = MedianPruner(n_warmup_steps=5)
        study_kwargs = {
            "study_name": args.study_name,
            "direction": args.direction,
            "sampler": sampler,
            "pruner": pruner,
        }
        if args.storage:
            study = optuna.create_study(storage=args.storage, load_if_exists=True, **study_kwargs)
        else:
            study = optuna.create_study(**study_kwargs)

        print(f"[Optuna] Starting study '{study.study_name}' for {args.trials} trials ...")
        study.optimize(lambda t: optuna_objective(t, args), n_trials=args.trials, n_jobs=1)

        print("[Optuna] Best trial:")
        print("  value (mean_spearman_topk):", study.best_trial.value)
        for k,v in study.best_trial.params.items():
            print(f"  {k} = {v}")
        for k,v in study.best_trial.user_attrs.items():
            print(f"  {k}: {v}")
    else:
        # single run
        _ = run_training(args)

if __name__ == "__main__":
    main()