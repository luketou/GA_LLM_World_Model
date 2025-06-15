"""
GuacaMol Oracle wrapper
-----------------------
1. 依 TASK_BENCHMARK_MAPPING 建立 benchmark 實例
2. 全程統一使用 asyncio + ThreadPoolExecutor 呼叫 `score_list`
3. 內建：
   * `TOTAL_LIMIT`      – 500 次（含 evolution 階段；prescan 不計）
   * 令牌桶限流        – 每分鐘 10 次 (default，可覆寫)
   * CSV 紀錄          – score_log.csv，欄位 [epoch, smiles, score]
4. `prescan_lowest(smi_path)`: 先對 .txt 檔所有 SMILES 批次評分，
   找出最低分分子並回傳 `(smiles, score)`；**不寫入 CSV、不佔配額**。
"""

import csv
import time
import asyncio
import functools
import pathlib
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple, Optional

from guacamol.standard_benchmarks import (
    hard_osimertinib, hard_fexofenadine, ranolazine_mpo,
    amlodipine_rings, sitagliptin_replacement, zaleplon_with_other_formula,
    median_camphor_menthol, median_tadalafil_sildenafil, similarity,
    perindopril_rings, hard_cobimetinib, qed_benchmark, logP_benchmark,
    tpsa_benchmark, cns_mpo, scaffold_hop, decoration_hop, weird_physchem,
    isomers_c11h24, isomers_c9h10n2o2pf2cl, valsartan_smarts
)

CELECOXIB_SMILES = 'CC1=CC=C(C=C1)C1=CC(=NN1C1=CC=C(C=C1)S(N)(=O)=O)C(F)(F)F'
TROGLITAZONE_SMILES = 'Cc1c(C)c2OC(C)(COc3ccc(CC4SC(=O)NC4=O)cc3)CCc2c(C)c1O'
THIOTHIZENE_SMILES = 'CN(C)S(=O)(=O)c1ccc2Sc3ccccc3C(=CCCN4CCN(C)CC4)c2c1'

# Benchmark 映射：對應任務到 GuacaMol 基準函式
TASK_BENCHMARK_MAPPING = {
    # MPO
    'osimertinib': hard_osimertinib,
    'fexofenadine': hard_fexofenadine,
    'ranolazine': ranolazine_mpo,
    'amlodipine': amlodipine_rings,
    'perindopril': perindopril_rings,
    'sitagliptin': sitagliptin_replacement,
    'zaleplon': zaleplon_with_other_formula,
    'cobimetinib': hard_cobimetinib,
    'qed': qed_benchmark,
    'cns_mpo': cns_mpo,
    'scaffold_hop': scaffold_hop,
    'decoration_hop': decoration_hop,
    'weird_physchem': weird_physchem,
    'valsartan_smarts': valsartan_smarts,

    # median
    'median1': median_camphor_menthol,
    'median2': median_tadalafil_sildenafil,

    # isomer
    'isomer_c11h24': isomers_c11h24,
    'isomer_c9h10n2o2pf2cl': isomers_c9h10n2o2pf2cl,

    # property targets
    'logp_2.5': lambda: logP_benchmark(target_value=2.5),
    'tpsa_100': lambda: tpsa_benchmark(target_value=100),

    # rediscovery
    'celecoxib': lambda: similarity(
        CELECOXIB_SMILES, 'Celecoxib', fp_type='ECFP4', threshold=1.0),
    'troglitazone': lambda: similarity(
        TROGLITAZONE_SMILES, 'Troglitazone', fp_type='ECFP4', threshold=1.0),
    'thiothixene': lambda: similarity(
        THIOTHIZENE_SMILES, 'Thiothixene', fp_type='ECFP4', threshold=1.0)
}


class RateLimiter:
    """簡易令牌桶，每 `interval` 秒允許 `capacity` 次呼叫。"""
    def __init__(self, capacity: int, interval: int):
        self.capacity = capacity
        self.tokens = capacity
        self.interval = interval
        self.lock = threading.Lock()
        self.last = time.time()

    def acquire(self):
        with self.lock:
            now = time.time()
            refill = int((now - self.last) / self.interval)
            if refill:
                self.tokens = min(self.capacity, self.tokens + refill)
                self.last = now
            if self.tokens == 0:
                return False
            self.tokens -= 1
            return True


class GuacaMolOracle:
    """
    GuacaMol Oracle 客戶端
    - 呼叫限額與速率控制
    - 非同步評分
    - Prescan 功能
    """
    TOTAL_LIMIT = 500                    # evolution 階段配額
    CSV_PATH = pathlib.Path("score_log.csv")
    _header_written = False

    def __init__(self, task_name: str,
                 rate_cap: int = 10, interval: int = 60):
        if task_name not in TASK_BENCHMARK_MAPPING:
            raise ValueError(f"Unknown task '{task_name}'. Available tasks: {list(TASK_BENCHMARK_MAPPING.keys())}")
        
        self.task_name = task_name
        self.benchmark = TASK_BENCHMARK_MAPPING[task_name]()
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.calls_left = GuacaMolOracle.TOTAL_LIMIT
        self.rate_limiter = RateLimiter(rate_cap, interval)
        self.epoch = 0
        
        # ensure CSV header
        if not GuacaMolOracle._header_written and not self.CSV_PATH.exists():
            with self.CSV_PATH.open("w", newline="") as f:
                csv.writer(f).writerow(["epoch", "smiles", "score"])
            GuacaMolOracle._header_written = True

    # ---------- Prescan ---------- #
    def prescan_lowest(self, smi_file: str) -> Tuple[str, float]:
        """
        Prescan：同步批次評分所有 SMILES
        回傳最低分分子 + 分數（不計進額度、不寫 CSV）
        """
        smi_path = pathlib.Path(smi_file)
        if not smi_path.exists():
            raise FileNotFoundError(f"SMILES file not found: {smi_file}")
        
        smiles_all = [l.strip() for l in smi_path.open() if l.strip()]
        if not smiles_all:
            raise ValueError(f"No valid SMILES found in {smi_file}")
        
        print(f"[Prescan] Evaluating {len(smiles_all)} SMILES from {smi_file}")
        scores = self.benchmark.score_list(smiles_all)
        idx_low = scores.index(min(scores))
        
        return smiles_all[idx_low], scores[idx_low]

    # ---------- Async scoring ---------- #
    async def score_async(self, smiles: List[str],
                          epoch: Optional[int] = None) -> List[float]:
        """
        非同步評分：
        - 檢查配額、令牌桶
        - 呼叫 benchmark.score_list
        - 寫入 score_log.csv（欄位：epoch, smiles, score）
        """
        if self.calls_left <= 0:
            raise RuntimeError("Oracle call limit exhausted (500).")
        
        if not self.rate_limiter.acquire():
            # 若令牌不足，非同步 sleep
            await asyncio.sleep(self.rate_limiter.interval)
            if not self.rate_limiter.acquire():
                raise RuntimeError("Rate limit exceeded")

        self.calls_left -= 1
        self.epoch += 1
        current_epoch = epoch if epoch is not None else self.epoch
        
        loop = asyncio.get_event_loop()
        scores = await loop.run_in_executor(
            self.executor,
            functools.partial(self.benchmark.score_list, smiles)
        )

        # CSV logging
        with self.CSV_PATH.open("a", newline="") as f:
            wr = csv.writer(f)
            for sm, sc in zip(smiles, scores):
                wr.writerow([current_epoch, sm, sc])

        return scores

    def close(self):
        """清理資源"""
        self.executor.shutdown(wait=True)
