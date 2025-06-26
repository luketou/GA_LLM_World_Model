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
import threading, logging

logger = logging.getLogger(__name__)

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
        
        logger.info(f"[Prescan] Evaluating {len(smiles_all)} SMILES from {smi_file}")
        
        # 限制 prescan 的 SMILES 數量以提高速度
        if len(smiles_all) > 1000:
            logger.info(f"[Prescan] Too many SMILES ({len(smiles_all)}), sampling 1000 for prescan")
            import random
            smiles_all = random.sample(smiles_all, 1000)
        
        # 使用 GuacaMol benchmark 的評分方法
        try:
            # 嘗試使用 objective 屬性進行逐一評分
            if hasattr(self.benchmark, 'objective') and hasattr(self.benchmark.objective, 'score'):
                scores = []
                for smiles in smiles_all:
                    try:
                        score = float(self.benchmark.objective.score(smiles)) # 確保分數是浮點數
                        scores.append(score)
                    except Exception as e:
                        logger.error(f"[Prescan] Error scoring {smiles}: {e}")
                        scores.append(0.0)
            elif hasattr(self.benchmark, 'score_list'):
                scores = self.benchmark.score_list(smiles_all)
            elif hasattr(self.benchmark, 'score'): # Fallback to single score method
                # 使用單一評分方法逐一評分
                scores = [self.benchmark.score(smiles) for smiles in smiles_all]
            else:
                # 最後回退：返回默認分數
                logger.warning(f"[Prescan] No suitable scoring method found, using default scores")
                scores = [0.0] * len(smiles_all)

        except Exception as e:
            logger.error(f"[Prescan] Error during scoring: {e}")
            logger.debug(f"[Prescan] Benchmark type: {type(self.benchmark)}")
            logger.debug(f"[Prescan] Available methods: {[attr for attr in dir(self.benchmark) if not attr.startswith('_')]}")
            
            # 嘗試檢查 objective 的方法
            if hasattr(self.benchmark, 'objective'):
                logger.debug(f"[Prescan] Objective methods: {[attr for attr in dir(self.benchmark.objective) if not attr.startswith('_')]}")
            
            # 回退：選擇第一個 SMILES 作為種子
            return smiles_all[0], 0.0
        
        idx_low = scores.index(min(scores))
        return smiles_all[idx_low], scores[idx_low]

    # ---------- Async scoring ---------- #
    async def score_async(self, smiles: List[str],
                          epoch: Optional[int] = None) -> List[float]:
        """
        非同步評分：
        - 檢查配額、令牌桶
        - 呼叫 benchmark 評分方法
        - 每個分子評分後立即寫入 score_log.csv（欄位：epoch, smiles, score）
        """
        import time
        start_time = time.time()
        logger.debug(f"[ORACLE-DEBUG] Starting score_async for {len(smiles)} SMILES")
        
        if self.calls_left <= 0:
            raise RuntimeError("Oracle call limit exhausted (500).")
        
        # 檢查是否有足夠的配額評分所有分子
        if len(smiles) > self.calls_left:
            raise RuntimeError(f"Not enough oracle calls remaining. Need {len(smiles)}, have {self.calls_left}")
        
        # Rate limit check disabled as per request.
        # if not self.rate_limiter.acquire():
        #     logger.warning(f"[ORACLE-DEBUG] Rate limit hit, waiting {self.rate_limiter.interval}s...")
        #     await asyncio.sleep(self.rate_limiter.interval)
        #     ...
        
        # 更新調用次數和 epoch（每批次調用算一次）
        self.calls_left -= len(smiles)  # 每個分子都要消耗配額
        self.epoch += 1
        current_epoch = epoch if epoch is not None else self.epoch
        
        logger.debug(f"[ORACLE-DEBUG] Oracle calls remaining: {self.calls_left}")
        logger.debug(f"[ORACLE-DEBUG] Current epoch: {current_epoch}")
        
        loop = asyncio.get_event_loop()
        
        # 定義評分函數，每個分子評分後立即記錄
        def score_and_log_molecule(smile_str):
            mol_start = time.time()
            logger.debug(f"[ORACLE-DEBUG] Scoring molecule: {smile_str[:50]}...")
            
            try:
                # 嘗試使用 objective 屬性進行單一評分
                if hasattr(self.benchmark, 'objective') and hasattr(self.benchmark.objective, 'score'):
                    try:
                        score_start = time.time()
                        score = float(self.benchmark.objective.score(smile_str)) # 確保分數是浮點數
                        score_time = time.time() - score_start
                        logger.debug(f"[ORACLE-DEBUG] Objective.score completed in {score_time:.3f}s, score: {score:.6f}")
                    except Exception as e:
                        logger.error(f"[ORACLE-DEBUG] Error in objective.score for {smile_str}: {e}")
                        score = 0.0
                elif hasattr(self.benchmark, 'score'):
                    # 使用單一評分方法
                    score_start = time.time()
                    score = self.benchmark.score(smile_str)
                    score_time = time.time() - score_start
                    logger.debug(f"[ORACLE-DEBUG] Benchmark.score completed in {score_time:.3f}s, score: {score:.6f}")
                else:
                    # 最後回退：返回默認分數
                    logger.warning(f"[ORACLE-DEBUG] No suitable scoring method found, using default score")
                    score = 0.0
                
                # 立即記錄到 CSV
                csv_start = time.time()
                with self.CSV_PATH.open("a", newline="") as f:
                    csv.writer(f).writerow([current_epoch, smile_str, score])
                csv_time = time.time() - csv_start
                logger.debug(f"[ORACLE-DEBUG] CSV logging completed in {csv_time:.3f}s")
                
                mol_time = time.time() - mol_start
                logger.debug(f"[ORACLE-DEBUG] Molecule scoring total time: {mol_time:.3f}s")
                
                return score
            except Exception as e:
                mol_time = time.time() - mol_start
                logger.error(f"[ORACLE-DEBUG] Scoring error for {smile_str} after {mol_time:.3f}s: {e}")
                logger.debug(f"[ORACLE-DEBUG] Benchmark type: {type(self.benchmark)}")
                logger.debug(f"[ORACLE-DEBUG] Available methods: {[attr for attr in dir(self.benchmark) if not attr.startswith('_')]}")
                # 返回默認分數並記錄
                score = 0.0
                with self.CSV_PATH.open("a", newline="") as f:
                    csv.writer(f).writerow([current_epoch, smile_str, score])
                return score
        
        # 對每個 SMILES 逐一評分和記錄
        logger.debug(f"[ORACLE-DEBUG] Starting individual molecule scoring...")
        scores = []
        for i, smile_str in enumerate(smiles):
            logger.debug(f"[ORACLE-DEBUG] Processing molecule {i+1}/{len(smiles)}")
            exec_start = time.time()
            
            score = await loop.run_in_executor(
                self.executor,
                score_and_log_molecule,
                smile_str
            )
            
            exec_time = time.time() - exec_start
            logger.debug(f"[ORACLE-DEBUG] Executor call {i+1} completed in {exec_time:.3f}s")
            
            scores.append(score)
        
        total_time = time.time() - start_time
        logger.debug(f"[ORACLE-DEBUG] score_async completed in {total_time:.2f}s for {len(smiles)} molecules")
        logger.debug(f"[ORACLE-DEBUG] Average time per molecule: {total_time/len(smiles):.3f}s")
        
        return scores

    def close(self):
        """清理資源"""
        self.executor.shutdown(wait=True)
