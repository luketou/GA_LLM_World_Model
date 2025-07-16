"""
SMILES 工具函式
⚠️ 重要警告：這些函數使用 RDKit 進行分子特性計算
❌ 禁止在 Oracle (GuacaMol) 評分前使用這些函數！
✅ 只能在 Oracle 評分完成後用於後處理或分析

包含的函數：
- tanimoto: 計算 Morgan fingerprint 相似度
- valid_smiles: 驗證 SMILES 有效性  
- canonicalize: SMILES 標準化
- molecular_weight: 計算分子量
- logp: 計算 LogP 值
- qed: 計算 QED 值

使用場景：
✅ Oracle 評分後的結果分析
✅ 後處理和數據清理
✅ 知識圖譜查詢和比較
❌ LLM 生成後的即時驗證
❌ Oracle 評分前的任何計算
"""
from typing import Optional
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit import DataStructs
import pubchempy as pcp
import time
from diskcache import Cache
from utils.concurrency import RateLimiterAsync
import asyncio


def tanimoto(smiles1: str, smiles2: str, radius: int = 2, nBits: int = 2048) -> float:
    """
    計算兩個 SMILES 的 Morgan fingerprint 相似度 (Tanimoto 係數)
    
    Args:
        smiles1: 第一個 SMILES
        smiles2: 第二個 SMILES  
        radius: Morgan fingerprint 半徑
        nBits: fingerprint 位數
        
    Returns:
        Tanimoto 相似度 (0-1 之間)
    """
    try:
        mol1 = Chem.MolFromSmiles(smiles1)
        mol2 = Chem.MolFromSmiles(smiles2)
        
        if mol1 is None or mol2 is None:
            return 0.0
        
        fp1 = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol1, radius, nBits=nBits)
        fp2 = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol2, radius, nBits=nBits)
        
        return DataStructs.TanimotoSimilarity(fp1, fp2)
    
    except Exception:
        return 0.0


def valid_smiles(smiles: str) -> bool:
    """
    驗證 SMILES 的有效性
    
    Args:
        smiles: 要驗證的 SMILES 字串
        
    Returns:
        True 如果 SMILES 有效，否則 False
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
    except Exception:
        return False


def canonicalize(smiles: str) -> Optional[str]:
    """
    SMILES 標準化
    
    Args:
        smiles: 要標準化的 SMILES 字串
        
    Returns:
        標準化的 SMILES，如果無效則返回 None
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol)
    except Exception:
        return None


def molecular_weight(smiles: str) -> Optional[float]:
    """
    計算分子量
    
    Args:
        smiles: SMILES 字串
        
    Returns:
        分子量，如果無效則返回 None
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Chem.Descriptors.MolWt(mol)
    except Exception:
        return None


def logp(smiles: str) -> Optional[float]:
    """
    計算 LogP 值
    
    Args:
        smiles: SMILES 字串
        
    Returns:
        LogP 值，如果無效則返回 None
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Chem.Descriptors.MolLogP(mol)
    except Exception:
        return None


def qed(smiles: str) -> Optional[float]:
    """
    計算 QED (Quantitative Estimate of Drug-likeness) 值
    
    Args:
        smiles: SMILES 字串
        
    Returns:
        QED 值 (0-1 之間)，如果無效則返回 None
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        from rdkit.Chem import QED
        return QED.qed(mol)
    except Exception:
        return None


# --- PubChem RAG + Cache ---
# 初始化一個持久化快取（建議用相對路徑，跨平台）
effective_cache = Cache('./pubchem_cache')
# 建立一個異步速率限制器（每秒最多 5 次）
rate_limiter = RateLimiterAsync(rate=5, per=1)

def get_pubchem_data_v2(smiles: str) -> dict:
    """
    查詢 PubChem 並快取 SMILES 對應的分子資訊。
    Args:
        smiles: SMILES 字串
    Returns:
        dict: 包含 PubChem 查詢結果或錯誤訊息
    """
    if smiles in effective_cache:
        print(f"✅ 從「硬碟」快取命中: {smiles}")
        return effective_cache[smiles]

    print(f"🚀 正在向 PubChem API 查詢: {smiles}")
    try:
        # 強制速率限制（同步版本，若需異步請改用 asyncio）
        loop = asyncio.get_event_loop() if asyncio.get_event_loop().is_running() else None
        if loop:
            loop.run_until_complete(rate_limiter.acquire())
        else:
            time.sleep(0.2)  # 最簡單的速率限制

        compounds = pcp.get_compounds(smiles, 'smiles')
        if not compounds:
            data = {"error": "Not found"}
        else:
            compound = compounds[0]
            data = {
                "cid": compound.cid,
                "iupac_name": compound.iupac_name,
                "molecular_formula": compound.molecular_formula,
                "molecular_weight": float(compound.molecular_weight) if compound.molecular_weight else None,
                "canonical_smiles": compound.canonical_smiles,
                "xlogp": float(compound.xlogp) if compound.xlogp else None,
                "h_bond_donor_count": compound.h_bond_donor_count,
                "h_bond_acceptor_count": compound.h_bond_acceptor_count,
                "pubchem_url": f"https://pubchem.ncbi.nlm.nih.gov/compound/{compound.cid}"
            }
        # 存入持久化快取
        effective_cache[smiles] = data
        return data
    except Exception as e:
        return {"error": str(e)}
