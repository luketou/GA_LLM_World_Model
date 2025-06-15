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
