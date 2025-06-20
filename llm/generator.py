"""
LLM Generator
LLM 客戶端封裝：
- 初始化 Cerebras 客戶端
- 批次生成介面
- 組裝 system + action messages
- 呼叫 LLM
- 解析並簡單檢查 SMILES 列表 (無 RDKit 驗證)
- 回傳 SMILES (驗證交給 Oracle)
"""
import json
import yaml
import pathlib
import logging
import os
import re
from typing import List, Dict, Any, Optional
from langsmith import traceable
from .cerebras_client import CerebrasClient
from .prompt import create_llm_messages

logger = logging.getLogger(__name__)


class LLMGenerator:
    """LLM 客戶端封裝 - 使用 Cerebras"""
    
    def __init__(self, 
                 provider: str = "cerebras", 
                 model_name: str = "qwen-3-32b", 
                 temperature: float = 0.2,
                 max_completion_tokens: int = 2048,
                 max_smiles_length: int = 100,
                 top_p: float = 1.0,
                 stream: bool = True,
                 api_key: Optional[str] = None):
        """
        初始化 LLM Generator
        
        Args:
            provider: LLM 提供商
            model_name: 模型名稱
            temperature: 溫度參數
            max_completion_tokens: 最大完成 token 數
            max_smiles_length: SMILES 最大長度
            top_p: top-p 參數
            stream: 是否使用流式輸出
            api_key: API 金鑰
        """
        self.provider = provider
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_completion_tokens
        self.max_smiles_length = max_smiles_length
        self.top_p = top_p
        self.stream = stream
        
        if provider == "cerebras":
            from cerebras.cloud.sdk import Cerebras
            # 使用傳入的 api_key 或環境變數
            api_key = api_key or os.getenv("CEREBRAS_API_KEY")
            if not api_key:
                raise ValueError("CEREBRAS_API_KEY is required")
            
            self.client = Cerebras(api_key=api_key)
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    @traceable
    def generate_batch(self, parent_smiles: str, actions: List[Dict], max_retries: int = 3) -> List[str]:
        """
        根據動作列表生成分子的 SMILES
        """
        if not actions:
            logger.warning("No actions provided for generation")
            return []
        
        results = []
        
        for retry in range(max_retries):
            try:
                # 構建提示
                action_descriptions = []
                for i, action in enumerate(actions):
                    description = action.get('description', action.get('name', f'action_{i}'))
                    action_descriptions.append(f"{i+1}. {description}")
                
                prompt = f"""You are an expert medicinal chemist. Given a parent molecule and a list of chemical modifications, generate new valid SMILES.

Parent molecule: {parent_smiles}

Chemical modifications to apply:
{chr(10).join(action_descriptions)}

For each modification, generate a new SMILES that represents the parent molecule after applying that specific chemical change. Return exactly {len(actions)} SMILES, one per line, in the same order as the modifications.

Requirements:
- Each SMILES must be chemically valid
- Modifications should be realistic and chemically feasible
- Keep molecular weight reasonable (< 800 Da)
- Avoid overly complex structures
- SMILES length should be under {self.max_smiles_length} characters

SMILES (one per line):"""

                # 使用正確的 Cerebras API 調用方式
                if self.provider == "cerebras":
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=self.temperature,
                        max_completion_tokens=self.max_tokens,
                        top_p=self.top_p,
                        stream=False  # 為了簡化，先不使用 stream
                    )
                    content = response.choices[0].message.content
                
                # 解析生成的 SMILES
                lines = [line.strip() for line in content.strip().split('\n') if line.strip()]
                
                # 過濾並驗證 SMILES
                valid_smiles = []
                for line in lines:
                    # 移除數字編號和其他前綴
                    smiles = re.sub(r'^\d+\.\s*', '', line.strip())
                    smiles = re.sub(r'^[-*]\s*', '', smiles.strip())  # 移除項目符號
                    smiles = smiles.split()[0] if smiles.split() else ""
                    
                    if smiles and self.validate_smiles(smiles):
                        valid_smiles.append(smiles)
                
                logger.info(f"Generated {len(valid_smiles)} valid SMILES from {len(lines)} total lines")
                
                # 檢查成功率
                success_rate = len(valid_smiles) / len(actions) if actions else 0
                min_success_rate = 0.5  # 降低最低成功率要求
                
                if success_rate >= min_success_rate:
                    # 補齊到所需數量
                    while len(valid_smiles) < len(actions):
                        fallback_smiles = self.fallback_smiles_generation(parent_smiles)
                        if fallback_smiles not in valid_smiles:  # 避免重複
                            valid_smiles.append(fallback_smiles)
                    
                    results = valid_smiles[:len(actions)]
                    logger.info(f"Successfully generated {len(results)} SMILES (success rate: {success_rate:.2f})")
                    break
                else:
                    logger.warning(f"Low success rate: {success_rate:.2f}, retrying...")
                    
            except Exception as e:
                logger.error(f"Error in generate_batch (retry {retry + 1}): {e}")
                if retry == max_retries - 1:
                    # 最後一次重試失敗，使用回退方法
                    logger.warning("All retries failed, using fallback generation")
                    results = [self.fallback_smiles_generation(parent_smiles) for _ in actions]
        
        # 確保返回正確數量的 SMILES
        if len(results) != len(actions):
            logger.warning(f"Generated fewer SMILES ({len(results)}) than actions ({len(actions)}) after {max_retries} retries")
            # 補齊不足的部分
            while len(results) < len(actions):
                results.append(self.fallback_smiles_generation(parent_smiles))
        
        logger.info(f"Final result: {len(results)} SMILES from {len(actions)} actions")
        return results

    def fallback_smiles_generation(self, parent_smiles: str) -> str:
        """
        當 LLM 生成失敗時的回退方法
        """
        try:
            from rdkit import Chem
            from rdkit.Chem import rdMolDescriptors
            
            mol = Chem.MolFromSmiles(parent_smiles)
            if mol is None:
                return "CCO"  # 簡單的回退分子
            
            # 簡單的結構修改
            modifications = [
                lambda m: self.add_simple_group(m, "C"),  # 添加甲基
                lambda m: self.add_simple_group(m, "O"),  # 添加羥基
                lambda m: self.add_simple_group(m, "N"),  # 添加氨基
                lambda m: self.remove_atom(m),  # 移除一個原子
                lambda m: self.modify_bond(m),  # 修改鍵
            ]
            
            import random
            modification = random.choice(modifications)
            modified_mol = modification(mol)
            
            if modified_mol:
                modified_smiles = Chem.MolToSmiles(modified_mol)
                if self.validate_smiles(modified_smiles):
                    return modified_smiles
            
            # 如果修改失敗，返回稍微變化的原分子
            return self.simple_variation(parent_smiles)
                
        except Exception as e:
            logger.error(f"Fallback generation failed: {e}")
            return "CCO"  # 最終的回退分子

    def simple_variation(self, smiles: str) -> str:
        """
        對 SMILES 進行簡單變化
        """
        try:
            # 簡單的字符替換方法
            variations = [
                smiles.replace('C', 'N', 1) if 'C' in smiles else smiles,
                smiles.replace('N', 'O', 1) if 'N' in smiles else smiles,
                smiles + 'C' if len(smiles) < self.max_smiles_length - 1 else smiles,
                smiles[:-1] if len(smiles) > 5 else smiles,
            ]
            
            import random
            return random.choice([v for v in variations if v != smiles and len(v) <= self.max_smiles_length]) or smiles
        except Exception:
            return smiles

    def add_simple_group(self, mol, atom_type):
        """
        添加簡單基團的輔助方法
        """
        try:
            from rdkit import Chem
            from rdkit.Chem import Descriptors
            
            # 獲取分子的 SMILES
            smiles = Chem.MolToSmiles(mol)
            
            # 簡單的字符串操作來添加原子
            if atom_type == "C" and len(smiles) < self.max_smiles_length - 1:
                return Chem.MolFromSmiles(smiles + "C")
            elif atom_type == "O" and len(smiles) < self.max_smiles_length - 1:
                return Chem.MolFromSmiles(smiles + "O")
            elif atom_type == "N" and len(smiles) < self.max_smiles_length - 1:
                return Chem.MolFromSmiles(smiles + "N")
            else:
                return mol
                
        except Exception:
            return mol

    def remove_atom(self, mol):
        """
        移除一個原子
        """
        try:
            from rdkit import Chem
            
            if mol.GetNumAtoms() <= 3:  # 保持最小分子大小
                return mol
            
            smiles = Chem.MolToSmiles(mol)
            if len(smiles) > 5:
                # 簡單地移除最後一個字符（如果它是原子）
                last_char = smiles[-1]
                if last_char in 'CNOSP':
                    modified_smiles = smiles[:-1]
                    return Chem.MolFromSmiles(modified_smiles)
            
            return mol
        except Exception:
            return mol

    def modify_bond(self, mol):
        """
        修改鍵
        """
        try:
            from rdkit import Chem
            
            smiles = Chem.MolToSmiles(mol)
            
            # 簡單的鍵修改：單鍵變雙鍵，雙鍵變單鍵
            if '=' in smiles:
                modified_smiles = smiles.replace('=', '', 1)
            else:
                # 在適當位置添加雙鍵
                if 'C' in smiles:
                    modified_smiles = smiles.replace('C', 'C=C', 1)
                else:
                    modified_smiles = smiles
            
            return Chem.MolFromSmiles(modified_smiles)
        except Exception:
            return mol

    def validate_smiles(self, smiles: str) -> bool:
        """
        增強的 SMILES 驗證機制
        """
        try:
            from rdkit import Chem
            from rdkit.Chem import rdMolDescriptors
            
            if not smiles or len(smiles) < 2:
                return False
            
            # 檢查長度限制
            if len(smiles) > self.max_smiles_length:
                return False
            
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return False
            
            # 檢查分子量
            mw = rdMolDescriptors.CalcExactMolWt(mol)
            if mw > 1000 or mw < 30:  # 合理的分子量範圍
                return False
            
            # 檢查原子數
            num_atoms = mol.GetNumAtoms()
            if num_atoms > 100 or num_atoms < 2:
                return False
            
            # 檢查是否包含異常結構
            if self.has_unusual_structures(mol):
                return False
            
            return True
            
        except Exception as e:
            logger.debug(f"SMILES validation failed for '{smiles}': {e}")
            return False

    def has_unusual_structures(self, mol) -> bool:
        """
        檢查是否包含異常結構
        """
        try:
            from rdkit.Chem import rdMolDescriptors
            
            # 檢查過多的環
            num_rings = rdMolDescriptors.CalcNumRings(mol)
            if num_rings > 8:
                return True
            
            # 檢查過多的雜原子
            heavy_atoms = mol.GetNumHeavyAtoms()
            heteroatoms = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() not in [1, 6])
            if heavy_atoms > 0 and heteroatoms / heavy_atoms > 0.7:  # 雜原子比例過高
                return True
            
            return False
            
        except Exception:
            return True  # 出錯時認為是異常結構