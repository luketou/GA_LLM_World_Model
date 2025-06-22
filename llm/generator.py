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
        import time
        start_time = time.time()
        print(f"[LLM-DEBUG] Starting generate_batch with {len(actions)} actions")
        
        if not actions:
            logger.warning("No actions provided for generation")
            return []
        
        results = []
        
        for retry in range(max_retries):
            retry_start = time.time()
            print(f"[LLM-DEBUG] Retry {retry + 1}/{max_retries} started")
            
            try:
                # 構建提示
                print(f"[LLM-DEBUG] Building prompt for {len(actions)} actions")
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
                
                print(f"[LLM-DEBUG] Calling {self.provider} API...")
                api_start = time.time()
                
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
                
                api_time = time.time() - api_start
                print(f"[LLM-DEBUG] API call completed in {api_time:.2f}s")
                
                # 解析生成的 SMILES
                print(f"[LLM-DEBUG] Parsing generated content...")
                parse_start = time.time()
                lines = [line.strip() for line in content.strip().split('\n') if line.strip()]
                print(f"[LLM-DEBUG] Found {len(lines)} lines in response")
                
                # 過濾並驗證 SMILES
                print(f"[LLM-DEBUG] Starting SMILES validation for {len(lines)} lines...")
                validation_start = time.time()
                valid_smiles = []
                
                for i, line in enumerate(lines):
                    line_start = time.time()
                    print(f"[LLM-DEBUG] Processing line {i+1}/{len(lines)}: {line[:50]}...")
                    
                    # 移除數字編號和其他前綴
                    smiles = re.sub(r'^\d+\.\s*', '', line.strip())
                    smiles = re.sub(r'^[-*]\s*', '', smiles.strip())  # 移除項目符號
                    smiles = smiles.split()[0] if smiles.split() else ""
                    
                    if smiles:
                        print(f"[LLM-DEBUG] Validating SMILES: {smiles}")
                        validation_result = self.validate_smiles(smiles)
                        line_time = time.time() - line_start
                        print(f"[LLM-DEBUG] Line {i+1} validation: {'VALID' if validation_result else 'INVALID'} ({line_time:.3f}s)")
                        
                        if validation_result:
                            valid_smiles.append(smiles)
                    else:
                        print(f"[LLM-DEBUG] Line {i+1}: Empty SMILES after cleaning")
                
                validation_time = time.time() - validation_start
                print(f"[LLM-DEBUG] Validation completed in {validation_time:.2f}s")
                
                logger.info(f"Generated {len(valid_smiles)} valid SMILES from {len(lines)} total lines")
                
                # 檢查成功率
                print(f"[LLM-DEBUG] Checking success rate...")
                success_rate = len(valid_smiles) / len(actions) if actions else 0
                min_success_rate = 0.5  # 降低最低成功率要求
                
                print(f"[LLM-DEBUG] Success rate: {success_rate:.2f} (required: {min_success_rate})")
                
                if success_rate >= min_success_rate:
                    print(f"[LLM-DEBUG] Success rate acceptable, filling remaining slots...")
                    
                    # 修正：使用集合來避免重複，並正確控制循環
                    valid_smiles_set = set(valid_smiles)
                    fallback_attempts = 0
                    max_fallback_attempts = 50  # 限制回退嘗試次數
                    
                    while len(valid_smiles) < len(actions) and fallback_attempts < max_fallback_attempts:
                        fallback_start = time.time()
                        fallback_attempts += 1
                        print(f"[LLM-DEBUG] Generating fallback SMILES {len(valid_smiles)+1}/{len(actions)} (attempt {fallback_attempts})")
                        
                        fallback_smiles = self.fallback_smiles_generation(parent_smiles)
                        fallback_time = time.time() - fallback_start
                        print(f"[LLM-DEBUG] Fallback generated: {fallback_smiles} ({fallback_time:.3f}s)")
                        
                        # 檢查是否重複
                        if fallback_smiles not in valid_smiles_set:
                            valid_smiles.append(fallback_smiles)
                            valid_smiles_set.add(fallback_smiles)
                            print(f"[LLM-DEBUG] Added unique fallback SMILES, now have {len(valid_smiles)}/{len(actions)}")
                        else:
                            print(f"[LLM-DEBUG] Fallback SMILES is duplicate, skipping")
                    
                    # 如果仍然不夠，用簡單的變體填充
                    if len(valid_smiles) < len(actions):
                        print(f"[LLM-DEBUG] Still need {len(actions) - len(valid_smiles)} more SMILES, using simple variations")
                        while len(valid_smiles) < len(actions):
                            # 使用更簡單的變體生成方法
                            simple_variation = self.generate_simple_variation(parent_smiles, len(valid_smiles))
                            if simple_variation not in valid_smiles_set:
                                valid_smiles.append(simple_variation)
                                valid_smiles_set.add(simple_variation)
                            else:
                                # 如果還是重複，就用索引變體
                                indexed_smiles = f"{parent_smiles}_{len(valid_smiles)}"[:self.max_smiles_length]
                                # 簡化為有效的 SMILES
                                if self.validate_smiles(parent_smiles):
                                    valid_smiles.append(parent_smiles)
                                else:
                                    valid_smiles.append("CCO")  # 最後的最後回退
                    
                    results = valid_smiles[:len(actions)]
                    retry_time = time.time() - retry_start
                    print(f"[LLM-DEBUG] Retry {retry+1} successful in {retry_time:.2f}s")
                    logger.info(f"Successfully generated {len(results)} SMILES (success rate: {success_rate:.2f})")
                    break
                else:
                    retry_time = time.time() - retry_start
                    print(f"[LLM-DEBUG] Retry {retry+1} failed (low success rate) in {retry_time:.2f}s")
                    logger.warning(f"Low success rate: {success_rate:.2f}, retrying...")
                    
            except Exception as e:
                retry_time = time.time() - retry_start
                print(f"[LLM-DEBUG] Retry {retry+1} failed with exception in {retry_time:.2f}s: {e}")
                logger.error(f"Error in generate_batch (retry {retry + 1}): {e}")
                if retry == max_retries - 1:
                    # 最後一次重試失敗，使用回退方法
                    print(f"[LLM-DEBUG] All retries failed, using fallback generation for all {len(actions)} actions")
                    logger.warning("All retries failed, using fallback generation")
                    results = []
                    for i in range(len(actions)):
                        fallback_smiles = self.fallback_smiles_generation(parent_smiles)
                        if fallback_smiles not in results:  # 避免重複
                            results.append(fallback_smiles)
                        else:
                            results.append(f"CCO_{i}")  # 簡單的唯一回退
        
        # 確保返回正確數量的 SMILES
        if len(results) != len(actions):
            print(f"[LLM-DEBUG] Mismatch: generated {len(results)} SMILES for {len(actions)} actions")
            logger.warning(f"Generated fewer SMILES ({len(results)}) than actions ({len(actions)}) after {max_retries} retries")
            # 補齊不足的部分
            while len(results) < len(actions):
                simple_smiles = f"CCO_{len(results)}"  # 用索引保證唯一性
                results.append(simple_smiles)
        
        total_time = time.time() - start_time
        print(f"[LLM-DEBUG] generate_batch completed in {total_time:.2f}s")
        logger.info(f"Final result: {len(results)} SMILES from {len(actions)} actions")
        return results

    def generate_simple_variation(self, parent_smiles: str, index: int) -> str:
        """
        生成簡單的分子變體，確保唯一性
        """
        try:
            # 基於索引的簡單變體
            simple_molecules = [
                "CCO",  # 乙醇
                "CCN",  # 乙胺
                "CCC",  # 丙烷
                "C=C",  # 乙烯
                "C#C",  # 乙炔
                "c1ccccc1",  # 苯
                "CC(C)C",  # 異丁烷
                "CCCC",  # 丁烷
                "CNC",  # 二甲胺
                "COC",  # 二甲醚
                "C1CCCCC1",  # 環己烷
                "CC=O",  # 乙醛
                "CCO",  # 乙醇（重複，但會被去重）
            ]
            
            # 選擇一個基礎分子
            base_index = index % len(simple_molecules)
            base_molecule = simple_molecules[base_index]
            
            # 添加索引以確保唯一性（通過簡單的分子修改）
            variations = [
                base_molecule,
                base_molecule.replace('C', 'N', 1) if 'C' in base_molecule else base_molecule,
                base_molecule.replace('C', 'O', 1) if 'C' in base_molecule else base_molecule,
                base_molecule + 'C' if len(base_molecule) < self.max_smiles_length - 1 else base_molecule,
            ]
            
            variation_index = index % len(variations)
            result = variations[variation_index]
            
            # 驗證生成的SMILES
            if self.validate_smiles(result):
                return result
            else:
                return "CCO"  # 如果驗證失敗，返回簡單的乙醇
                
        except Exception as e:
            logger.debug(f"Error in generate_simple_variation: {e}")
            return "CCO"

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
        import time
        start_time = time.time()
        
        try:
            # Quick checks first
            if not smiles or len(smiles) < 2:
                return False
            
            # 檢查長度限制
            if len(smiles) > self.max_smiles_length:
                return False
            
            # RDKit import and mol creation - potential bottleneck
            rdkit_start = time.time()
            from rdkit import Chem
            from rdkit.Chem import rdMolDescriptors
            
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return False
            rdkit_time = time.time() - rdkit_start
            
            # 檢查分子量 - expensive calculation
            mw_start = time.time()
            mw = rdMolDescriptors.CalcExactMolWt(mol)
            if mw > 1000 or mw < 30:  # 合理的分子量範圍
                return False
            mw_time = time.time() - mw_start
            
            # 檢查原子數 - less expensive
            atoms_start = time.time()
            num_atoms = mol.GetNumAtoms()
            if num_atoms > 100 or num_atoms < 2:
                return False
            atoms_time = time.time() - atoms_start
            
            # 檢查是否包含異常結構 - potentially expensive
            struct_start = time.time()
            if self.has_unusual_structures(mol):
                return False
            struct_time = time.time() - struct_start
            
            total_time = time.time() - start_time
            if total_time > 0.1:  # Only log if validation takes > 100ms
                print(f"[VALIDATION-DEBUG] SMILES validation took {total_time:.3f}s "
                      f"(rdkit: {rdkit_time:.3f}s, mw: {mw_time:.3f}s, "
                      f"atoms: {atoms_time:.3f}s, struct: {struct_time:.3f}s) for: {smiles[:30]}...")
            
            return True
            
        except Exception as e:
            total_time = time.time() - start_time
            if total_time > 0.05:  # Log errors that take time
                print(f"[VALIDATION-DEBUG] SMILES validation failed in {total_time:.3f}s for '{smiles[:30]}...': {e}")
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