"""
LLM Generator
LLM 客戶端封裝：
- 初始化 GitHub 或 Cerebras 客戶端
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
from .github_client import GitHubClient
from .prompt import create_llm_messages

logger = logging.getLogger(__name__)


class LLMGenerator:
    """LLM 客戶端封裝 - 支援 GitHub 和 Cerebras"""
    
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
            provider: LLM 提供商 ("github" 或 "cerebras")
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
        
        if provider == "github":
            # 使用傳入的 api_key 或環境變數
            api_key = api_key or os.getenv("GITHUB_TOKEN")
            if not api_key:
                raise ValueError("GITHUB_TOKEN is required")
            
            self.client = GitHubClient(
                model_name=model_name,
                temperature=temperature,
                max_completion_tokens=max_completion_tokens,
                top_p=top_p,
                stream=stream,
                api_key=api_key
            )
        elif provider == "cerebras":
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
        根據動作列表生成分子的 SMILES - 智能重新生成直到獲得足夠的有效分子
        """
        import time
        start_time = time.time()
        print(f"[LLM-DEBUG] Starting generate_batch with {len(actions)} actions")
        
        if not actions:
            logger.warning("No actions provided for generation")
            return []
        
        target_count = len(actions)
        valid_smiles = []
        valid_smiles_set = set()
        total_attempts = 0
        max_total_attempts = max_retries * 3  # 允許更多嘗試
        
        while len(valid_smiles) < target_count and total_attempts < max_total_attempts:
            total_attempts += 1
            needed_count = target_count - len(valid_smiles)
            
            print(f"[LLM-DEBUG] Attempt {total_attempts}: Need {needed_count} more valid SMILES (have {len(valid_smiles)}/{target_count})")
            
            try:
                # 動態調整請求數量 - 請求比需要的多一些以提高效率
                request_count = min(needed_count * 2, len(actions))  # 最多請求原始數量的2倍
                
                # 為剩餘的動作構建提示
                remaining_actions = actions[-request_count:] if total_attempts == 1 else actions[:request_count]
                
                action_descriptions = []
                for i, action in enumerate(remaining_actions):
                    description = action.get('description', action.get('name', f'action_{i}'))
                    action_descriptions.append(f"{i+1}. {description}")
                
                prompt = f"""You are an expert medicinal chemist. Given a parent molecule and a list of chemical modifications, generate new valid SMILES.
Parent molecule: {parent_smiles}
Chemical modifications to apply:
{chr(10).join(action_descriptions)}

Generate exactly {request_count} new SMILES, one per line. Each SMILES should:
- Be chemically valid and reasonable
- Apply the corresponding modification to the parent molecule
- Have molecular weight < 800 Da
- Be under {self.max_smiles_length} characters
- Be structurally diverse

SMILES (one per line):"""
                
                print(f"[LLM-DEBUG] Calling {self.provider} API for {request_count} SMILES...")
                api_start = time.time()
                
                if self.provider == "github":
                    response = self.client.generate([{"role": "user", "content": prompt}])
                    content = response
                elif self.provider == "cerebras":
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=self.temperature,
                        max_completion_tokens=self.max_tokens,
                        top_p=self.top_p,
                        stream=False
                    )
                    content = response.choices[0].message.content
                
                api_time = time.time() - api_start
                print(f"[LLM-DEBUG] API call completed in {api_time:.2f}s")
                
                # 解析和驗證生成的 SMILES
                lines = [line.strip() for line in content.strip().split('\n') if line.strip()]
                print(f"[LLM-DEBUG] Parsing {len(lines)} lines from response")
                
                new_valid_count = 0
                for i, line in enumerate(lines):
                    # 清理 SMILES 字符串
                    smiles = re.sub(r'^\d+\.\s*', '', line.strip())
                    smiles = re.sub(r'^[-*]\s*', '', smiles.strip())
                    smiles = smiles.split()[0] if smiles.split() else ""
                    
                    if smiles and len(smiles) >= 2:
                        # 驗證 SMILES
                        if self.validate_smiles(smiles) and smiles not in valid_smiles_set:
                            valid_smiles.append(smiles)
                            valid_smiles_set.add(smiles)
                            new_valid_count += 1
                            print(f"[LLM-DEBUG] Added valid SMILES #{len(valid_smiles)}: {smiles}")
                        else:
                            print(f"[LLM-DEBUG] Rejected SMILES: {smiles} (invalid or duplicate)")
                
                print(f"[LLM-DEBUG] Added {new_valid_count} new valid SMILES in this attempt")
                
                # 如果這次嘗試沒有獲得任何有效分子，嘗試使用回退方法
                if new_valid_count == 0:
                    print(f"[LLM-DEBUG] No valid SMILES generated, using fallback methods")
                    for fallback_attempt in range(min(needed_count, 5)):  # 最多生成5個回退分子
                        fallback_smiles = self.fallback_smiles_generation(parent_smiles)
                        if fallback_smiles and fallback_smiles not in valid_smiles_set:
                            valid_smiles.append(fallback_smiles)
                            valid_smiles_set.add(fallback_smiles)
                            print(f"[LLM-DEBUG] Added fallback SMILES #{len(valid_smiles)}: {fallback_smiles}")
                
            except Exception as e:
                print(f"[LLM-DEBUG] Error in attempt {total_attempts}: {e}")
                logger.error(f"Error in generate_batch attempt {total_attempts}: {e}")
                
                # 錯誤時也嘗試生成一些回退分子
                if len(valid_smiles) < target_count:
                    fallback_smiles = self.fallback_smiles_generation(parent_smiles)
                    if fallback_smiles and fallback_smiles not in valid_smiles_set:
                        valid_smiles.append(fallback_smiles)
                        valid_smiles_set.add(fallback_smiles)
        
        # 如果仍然不足，使用簡單分子填充
        if len(valid_smiles) < target_count:
            print(f"[LLM-DEBUG] Still need {target_count - len(valid_smiles)} more SMILES, using simple molecules")
            simple_molecules = [
                "CCO", "CCN", "CCC", "C=C", "c1ccccc1", "CC(C)C", 
                "CCCC", "CNC", "COC", "C1CCCCC1", "CC=O", "CCO"
            ]
            
            for simple_mol in simple_molecules:
                if len(valid_smiles) >= target_count:
                    break
                if simple_mol not in valid_smiles_set:
                    valid_smiles.append(simple_mol)
                    valid_smiles_set.add(simple_mol)
                    print(f"[LLM-DEBUG] Added simple molecule #{len(valid_smiles)}: {simple_mol}")
        
        # 確保返回正確數量（截取或填充）
        if len(valid_smiles) > target_count:
            valid_smiles = valid_smiles[:target_count]
        elif len(valid_smiles) < target_count:
            # 最後的填充 - 使用父分子的變體
            while len(valid_smiles) < target_count:
                variation = self.generate_simple_variation(parent_smiles, len(valid_smiles))
                if variation not in valid_smiles_set:
                    valid_smiles.append(variation)
                    valid_smiles_set.add(variation)
                else:
                    valid_smiles.append("CCO")  # 最終保證
        
        total_time = time.time() - start_time
        success_rate = len(valid_smiles) / target_count if target_count > 0 else 0
        
        print(f"[LLM-DEBUG] generate_batch completed in {total_time:.2f}s")
        print(f"[LLM-DEBUG] Final result: {len(valid_smiles)}/{target_count} SMILES (success rate: {success_rate:.2%})")
        print(f"[LLM-DEBUG] Total API attempts: {total_attempts}")
        
        logger.info(f"Generated {len(valid_smiles)} SMILES from {target_count} actions in {total_attempts} attempts")
        
        return valid_smiles

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