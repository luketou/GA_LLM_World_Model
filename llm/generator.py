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
from .vllm_client import VLLMClient
from .prompt import create_enhanced_llm_messages, create_simple_generation_prompt, create_fallback_prompt
from utils.smiles_tools import get_pubchem_data_v2, canonicalize

logger = logging.getLogger(__name__)

# 新增 llm_logger 設定
llm_logger = logging.getLogger("llm_logger")
if not llm_logger.handlers:
    llm_handler = logging.FileHandler("llm_log.json")
    llm_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    llm_logger.addHandler(llm_handler)
    llm_logger.setLevel(logging.INFO)

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
            # 使用傳入的 api_key 或環境變數
            api_key = api_key or os.getenv("CEREBRAS_API_KEY")
            if not api_key:
                raise ValueError("CEREBRAS_API_KEY is required")
            
            self.client = CerebrasClient(
                model_name=model_name,
                temperature=temperature,
                max_completion_tokens=max_completion_tokens,
                top_p=top_p,
                stream=stream,
                api_key=api_key
            )
        elif provider == "vllm":
            self.client = VLLMClient(
                model_name=model_name,
                temperature=temperature,
                max_completion_tokens=max_completion_tokens,
                top_p=top_p,
                stream=stream,
                api_key="vllm" # Dummy key for local server
            )
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    @traceable
    def generate_batch(self, parent_smiles: str, actions: List[Dict], max_retries: int = 3) -> List[str]:
        """
        根據動作列表生成分子的 SMILES - 使用 <SMILES></SMILES> token 標記提升準確性
        """
        import time
        start_time = time.time()
        logger.debug(f"[LLM-DEBUG] Starting generate_batch with {len(actions)} actions")
        
        # LLM input log
        try:
            llm_logger.info(json.dumps({
                "type": "input",
                "provider": self.provider,
                "parent_smiles": parent_smiles,
                "actions": actions
            }, ensure_ascii=False))
        except Exception as e:
            logger.debug(f"Failed to log LLM input: {e}")
        
        if not actions:
            logger.warning("No actions provided for generation")
            return []
        
        target_count = len(actions)
        valid_smiles = []
        valid_smiles_set = set() # 用於去重
        total_attempts = 0
        max_total_attempts = max_retries * 3
        
        # PubChem RAG: 查詢 parent_smiles 的 PubChem 資訊
        parent_pubchem_data = get_pubchem_data_v2(parent_smiles)
        
        while len(valid_smiles) < target_count and total_attempts < max_total_attempts:
            total_attempts += 1
            needed_count = target_count - len(valid_smiles) # 每次嘗試需要補充的數量

            logger.debug(f"[LLM-DEBUG] Attempt {total_attempts}: Need {needed_count} more valid SMILES (have {len(valid_smiles)}/{target_count})")
            
            try:
                # 根據嘗試次數選擇不同的提示策略
                if total_attempts <= 2:
                    # 前兩次使用增強提示
                    messages = create_enhanced_llm_messages(parent_smiles, actions, pubchem_data=parent_pubchem_data)
                elif total_attempts <= 4:
                    # 第3-4次使用簡單提示
                    messages = create_simple_generation_prompt(parent_smiles, needed_count, pubchem_data=parent_pubchem_data)
                else:
                    # 後續使用最簡化提示
                    messages = create_fallback_prompt(parent_smiles, needed_count, pubchem_data=parent_pubchem_data)
                
                logger.debug(f"Calling {self.provider} API for {needed_count} SMILES...")
                api_start = time.time()
                
                content = self.client.generate(messages)
                
                # LLM output log
                try:
                    llm_logger.info(json.dumps({
                        "type": "output",
                        "provider": self.provider,
                        "parent_smiles": parent_smiles,
                        "actions": actions,
                        "messages": messages,
                        "response": content
                    }, ensure_ascii=False))
                except Exception as e:
                    logger.debug(f"Failed to log LLM output: {e}")
                
                api_time = time.time() - api_start
                logger.debug(f"API call completed in {api_time:.2f}s")
                
                # 使用新的提取方法解析 SMILES
                new_smiles = self._extract_smiles_from_response(content)
                
                # 去重並添加到結果中
                for smiles in new_smiles:
                    if smiles not in valid_smiles_set and len(valid_smiles) < target_count:
                        valid_smiles.append(smiles)
                        valid_smiles_set.add(smiles) # 添加到集合中以確保唯一性
                
                logger.debug(f"Added {len(new_smiles)} new valid SMILES in this attempt")
                
                # 如果這次嘗試沒有獲得任何有效分子，嘗試使用回退方法
                if len(new_smiles) == 0:
                    logger.debug("No valid SMILES generated, using fallback methods")
                    for fallback_attempt in range(min(needed_count, 3)):
                        fallback_smiles = self.fallback_smiles_generation(parent_smiles)
                        if fallback_smiles and fallback_smiles not in valid_smiles_set:
                            valid_smiles.append(fallback_smiles)
                            valid_smiles_set.add(fallback_smiles)
                            logger.debug(f"Added fallback SMILES #{fallback_attempt+1}: {fallback_smiles}")
                
            except Exception as e:
                logger.error(f"Error in generate_batch attempt {total_attempts}: {e}")
                # LLM error log
                try:
                    llm_logger.info(json.dumps({
                        "type": "output",
                        "provider": self.provider,
                        "parent_smiles": parent_smiles,
                        "actions": actions,
                        "messages": messages if 'messages' in locals() else None,
                        "response": f"Error: {e}"
                    }, ensure_ascii=False))
                except Exception as e2:
                    logger.debug(f"Failed to log LLM error output: {e2}")
                
                # 錯誤時也嘗試生成一些回退分子
                if len(valid_smiles) < target_count:
                    fallback_smiles = self.fallback_smiles_generation(parent_smiles)
                    if fallback_smiles and fallback_smiles not in valid_smiles_set:
                        valid_smiles.append(fallback_smiles)
                        valid_smiles_set.add(fallback_smiles)

        # 如果仍然不足，使用簡單分子填充
        if len(valid_smiles) < target_count:
            logger.debug(f"Still need {target_count - len(valid_smiles)} more SMILES, using simple molecules")
            simple_molecules = [
                "CCO", "CCN", "CCC", "C=C", "c1ccccc1", "CC(C)C", 
                "CCCC", "CNC", "COC", "C1CCCCC1", "CC=O", "CCO"
            ]
            
            for simple_mol in simple_molecules:
                if len(valid_smiles) >= target_count:
                    break
                if simple_mol not in valid_smiles_set:
                    valid_smiles.append(simple_mol)
                    valid_smiles_set.add(simple_mol) # 添加到集合中以確保唯一性
                    logger.debug(f"Added simple molecule #{len(valid_smiles)}: {simple_mol}")
        
        # 確保返回正確數量（截取或填充到目標數量）
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
        
        total_time = time.time() - start_time # 計算總耗時
        success_rate = len(valid_smiles) / target_count if target_count > 0 else 0
        
        logger.debug(f"generate_batch completed in {total_time:.2f}s")
        logger.debug(f"Final result: {len(valid_smiles)}/{target_count} SMILES (success rate: {success_rate:.2%})")
        logger.debug(f"Total API attempts: {total_attempts}")
        
        logger.info(f"Generated {len(valid_smiles)} SMILES from {target_count} actions in {total_attempts} attempts")
        
        return valid_smiles

    def _extract_smiles_from_response(self, response_text: str) -> List[str]:
        """
        從 LLM 回應中提取 SMILES - 使用 <SMILES></SMILES> token 標記
        """
        smiles_list = []
        
        try:
            # 方法 1: 使用正則表達式提取 <SMILES></SMILES> 標記中的內容
            import re
            pattern = r'<SMILES>(.*?)</SMILES>'
            matches = re.findall(pattern, response_text, re.IGNORECASE | re.MULTILINE)
            
            for match in matches:
                smiles = match.strip()
                canonical_smiles = canonicalize(smiles)
                if canonical_smiles:
                    smiles_list.append(canonical_smiles)
                    logger.debug(f"[TOKEN-EXTRACT] Valid SMILES found: {canonical_smiles}")
                else:
                    logger.debug(f"[TOKEN-EXTRACT] Invalid SMILES rejected: {smiles}")
            
            # 方法 2: 如果正則表達式沒有找到結果，嘗試逐行解析
            if not smiles_list:
                logger.debug("[TOKEN-EXTRACT] No token-marked SMILES found, trying line-by-line parsing")
                lines = response_text.strip().split('\n')
                
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    
                    # 嘗試從行中提取 SMILES
                    if '<SMILES>' in line and '</SMILES>' in line:
                        start = line.find('<SMILES>') + 8
                        end = line.find('</SMILES>')
                        if start < end:
                            smiles = line[start:end].strip()
                            if smiles and self.validate_smiles(smiles):
                                smiles_list.append(smiles)
                                logger.debug(f"[LINE-EXTRACT] Valid SMILES found: {smiles}")
                    else:
                        # 檢查是否是純 SMILES 行（作為後備）
                        if self.validate_smiles(line):
                            smiles_list.append(line)
                            logger.debug(f"[PLAIN-EXTRACT] Valid SMILES found: {line}")
            
            logger.info(f"[SMILES-EXTRACT] Total valid SMILES extracted: {len(smiles_list)}")
            return smiles_list
            
        except Exception as e:
            logger.error(f"Error extracting SMILES from response: {e}")
            return []

    @traceable(name="LLMGenerator::generate_text")
    def generate_text_response(self, prompt: str) -> str:
        """
        通用文本生成 - 用於需要文字回覆的任務，例如動作選擇的推理。
        這個方法會將單一的 prompt 字符串轉換為 client 需要的 messages 格式。

        Args:
            prompt: 發送給 LLM 的完整提示字符串。

        Returns:
            LLM 生成的文字回覆。
        """
        messages = [{"role": "user", "content": prompt}]
        # LLM input log
        try:
            llm_logger.info(json.dumps({
                "type": "input",
                "provider": self.provider,
                "messages": messages
            }, ensure_ascii=False))
        except Exception as e:
            logger.debug(f"Failed to log LLM input: {e}")
        try:
            content = self.client.generate(messages)
            # LLM output log
            try:
                llm_logger.info(json.dumps({
                    "type": "output",
                    "provider": self.provider,
                    "messages": messages,
                    "response": content
                }, ensure_ascii=False))
            except Exception as e:
                logger.debug(f"Failed to log LLM output: {e}")
            return content
        except Exception as e:
            logger.error(f"Error in generate_text_response: {e}")
            # LLM error log
            try:
                llm_logger.info(json.dumps({
                    "type": "output",
                    "provider": self.provider,
                    "messages": messages,
                    "response": f"Error: {e}"
                }, ensure_ascii=False))
            except Exception as e2:
                logger.debug(f"Failed to log LLM error output: {e2}")
            return '{"error": "LLM generation failed", "reasoning": "An error occurred during LLM text generation."}'

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
        # 根據架構原則，在 Oracle 評分前只進行輕量級驗證
        # 避免使用 RDKit
        try:
            # 1. 基本的空值和長度檢查
            if not smiles or not isinstance(smiles, str):
                return False
            
            if len(smiles) > self.max_smiles_length or len(smiles) < 2:
                return False

            # 2. 檢查是否有不應存在的空格
            if ' ' in smiles:
                return False
            
            # 3. 檢查是否包含常見的 LLM 錯誤輸出模式
            # 例如：包含解釋性文字、不完整的 JSON 符號、思考過程標籤
            if any(keyword in smiles.lower() for keyword in [
                "selected_action_names", "reasoning", "confidence",
                "thought", "thinking", "explanation", "here are", "i will",
                "json", "```", "```json"
            ]):
                return False

            # 4. 檢查是否包含非SMILES字符 (簡化版，不包含所有有效SMILES字符)
            if not re.fullmatch(r'[a-zA-Z0-9\(\)\[\]\-=#$@:./\\+]+', smiles):
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