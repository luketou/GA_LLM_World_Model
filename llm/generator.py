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
                 model_name: str = "llama-4-scout-17b-16e-instruct",
                 temperature: float = 0.2,
                 max_completion_tokens: int = 2048,
                 top_p: float = 1.0,
                 stream: bool = True,
                 api_key: Optional[str] = None,
                 **kwargs):
        """
        初始化 LLM 客戶端
        
        Args:
            provider: LLM 提供者 ("cerebras" 或 "openai")
            model_name: 模型名稱
            temperature: 溫度參數
            max_completion_tokens: 最大完成令牌數
            top_p: Top-p 參數
            stream: 是否使用串流
            api_key: API 金鑰
            **kwargs: 其他參數
        """
        self.provider = provider
        self.model_name = model_name
        self.temperature = temperature
        self.max_completion_tokens = max_completion_tokens
        
        if provider.lower() == "cerebras":
            self.client = CerebrasClient(
                model_name=model_name,
                temperature=temperature,
                max_completion_tokens=max_completion_tokens,
                top_p=top_p,
                stream=stream,
                api_key=api_key,
                **kwargs
            )
        else:
            # 回退到 OpenAI (保持向後兼容)
            from langchain_openai import ChatOpenAI
            llm_kwargs = {
                "model_name": model_name,
                "temperature": temperature,
                **kwargs
            }
            if max_completion_tokens:
                llm_kwargs["max_tokens"] = max_completion_tokens
            self.client = ChatOpenAI(**llm_kwargs)
        
        logger.info(f"LLM Generator initialized with {provider} provider, model: {model_name}")

    @traceable(name="LLM::generate_batch")
    def generate_batch(self,
                       parent_smiles: str,
                       actions: List[Dict[str, Any]],
                       max_retries: int = 3) -> List[str]:
        """
        批次生成介面：
        - 組裝 system + action messages  
        - 呼叫 LLM
        - 解析並驗證 SMILES 列表
        - 如果生成數量不足，重新嘗試生成補齊
        - 回傳合法 SMILES
        """
        target_count = len(actions)
        all_valid_smiles = []
        retry_count = 0
        
        while len(all_valid_smiles) < target_count and retry_count < max_retries:
            try:
                # 計算還需要生成多少個
                remaining_actions = actions[len(all_valid_smiles):]
                
                # 創建消息
                messages = create_llm_messages(parent_smiles, remaining_actions)
                
                # 呼叫 LLM
                response = self.client(messages)
                
                # 解析回應
                if hasattr(response, 'content'):
                    response_content = response.content
                else:
                    response_content = str(response)
                
                smiles_list = self._parse_response(response_content)
                
                # 基本檢查 SMILES
                basic_smiles = self._basic_smiles_check(smiles_list)
                
                # 添加有效的 SMILES
                all_valid_smiles.extend(basic_smiles)
                
                logger.info(f"Retry {retry_count + 1}: Generated {len(basic_smiles)} valid SMILES, total: {len(all_valid_smiles)}/{target_count}")
                
                retry_count += 1
                
            except Exception as e:
                logger.error(f"Error in generate_batch (retry {retry_count + 1}): {e}")
                retry_count += 1
        
        # 如果仍然不足，用變異的分子填充而不是完全相同的父分子
        if len(all_valid_smiles) < target_count:
            logger.warning(f"Generated fewer SMILES ({len(all_valid_smiles)}) than actions ({target_count}) after {max_retries} retries")
            
            # 創建變異版本而不是重複相同的分子
            import random
            
            while len(all_valid_smiles) < target_count:
                # 使用已生成的分子或父分子作為基礎
                base_smiles = all_valid_smiles[0] if all_valid_smiles else parent_smiles
                
                # 創建簡單的變異
                # 方法1：嘗試替換一些字符
                varied_smiles = self._create_simple_variation(base_smiles)
                
                all_valid_smiles.append(varied_smiles)
        
        # 確保返回的 SMILES 數量與 actions 數量相符
        result = all_valid_smiles[:target_count]
        logger.info(f"Final result: {len(result)} SMILES from {target_count} actions")
        
        return result

    def _parse_response(self, response_content: str) -> List[str]:
        """解析 LLM 回應，提取 SMILES 列表"""
        try:
            # 清理回應內容
            content = response_content.strip()
            
            # 嘗試找到 JSON 數組
            start_idx = content.find('[')
            end_idx = content.rfind(']')
            
            if start_idx != -1 and end_idx != -1:
                json_str = content[start_idx:end_idx+1]
                smiles_list = json.loads(json_str)
                if isinstance(smiles_list, list):
                    return [str(s).strip() for s in smiles_list if s]
            
            # 如果 JSON 解析失敗，嘗試直接解析 JSON
            smiles_list = json.loads(content)
            if isinstance(smiles_list, list):
                return [str(s).strip() for s in smiles_list if s]
                
        except json.JSONDecodeError:
            logger.warning("Failed to parse JSON, trying line-by-line parsing")
        
        # 如果不是 JSON，嘗試按行解析
        lines = response_content.strip().split('\n')
        smiles_list = []
        
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#') and not line.startswith('//'):
                # 移除可能的前綴和後綴
                line = line.strip('",[]() ')
                if line and len(line) > 1:  # 基本的 SMILES 長度檢查
                    smiles_list.append(line)
        
        return smiles_list

    def _basic_smiles_check(self, smiles_list: List[str]) -> List[str]:
        """
        基本 SMILES 檢查（不使用 RDKit）
        只進行簡單的字符串格式檢查，真正的驗證交給 Oracle
        """
        checked_smiles = []
        
        for smiles in smiles_list:
            try:
                smiles = smiles.strip()
                if not smiles:
                    continue
                
                # 基本格式檢查：
                # 1. 長度合理 (至少1個字符，不超過150個字符以避免過度複雜)
                # 2. 包含合理的化學元素字符
                # 3. 不包含明顯的非SMILES字符
                if (1 <= len(smiles) <= 150 and
                    self._basic_smiles_format_check(smiles)):
                    checked_smiles.append(smiles)
                else:
                    logger.warning(f"Basic format check failed for: {smiles}")
                    
            except Exception as e:
                logger.warning(f"Error in basic check for '{smiles}': {e}")
        
        return checked_smiles

    def _basic_smiles_format_check(self, smiles: str) -> bool:
        """
        基本 SMILES 格式檢查（不使用 RDKit）
        檢查是否包含合理的 SMILES 字符
        """
        # 常見的 SMILES 字符集合 - 擴展版本
        # 元素：C, N, O, S, P, F, Cl, Br, I, H, B, Si, As, Se 等
        # 結構：()[]@=#-+.:\/
        # 數字：0-9
        # 額外：空格和其他可能的字符
        valid_chars = set('CNOSPFHBIclbrase0123456789()[]@=#-+.:\\/\\ \t*%')
        
        # 檢查是否大部分字符都是有效的 SMILES 字符
        valid_char_count = sum(1 for c in smiles.lower() if c in valid_chars)
        total_chars = len(smiles)
        
        # 放寬到 70% 的字符是有效的就認為格式基本正確
        if total_chars > 0 and (valid_char_count / total_chars) >= 0.7:
            return True
        
        # 如果包含基本的化學元素字符，也認為是有效的
        has_carbon = 'c' in smiles.lower()
        has_nitrogen = 'n' in smiles.lower()
        has_oxygen = 'o' in smiles.lower()
        
        if has_carbon or has_nitrogen or has_oxygen:
            return True
        
        return False

    def _create_simple_variation(self, base_smiles: str) -> str:
        """創建分子的簡單變異，避免過度複雜化"""
        import random
        
        # 如果基礎分子已經很長，使用更保守的變異
        if len(base_smiles) > 100:
            # 對於長分子，只做簡單的原子替換
            variations = [
                base_smiles,  # 保持原樣
                base_smiles.replace("C", "N", 1),  # 替換一個碳為氮
                base_smiles.replace("N", "O", 1),  # 替換一個氮為氧
                base_smiles.replace("O", "S", 1),  # 替換一個氧為硫
            ]
        else:
            # 對於較短分子，可以做更多變異
            variations = [
                base_smiles,  # 保持原樣
                base_smiles + "C",  # 添加甲基
                base_smiles + "O",  # 添加羥基
                base_smiles + "N",  # 添加氨基
                base_smiles.replace("C", "N", 1),  # 替換一個碳為氮
                base_smiles.replace("N", "O", 1),  # 替換一個氮為氧
            ]
        
        # 移除可能無效的變異
        valid_variations = [v for v in variations if 1 <= len(v) <= 150]
        
        if valid_variations:
            return random.choice(valid_variations)
        else:
            return base_smiles