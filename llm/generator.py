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
                       actions: List[Dict[str, Any]]) -> List[str]:
        """
        批次生成介面：
        - 組裝 system + action messages  
        - 呼叫 LLM
        - 解析並驗證 SMILES 列表
        - 回傳合法 SMILES
        """
        try:
            # 創建消息
            messages = create_llm_messages(parent_smiles, actions)
            
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
            
            logger.info(f"Generated {len(basic_smiles)} SMILES from {len(actions)} actions (no RDKit validation)")
            
            # 確保返回的 SMILES 數量與 actions 數量相符
            if len(basic_smiles) < len(actions):
                logger.warning(f"Generated fewer SMILES ({len(basic_smiles)}) than actions ({len(actions)})")
                # 用父分子填充不足的部分作為回退
                while len(basic_smiles) < len(actions):
                    basic_smiles.append(parent_smiles)
            
            return basic_smiles[:len(actions)]  # 確保不超過 actions 數量
            
        except Exception as e:
            logger.error(f"Error in generate_batch: {e}")
            # 回退：返回父分子的副本
            return [parent_smiles] * len(actions)

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
                # 1. 長度合理 (至少1個字符，不超過500個字符)
                # 2. 包含合理的化學元素字符
                # 3. 不包含明顯的非SMILES字符
                if (1 <= len(smiles) <= 500 and
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
        # 常見的 SMILES 字符集合
        # 元素：C, N, O, S, P, F, Cl, Br, I, H, B, Si 等
        # 結構：()[]@=#-+.:\/
        # 數字：0-9
        valid_chars = set('CNOSPFHBIclbr0123456789()[]@=#-+.:\\/\\')
        
        # 檢查是否大部分字符都是有效的 SMILES 字符
        valid_char_count = sum(1 for c in smiles.lower() if c in valid_chars)
        total_chars = len(smiles)
        
        # 如果超過 90% 的字符是有效的，認為格式基本正確
        if total_chars > 0 and (valid_char_count / total_chars) >= 0.9:
            return True
        
        return False