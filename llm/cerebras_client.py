"""
Cerebras LLM Client
支援 Cerebras Cloud SDK 的 LLM 客戶端
"""
import os
import json
import logging
from typing import List, Dict, Any, Optional
from cerebras.cloud.sdk import Cerebras

logger = logging.getLogger(__name__)


class CerebrasClient:
    """Cerebras Cloud SDK 客戶端"""
    
    def __init__(self, 
                 model_name: str = "llama-4-scout-17b-16e-instruct",
                 temperature: float = 0.2,
                 max_completion_tokens: int = 2048,
                 top_p: float = 1.0,
                 stream: bool = True,
                 api_key: Optional[str] = None,
                 **kwargs):
        """
        初始化 Cerebras 客戶端
        
        Args:
            model_name: Cerebras 模型名稱
            temperature: 溫度參數
            max_completion_tokens: 最大完成令牌數
            top_p: Top-p 參數
            stream: 是否使用串流
            api_key: API 金鑰，如果未提供則從環境變數讀取
            **kwargs: 其他參數
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_completion_tokens = max_completion_tokens
        self.top_p = top_p
        self.stream = stream
        
        # 設定 API 金鑰
        if api_key:
            os.environ["CEREBRAS_API_KEY"] = api_key
        
        # 初始化 Cerebras 客戶端
        self.client = Cerebras(
            api_key=api_key or os.environ.get("CEREBRAS_API_KEY")
        )
        
        logger.info(f"Cerebras client initialized with model: {model_name}")

    def generate(self, messages: List[Dict[str, str]]) -> str:
        """
        生成文本
        
        Args:
            messages: 對話消息列表
            
        Returns:
            生成的文本
        """
        try:
            if self.stream:
                return self._generate_stream(messages)
            else:
                return self._generate_sync(messages)
                
        except Exception as e:
            logger.error(f"Error generating text with Cerebras: {e}")
            raise

    def _generate_stream(self, messages: List[Dict[str, str]]) -> str:
        """串流生成"""
        try:
            stream = self.client.chat.completions.create(
                messages=messages,
                model=self.model_name,
                stream=True,
                max_completion_tokens=self.max_completion_tokens,
                temperature=self.temperature,
                top_p=self.top_p
            )
            
            response_text = ""
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    response_text += chunk.choices[0].delta.content
            
            return response_text.strip()
            
        except Exception as e:
            logger.error(f"Error in stream generation: {e}")
            raise

    def _generate_sync(self, messages: List[Dict[str, str]]) -> str:
        """同步生成"""
        try:
            response = self.client.chat.completions.create(
                messages=messages,
                model=self.model_name,
                stream=False,
                max_completion_tokens=self.max_completion_tokens,
                temperature=self.temperature,
                top_p=self.top_p
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Error in sync generation: {e}")
            raise

    def __call__(self, messages: List[Dict[str, str]]) -> Any:
        """
        使呼叫符合 LangChain 風格的介面
        返回類似 ChatOpenAI 的回應物件
        """
        content = self.generate(messages)
        
        # 創建類似 LangChain 回應的物件
        class MockResponse:
            def __init__(self, content: str):
                self.content = content
        
        return MockResponse(content)
