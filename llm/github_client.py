"""
GitHub Models API Client
支援 GitHub Models API 的 LLM 客戶端，並整合 LangSmith 追蹤
"""
import os
import json
import logging
from typing import List, Dict, Any, Optional
from openai import OpenAI
from langsmith import traceable

logger = logging.getLogger(__name__)


class GitHubClient:
    """GitHub Models API 客戶端"""
    
    def __init__(self, 
                 model_name: str = "openai/gpt-4.1",
                 temperature: float = 0.2,
                 max_completion_tokens: int = 4000,
                 top_p: float = 1.0,
                 stream: bool = False,
                 api_key: Optional[str] = None,
                 **kwargs):
        """
        初始化 GitHub 客戶端
        
        Args:
            model_name: GitHub 模型名稱
            temperature: 溫度參數
            max_completion_tokens: 最大完成令牌數
            top_p: Top-p 參數
            stream: 是否使用串流
            api_key: GitHub API 金鑰，如果未提供則從環境變數讀取
            **kwargs: 其他參數
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_completion_tokens = max_completion_tokens
        self.top_p = top_p
        self.stream = stream
        
        # 設定 API 金鑰
        if api_key:
            os.environ["GITHUB_TOKEN"] = api_key
        
        # 初始化 GitHub 客戶端（使用 OpenAI 客戶端）
        self.client = OpenAI(
            base_url="https://models.github.ai/inference",
            api_key=api_key or os.environ.get("GITHUB_TOKEN")
        )
        
        logger.info(f"GitHub client initialized with model: {model_name}")

    @traceable(name="GitHub::generate")
    def generate(self, messages: List[Dict[str, str]]) -> str:
        """
        生成文本 - 增強 LangSmith 追蹤
        
        Args:
            messages: 對話消息列表
            
        Returns:
            生成的文本
        """
        try:
            # 記錄請求詳情到 LangSmith
            logger.info(f"GitHub API Request - Model: {self.model_name}")
            logger.info(f"Request parameters - Temperature: {self.temperature}, Max tokens: {self.max_completion_tokens}")
            logger.debug(f"Input messages count: {len(messages)}")
            
            # 記錄輸入內容預覽
            for i, msg in enumerate(messages):
                content_preview = msg.get('content', '')
                logger.debug(f"Message {i}: {msg.get('role', 'unknown')} - {content_preview}")
            
            if self.stream:
                response = self._generate_stream(messages)
            else:
                response = self._generate_sync(messages)
            
            # 記錄響應詳情到 LangSmith
            logger.info(f"GitHub API Response - Length: {len(response)} characters")
            response_preview = response
            logger.debug(f"Response preview: {response_preview}")
            
            return response
                
        except Exception as e:
            logger.error(f"Error generating text with GitHub: {e}")
            raise

    @traceable(name="GitHub::generate_stream")
    def _generate_stream(self, messages: List[Dict[str, str]]) -> str:
        """串流生成"""
        try:
            stream = self.client.chat.completions.create(
                messages=messages,
                model=self.model_name,
                stream=True,
                max_tokens=self.max_completion_tokens,
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

    @traceable(name="GitHub::generate_sync")
    def _generate_sync(self, messages: List[Dict[str, str]]) -> str:
        """同步生成"""
        try:
            response = self.client.chat.completions.create(
                messages=messages,
                model=self.model_name,
                stream=False,
                max_tokens=self.max_completion_tokens,
                temperature=self.temperature,
                top_p=self.top_p
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Error in sync generation: {e}")
            raise

    def __call__(self, messages: List[Dict[str, str]]) -> Any:
        """使客戶端可直接調用"""
        return self.generate(messages)