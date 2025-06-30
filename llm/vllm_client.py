"""
vLLM Client
- Interacts with a vLLM server that provides an OpenAI-compatible API.
"""
import os
import requests
import json
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class VLLMClient:
    """Client for interacting with a vLLM OpenAI-compatible server."""

    def __init__(
        self,
        model_name: str,
        api_url: str = "http://localhost:8000/v1/chat/completions",
        temperature: float = 0.2,
        max_completion_tokens: int = 2048,
        top_p: float = 1.0,
        stream: bool = False, # vLLM OpenAI endpoint does not support streaming in the same way
        api_key: str = "vllm"
    ):
        """
        Initializes the vLLM client.

        Args:
            model_name: The name of the model being served by vLLM.
            api_url: The URL of the vLLM server's chat completions endpoint.
            temperature: The temperature for sampling.
            max_completion_tokens: The maximum number of tokens to generate.
            top_p: The top-p value for sampling.
            stream: Whether to use streaming (not fully supported).
            api_key: API key (can be a dummy key for local servers).
        """
        self.model = model_name
        self.api_url = api_url
        self.temperature = temperature
        self.max_tokens = max_completion_tokens
        self.top_p = top_p
        self.stream = stream
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        logger.info(f"vLLM client initialized for model {self.model} at {self.api_url}")

    def generate(self, messages: List[Dict[str, Any]]) -> str:
        """
        Generates a response from the vLLM server.

        Args:
            messages: A list of messages in the conversation.

        Returns:
            The generated text content.
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "stream": self.stream
        }

        try:
            response = requests.post(self.api_url, headers=self.headers, data=json.dumps(payload))
            response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
            response_data = response.json()
            content = response_data['choices'][0]['message']['content']
            return content
        except requests.exceptions.RequestException as e:
            logger.error(f"Error calling vLLM API: {e}")
            return f"Error: Could not connect to vLLM server at {self.api_url}."
        except (KeyError, IndexError) as e:
            logger.error(f"Error parsing vLLM response: {e}")
            logger.error(f"Received response: {response.text}")
            return "Error: Invalid response format from vLLM server."
