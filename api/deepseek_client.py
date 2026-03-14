"""
DeepSeek LLM Client Adapter

Implements the BaseLLM interface for DeepSeek's API.

Installation:
    pip install openai

Environment Variables:
    DEEPSEEK_API_KEY: API key for DeepSeek
"""

import os
import time
from typing import Optional, Dict, Any, List, Iterator, AsyncIterator

from .base import (
    BaseLLM, ProviderType, GenerationConfig, LLMResponse, 
    ModelInfo, ResponseFormat, _estimate_cost
)

try:
    from openai import OpenAI
    DEEPSEEK_AVAILABLE = True
except ImportError:
    DEEPSEEK_AVAILABLE = False


class DeepSeekLLMClient(BaseLLM):
    """
    DeepSeek LLM client using the OpenAI-compatible API.
    
    Usage:
        client = DeepSeekLLMClient(api_key="your-api-key")
        response = client.generate("Explain quantum computing")
        print(response.content)
    
    Latest Models (as of 2025):
        - deepseek-chat: DeepSeek V3.2 - General purpose chat
        - deepseek-reasoner: DeepSeek R1 - Advanced reasoning model
    """

    DEFAULT_MODEL = "deepseek-chat"
    
    MODEL_CONTEXT_WINDOWS = {
        "deepseek-chat": 131_072,
        "deepseek-reasoner": 131_072,
        "deepseek-coder": 131_072,
    }
    
    VISION_MODELS = set()

    def __init__(self, api_key: Optional[str] = None, **kwargs):
        self._api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        self._config = kwargs
        self._client = None
        
        if not DEEPSEEK_AVAILABLE:
            raise ImportError(
                "openai package is required. "
                "Install with: pip install openai"
            )

    @property
    def provider_type(self) -> ProviderType:
        return ProviderType.DEEPSEEK
    
    @property
    def default_model(self) -> str:
        return self.DEFAULT_MODEL

    def _initialize_client(self) -> None:
        if not self._api_key:
            raise ValueError(
                "DeepSeek API key is required. Provide it as an argument or "
                "set DEEPSEEK_API_KEY environment variable."
            )
        self._client = OpenAI(
            api_key=self._api_key,
            base_url="https://api.deepseek.com"
        )

    def _convert_config(self, config: Optional[GenerationConfig]) -> Dict[str, Any]:
        if config is None:
            config = GenerationConfig()
        
        deepseek_config = {}
        
        if config.max_tokens is not None:
            deepseek_config["max_tokens"] = config.max_tokens
        
        if config.temperature is not None:
            deepseek_config["temperature"] = config.temperature
        
        if config.top_p is not None:
            deepseek_config["top_p"] = config.top_p
        
        if config.stop_sequences:
            deepseek_config["stop"] = config.stop_sequences
        
        if config.response_format == ResponseFormat.JSON:
            deepseek_config["response_format"] = {"type": "json_object"}
        
        return deepseek_config

    def generate(
        self, 
        prompt: str, 
        config: Optional[GenerationConfig] = None,
        model: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        start_time = time.time()
        
        try:
            self._ensure_initialized()
            
            model_id = model or self.default_model
            deepseek_config = self._convert_config(config)
            
            messages = [{"role": "user", "content": prompt}]
            if config and config.system_instruction:
                messages.insert(0, {"role": "system", "content": config.system_instruction})
            
            response = self._client.chat.completions.create(
                model=model_id,
                messages=messages,
                **deepseek_config
            )
            
            latency = time.time() - start_time
            choice = response.choices[0]
            
            return LLMResponse(
                success=True,
                content=choice.message.content or "",
                model=model_id,
                provider="deepseek",
                tokens_used=response.usage.total_tokens,
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                latency=latency,
                finish_reason=choice.finish_reason,
                raw_response=response
            )
            
        except Exception as e:
            return LLMResponse(
                success=False,
                content="",
                model=model or self.default_model,
                provider="deepseek",
                error=str(e)
            )

    def generate_stream(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
        model: Optional[str] = None,
        **kwargs
    ) -> Iterator[str]:
        self._ensure_initialized()
        
        model_id = model or self.default_model
        deepseek_config = self._convert_config(config)
        
        messages = [{"role": "user", "content": prompt}]
        if config and config.system_instruction:
            messages.insert(0, {"role": "system", "content": config.system_instruction})
        
        response = self._client.chat.completions.create(
            model=model_id,
            messages=messages,
            stream=True,
            **deepseek_config
        )
        
        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    async def generate_async(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
        model: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        return self.generate(prompt, config, model, **kwargs)

    async def generate_stream_async(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
        model: Optional[str] = None,
        **kwargs
    ) -> AsyncIterator[str]:
        for chunk in self.generate_stream(prompt, config, model, **kwargs):
            yield chunk

    def list_models(self) -> List[ModelInfo]:
        return [
            ModelInfo(
                id="deepseek-chat",
                name="DeepSeek V3.2",
                provider="deepseek",
                context_window=131_072,
                max_output_tokens=8_192,
                supports_vision=False,
                description="General purpose chat model"
            ),
            ModelInfo(
                id="deepseek-reasoner",
                name="DeepSeek R1",
                provider="deepseek",
                context_window=131_072,
                max_output_tokens=8_192,
                supports_vision=False,
                description="Advanced reasoning model"
            ),
            ModelInfo(
                id="deepseek-coder",
                name="DeepSeek Coder",
                provider="deepseek",
                context_window=131_072,
                max_output_tokens=8_192,
                supports_vision=False,
                description="Code generation model"
            ),
        ]

    def get_model_info(self, model_id: str) -> Optional[ModelInfo]:
        for model in self.list_models():
            if model.id == model_id:
                return model
        return None

    def count_tokens(self, text: str, model: Optional[str] = None) -> int:
        return len(text) // 4
