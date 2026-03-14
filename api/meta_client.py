"""
Meta Llama LLM Client Adapter

Implements the BaseLLM interface for Meta's Llama API.

Installation:
    pip install openai

Environment Variables:
    META_API_KEY: API key for Meta Llama API
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
    META_AVAILABLE = True
except ImportError:
    META_AVAILABLE = False


class MetaLLMClient(BaseLLM):
    """
    Meta Llama LLM client using the OpenAI-compatible Llama API.
    
    This adapter implements the BaseLLM interface for Meta's Llama models,
    handling parameter mapping and response normalization.
    
    Usage:
        client = MetaLLMClient(api_key="your-api-key")
        response = client.generate("Explain quantum computing")
        print(response.content)
    
    Latest Models (as of 2025):
        - llama-4-scout-17b-16e-instruct: Llama 4 Scout - Efficient, 16 experts
        - llama-4-maverick-17b-128e-instruct: Llama 4 Maverick - Powerful, 128 experts
        - llama-3.3-70b-instruct: Llama 3.3 70B - Strong performance
        - llama-3.1-405b-instruct: Llama 3.1 405B - Large model
        - llama-3.1-70b-instruct: Llama 3.1 70B - Balanced
        - llama-3.1-8b-instruct: Llama 3.1 8B - Lightweight
    """

    DEFAULT_MODEL = "llama-4-scout-17b-16e-instruct"
    
    MODEL_CONTEXT_WINDOWS = {
        "llama-4-scout-17b-16e-instruct": 1_048_576,
        "llama-4-maverick-17b-128e-instruct": 1_048_576,
        "llama-3.3-70b-instruct": 128_000,
        "llama-3.1-405b-instruct": 128_000,
        "llama-3.1-70b-instruct": 128_000,
        "llama-3.1-8b-instruct": 128_000,
        "llama-3-70b-instruct": 8_192,
        "llama-3-8b-instruct": 8_192,
    }
    
    VISION_MODELS = {"llama-4-scout-17b-16e-instruct", "llama-4-maverick-17b-128e-instruct"}

    def __init__(self, api_key: Optional[str] = None, **kwargs):
        self._api_key = api_key or os.getenv("META_API_KEY")
        self._config = kwargs
        self._client = None
        
        if not META_AVAILABLE:
            raise ImportError(
                "openai package is required. "
                "Install with: pip install openai"
            )

    @property
    def provider_type(self) -> ProviderType:
        return ProviderType.META
    
    @property
    def default_model(self) -> str:
        return self.DEFAULT_MODEL

    def _initialize_client(self) -> None:
        if not self._api_key:
            raise ValueError(
                "Meta API key is required. Provide it as an argument or "
                "set META_API_KEY environment variable."
            )
        self._client = OpenAI(
            api_key=self._api_key,
            base_url="https://api.meta.ai/v1"
        )

    def _convert_config(self, config: Optional[GenerationConfig]) -> Dict[str, Any]:
        if config is None:
            config = GenerationConfig()
        
        meta_config = {}
        
        if config.max_tokens is not None:
            meta_config["max_tokens"] = config.max_tokens
        
        if config.temperature is not None:
            meta_config["temperature"] = config.temperature
        
        if config.top_p is not None:
            meta_config["top_p"] = config.top_p
        
        if config.stop_sequences:
            meta_config["stop"] = config.stop_sequences
        
        if config.response_format == ResponseFormat.JSON:
            meta_config["response_format"] = {"type": "json_object"}
        
        return meta_config

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
            meta_config = self._convert_config(config)
            
            messages = [{"role": "user", "content": prompt}]
            if config and config.system_instruction:
                messages.insert(0, {"role": "system", "content": config.system_instruction})
            
            response = self._client.chat.completions.create(
                model=model_id,
                messages=messages,
                **meta_config
            )
            
            latency = time.time() - start_time
            choice = response.choices[0]
            
            return LLMResponse(
                success=True,
                content=choice.message.content or "",
                model=model_id,
                provider="meta",
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
                provider="meta",
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
        meta_config = self._convert_config(config)
        
        messages = [{"role": "user", "content": prompt}]
        if config and config.system_instruction:
            messages.insert(0, {"role": "system", "content": config.system_instruction})
        
        response = self._client.chat.completions.create(
            model=model_id,
            messages=messages,
            stream=True,
            **meta_config
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
                id="llama-4-scout-17b-16e-instruct",
                name="Llama 4 Scout",
                provider="meta",
                context_window=1_048_576,
                max_output_tokens=32_768,
                supports_vision=True,
                description="Efficient model with 16 experts, 1M context"
            ),
            ModelInfo(
                id="llama-4-maverick-17b-128e-instruct",
                name="Llama 4 Maverick",
                provider="meta",
                context_window=1_048_576,
                max_output_tokens=32_768,
                supports_vision=True,
                description="Powerful model with 128 experts, 1M context"
            ),
            ModelInfo(
                id="llama-3.3-70b-instruct",
                name="Llama 3.3 70B",
                provider="meta",
                context_window=128_000,
                max_output_tokens=32_768,
                supports_vision=False,
                description="Strong performance, 128K context"
            ),
            ModelInfo(
                id="llama-3.1-405b-instruct",
                name="Llama 3.1 405B",
                provider="meta",
                context_window=128_000,
                max_output_tokens=32_768,
                supports_vision=False,
                description="Large model for complex tasks"
            ),
            ModelInfo(
                id="llama-3.1-70b-instruct",
                name="Llama 3.1 70B",
                provider="meta",
                context_window=128_000,
                max_output_tokens=32_768,
                supports_vision=False,
                description="Balanced performance"
            ),
            ModelInfo(
                id="llama-3.1-8b-instruct",
                name="Llama 3.1 8B",
                provider="meta",
                context_window=128_000,
                max_output_tokens=32_768,
                supports_vision=False,
                description="Lightweight, fast model"
            ),
        ]

    def get_model_info(self, model_id: str) -> Optional[ModelInfo]:
        for model in self.list_models():
            if model.id == model_id:
                return model
        return None

    def count_tokens(self, text: str, model: Optional[str] = None) -> int:
        return len(text) // 4
