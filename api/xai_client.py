"""
xAI Grok LLM Client Adapter

Implements the BaseLLM interface for xAI's Grok API.

Installation:
    pip install openai

Environment Variables:
    XAI_API_KEY: API key for xAI API
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
    XAI_AVAILABLE = True
except ImportError:
    XAI_AVAILABLE = False


class XAILLMClient(BaseLLM):
    """
    xAI Grok LLM client using the OpenAI-compatible API.
    
    This adapter implements the BaseLLM interface for xAI's Grok models,
    handling parameter mapping and response normalization.
    
    Parameter Mapping:
        - max_tokens -> max_tokens
        - temperature -> temperature
        - top_p -> top_p
        - stop_sequences -> stop
    
    Usage:
        client = XAILLMClient(api_key="your-api-key")
        response = client.generate("Explain quantum computing")
        print(response.content)
    
    Latest Models (as of 2025):
        - grok-4-2025-11-04: Grok 4 - Latest flagship with advanced reasoning
        - grok-4-mini: Grok 4 Mini - Efficient version
        - grok-3-mini: Grok 3 Mini - Fast, cost-effective
        - grok-2: Grok 2 - Previous generation
    """

    DEFAULT_MODEL = "grok-4.20"
    
    MODEL_CONTEXT_WINDOWS = {
        "grok-4.20": 128_000,
        "grok-4.20-beta": 128_000,
        "grok-4-2025-11-04": 32_768,
        "grok-4-mini": 32_768,
        "grok-3-mini": 32_768,
        "grok-3": 32_768,
        "grok-2": 32_768,
    }
    
    VISION_MODELS = {"grok-4.20", "grok-4.20-beta", "grok-4-2025-11-04", "grok-4-mini"}

    def __init__(self, api_key: Optional[str] = None, **kwargs):
        self._api_key = api_key or os.getenv("XAI_API_KEY")
        self._config = kwargs
        self._client = None
        
        if not XAI_AVAILABLE:
            raise ImportError(
                "openai package is required. "
                "Install with: pip install openai"
            )

    @property
    def provider_type(self) -> ProviderType:
        return ProviderType.XAI
    
    @property
    def default_model(self) -> str:
        return self.DEFAULT_MODEL

    def _initialize_client(self) -> None:
        if not self._api_key:
            raise ValueError(
                "xAI API key is required. Provide it as an argument or "
                "set XAI_API_KEY environment variable."
            )
        self._client = OpenAI(
            api_key=self._api_key,
            base_url="https://api.x.ai/v1"
        )

    def _convert_config(self, config: Optional[GenerationConfig]) -> Dict[str, Any]:
        if config is None:
            config = GenerationConfig()
        
        xai_config = {}
        
        if config.max_tokens is not None:
            xai_config["max_tokens"] = config.max_tokens
        
        if config.temperature is not None:
            xai_config["temperature"] = config.temperature
        
        if config.top_p is not None:
            xai_config["top_p"] = config.top_p
        
        if config.stop_sequences:
            xai_config["stop"] = config.stop_sequences
        
        if config.system_instruction:
            xai_config["messages"] = [{"role": "system", "content": config.system_instruction}]
        
        if config.response_format == ResponseFormat.JSON:
            xai_config["response_format"] = {"type": "json_object"}
        
        return xai_config

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
            xai_config = self._convert_config(config)
            
            messages = [{"role": "user", "content": prompt}]
            if config and config.system_instruction:
                messages.insert(0, {"role": "system", "content": config.system_instruction})
            
            response = self._client.chat.completions.create(
                model=model_id,
                messages=messages,
                **xai_config
            )
            
            latency = time.time() - start_time
            choice = response.choices[0]
            
            return LLMResponse(
                success=True,
                content=choice.message.content or "",
                model=model_id,
                provider="xai",
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
                provider="xai",
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
        xai_config = self._convert_config(config)
        
        messages = [{"role": "user", "content": prompt}]
        if config and config.system_instruction:
            messages.insert(0, {"role": "system", "content": config.system_instruction})
        
        response = self._client.chat.completions.create(
            model=model_id,
            messages=messages,
            stream=True,
            **xai_config
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
                id="grok-4-2025-11-04",
                name="Grok 4",
                provider="xai",
                context_window=32_768,
                max_output_tokens=32_768,
                supports_vision=True,
                description="Latest flagship with advanced reasoning"
            ),
            ModelInfo(
                id="grok-4-mini",
                name="Grok 4 Mini",
                provider="xai",
                context_window=32_768,
                max_output_tokens=32_768,
                supports_vision=True,
                description="Efficient version of Grok 4"
            ),
            ModelInfo(
                id="grok-3-mini",
                name="Grok 3 Mini",
                provider="xai",
                context_window=32_768,
                max_output_tokens=32_768,
                supports_vision=False,
                description="Fast, cost-effective reasoning"
            ),
            ModelInfo(
                id="grok-3",
                name="Grok 3",
                provider="xai",
                context_window=32_768,
                max_output_tokens=32_768,
                supports_vision=False,
                description="Previous generation flagship"
            ),
        ]

    def get_model_info(self, model_id: str) -> Optional[ModelInfo]:
        for model in self.list_models():
            if model.id == model_id:
                return model
        return None

    def count_tokens(self, text: str, model: Optional[str] = None) -> int:
        return len(text) // 4
