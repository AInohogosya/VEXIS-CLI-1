"""
Mistral AI LLM Client Adapter

Implements the BaseLLM interface for Mistral AI's API.

Installation:
    pip install mistralai

Environment Variables:
    MISTRAL_API_KEY: API key for Mistral AI
"""

import os
import time
from typing import Optional, Dict, Any, List, Iterator, AsyncIterator

from .base import (
    BaseLLM, ProviderType, GenerationConfig, LLMResponse, 
    ModelInfo, ResponseFormat, _estimate_cost
)

try:
    from mistralai import Mistral
    MISTRAL_AVAILABLE = True
except ImportError:
    MISTRAL_AVAILABLE = False


class MistralLLMClient(BaseLLM):
    """
    Mistral AI LLM client using the official Mistral SDK.
    
    Usage:
        client = MistralLLMClient(api_key="your-api-key")
        response = client.generate("Explain quantum computing")
        print(response.content)
    
    Latest Models (as of 2025):
        - mistral-large-3: Latest flagship with strong reasoning
        - mistral-small-3.1: Efficient model for simple tasks
        - codestral-2501: Code generation model
        - pixtral-large-2411: Vision model
    """

    DEFAULT_MODEL = "mistral-large-3"
    
    MODEL_CONTEXT_WINDOWS = {
        "mistral-large-3": 1_048_576,
        "mistral-small-3.1": 128_000,
        "mistral-small-3": 128_000,
        "codestral-2501": 256_000,
        "codestral-2405": 32_768,
        "pixtral-large-2411": 1_048_576,
        "pixtral-12b-2409": 1_048_576,
    }
    
    VISION_MODELS = {"pixtral-large-2411", "pixtral-12b-2409"}

    def __init__(self, api_key: Optional[str] = None, **kwargs):
        self._api_key = api_key or os.getenv("MISTRAL_API_KEY")
        self._config = kwargs
        self._client = None
        
        if not MISTRAL_AVAILABLE:
            raise ImportError(
                "mistralai package is required. "
                "Install with: pip install mistralai"
            )

    @property
    def provider_type(self) -> ProviderType:
        return ProviderType.MISTRAL
    
    @property
    def default_model(self) -> str:
        return self.DEFAULT_MODEL

    def _initialize_client(self) -> None:
        if not self._api_key:
            raise ValueError(
                "Mistral API key is required. Provide it as an argument or "
                "set MISTRAL_API_KEY environment variable."
            )
        self._client = Mistral(api_key=self._api_key)

    def _convert_config(self, config: Optional[GenerationConfig]) -> Dict[str, Any]:
        if config is None:
            config = GenerationConfig()
        
        mistral_config = {}
        
        if config.max_tokens is not None:
            mistral_config["max_tokens"] = config.max_tokens
        
        if config.temperature is not None:
            mistral_config["temperature"] = config.temperature
        
        if config.top_p is not None:
            mistral_config["top_p"] = config.top_p
        
        if config.stop_sequences:
            mistral_config["stop"] = config.stop_sequences
        
        if config.response_format == ResponseFormat.JSON:
            mistral_config["response_format"] = {"type": "json_object"}
        
        return mistral_config

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
            mistral_config = self._convert_config(config)
            
            messages = [{"role": "user", "content": prompt}]
            if config and config.system_instruction:
                messages.insert(0, {"role": "system", "content": config.system_instruction})
            
            response = self._client.chat.complete(
                model=model_id,
                messages=messages,
                **mistral_config
            )
            
            latency = time.time() - start_time
            
            return LLMResponse(
                success=True,
                content=response.choices[0].message.content or "",
                model=model_id,
                provider="mistral",
                tokens_used=response.usage.total_tokens,
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                latency=latency,
                finish_reason=response.choices[0].finish_reason,
                raw_response=response
            )
            
        except Exception as e:
            return LLMResponse(
                success=False,
                content="",
                model=model or self.default_model,
                provider="mistral",
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
        mistral_config = self._convert_config(config)
        
        messages = [{"role": "user", "content": prompt}]
        if config and config.system_instruction:
            messages.insert(0, {"role": "system", "content": config.system_instruction})
        
        response = self._client.chat.stream(
            model=model_id,
            messages=messages,
            **mistral_config
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
                id="mistral-large-3",
                name="Mistral Large 3",
                provider="mistral",
                context_window=1_048_576,
                max_output_tokens=32_768,
                supports_vision=False,
                description="Latest flagship with strong reasoning"
            ),
            ModelInfo(
                id="mistral-small-3.1",
                name="Mistral Small 3.1",
                provider="mistral",
                context_window=128_000,
                max_output_tokens=32_768,
                supports_vision=False,
                description="Efficient model for simple tasks"
            ),
            ModelInfo(
                id="codestral-2501",
                name="Codestral 25.01",
                provider="mistral",
                context_window=256_000,
                max_output_tokens=32_768,
                supports_vision=False,
                description="Code generation model"
            ),
            ModelInfo(
                id="pixtral-large-2411",
                name="Pixtral Large 24.11",
                provider="mistral",
                context_window=1_048_576,
                max_output_tokens=32_768,
                supports_vision=True,
                description="Vision model with 1M context"
            ),
        ]

    def get_model_info(self, model_id: str) -> Optional[ModelInfo]:
        for model in self.list_models():
            if model.id == model_id:
                return model
        return None

    def count_tokens(self, text: str, model: Optional[str] = None) -> int:
        return len(text) // 4
