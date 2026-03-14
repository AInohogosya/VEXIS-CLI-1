"""
Cohere AI LLM Client Adapter

Implements the BaseLLM interface for Cohere's API.

Installation:
    pip install cohere

Environment Variables:
    COHERE_API_KEY: API key for Cohere
"""

import os
import time
from typing import Optional, Dict, Any, List, Iterator, AsyncIterator

from .base import (
    BaseLLM, ProviderType, GenerationConfig, LLMResponse, 
    ModelInfo, ResponseFormat, _estimate_cost
)

try:
    import cohere
    COHERE_AVAILABLE = True
except ImportError:
    COHERE_AVAILABLE = False


class CohereLLMClient(BaseLLM):
    """
    Cohere AI LLM client using the official Cohere SDK.
    
    Usage:
        client = CohereLLMClient(api_key="your-api-key")
        response = client.generate("Explain quantum computing")
        print(response.content)
    
    Latest Models (as of 2025):
        - command-a-03-2025: Command A - Most performant enterprise model
        - command-r-08-2024: Command R - Strong reasoning model
        - command-r-plus-08-2024: Command R+ - Enhanced reasoning
        - command-7b-2024-08: Command - Fast, lightweight
    """

    DEFAULT_MODEL = "command-a-03-2025"
    
    MODEL_CONTEXT_WINDOWS = {
        "command-a-03-2025": 128_000,
        "command-r-08-2024": 128_000,
        "command-r-plus-08-2024": 128_000,
        "command-7b-2024-08": 4_096,
    }
    
    VISION_MODELS = set()

    def __init__(self, api_key: Optional[str] = None, **kwargs):
        self._api_key = api_key or os.getenv("COHERE_API_KEY")
        self._config = kwargs
        self._client = None
        
        if not COHERE_AVAILABLE:
            raise ImportError(
                "cohere package is required. "
                "Install with: pip install cohere"
            )

    @property
    def provider_type(self) -> ProviderType:
        return ProviderType.COHERE
    
    @property
    def default_model(self) -> str:
        return self.DEFAULT_MODEL

    def _initialize_client(self) -> None:
        if not self._api_key:
            raise ValueError(
                "Cohere API key is required. Provide it as an argument or "
                "set COHERE_API_KEY environment variable."
            )
        self._client = cohere.Client(self._api_key)

    def _convert_config(self, config: Optional[GenerationConfig]) -> Dict[str, Any]:
        if config is None:
            config = GenerationConfig()
        
        cohere_config = {}
        
        if config.max_tokens is not None:
            cohere_config["max_tokens"] = config.max_tokens
        
        if config.temperature is not None:
            cohere_config["temperature"] = config.temperature
        
        if config.top_p is not None:
            cohere_config["p"] = config.top_p
        
        if config.stop_sequences:
            cohere_config["stop_sequences"] = config.stop_sequences
        
        return cohere_config

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
            cohere_config = self._convert_config(config)
            
            response = self._client.generate(
                model=model_id,
                prompt=prompt,
                **cohere_config
            )
            
            latency = time.time() - start_time
            
            return LLMResponse(
                success=True,
                content=response.text,
                model=model_id,
                provider="cohere",
                tokens_used=response.token_count if hasattr(response, 'token_count') else None,
                latency=latency,
                finish_reason=response.finish_reason if hasattr(response, 'finish_reason') else None,
                raw_response=response
            )
            
        except Exception as e:
            return LLMResponse(
                success=False,
                content="",
                model=model or self.default_model,
                provider="cohere",
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
        cohere_config = self._convert_config(config)
        
        response = self._client.generate(
            model=model_id,
            prompt=prompt,
            stream=True,
            **cohere_config
        )
        
        for event in response:
            if event.text:
                yield event.text

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
                id="command-a-03-2025",
                name="Command A",
                provider="cohere",
                context_window=128_000,
                max_output_tokens=4_096,
                supports_vision=False,
                description="Most performant enterprise model"
            ),
            ModelInfo(
                id="command-r-08-2024",
                name="Command R",
                provider="cohere",
                context_window=128_000,
                max_output_tokens=4_096,
                supports_vision=False,
                description="Strong reasoning model"
            ),
            ModelInfo(
                id="command-r-plus-08-2024",
                name="Command R+",
                provider="cohere",
                context_window=128_000,
                max_output_tokens=4_096,
                supports_vision=False,
                description="Enhanced reasoning model"
            ),
            ModelInfo(
                id="command-7b-2024-08",
                name="Command",
                provider="cohere",
                context_window=4_096,
                max_output_tokens=4_096,
                supports_vision=False,
                description="Fast, lightweight model"
            ),
        ]

    def get_model_info(self, model_id: str) -> Optional[ModelInfo]:
        for model in self.list_models():
            if model.id == model_id:
                return model
        return None

    def count_tokens(self, text: str, model: Optional[str] = None) -> int:
        return len(text) // 4
