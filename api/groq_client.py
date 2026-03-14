"""
Groq LLM Client Adapter

Implements the BaseLLM interface for Groq's API.

Installation:
    pip install openai

Environment Variables:
    GROQ_API_KEY: API key for Groq
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
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False


class GroqLLMClient(BaseLLM):
    """
    Groq LLM client using the OpenAI-compatible API.
    
    Usage:
        client = GroqLLMClient(api_key="your-api-key")
        response = client.generate("Explain quantum computing")
        print(response.content)
    
    Latest Models (as of 2026):
        - llama-3.3-70b-versatile: Llama 3.3 70B - High performance
        - openai/gpt-oss-120b: OpenAI GPT-OSS 120B - Open source powerhouse
        - llama-3.1-70b-instruct: Llama 3.1 70B - Strong reasoning
        - llama-3.1-8b-instant: Llama 3.1 8B - Fast, efficient
    """

    DEFAULT_MODEL = "llama-3.3-70b-versatile"
    
    MODEL_CONTEXT_WINDOWS = {
        "llama-3.3-70b-versatile": 128_000,
        "openai/gpt-oss-120b": 128_000,
        "llama-3.1-70b-instruct": 128_000,
        "llama-3.1-8b-instruct": 128_000,
        "llama-3.1-8b-instant": 128_000,
        "gemma2-9b-it": 8_192,
    }
    
    VISION_MODELS = set()

    def __init__(self, api_key: Optional[str] = None, **kwargs):
        self._api_key = api_key or os.getenv("GROQ_API_KEY")
        self._config = kwargs
        self._client = None
        
        if not GROQ_AVAILABLE:
            raise ImportError(
                "openai package is required. "
                "Install with: pip install openai"
            )

    @property
    def provider_type(self) -> ProviderType:
        return ProviderType.GROQ
    
    @property
    def default_model(self) -> str:
        return self.DEFAULT_MODEL

    def _initialize_client(self) -> None:
        if not self._api_key:
            raise ValueError(
                "Groq API key is required. Provide it as an argument or "
                "set GROQ_API_KEY environment variable."
            )
        self._client = OpenAI(
            api_key=self._api_key,
            base_url="https://api.groq.com/openai/v1"
        )

    def _convert_config(self, config: Optional[GenerationConfig]) -> Dict[str, Any]:
        if config is None:
            config = GenerationConfig()
        
        groq_config = {}
        
        if config.max_tokens is not None:
            groq_config["max_tokens"] = config.max_tokens
        
        if config.temperature is not None:
            groq_config["temperature"] = config.temperature
        
        if config.top_p is not None:
            groq_config["top_p"] = config.top_p
        
        if config.stop_sequences:
            groq_config["stop"] = config.stop_sequences
        
        if config.response_format == ResponseFormat.JSON:
            groq_config["response_format"] = {"type": "json_object"}
        
        return groq_config

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
            groq_config = self._convert_config(config)
            
            messages = [{"role": "user", "content": prompt}]
            if config and config.system_instruction:
                messages.insert(0, {"role": "system", "content": config.system_instruction})
            
            response = self._client.chat.completions.create(
                model=model_id,
                messages=messages,
                **groq_config
            )
            
            latency = time.time() - start_time
            choice = response.choices[0]
            
            return LLMResponse(
                success=True,
                content=choice.message.content or "",
                model=model_id,
                provider="groq",
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
                provider="groq",
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
        groq_config = self._convert_config(config)
        
        messages = [{"role": "user", "content": prompt}]
        if config and config.system_instruction:
            messages.insert(0, {"role": "system", "content": config.system_instruction})
        
        response = self._client.chat.completions.create(
            model=model_id,
            messages=messages,
            stream=True,
            **groq_config
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
                id="llama-3.3-70b-versatile",
                name="Llama 3.3 70B Versatile",
                provider="groq",
                context_window=128_000,
                max_output_tokens=32_768,
                supports_vision=False,
                description="High performance with 128K context"
            ),
            ModelInfo(
                id="llama-3.1-70b-instruct",
                name="Llama 3.1 70B Instruct",
                provider="groq",
                context_window=128_000,
                max_output_tokens=32_768,
                supports_vision=False,
                description="Strong reasoning capabilities"
            ),
            ModelInfo(
                id="llama-3.1-8b-instant",
                name="Llama 3.1 8B Instant",
                provider="groq",
                context_window=128_000,
                max_output_tokens=32_768,
                supports_vision=False,
                description="Fast, efficient for simple tasks"
            ),
            ModelInfo(
                id="openai/gpt-oss-120b",
                name="GPT-OSS 120B",
                provider="groq",
                context_window=128_000,
                max_output_tokens=32_768,
                supports_vision=False,
                supports_streaming=True,
                description="OpenAI's open source 120B parameter model",
                capabilities=["streaming", "function_calling"]
            ),
        ]

    def get_model_info(self, model_id: str) -> Optional[ModelInfo]:
        for model in self.list_models():
            if model.id == model_id:
                return model
        return None

    def count_tokens(self, text: str, model: Optional[str] = None) -> int:
        return len(text) // 4
