"""
Microsoft Azure OpenAI LLM Client Adapter

Implements the BaseLLM interface for Microsoft Azure OpenAI.

Installation:
    pip install openai

Environment Variables:
    AZURE_OPENAI_API_KEY: API key for Azure OpenAI
    AZURE_OPENAI_ENDPOINT: Azure endpoint URL
    AZURE_OPENAI_DEPLOYMENT: Deployment name
"""

import os
import time
from typing import Optional, Dict, Any, List, Iterator, AsyncIterator

from .base import (
    BaseLLM, ProviderType, GenerationConfig, LLMResponse, 
    ModelInfo, ResponseFormat, _estimate_cost
)

try:
    from openai import AzureOpenAI
    AZURE_AVAILABLE = True
except ImportError:
    AZURE_AVAILABLE = False


class MicrosoftLLMClient(BaseLLM):
    """
    Microsoft Azure OpenAI client using the OpenAI SDK.
    
    Usage:
        client = MicrosoftLLMClient(
            api_key="your-api-key",
            endpoint="https://your-resource.openai.azure.com/",
            deployment="your-deployment"
        )
        response = client.generate("Explain quantum computing")
        print(response.content)
    
    Latest Models (as of 2025):
        - gpt-5.4: Latest flagship model
        - gpt-4o: Powerful general purpose
        - gpt-4o-mini: Efficient model
    """

    DEFAULT_MODEL = "gpt-5.4"
    
    MODEL_CONTEXT_WINDOWS = {
        "gpt-5.4": 1_048_576,
        "gpt-5.4-pro": 1_048_576,
        "gpt-5.4-mini": 400_000,
        "gpt-4o": 128_000,
        "gpt-4o-mini": 128_000,
    }
    
    VISION_MODELS = {"gpt-5.4", "gpt-5.4-pro", "gpt-4o", "gpt-4o-mini"}

    def __init__(self, api_key: Optional[str] = None, **kwargs):
        self._api_key = api_key or os.getenv("AZURE_OPENAI_API_KEY")
        self._endpoint = kwargs.get("endpoint") or os.getenv("AZURE_OPENAI_ENDPOINT")
        self._deployment = kwargs.get("deployment") or os.getenv("AZURE_OPENAI_DEPLOYMENT")
        self._api_version = kwargs.get("api_version", "2024-12-01-preview")
        self._config = kwargs
        self._client = None
        
        if not AZURE_AVAILABLE:
            raise ImportError(
                "openai package is required. "
                "Install with: pip install openai"
            )

    @property
    def provider_type(self) -> ProviderType:
        return ProviderType.MICROSOFT
    
    @property
    def default_model(self) -> str:
        return self.DEFAULT_MODEL

    def _initialize_client(self) -> None:
        if not self._api_key:
            raise ValueError(
                "Azure API key is required. Provide it as an argument or "
                "set AZURE_OPENAI_API_KEY environment variable."
            )
        if not self._endpoint:
            raise ValueError(
                "Azure endpoint is required. Provide it as an argument or "
                "set AZURE_OPENAI_ENDPOINT environment variable."
            )
        
        self._client = AzureOpenAI(
            api_key=self._api_key,
            api_version=self._api_version,
            azure_endpoint=self._endpoint
        )

    def _convert_config(self, config: Optional[GenerationConfig]) -> Dict[str, Any]:
        if config is None:
            config = GenerationConfig()
        
        azure_config = {}
        
        if config.max_tokens is not None:
            azure_config["max_tokens"] = config.max_tokens
        
        if config.temperature is not None:
            azure_config["temperature"] = config.temperature
        
        if config.top_p is not None:
            azure_config["top_p"] = config.top_p
        
        if config.stop_sequences:
            azure_config["stop"] = config.stop_sequences
        
        if config.response_format == ResponseFormat.JSON:
            azure_config["response_format"] = {"type": "json_object"}
        
        return azure_config

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
            
            model_id = model or self._deployment or self.default_model
            azure_config = self._convert_config(config)
            
            messages = [{"role": "user", "content": prompt}]
            if config and config.system_instruction:
                messages.insert(0, {"role": "system", "content": config.system_instruction})
            
            response = self._client.chat.completions.create(
                model=model_id,
                messages=messages,
                **azure_config
            )
            
            latency = time.time() - start_time
            choice = response.choices[0]
            
            return LLMResponse(
                success=True,
                content=choice.message.content or "",
                model=model_id,
                provider="microsoft",
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
                provider="microsoft",
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
        
        model_id = model or self._deployment or self.default_model
        azure_config = self._convert_config(config)
        
        messages = [{"role": "user", "content": prompt}]
        if config and config.system_instruction:
            messages.insert(0, {"role": "system", "content": config.system_instruction})
        
        response = self._client.chat.completions.create(
            model=model_id,
            messages=messages,
            stream=True,
            **azure_config
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
                id="gpt-5.4",
                name="GPT-5.4",
                provider="microsoft",
                context_window=1_048_576,
                max_output_tokens=32_768,
                supports_vision=True,
                description="Latest flagship model"
            ),
            ModelInfo(
                id="gpt-5.4-pro",
                name="GPT-5.4 Pro",
                provider="microsoft",
                context_window=1_048_576,
                max_output_tokens=32_768,
                supports_vision=True,
                description="Professional GPT-5.4 with enhanced capabilities"
            ),
            ModelInfo(
                id="gpt-5.4-mini",
                name="GPT-5.4 Mini",
                provider="microsoft",
                context_window=400_000,
                max_output_tokens=32_768,
                supports_vision=True,
                description="Efficient reasoning model"
            ),
            ModelInfo(
                id="gpt-4o",
                name="GPT-4o",
                provider="microsoft",
                context_window=128_000,
                max_output_tokens=16_384,
                supports_vision=True,
                description="Powerful general purpose"
            ),
            ModelInfo(
                id="gpt-4o-mini",
                name="GPT-4o Mini",
                provider="microsoft",
                context_window=128_000,
                max_output_tokens=16_384,
                supports_vision=True,
                description="Efficient model"
            ),
        ]

    def get_model_info(self, model_id: str) -> Optional[ModelInfo]:
        for model in self.list_models():
            if model.id == model_id:
                return model
        return None

    def count_tokens(self, text: str, model: Optional[str] = None) -> int:
        return len(text) // 4
