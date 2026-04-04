"""
OpenRouter API Provider for VEXIS-CLI-2 AI Agent
Handles communication with OpenRouter API - provides access to 300+ AI models
"""

import base64
import io
import json
import time
import uuid
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

try:
    import requests
except ImportError:
    raise ImportError("requests is required for OpenRouter API provider")

try:
    from PIL import Image
except ImportError:
    raise ImportError("PIL (Pillow) is required for OpenRouter API provider")

from ai_agent.utils.exceptions import APIError, ValidationError
from ai_agent.utils.logger import get_logger


@dataclass
class OpenRouterResponse:
    """OpenRouter response structure"""
    success: bool
    content: str
    model: str
    provider: str
    tokens_used: Optional[int] = None
    cost: Optional[float] = None
    error: Optional[str] = None


class OpenRouterProvider:
    """OpenRouter API provider - access to 300+ AI models"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.api_key = config.get("openrouter_api_key")
        self.endpoint = "https://openrouter.ai/api/v1/chat/completions"
        self.timeout = config.get("timeout", 120)
        self.logger = get_logger("openrouter_provider")
        
        # Popular OpenRouter models (verified official names)
        self.popular_models = [
            # OpenAI Models
            "openai/gpt-4o",
            "openai/gpt-4o-mini", 
            "openai/gpt-4-turbo",
            "openai/gpt-3.5-turbo",
            
            # Anthropic Models
            "anthropic/claude-3.5-sonnet",
            "anthropic/claude-3.5-haiku",
            "anthropic/claude-3-opus",
            
            # Google Models
            "google/gemini-2.0-flash-exp",
            "google/gemini-1.5-pro",
            "google/gemini-1.5-flash",
            
            # Meta Models
            "meta-llama/llama-3.1-405b-instruct",
            "meta-llama/llama-3.1-70b-instruct",
            "meta-llama/llama-3.1-8b-instruct",
            
            # Mistral Models
            "mistralai/mistral-large",
            "mistralai/mixtral-8x7b",
            "mistralai/mistral-7b-instruct",
            
            # DeepSeek Models
            "deepseek/deepseek-r1",
            "deepseek/deepseek-chat",
            
            # Groq Models
            "groq/llama-3.1-405b-reasoning",
            "groq/llama-3.1-70b-versatile",
            
            # Other Popular Models
            "qwen/qwen-2.5-72b-instruct",
            "cohere/command-r-plus",
            "perplexity/llama-3.1-sonar-large-128k-online",
            "nvidia/nemotron-4-340b-instruct"
        ]
        
    @property
    def name(self) -> str:
        return "openrouter"
    
    @property
    def default_model(self) -> str:
        return "openai/gpt-4o-mini"
    
    def get_available_models(self) -> List[str]:
        """Get list of available models"""
        return self.popular_models + ["Other Models"]
    
    def chat(self, prompt: str, model: Optional[str] = None, temperature: float = 1.0, max_tokens: int = 5000, system_instructions: Optional[str] = None, image_data: Optional[bytes] = None, image_format: str = "PNG") -> OpenRouterResponse:
        """Send a chat request to OpenRouter"""
        if not self.api_key:
            return OpenRouterResponse(
                success=False,
                content="",
                model=self.default_model,
                provider=self.name,
                error="OpenRouter API key not configured"
            )
        
        # Handle "Other Models" option
        model_name = model or self.default_model
        if model_name == "Other Models" or model_name not in self.popular_models:
            # Use the exact model name provided by user
            if model_name == "Other Models":
                # This shouldn't happen in normal flow, but handle it gracefully
                return OpenRouterResponse(
                    success=False,
                    content="",
                    model="unknown",
                    provider=self.name,
                    error="Please specify a model name when using 'Other Models' option"
                )
        
        try:
            # Prepare the request payload for OpenRouter API
            # Add unique identifier at the beginning to bypass caching
            unique_id = str(uuid.uuid4())
            unique_prompt = f"Unique Request ID: {unique_id}\n\n{prompt}"
            
            # Build messages array
            messages = []
            if system_instructions:
                messages.append({"role": "system", "content": system_instructions})
            messages.append({"role": "user", "content": unique_prompt})
            
            payload = {
                "model": model_name,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
            
            # Add image if provided
            if image_data:
                # Convert image to base64
                image_base64 = base64.b64encode(image_data).decode('utf-8')
                # Determine MIME type
                image_format_lower = image_format.lower()
                mime_type = f"image/{image_format_lower}" if image_format_lower in ["jpeg", "jpg", "png", "webp", "gif"] else "image/png"
                
                # Add image to the last message
                messages[-1]["content"] = [
                    {"type": "text", "text": unique_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{image_base64}"}}
                ]
                payload["messages"] = messages
            
            # Make API call
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
                "HTTP-Referer": "https://github.com/AInohogosya/VEXIS-CLI-2",
                "X-Title": "VEXIS-CLI-2 AI Agent"
            }
            
            response = requests.post(
                self.endpoint,
                headers=headers,
                json=payload,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                # Extract content from OpenRouter response format
                content = ""
                if "choices" in result and len(result["choices"]) > 0:
                    choice = result["choices"][0]
                    if "message" in choice and "content" in choice["message"]:
                        content = choice["message"]["content"]
                
                # Extract token usage if available
                tokens_used = None
                if "usage" in result:
                    tokens_used = result["usage"].get("total_tokens")
                
                return OpenRouterResponse(
                    success=True,
                    content=content,
                    model=model_name,
                    provider=self.name,
                    cost=self._calculate_cost(model_name, tokens_used),
                    tokens_used=tokens_used,
                )
            else:
                error_msg = f"HTTP {response.status_code}: {response.text}"
                return OpenRouterResponse(
                    success=False,
                    content="",
                    model=model_name,
                    provider=self.name,
                    error=error_msg,
                )
            
        except Exception as e:
            return OpenRouterResponse(
                success=False,
                content="",
                model=model_name,
                provider=self.name,
                error=str(e),
            )
    
    def analyze_image(self, request):  # Compatibility method
        """Analyze image using OpenRouter API - compatibility wrapper"""
        return self.chat(
            prompt=request.prompt,
            model=request.model,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            system_instructions=getattr(request, 'system_instruction', None),
            image_data=getattr(request, 'image_data', None),
            image_format=getattr(request, 'image_format', 'PNG')
        )
    
    def _calculate_cost(self, model: str, tokens: Optional[int] = None) -> Optional[float]:
        """Calculate cost for OpenRouter API"""
        # OpenRouter pricing varies by model
        # These are approximate costs and may change
        if tokens and model:
            # Rough cost estimates for popular models (per 1M tokens)
            pricing = {
                # OpenAI models
                "openai/gpt-4o": 5.0,      # $5 per 1M tokens (average)
                "openai/gpt-4o-mini": 0.15, # $0.15 per 1M tokens
                "openai/gpt-4-turbo": 10.0, # $10 per 1M tokens
                "openai/gpt-3.5-turbo": 0.5, # $0.5 per 1M tokens
                
                # Anthropic models
                "anthropic/claude-3.5-sonnet": 3.0,
                "anthropic/claude-3.5-haiku": 0.25,
                "anthropic/claude-3-opus": 15.0,
                
                # Google models
                "google/gemini-2.0-flash-exp": 0.5,
                "google/gemini-1.5-pro": 3.5,
                "google/gemini-1.5-flash": 0.2,
                
                # Meta models
                "meta-llama/llama-3.1-405b-instruct": 1.0,
                "meta-llama/llama-3.1-70b-instruct": 0.8,
                "meta-llama/llama-3.1-8b-instruct": 0.2,
                
                # Mistral models
                "mistralai/mistral-large": 4.0,
                "mistralai/mixtral-8x7b": 1.0,
                "mistralai/mistral-7b-instruct": 0.3,
                
                # DeepSeek models
                "deepseek/deepseek-r1": 0.5,
                "deepseek/deepseek-chat": 0.3,
            }
            
            # Find matching price (use closest match if exact not found)
            cost_per_million = None
            if model in pricing:
                cost_per_million = pricing[model]
            else:
                # Try to find a price based on model family
                for model_pattern, price in pricing.items():
                    if model_pattern.split('/')[0] in model:
                        cost_per_million = price
                        break
            
            if cost_per_million:
                estimated_cost = (tokens * cost_per_million) / 1000000
                return round(estimated_cost, 6)
        
        return None
    
    def validate_model(self, model: str) -> bool:
        """Validate if model name is supported"""
        # Allow any model name (including custom ones)
        return True
    
    def get_model_info(self, model: str) -> Dict[str, Any]:
        """Get information about a specific model"""
        return {
            "name": model,
            "provider": self.name,
            "supports_vision": True,  # Most OpenRouter models support vision
            "supports_tools": True,  # Most support tool calling
            "context_window": self._estimate_context_window(model)
        }
    
    def _estimate_context_window(self, model: str) -> int:
        """Estimate context window for model"""
        # Rough estimates based on model families
        if "gpt-4" in model:
            return 128000
        elif "claude-3" in model:
            return 200000
        elif "gemini" in model:
            return 1000000
        elif "llama-3.1" in model:
            return 128000
        elif "mistral" in model:
            return 32000
        else:
            return 8000  # Conservative default
