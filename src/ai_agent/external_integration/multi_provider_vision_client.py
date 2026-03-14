"""
Multi-Provider Vision API Client for AI Agent System
Supports 13+ AI providers while maintaining current architecture
"""

import io
import time
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum

try:
    from PIL import Image
except ImportError:
    raise ImportError("PIL (Pillow) is required for Vision API client")

from ..utils.exceptions import ValidationError
from ..utils.logger import get_logger
from ..utils.config import load_config
from .ollama_provider import SimpleOllamaProvider

# Import API clients with error handling
try:
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
    from api import LLMFactory, ProviderType, GenerationConfig, LLMResponse
    API_AVAILABLE = True
except ImportError:
    API_AVAILABLE = False

# Import SDK installer
try:
    from ..utils.sdk_installer import create_installer
    SDK_INSTALLER_AVAILABLE = True
except ImportError:
    SDK_INSTALLER_AVAILABLE = False
    print("Warning: Multi-provider API not available, falling back to Ollama + Google only")


class APIProvider(Enum):
    """Supported API providers - Extended to 13+ providers"""
    OLLAMA = "ollama"
    GOOGLE = "google"
    
    # Additional providers (when API package is available)
    if API_AVAILABLE:
        OPENAI = "openai"
        ANTHROPIC = "anthropic"
        XAI = "xai"
        META = "meta"
        MISTRAL = "mistral"
        AZURE = "azure"
        AMAZON = "amazon"
        COHERE = "cohere"
        DEEPSEEK = "deepseek"
        GROQ = "groq"
        TOGETHER = "together"


@dataclass
class APIResponse:
    """API response structure"""
    success: bool
    content: str
    model: str
    provider: str
    tokens_used: Optional[int] = None
    cost: Optional[float] = None
    latency: Optional[float] = None
    error: Optional[str] = None


@dataclass
class APIRequest:
    """API request structure"""
    prompt: str
    image_data: Optional[bytes] = None
    image_format: str = "PNG"
    max_tokens: int = 5000
    temperature: float = 1.0
    model: Optional[str] = None
    provider: Optional[str] = None


class MultiProviderVisionAPIClient:
    """Multi-provider Vision API Client"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, auto_install_sdks: bool = False):
        # Handle config properly
        if config is None:
            config = load_config()
        elif hasattr(config, 'api'):
            # It's a Config object, get the api dict
            config = config.api.__dict__
        
        self.config = config or load_config().api.__dict__
        self.logger = get_logger(__name__)
        
        # Initialize Ollama provider (always available)
        self.ollama_provider = SimpleOllamaProvider()
        
        # Initialize SDK installer if available
        self.sdk_installer = None
        if SDK_INSTALLER_AVAILABLE:
            self.sdk_installer = create_installer(auto_install=auto_install_sdks)
        
        # Initialize API clients for other providers
        self.api_clients = {}
        if API_AVAILABLE:
            self._initialize_api_clients()
        
        self.logger.info(f"Multi-provider Vision API client initialized with {len(self.api_clients)} providers")
    
    def _initialize_api_clients(self):
        """Initialize all available API clients"""
        provider_mappings = {
            'google': ProviderType.GOOGLE,
            'openai': ProviderType.OPENAI,
            'anthropic': ProviderType.ANTHROPIC,
            'xai': ProviderType.XAI,
            'meta': ProviderType.META,
            'mistral': ProviderType.MISTRAL,
            'microsoft': ProviderType.MICROSOFT,
            'amazon': ProviderType.AMAZON,
            'cohere': ProviderType.COHERE,
            'deepseek': ProviderType.DEEPSEEK,
            'groq': ProviderType.GROQ,
            'together': ProviderType.TOGETHER,
        }
        
        for provider_name, provider_type in provider_mappings.items():
            try:
                client = LLMFactory.create(provider_type)
                self.api_clients[provider_name] = client
                self.logger.info(f"Initialized {provider_name} client")
            except ValueError as e:
                # Provider not registered (missing SDK dependencies)
                self.logger.warning(f"Provider {provider_name} not available: {e}")
                
                # Offer to install SDK if installer is available
                if self.sdk_installer and self.sdk_installer.check_sdk_availability(provider_name) == False:
                    self._offer_sdk_installation(provider_name)
                
                # Don't add to api_clients dict - this provider is unavailable
            except Exception as e:
                self.logger.warning(f"Failed to initialize {provider_name} client: {e}")
                # Don't add to api_clients dict - this provider failed to initialize
    
    def _offer_sdk_installation(self, provider: str):
        """Offer to install SDK for a missing provider"""
        if not self.sdk_installer:
            return
        
        try:
            # Check if we're in an interactive terminal
            if not sys.stdin.isatty():
                return  # Not interactive, don't offer installation
            
            print(f"\n🔍 Missing SDK detected for {provider}")
            print(f"   This provider requires additional installation to work.")
            
            # Show what will be installed
            missing_sdks = self.sdk_installer.get_missing_sdks([provider])
            if provider in missing_sdks:
                sdk_info = missing_sdks[provider]
                print(f"   Package: {sdk_info['package']}")
                print(f"   Description: {sdk_info['description']}")
                print(f"   Command: {sdk_info['install_command']}")
                
                try:
                    choice = input(f"\nWould you like to install {sdk_info['package']} now? (y/n): ").lower().strip()
                    if choice in ['y', 'yes']:
                        success = self.sdk_installer.install_sdk(provider, interactive=False)
                        
                        if success:
                            # Try to initialize the provider again
                            try:
                                from api import LLMFactory, ProviderType
                                provider_mappings = {
                                    'google': ProviderType.GOOGLE,
                                    'openai': ProviderType.OPENAI,
                                    'anthropic': ProviderType.ANTHROPIC,
                                    'xai': ProviderType.XAI,
                                    'meta': ProviderType.META,
                                    'mistral': ProviderType.MISTRAL,
                                    'microsoft': ProviderType.MICROSOFT,
                                    'amazon': ProviderType.AMAZON,
                                    'cohere': ProviderType.COHERE,
                                    'deepseek': ProviderType.DEEPSEEK,
                                    'groq': ProviderType.GROQ,
                                    'together': ProviderType.TOGETHER,
                                }
                                
                                if provider in provider_mappings:
                                    client = LLMFactory.create(provider_mappings[provider])
                                    self.api_clients[provider] = client
                                    self.logger.info(f"Successfully initialized {provider} client after SDK installation")
                                    print(f"✅ {provider} is now available!")
                                    
                            except Exception as e:
                                self.logger.warning(f"Still failed to initialize {provider} after SDK installation: {e}")
                                print(f"⚠️ SDK installed but {provider} still not available: {e}")
                        else:
                            print(f"❌ Failed to install SDK for {provider}")
                except (KeyboardInterrupt, EOFError):
                    print("\nInstallation cancelled.")
                    
        except Exception as e:
            self.logger.error(f"Error offering SDK installation for {provider}: {e}")
    
    def generate_response(self, request: APIRequest) -> APIResponse:
        """Generate response using specified or preferred provider"""
        start_time = time.time()
        
        try:
            # Use provider from request - no fallbacks
            provider = request.provider
            if not provider:
                raise ValidationError("No provider specified in request")
            
            # Handle Ollama provider
            if provider == 'ollama':
                return self._handle_ollama_request(request, start_time)
            
            # Handle multi-provider API
            if API_AVAILABLE and provider in self.api_clients:
                return self._handle_api_request(request, provider, start_time)
            
            # No fallback - raise error if provider not handled
            raise ValidationError(f"Provider '{provider}' is not configured or available")
            
        except Exception as e:
            self.logger.error(f"Error generating response: {e}")
            return APIResponse(
                success=False,
                content="",
                model=request.model or "unknown",
                provider=provider,
                error=str(e),
                latency=time.time() - start_time
            )
    
    def _handle_ollama_request(self, request: APIRequest, start_time: float) -> APIResponse:
        """Handle Ollama requests"""
        try:
            # Use existing Ollama provider logic
            ollama_response = self.ollama_provider.chat(
                prompt=request.prompt,
                model=request.model,
                temperature=request.temperature,
                max_tokens=request.max_tokens
            )
            
            return APIResponse(
                success=ollama_response.success,
                content=ollama_response.content,
                model=ollama_response.model,
                provider='ollama',
                error=ollama_response.error,
                latency=time.time() - start_time
            )
        except Exception as e:
            return APIResponse(
                success=False,
                content="",
                model=request.model or 'unknown',
                provider='ollama',
                error=str(e),
                latency=time.time() - start_time
            )
    
    def _handle_api_request(self, request: APIRequest, provider: str, start_time: float) -> APIResponse:
        """Handle multi-provider API requests"""
        try:
            client = self.api_clients[provider]
            
            # Prepare generation config
            config = GenerationConfig(
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                model=request.model
            )
            
            # Handle image data if present
            if request.image_data:
                # For vision models, we'd need to handle image encoding
                # This is a simplified version - full implementation would vary by provider
                prompt_with_image = self._prepare_vision_prompt(request)
            else:
                prompt_with_image = request.prompt
            
            # Generate response
            response = client.generate(prompt_with_image, config)
            
            return APIResponse(
                success=True,
                content=response.content,
                model=response.model or request.model or 'unknown',
                provider=provider,
                tokens_used=response.tokens_used,
                cost=response.cost,
                latency=time.time() - start_time
            )
        except Exception as e:
            return APIResponse(
                success=False,
                content="",
                model=request.model or 'unknown',
                provider=provider,
                error=str(e),
                latency=time.time() - start_time
            )
    
    def _prepare_vision_prompt(self, request: APIRequest) -> str:
        """Prepare prompt for vision models"""
        # This is a simplified implementation
        # Full implementation would encode images properly for each provider
        if request.image_data:
            return f"[IMAGE: {request.image_format} data] {request.prompt}"
        return request.prompt
    
    def get_available_providers(self) -> List[str]:
        """Get list of actually available providers (with SDK dependencies installed)"""
        available = []
        if API_AVAILABLE:
            available.extend(list(self.api_clients.keys()))
        # Only include ollama if explicitly configured or available
        if self.ollama_provider.is_available():
            available.append('ollama')
        return available
    
    def install_missing_sdks(self, providers: Optional[List[str]] = None, interactive: bool = True) -> Dict[str, bool]:
        """Install missing SDKs for specified providers or all providers"""
        if not self.sdk_installer:
            print("❌ SDK installer not available")
            return {}
        
        if providers is None:
            # Install for all known providers
            from ..utils.sdk_installer import PROVIDER_SDKS
            providers = list(PROVIDER_SDKS.keys())
        
        return self.sdk_installer.install_missing_sdks(providers, interactive)
    
    def show_sdk_status(self, providers: Optional[List[str]] = None):
        """Show SDK installation status"""
        if not self.sdk_installer:
            print("❌ SDK installer not available")
            return
        
        if providers is None:
            from ..utils.sdk_installer import PROVIDER_SDKS
            providers = list(PROVIDER_SDKS.keys())
        
        self.sdk_installer.show_provider_status(providers)
    
    def get_provider_models(self, provider: str) -> List[str]:
        """Get available models for a provider"""
        if provider == 'ollama':
            return self.ollama_provider.get_available_models()
        
        if API_AVAILABLE and provider in self.api_clients:
            try:
                client = self.api_clients[provider]
                return [model.name for model in client.list_models()]
            except Exception as e:
                self.logger.warning(f"Failed to get models for {provider}: {e}")
                return []
        
        return []


# Factory function for backward compatibility
def create_vision_api_client(config: Optional[Dict[str, Any]] = None, auto_install_sdks: bool = False) -> MultiProviderVisionAPIClient:
    """Create a vision API client instance"""
    return MultiProviderVisionAPIClient(config, auto_install_sdks=auto_install_sdks)


# Legacy compatibility
VisionAPIClient = MultiProviderVisionAPIClient
