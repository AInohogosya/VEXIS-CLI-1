"""
AI Agent System - CLI-based automation
Production-ready implementation with zero-defect policy

Version Note: VEXIS-CLI uses semantic versioning (2.1.0)
with 5-phase pipeline architecture and 16+ AI provider support.
"""

__version__ = "2.1.0"
__author__ = "AInohogosya"
__email__ = "AInohogosya@proton.me"
__description__ = "Advanced 5-phase AI-powered terminal automation system"

# Import core components for easy access
from .core_processing.five_phase_engine import FivePhaseEngine
from .platform_abstraction.platform_detector import PlatformDetector
from .external_integration.vision_api_client import VisionAPIClient
from .external_integration.model_runner import ModelRunner

__all__ = [
		"FivePhaseEngine",
		"PlatformDetector",
		"VisionAPIClient",
		"ModelRunner",
]
