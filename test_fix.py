#!/usr/bin/env python3
"""Test script to verify the fix"""

import sys
from pathlib import Path

# Add src to Python path
current_dir = Path(__file__).parent
src_dir = current_dir / "src"
sys.path.insert(0, str(src_dir))

try:
    from ai_agent.external_integration.multi_provider_vision_client import MultiProviderVisionAPIClient, APIRequest
    
    # Create a test request
    config = {
        'preferred_provider': 'ollama',
        'local_model': 'llama3.2:latest'
    }
    
    client = MultiProviderVisionAPIClient(config)
    
    request = APIRequest(
        prompt="Test prompt",
        model="llama3.2:latest",
        provider="ollama",
        max_tokens=100,
        temperature=1.0
    )
    
    print("Testing MultiProviderVisionAPIClient...")
    response = client.generate_response(request)
    
    print(f"Success: {response.success}")
    print(f"Content: {response.content}")
    print(f"Error: {response.error}")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
