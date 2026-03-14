#!/usr/bin/env python3
"""
Test script for cloud model error handling
"""

import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from ai_agent.external_integration.ollama_provider import SimpleOllamaProvider

def test_cloud_model_errors():
    """Test cloud model error handling"""
    
    print("🧪 Testing Cloud Model Error Handling")
    print("=" * 50)
    
    provider = SimpleOllamaProvider()
    
    # Test cases
    test_cases = [
        ("qwen3.5:397b-cloud", "Qwen 3.5 397B Cloud"),
        ("mistral-large-3:675b-cloud", "Mistral Large 3 Cloud"),
        ("glm-5:cloud", "GLM-5 Cloud"),
        ("deepseek-r1:671b-cloud", "DeepSeek R1 Cloud (non-existent)"),
    ]
    
    for model, description in test_cases:
        print(f"\n🔍 Testing: {description}")
        print(f"   Model: {model}")
        print("-" * 40)
        
        # This should trigger the cloud model validation
        response = provider.chat("Hello, test message", model=model)
        
        if not response.success:
            print(f"✅ Error handled correctly:")
            print(f"   {response.error}")
        else:
            print(f"❌ Unexpected success: {response.content[:100]}...")
    
    print("\n" + "=" * 50)
    print("🎯 Test completed!")
    print("\n💡 If you see authentication errors, run:")
    print("   ollama signin")
    print("\n📋 Then verify with:")
    print("   ollama whoami")

if __name__ == "__main__":
    test_cloud_model_errors()
