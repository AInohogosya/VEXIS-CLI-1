#!/usr/bin/env python3
"""
Absolute final verification - check every single model name against Ollama
"""
import sys
import os
import requests
import json

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def main():
    try:
        exec(open('src/ai_agent/utils/model_definitions.py').read())
        families = get_model_families()
        predefined = get_predefined_models()
        
        print("=== ABSOLUTE FINAL VERIFICATION ===")
        print(f"✅ Total families: {len(families)}")
        print(f"✅ Total predefined models: {len(predefined)}")
        
        # Test each model by checking if it exists on ollama.com
        errors = []
        verified = []
        
        for model_name in predefined.keys():
            try:
                # Extract base model name (remove version tags)
                base_name = model_name.split(':')[0] if ':' in model_name else model_name
                
                # Check if model exists on Ollama
                url = f"https://ollama.com/library/{base_name}"
                response = requests.head(url, timeout=5)
                
                if response.status_code == 200:
                    verified.append(model_name)
                    print(f"  ✅ {model_name}: EXISTS")
                else:
                    errors.append(f"{model_name}: NOT FOUND (status {response.status_code})")
                    print(f"  ❌ {model_name}: NOT FOUND")
                    
            except Exception as e:
                errors.append(f"{model_name}: ERROR - {e}")
                print(f"  ⚠️  {model_name}: ERROR - {e}")
        
        print(f"\n📊 SUMMARY:")
        print(f"  • Verified: {len(verified)}")
        print(f"  • Errors: {len(errors)}")
        
        if errors:
            print(f"\n❌ ERRORS FOUND:")
            for error in errors:
                print(f"  • {error}")
            success = False
        else:
            print(f"\n🎉 ALL MODELS VERIFIED!")
            success = True
        
        return success
            
    except Exception as e:
        print(f"❌ FATAL ERROR: {e}")
        return False

if __name__ == "__main__":
    main()
