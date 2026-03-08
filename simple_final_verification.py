#!/usr/bin/env python3
"""
Simple final verification without requests dependency
"""
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def main():
    try:
        exec(open('src/ai_agent/utils/model_definitions.py').read())
        families = get_model_families()
        predefined = get_predefined_models()
        
        print("=== FINAL VERIFICATION SUMMARY ===")
        print(f"✅ Total families: {len(families)}")
        print(f"✅ Total predefined models: {len(predefined)}")
        
        # Check for obvious issues in model names
        issues = []
        
        # Check for models that shouldn't have :latest separately
        latest_models = [name for name in predefined.keys() if name.endswith(':latest')]
        for model in latest_models:
            base_name = model.replace(':latest', '')
            if base_name in predefined:
                issues.append(f"{model}: Should not exist separately (use {base_name})")
        
        # Check for models that should exist but don't
        should_exist = [
            'mistral-large:123b',
            'qwen2.5:coder',
            'yi:latest',
            'yi-coder:latest',
            'command-r7b:latest',
            'granite-code:latest',
            'llava:latest',
            'moondream:latest',
            'starcoder2:latest'
        ]
        
        for model in should_exist:
            if model not in predefined:
                issues.append(f"{model}: Missing from predefined")
        
        print(f"\n📊 VERIFICATION RESULTS:")
        print(f"  • Total models checked: {len(predefined)}")
        print(f"  • Issues found: {len(issues)}")
        
        if issues:
            print(f"\n❌ ISSUES FOUND:")
            for issue in issues:
                print(f"  • {issue}")
            return False
        else:
            print(f"\n🎉 NO OBVIOUS ISSUES FOUND!")
            print(f"  • All {len(predefined)} models appear correctly defined")
            return True
            
    except Exception as e:
        print(f"❌ FATAL ERROR: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
