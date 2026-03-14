#!/usr/bin/env python3
"""
Quick Setup Script for VEXIS-CLI
Automatically checks and fixes common setup issues
"""

import subprocess
import sys
import os
from pathlib import Path


def run_command(cmd, description, check=True):
    """Run a command and handle errors"""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=check)
        if result.stdout.strip():
            print(f"✅ {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed: {e.stderr.strip()}")
        return False


def check_ollama():
    """Check and install Ollama if needed"""
    print("\n=== Checking Ollama ===")
    
    try:
        result = subprocess.run(['ollama', '--version'], capture_output=True, text=True, timeout=5)
        print(f"✅ Ollama is installed: {result.stdout.strip()}")
        return True
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("❌ Ollama is not installed")
        response = input("Would you like to install Ollama? (y/n): ")
        if response.lower() == 'y':
            return run_command("curl -fsSL https://ollama.com/install.sh | sh", "Installing Ollama")
        return False


def check_models():
    """Check and install basic models if needed"""
    print("\n=== Checking Models ===")
    
    try:
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')[1:]  # Skip header
            models = [line.split()[0] for line in lines if line.strip()]
            
            if models:
                print(f"✅ Found {len(models)} models: {', '.join(models[:3])}")
                return True
            else:
                print("❌ No models found")
                response = input("Would you like to install a lightweight model? (y/n): ")
                if response.lower() == 'y':
                    return run_command("ollama pull llama3.2:1b", "Installing llama3.2:1b model")
                return False
    except Exception as e:
        print(f"❌ Error checking models: {e}")
        return False


def check_python_deps():
    """Check Python dependencies"""
    print("\n=== Checking Python Dependencies ===")
    
    # Check if we're in a virtual environment
    if sys.prefix == sys.base_prefix:
        print("⚠️  Not in a virtual environment")
        response = input("Would you like to create a project-level virtual environment? (y/n): ")
        if response.lower() == 'y':
            if not run_command("python3 -m venv venv", "Creating project-level virtual environment"):
                return False
            print("✅ Project-level virtual environment created at ./venv")
            print("Please run: source venv/bin/activate && python3 run.py 'your command'")
            return False
    else:
        print("✅ Running in project-level virtual environment")
    
    # Check requirements
    requirements_file = Path("requirements.txt")
    if requirements_file.exists():
        # Use --break-system-packages only if in venv or explicitly allowed
        venv_python = sys.executable
        pip_cmd = f"{venv_python} -m pip install -r requirements.txt"
        return run_command(pip_cmd, "Installing Python dependencies")
    else:
        print("⚠️  requirements.txt not found")
        return False


def main():
    """Main setup function"""
    print("=== VEXIS-CLI Quick Setup ===")
    print("This script will check and fix common setup issues.\n")
    
    # Change to project directory if needed
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    success_count = 0
    total_checks = 3
    
    # Check each component
    if check_ollama():
        success_count += 1
    
    if check_models():
        success_count += 1
    
    if check_python_deps():
        success_count += 1
    
    # Summary
    print(f"\n=== Setup Summary ===")
    print(f"✅ {success_count}/{total_checks} components ready")
    
    if success_count == total_checks:
        print("🎉 Setup complete! You can now run:")
        print("python3 run.py \"your instruction here\"")
    else:
        print("⚠️  Some issues remain. Please fix them manually.")
        print("You can also run: python3 check_models.py for detailed model status.")


if __name__ == "__main__":
    main()
