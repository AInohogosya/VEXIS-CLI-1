#!/usr/bin/env python3
"""
Quick Setup Script for VEXIS-CLI-2
Automatically checks and fixes common setup issues
"""

import subprocess
import sys
import os
from pathlib import Path


def run_command(cmd, description, check=True):
    """Run a command and handle errors (shell=False for security)"""
    import tempfile
    
    print(f"🔧 {description}...")
    try:
        # Handle special case: curl pipe to shell (insecure pattern)
        if isinstance(cmd, str) and 'curl' in cmd and '| sh' in cmd:
            return _safe_curl_install(cmd, description)
        
        # Convert string command to list for security (no shell=True)
        if isinstance(cmd, str):
            # Use shlex to safely split the command
            import shlex
            cmd_list = shlex.split(cmd)
        else:
            cmd_list = cmd
        
        result = subprocess.run(cmd_list, shell=False, capture_output=True, text=True, check=check)
        if result.stdout.strip():
            print(f"✅ {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed: {e.stderr.strip()}")
        return False


def _safe_curl_install(cmd_str, description):
    """Safely execute curl | sh pattern by downloading first, then executing separately"""
    import tempfile
    
    print(f"🔧 {description}...")
    # Extract URL from command like "curl -fsSL https://ollama.com/install.sh | sh"
    parts = cmd_str.split()
    url = None
    for i, part in enumerate(parts):
        if part.startswith('http'):
            url = part
            break
    
    if not url:
        print(f"❌ Failed: Could not extract URL from command")
        return False
    
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as tmp_script:
            script_path = tmp_script.name
        
        try:
            # Download the script
            download_result = subprocess.run(
                ['curl', '-fsSL', url],
                capture_output=True,
                text=True,
                timeout=120
            )
            
            if download_result.returncode != 0:
                print(f"❌ Failed to download script: {download_result.stderr.strip()}")
                return False
            
            # Write and execute
            with open(script_path, 'w') as f:
                f.write(download_result.stdout)
            os.chmod(script_path, 0o755)
            
            result = subprocess.run(
                ['bash', script_path],
                capture_output=True,
                text=True,
                timeout=300
            )
            
            if result.returncode == 0:
                if result.stdout.strip():
                    print(f"✅ {result.stdout.strip()}")
                return True
            else:
                print(f"❌ Failed: {result.stderr.strip()}")
                return False
        finally:
            try:
                os.unlink(script_path)
            except Exception:
                pass
    except Exception as e:
        print(f"❌ Failed: {str(e)}")
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
                    return run_command(['ollama', 'pull', 'llama3.2:1b'], "Installing llama3.2:1b model")
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
            if not run_command([sys.executable, '-m', 'venv', 'venv'], "Creating project-level virtual environment"):
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
        return run_command([venv_python, '-m', 'pip', 'install', '-r', 'requirements.txt'], "Installing Python dependencies")
    else:
        print("⚠️  requirements.txt not found")
        return False


def main():
    """Main setup function"""
    print("=== VEXIS-CLI-2 Quick Setup ===")
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
