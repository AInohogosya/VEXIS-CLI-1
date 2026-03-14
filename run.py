#!/usr/bin/env python3
"""
Ultimate Zero-Configuration AI Agent Runner
Usage: python3 run.py "your instruction here"

This script automatically:
1. Detects if running in virtual environment
2. Creates virtual environment if needed
3. Installs all dependencies automatically
4. Restarts itself in the virtual environment
5. Prompts for model selection (Ollama with model options or Google API)
6. Runs the AI agent with the provided instruction
"""

import sys
import os
import subprocess
import platform
import shutil
from pathlib import Path
from typing import Optional

# Global constants
VENV_DIR = "venv"
VENV_RESTART_FLAG = "--__venv_restarted__"

def is_in_virtual_environment():
    """Check if currently running in a virtual environment"""
    return (
        hasattr(sys, 'real_prefix') or 
        (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix) or
        os.getenv('VIRTUAL_ENV') is not None
    )

def get_venv_python_path():
    """Get the Python executable path in the virtual environment"""
    project_root = Path(__file__).parent
    venv_path = project_root / VENV_DIR
    
    if not venv_path.exists():
        return None
    
    if platform.system() == "Windows":
        python_exe = venv_path / "Scripts" / "python.exe"
        if not python_exe.exists():
            python_exe = venv_path / "Scripts" / "pythonw.exe"
    else:
        python_exe = venv_path / "bin" / "python"
        if not python_exe.exists():
            python_exe = venv_path / "bin" / "python3"
    
    return str(python_exe) if python_exe.exists() else None

def check_venv_prerequisites():
    """Check if virtual environment creation prerequisites are met"""
    print("Checking virtual environment prerequisites...")
    
    # Test if venv module is available
    try:
        import venv
        print("✓ venv module is available")
        return True
    except ImportError:
        print("✗ venv module is not available")
        return False

def create_virtual_environment():
    """Create a virtual environment with robust error handling"""
    project_root = Path(__file__).parent
    venv_path = project_root / VENV_DIR
    
    print(f"Creating virtual environment at {venv_path}...")
    
    # Remove existing venv if it exists and appears broken
    if venv_path.exists():
        venv_python = get_venv_python_path()
        if venv_python:
            try:
                # Test if existing venv works
                result = subprocess.run([venv_python, "--version"], 
                                      capture_output=True, text=True, timeout=10)
                if result.returncode != 0:
                    print("Existing virtual environment appears broken, recreating...")
                    shutil.rmtree(venv_path)
                else:
                    print("Virtual environment already exists and is functional")
                    return True
            except Exception:
                print("Existing virtual environment appears broken, recreating...")
                shutil.rmtree(venv_path)
        else:
            print("Removing incomplete virtual environment...")
            shutil.rmtree(venv_path)
    
    try:
        # Create virtual environment
        result = subprocess.run(
            [sys.executable, "-m", "venv", str(venv_path)],
            capture_output=True,
            text=True,
            timeout=120
        )
        
        if result.returncode != 0:
            error_msg = result.stderr.strip() or result.stdout.strip()
            
            # Handle specific error cases
            if "ensurepip is not available" in error_msg or "python3-venv" in error_msg:
                print("✗ Virtual environment creation failed: python3-venv package not installed")
                print()
                print("To fix this issue, run one of the following commands:")
                print(f"  sudo apt install python3.{sys.version_info.minor}-venv")
                print("  # or for Ubuntu/Debian systems:")
                print("  sudo apt install python3-venv")
                print()
                print("After installing the package, run this script again.")
                return False
            elif "Permission denied" in error_msg:
                print("✗ Permission denied when creating virtual environment")
                print("Check that you have write permissions to the project directory")
                return False
            else:
                print(f"✗ Failed to create virtual environment: {error_msg}")
                print("Full error details:")
                print(f"  Return code: {result.returncode}")
                print(f"  Stderr: {result.stderr}")
                print(f"  Stdout: {result.stdout}")
                return False
        
        print("✓ Virtual environment created successfully")
        return True
        
    except subprocess.TimeoutExpired:
        print("✗ Virtual environment creation timed out")
        return False
    except Exception as e:
        print(f"✗ Error creating virtual environment: {e}")
        return False

def restart_in_venv():
    """Restart the current script in the virtual environment with robust error handling"""
    venv_python = get_venv_python_path()
    if not venv_python:
        print("Error: Could not find virtual environment Python executable")
        return False
    
    # Add restart flag to prevent infinite loops
    new_argv = [venv_python, str(__file__), VENV_RESTART_FLAG] + sys.argv[1:]
    
    print(f"Restarting in virtual environment: {venv_python}")
    
    try:
        # Use os.execv to replace current process
        # This is more reliable than subprocess on all platforms
        os.execv(venv_python, new_argv)
    except OSError as e:
        print(f"OS error restarting in virtual environment: {e}")
        print("This might be due to permissions or antivirus software.")
        return False
    except Exception as e:
        print(f"Unexpected error restarting in virtual environment: {e}")
        return False
    
    # This should never be reached if execv succeeds
    return True

def install_dependencies():
    """Install all dependencies in the virtual environment with enhanced error handling"""
    project_root = Path(__file__).parent
    venv_python = get_venv_python_path()
    
    if not venv_python:
        print("Error: Virtual environment Python not found")
        return False
    
    print("Installing dependencies...")
    
    # Check network connectivity first
    try:
        import socket
        socket.create_connection(("pypi.org", 443), timeout=10)
        print("✓ Network connectivity OK")
    except Exception as e:
        print(f"Warning: Network connectivity issue: {e}")
        print("Dependency installation may fail without internet access.")
    
    # Upgrade pip first with retry mechanism
    max_retries = 3
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                print(f"Retry {attempt + 1}/{max_retries} upgrading pip...")
            else:
                print("Upgrading pip...")
            
            result = subprocess.run([venv_python, "-m", "pip", "install", "--upgrade", "pip"],
                                  capture_output=True, text=True, timeout=300)
            if result.returncode == 0:
                print("✓ pip upgraded")
                break
            else:
                if attempt == max_retries - 1:
                    print(f"pip upgrade failed after {max_retries} attempts: {result.stderr}")
                    print("Continuing with current pip version...")
                else:
                    print(f"pip upgrade attempt {attempt + 1} failed, retrying...")
        except subprocess.TimeoutExpired:
            if attempt == max_retries - 1:
                print("pip upgrade timed out, continuing with current pip version...")
            else:
                print("pip upgrade timed out, retrying...")
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"pip upgrade error: {e}")
                print("Continuing with current pip version...")
            else:
                print(f"pip upgrade error: {e}, retrying...")
    
    # Install from requirements files if they exist
    requirements_files = [
        project_root / "requirements-core.txt",
        project_root / "requirements.txt"  # fallback to original
    ]
    
    for requirements_file in requirements_files:
        if requirements_file.exists():
            for attempt in range(max_retries):
                try:
                    if attempt > 0:
                        print(f"Retry {attempt + 1}/{max_retries} installing {requirements_file.name}...")
                    else:
                        print(f"Installing from {requirements_file.name}...")
                    
                    result = subprocess.run([venv_python, "-m", "pip", "install", "-r", str(requirements_file)],
                                          capture_output=True, text=True, timeout=600)
                    if result.returncode == 0:
                        print(f"✓ {requirements_file.name} installed")
                        
                        # If we successfully installed core requirements, we're done
                        if requirements_file.name == "requirements-core.txt":
                            print("✓ Core dependencies installed successfully")
                            print("Note: Optional ML/AI dependencies can be installed later with:")
                            print("  pip install -r requirements-optional.txt")
                            return True  # Success, exit the function
                        break
                    else:
                        error_msg = result.stderr.strip()
                        if attempt == max_retries - 1:
                            print(f"{requirements_file.name} installation failed after {max_retries} attempts: {error_msg}")
                            
                            # Provide helpful error messages
                            if "Permission denied" in error_msg:
                                print("Permission denied. Check antivirus software or file permissions.")
                            elif "Could not find a version" in error_msg:
                                print("Package version conflict. Check requirements file compatibility.")
                            elif "Network is unreachable" in error_msg or "Connection failed" in error_msg:
                                print("Network error. Check internet connection.")
                            else:
                                print("See error message above for details.")
                            
                            # If this was requirements-core.txt that failed, return False
                            # If it was requirements.txt that failed, we can continue (it's optional)
                            if requirements_file.name == "requirements-core.txt":
                                return False
                            else:
                                print("Continuing without optional dependencies...")
                                return True  # Continue without optional deps
                        else:
                            print(f"{requirements_file.name} attempt {attempt + 1} failed, retrying...")
                except subprocess.TimeoutExpired:
                    if attempt == max_retries - 1:
                        print(f"{requirements_file.name} installation timed out")
                        if requirements_file.name == "requirements-core.txt":
                            return False
                        else:
                            print("Continuing without optional dependencies...")
                            return True  # Continue without optional deps
                    else:
                        print(f"{requirements_file.name} installation timed out, retrying...")
                except Exception as e:
                    if attempt == max_retries - 1:
                        print(f"{requirements_file.name} installation error: {e}")
                        if requirements_file.name == "requirements-core.txt":
                            return False
                        else:
                            print("Continuing without optional dependencies...")
                            return True  # Continue without optional deps
                    else:
                        print(f"{requirements_file.name} installation error: {e}, retrying...")
    
    # Install project in editable mode if pyproject.toml exists
    pyproject_file = project_root / "pyproject.toml"
    if pyproject_file.exists():
        try:
            print("Installing project in editable mode...")
            result = subprocess.run([venv_python, "-m", "pip", "install", "-e", str(project_root)],
                                  capture_output=True, text=True, timeout=300)
            if result.returncode == 0:
                print("✓ project installed")
            else:
                print(f"project installation warning: {result.stderr}")
                print("Project installation failed, but dependencies may still work")
        except subprocess.TimeoutExpired:
            print("project installation timed out")
            print("Project installation failed, but dependencies may still work")
        except Exception as e:
            print(f"project installation error: {e}")
            print("Project installation failed, but dependencies may still work")
    
    return True

def bootstrap_environment():
    """Bootstrap the environment - create venv and install dependencies"""
    print("Bootstrapping environment...")
    
    # Check prerequisites first
    if not check_venv_prerequisites():
        print()
        print("Virtual environment prerequisites not met.")
        print("This is likely because the python3-venv package is not installed.")
        print()
        print("To fix this issue, run one of the following commands:")
        print(f"  sudo apt install python3.{sys.version_info.minor}-venv")
        print("  # or for Ubuntu/Debian systems:")
        print("  sudo apt install python3-venv")
        print()
        print("After installing the package, run this script again.")
        return False
    
    # Create virtual environment
    if not create_virtual_environment():
        print("Failed to create virtual environment")
        return False
    
    # Install dependencies
    if not install_dependencies():
        print("Failed to install dependencies")
        return False
    
    print("✓ Environment bootstrap complete")
    return True

def show_help():
    """Show help message"""
    print("VEXIS-1.1 AI Agent Runner")
    print("=" * 50)
    print("Usage: python3 run.py \"your instruction here\"")
    print()
    print("This script automatically handles:")
    print("  • Virtual environment creation and management")
    print("  • Dependency installation")
    print("  • Model selection (13 AI providers with model options)")
    print("    - Local: Ollama (privacy-focused)")
    print("    - Cloud: OpenAI, Anthropic, Google, xAI, Meta, Groq, DeepSeek, Together, Microsoft, Mistral, Amazon, Cohere")
    print("  • Cross-platform compatibility")
    print("  • Self-bootstrapping")
    print("  • Environment detection and adaptive execution")
    print()
    print("Model Options:")
    print("  🦊 Ollama: Local models (privacy-focused) - Stable")
    print("  🌐 Google: Gemini models (enterprise-grade) - Stable")
    print("  🤖 OpenAI: GPT models (advanced capabilities) - Beta")
    print("  🧠 Anthropic: Claude models (strong reasoning) - Beta")
    print("  🚀 xAI: Grok models (real-time knowledge) - Beta")
    print("  🦙 Meta: Llama models (via Meta API) - Beta")
    print("  ⚡ Groq: Fast inference (Llama/Mixtral) - Beta")
    print("  🔍 DeepSeek: Advanced reasoning models - Beta")
    print("  🤝 Together AI: Open-source model hosting - Beta")
    print("  ☁️ Microsoft: GPT models via Azure - Beta")
    print("  🌍 Mistral AI: Multilingual models - Beta")
    print("  🏭 Amazon Bedrock: Titan/Nova models via AWS - Beta")
    print("  🏢 Cohere: Command models for enterprise - Beta")
    print()
    print("Environment Commands:")
    print("  --check, -c         Run environment check and show recommendations")
    print("  --fix               Run environment check and auto-fix issues")
    print("  --install-sdks      Install missing AI provider SDKs")
    print("  --sdk-status        Show AI provider SDK installation status")
    print()
    print("Examples:")
    print("  python3 run.py \"Take a screenshot\"")
    print("  python3 run.py \"Open a web browser and search for AI\"")
    print("  python3 run.py --check")
    print("  python3 run.py --install-sdks")
    print()
    print("Options:")
    print("  --help, -h          Show this help message")
    print("  --debug             Enable debug mode")
    print("  --no-prompt         Use saved provider preference without prompting")
    print()
    print("SDK Management:")
    print("  python3 manage_sdks.py status          # Show SDK status")
    print("  python3 manage_sdks.py install         # Install all missing SDKs")
    print("  python3 manage_sdks.py install google  # Install specific SDK")
    print()
    print("Virtual Environment:")
    print("  Automatically creates and uses './venv' directory")
    print("  All dependencies are isolated within the virtual environment")
    print("  No manual setup required - just run and go!")

def check_ollama_login_with_fallback():
    """Check Ollama login with version-aware fallback"""
    from ai_agent.utils.interactive_menu import Colors, success_message, error_message, warning_message
    from ai_agent.utils.environment_detector import EnvironmentDetector
    
    detector = EnvironmentDetector()
    ollama_available = detector._detect_ollama_available()
    
    if not ollama_available:
        error_message("Ollama is not installed or not in PATH")
        print(f"{Colors.BRIGHT_CYAN}Please install Ollama first: https://ollama.com/{Colors.RESET}")
        print(f"{Colors.CYAN}Or run with --fix to auto-install{Colors.RESET}")
        return False, "not_installed"
    
    # Check version for cloud model support
    needs_update = detector._detect_needs_ollama_update()
    has_signin = detector._detect_ollama_has_signin()
    has_whoami = detector._detect_ollama_has_whoami()
    
    if needs_update:
        warning_message(f"Ollama version is outdated (cloud models require 0.17.0+)")
        print(f"{Colors.CYAN}Local models will work, but cloud models require update.{Colors.RESET}")
        print(f"{Colors.CYAN}Run with --fix to update Ollama automatically.{Colors.RESET}")
        # Return partial success - local models still work
        return True, "local_only"
    
    # Check if signed in (only for newer versions)
    if has_whoami:
        try:
            result = subprocess.run(["ollama", "whoami"], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode != 0 or "not signed in" in result.stderr.lower():
                warning_message("Ollama is available but you are not signed in.")
                print(f"{Colors.CYAN}Cloud models require signin. Local models will work.{Colors.RESET}")
                print(f"{Colors.CYAN}Run 'ollama signin' to enable cloud models.{Colors.RESET}")
                return True, "needs_signin"
            else:
                success_message("Ollama is signed in")
                return True, "full"
        except Exception:
            return True, "local_only"
    
    # Old version without whoami - assume local only
    return True, "local_only"

def run_environment_check(fix_mode=False):
    """Run environment detection and optionally fix issues"""
    from ai_agent.utils.environment_detector import detect_and_plan
    from ai_agent.utils.interactive_menu import Colors
    
    env_info, executor = detect_and_plan()
    
    # Save report
    import json
    from dataclasses import asdict
    from pathlib import Path
    
    report_path = Path("environment_report.json")
    with open(report_path, 'w') as f:
        json.dump(asdict(env_info), f, indent=2)
    
    print(f"\n📄 Detailed report saved to: {report_path}")
    
    # Execute fix plan if requested
    if fix_mode and executor.execution_plan:
        print(f"\n🔧 Fix mode enabled - executing {len(executor.execution_plan)} steps")
        executor.execute_plan(interactive=True)
    elif executor.execution_plan:
        print(f"\n💡 Run with --fix to automatically address these issues")
    
    return env_info, executor

def update_ollama():
    """Update Ollama to latest version"""
    from ai_agent.utils.interactive_menu import Colors, success_message, error_message, warning_message
    
    print(f"{Colors.CYAN}Updating Ollama...{Colors.RESET}")
    try:
        # Download and run install script
        result = subprocess.run(
            "curl -fsSL https://ollama.com/install.sh | sh",
            shell=True,
            capture_output=True,
            text=True,
            timeout=300
        )
        if result.returncode == 0:
            success_message("Ollama updated successfully")
            return True
        else:
            error_message(f"Ollama update failed: {result.stderr}")
            return False
    except Exception as e:
        error_message(f"Error updating Ollama: {e}")
        return False

def prompt_for_google_api_key():
    """Prompt user for Google API key and handle saving"""
    import getpass
    from ai_agent.utils.interactive_menu import Colors, success_message, error_message, warning_message
    
    print(f"\n{Colors.BOLD}{Colors.CYAN}Google API Key Setup{Colors.RESET}")
    print(f"{Colors.CYAN}{'-' * 25}{Colors.RESET}")
    print(f"{Colors.WHITE}To use Google's official Gemini API, you need an API key.{Colors.RESET}")
    print(f"{Colors.BRIGHT_CYAN}You can get one from: https://aistudio.google.com/app/apikey{Colors.RESET}")
    print()
    
    while True:
        try:
            api_key = getpass.getpass(f"{Colors.YELLOW}Enter your Google API key (or press Enter to cancel):{Colors.RESET} ")
            if not api_key.strip():
                warning_message("No API key provided. Skipping Google API setup.")
                return None
            
            # Basic validation (Google API keys are typically 39 characters starting with 'AIza')
            if len(api_key) < 20:
                error_message("API key seems too short. Please check your key.")
                continue
            
            # Ask if user wants to save the key
            save_key = input(f"{Colors.CYAN}Save this API key for future use? (y/n):{Colors.RESET} ").lower().strip()
            should_save = save_key.startswith('y')
            
            return api_key, should_save
            
        except KeyboardInterrupt:
            print(f"\n{Colors.BRIGHT_YELLOW}Operation cancelled.{Colors.RESET}")
            return None
        except Exception as e:
            error_message(f"Error reading input: {e}")
            return None

def select_google_model():
    """Prompt user to select Google model using curses arrow keys"""
    from ai_agent.utils.settings_manager import get_settings_manager
    from ai_agent.utils.curses_menu import get_curses_menu
    
    settings_manager = get_settings_manager()
    current_model = settings_manager.get_google_model()
    
    # Use curses-based menu with arrow keys
    menu = get_curses_menu(
        "🚀 Select Gemini Model",
        "Choose your preferred Gemini model:"
    )
    
    menu.add_item(
        "Gemini 3 Flash",
        "Fast and efficient • Cost-effective for most tasks",
        "gemini-3-flash-preview",
        "🚀"
    )
    
    menu.add_item(
        "Gemini 3.1 Pro",
        "Advanced reasoning • Best for complex problem-solving",
        "gemini-3.1-pro-preview",
        "🧠"
    )
    
    selected_model = menu.show()
    
    if selected_model is None:
        return current_model
    
    settings_manager.set_google_model(selected_model)
    return selected_model

def show_config_summary(provider: str, model: str = None):
    """Display a clean configuration summary"""
    from ai_agent.utils.interactive_menu import Colors
    from ai_agent.utils.settings_manager import get_settings_manager
    
    settings_manager = get_settings_manager()
    
    print(f"\n{Colors.BOLD}{Colors.BRIGHT_CYAN}{'─' * 50}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BRIGHT_GREEN}✓ Configuration Complete{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BRIGHT_CYAN}{'─' * 50}{Colors.RESET}")
    
    # Provider and model display mapping
    provider_info = {
        "ollama": ("Ollama (Local Models)", settings_manager.get_ollama_model()),
        "google": ("Google Official API", model or settings_manager.get_google_model()),
        "openai": ("OpenAI Official API", model or settings_manager.get_openai_model()),
        "anthropic": ("Anthropic Official API", model or settings_manager.get_anthropic_model()),
        "xai": ("xAI Official API", model or settings_manager.get_xai_model()),
        "meta": ("Meta Official API", model or settings_manager.get_meta_model()),
        "groq": ("Groq Official API", model or settings_manager.get_groq_model()),
        "deepseek": ("DeepSeek Official API", model or settings_manager.get_deepseek_model()),
        "together": ("Together AI Official API", model or settings_manager.get_together_model()),
        "microsoft": ("Microsoft Azure API", model or settings_manager.get_microsoft_model()),
        "mistral": ("Mistral Official API", model or settings_manager.get_mistral_model()),
        "amazon": ("Amazon Bedrock API", model or settings_manager.get_amazon_model()),
        "cohere": ("Cohere Official API", model or settings_manager.get_cohere_model())
    }
    
    if provider in provider_info:
        provider_name, model_name = provider_info[provider]
        print(f"{Colors.WHITE}  Provider: {Colors.BRIGHT_YELLOW}{provider_name}{Colors.RESET}")
        
        # Format model name for better display
        if model_name:
            display_model = format_model_display_name(provider, model_name)
            print(f"{Colors.WHITE}  Model:    {Colors.BRIGHT_YELLOW}{display_model}{Colors.RESET}")
    else:
        print(f"{Colors.WHITE}  Provider: {Colors.BRIGHT_YELLOW}Unknown Provider{Colors.RESET}")
        print(f"{Colors.WHITE}  Model:    {Colors.BRIGHT_YELLOW}{model or 'Unknown'}{Colors.RESET}")
    
    print(f"{Colors.BOLD}{Colors.BRIGHT_CYAN}{'─' * 50}{Colors.RESET}\n")

def format_model_display_name(provider: str, model: str) -> str:
    """Format model names for better display"""
    model_display_map = {
        "google": {
            "gemini-2.0-flash-exp": "Gemini 2.0 Flash",
            "gemini-3-flash-preview": "Gemini 3 Flash",
            "gemini-1.5-pro": "Gemini 1.5 Pro",
            "gemini-1.5-flash": "Gemini 1.5 Flash"
        },
        "openai": {
            "gpt-4o": "GPT-4o",
            "gpt-4o-mini": "GPT-4o Mini",
            "gpt-4-turbo": "GPT-4 Turbo",
            "gpt-3.5-turbo": "GPT-3.5 Turbo"
        },
        "anthropic": {
            "claude-3-5-sonnet-20241022": "Claude 3.5 Sonnet",
            "claude-3-opus-20240229": "Claude 3 Opus",
            "claude-3-sonnet-20240229": "Claude 3 Sonnet",
            "claude-3-haiku-20240307": "Claude 3 Haiku"
        }
    }
    
    if provider in model_display_map and model in model_display_map[provider]:
        return model_display_map[provider][model]
    
    return model

def configure_google_provider():
    """Configure Google provider with API key and model selection"""
    from ai_agent.utils.settings_manager import get_settings_manager
    from ai_agent.utils.interactive_menu import Colors
    
    settings_manager = get_settings_manager()
    
    # Check if API key already exists
    if not settings_manager.has_google_api_key():
        # Prompt for API key
        result = prompt_for_google_api_key()
        if result is None:
            return None, None
        
        api_key, should_save = result
        settings_manager.set_google_api_key(api_key, should_save)
    
    
    # Select model
    model = select_google_model()
    if model is None:
        model = settings_manager.get_google_model()
    
    settings_manager.set_preferred_provider("google")
    return "google", model

def ensure_ollama_model_available(model_name: str) -> bool:
    """Ensure the specified Ollama model is available locally, pull if necessary"""
    from ai_agent.utils.interactive_menu import Colors, success_message, error_message, warning_message
    from ai_agent.utils.ollama_error_handler import handle_ollama_error
    
    try:
        # Check if model is already available
        result = subprocess.run(["ollama", "list"], 
                              capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            available_models = result.stdout.strip().split('\n')
            if len(available_models) > 1:  # First line is header
                model_names = [line.split()[0] for line in available_models[1:] if line.strip()]
                if model_name in model_names:
                    success_message(f"Model {model_name} is already available")
                    return True
        
        # Model not available, try to pull it
        warning_message(f"Model {model_name} not found locally, pulling...")
        print(f"{Colors.CYAN}This may take several minutes depending on model size and network speed.{Colors.RESET}")
        print(f"{Colors.YELLOW}💡 Tip: You can press Ctrl+C to cancel if needed{Colors.RESET}")
        
        # Check available disk space for large models
        try:
            import shutil
            disk_usage = shutil.disk_usage("/")
            free_gb = disk_usage.free / (1024**3)
            if free_gb < 10:  # Less than 10GB free
                print(f"{Colors.YELLOW}⚠️ Low disk space warning: {free_gb:.1f}GB available{Colors.RESET}")
                print(f"{Colors.YELLOW}💡 Consider freeing up space before downloading large models{Colors.RESET}")
        except Exception:
            pass  # Disk space check is optional
        
        # Show progress indicator
        import threading
        import time
        
        stop_spinner = threading.Event()
        def spinner():
            spinner_chars = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
            i = 0
            while not stop_spinner.is_set():
                print(f"{Colors.CYAN}\r{spinner_chars[i % len(spinner_chars)]} Downloading {model_name}...{Colors.RESET}", end='', flush=True)
                time.sleep(0.1)
                i += 1
        
        spinner_thread = threading.Thread(target=spinner)
        spinner_thread.daemon = True
        spinner_thread.start()
        
        try:
            pull_result = subprocess.run(["ollama", "pull", model_name], 
                                       capture_output=False, text=True, timeout=600)  # 10 minutes timeout
        except KeyboardInterrupt:
            stop_spinner.set()
            print(f"\n{Colors.YELLOW}⚠ Download cancelled by user{Colors.RESET}")
            return False
        finally:
            stop_spinner.set()
            spinner_thread.join(timeout=0.5)
            print(f"\r{' ' * 50}\r", end='', flush=True)  # Clear spinner line
        
        if pull_result.returncode == 0:
            success_message(f"✅ Successfully pulled Ollama model: {model_name}")
            # Show model size info if available
            try:
                size_result = subprocess.run(["ollama", "list"], 
                                          capture_output=True, text=True, timeout=10)
                if size_result.returncode == 0:
                    lines = size_result.stdout.strip().split('\n')
                    for line in lines[1:]:  # Skip header
                        if model_name in line:
                            parts = line.split()
                            if len(parts) >= 2:
                                size_info = parts[1]
                                print(f"{Colors.GREEN}📊 Model size: {size_info}{Colors.RESET}")
                            break
            except Exception:
                pass  # Size info is optional
            return True
        else:
            # Use enhanced error handling for pull failures
            error_message(f"Failed to pull model {model_name}")
            
            # Offer retry option for network issues
            if "network" in str(pull_result.stderr).lower() or "connection" in str(pull_result.stderr).lower():
                print(f"{Colors.YELLOW}🔄 Network issue detected. Would you like to retry?{Colors.RESET}")
                try:
                    retry = input(f"{Colors.CYAN}Retry download? (y/N): {Colors.RESET}").strip().lower()
                    if retry in ['y', 'yes']:
                        print(f"{Colors.CYAN}🔄 Retrying download...{Colors.RESET}")
                        retry_result = subprocess.run(["ollama", "pull", model_name], 
                                                   capture_output=False, text=True, timeout=600)
                        if retry_result.returncode == 0:
                            success_message(f"✅ Successfully pulled Ollama model: {model_name} (retry)")
                            return True
                        else:
                            error_message(f"Retry also failed for model {model_name}")
                except KeyboardInterrupt:
                    print(f"{Colors.YELLOW}⚠ Retry cancelled by user{Colors.RESET}")
                except Exception:
                    pass
            
            # Try to get more specific error information
            try:
                error_result = subprocess.run(["ollama", "pull", model_name], 
                                          capture_output=True, text=True, timeout=30)
                if error_result.returncode != 0:
                    context = {
                        'model_name': model_name,
                        'operation': 'pull_model'
                    }
                    handle_ollama_error(error_result.stderr or error_result.stdout, context, display_to_user=True)
            except Exception as e:
                context = {
                    'model_name': model_name,
                    'operation': 'pull_model'
                }
                handle_ollama_error(str(e), context, display_to_user=True)
            
            return False
            
    except subprocess.TimeoutExpired:
        error_message(f"Timeout pulling model {model_name}")
        context = {
            'model_name': model_name,
            'operation': 'pull_model'
        }
        handle_ollama_error(f"Timeout pulling model {model_name}", context, display_to_user=True)
        return False
    except FileNotFoundError:
        error_message("Ollama command not found")
        context = {
            'operation': 'ollama_command'
        }
        handle_ollama_error("Ollama command not found", context, display_to_user=True)
        return False
    except Exception as e:
        error_message(f"Error ensuring model availability: {e}")
        context = {
            'model_name': model_name,
            'operation': 'ensure_model'
        }
        handle_ollama_error(str(e), context, display_to_user=True)
        return False

def configure_ollama_provider():
    """Configure Ollama provider with model selection"""
    from ai_agent.utils.settings_manager import get_settings_manager
    from ai_agent.utils.ollama_model_selector import select_ollama_model
    from ai_agent.utils.interactive_menu import Colors, warning_message, info_message
    from ai_agent.utils.ollama_error_handler import handle_ollama_error
    
    settings_manager = get_settings_manager()
    
    # Check Ollama with version-aware fallback
    try:
        login_ok, status = check_ollama_login_with_fallback()
        if not login_ok:
            return None
    except Exception as e:
        # Use enhanced error handling for Ollama check failures
        context = {
            'operation': 'check_ollama_status'
        }
        handle_ollama_error(str(e), context, display_to_user=True)
        return None
    
    # Handle different status levels
    if status == "not_installed":
        return None
    elif status == "local_only":
        info_message("Using Ollama with local models only (cloud models require update)")
    elif status == "needs_signin":
        info_message("Ollama available. Local models work; sign in for cloud models.")
    
    # Always show model selection for Ollama
    print(f"{Colors.CYAN}🦊 Selecting Ollama model...{Colors.RESET}")
    try:
        model = select_ollama_model()
    except Exception as e:
        # Use enhanced error handling for model selection failures
        context = {
            'operation': 'select_model'
        }
        handle_ollama_error(str(e), context, display_to_user=True)
        return None
    
    if model is None:
        # User cancelled or selection failed - show current model and continue
        current_model = settings_manager.get_ollama_model()
        warning_message(f"Using current model: {current_model}")
        model = current_model
    else:
        # Successfully selected new model
        from ai_agent.utils.interactive_menu import success_message
        success_message(f"Selected Ollama model: {model}")
    
    # Ensure the model is pulled locally
    if not ensure_ollama_model_available(model):
        info_message(f"Failed to pull Ollama model: {model}")
        return None
    
    # Don't automatically set preferred provider - let user choose explicitly
    return "ollama"

def select_model_provider():
    """Main configuration screen for model provider selection using curses arrow keys"""
    from ai_agent.utils.settings_manager import get_settings_manager
    from ai_agent.utils.curses_menu import get_curses_menu
    
    settings_manager = get_settings_manager()
    current_provider = settings_manager.get_preferred_provider()
    
    # Use curses-based menu with arrow keys
    menu = get_curses_menu(
        "🔧 Select AI Provider",
        "Choose how you want to run AI models:"
    )
    
    menu.add_item(
        "Ollama (Local)",
        "Run models locally via Ollama • Privacy-focused",
        "ollama",
        "🦊"
    )
    
    menu.add_item(
        "Google Official API",
        "Use Google's cloud Gemini models • Requires API key",
        "google",
        "🌐"
    )
    
    menu.add_item(
        "OpenAI (Beta)",
        "Use OpenAI's GPT models • Requires API key",
        "openai",
        "🤖"
    )
    
    menu.add_item(
        "Anthropic (Beta)",
        "Use Anthropic's Claude models • Requires API key",
        "anthropic",
        "🧠"
    )
    
    menu.add_item(
        "xAI/Grok (Beta)",
        "Use xAI's Grok models • Requires API key",
        "xai",
        "🚀"
    )
    
    menu.add_item(
        "Meta/Llama (Beta)",
        "Use Meta's Llama models • Requires API key",
        "meta",
        "🦙"
    )
    
    menu.add_item(
        "Groq (Beta)",
        "Use Groq's fast inference • Requires API key",
        "groq",
        "⚡"
    )
    
    menu.add_item(
        "DeepSeek (Beta)",
        "Use DeepSeek's reasoning models • Requires API key",
        "deepseek",
        "🔍"
    )
    
    menu.add_item(
        "Together AI (Beta)",
        "Use Together AI's open-source models • Requires API key",
        "together",
        "🤝"
    )
    
    menu.add_item(
        "Microsoft Azure (Beta)",
        "Use Azure's GPT models • Requires API key",
        "microsoft",
        "☁️"
    )
    
    menu.add_item(
        "Mistral AI (Beta)",
        "Use Mistral's multilingual models • Requires API key",
        "mistral",
        "🌍"
    )
    
    menu.add_item(
        "Amazon Bedrock (Beta)",
        "Use AWS Bedrock models • Requires API key",
        "amazon",
        "🏭"
    )
    
    menu.add_item(
        "Cohere (Beta)",
        "Use Cohere's enterprise models • Requires API key",
        "cohere",
        "🏢"
    )
    
    selected_provider = menu.show()
    
    if selected_provider is None:
        # User cancelled - use current settings
        if current_provider == "google":
            model = settings_manager.get_google_model()
            show_config_summary(current_provider, model)
        else:
            ollama_model = settings_manager.get_ollama_model()
            show_config_summary(current_provider, ollama_model)
        return current_provider
    
    
    # Handle provider selection
    if selected_provider == "ollama":
        result = configure_ollama_provider()
        if result is None:
            # Failed - retry configuration
            return select_model_provider()
        ollama_model = settings_manager.get_ollama_model()
        show_config_summary("ollama", ollama_model)
        return "ollama"
        
    elif selected_provider == "google":
        provider, model = configure_google_provider()
        if provider is None:
            # User cancelled API key entry - retry
            return select_model_provider()
        show_config_summary(provider, model)
        return "google"
        
    elif selected_provider in ["openai", "anthropic", "xai", "meta", "groq", "deepseek", "together", "microsoft", "mistral", "amazon", "cohere"]:
        # Generic handler for all other providers
        provider, model = configure_generic_provider(selected_provider)
        if provider is None:
            # User cancelled API key entry - retry
            return select_model_provider()
        show_config_summary(provider, model)
        return selected_provider

def configure_generic_provider(provider_name):
    """Generic configuration for cloud providers with arrow key model selection"""
    from ai_agent.utils.settings_manager import get_settings_manager
    from ai_agent.utils.interactive_menu import Colors, info_message, warning_message
    from ai_agent.utils.curses_menu import get_curses_menu
    
    settings_manager = get_settings_manager()
    
    # Provider-specific model options (updated to current 2026 models)
    provider_models = {
        "openai": ["gpt-5.4", "gpt-5.4-pro", "gpt-5-mini", "gpt-4o"],
        "anthropic": ["claude-opus-4.6", "claude-sonnet-4.6", "claude-sonnet-4.5"],
        "xai": ["grok-4.20", "grok-4.20-beta"],
        "meta": ["llama-4-scout-17b", "meta-llama-3.1-405b-instruct"],
        "groq": ["llama-3.3-70b-versatile", "openai/gpt-oss-120b", "llama-3.1-8b-instant"],
        "deepseek": ["deepseek-r1", "deepseek-v4"],
        "together": ["meta-llama/Llama-4-Scout-17B-Instruct", "meta-llama/Llama-3.3-70B-Instruct-Turbo"],
        "microsoft": ["gpt-5.4", "gpt-5.4-pro", "gpt-4o"],
        "mistral": ["mistral-large-2411", "mistral-small-2409"],
        "amazon": ["anthropic.claude-opus-4.6-v1:0", "anthropic.claude-sonnet-4.6-v1:0"],
        "cohere": ["command-r-plus-08-2024", "command-r-08-2024"]
    }
    
    # API key environment variables
    api_key_vars = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "xai": "XAI_API_KEY",
        "meta": "META_API_KEY",
        "groq": "GROQ_API_KEY",
        "deepseek": "DEEPSEEK_API_KEY",
        "together": "TOGETHER_API_KEY",
        "microsoft": "AZURE_API_KEY",
        "mistral": "MISTRAL_API_KEY",
        "amazon": "AWS_ACCESS_KEY_ID",
        "cohere": "COHERE_API_KEY"
    }
    
    info_message(f"🔑 Configuring {provider_name.upper()} Provider")
    print(f"Environment variable: {api_key_vars[provider_name]}")
    
    # Check for existing API key
    import os
    existing_key = os.getenv(api_key_vars[provider_name])
    if existing_key:
        print(f"{Colors.GREEN}✓ API key found in environment{Colors.RESET}")
        use_existing = input("Use existing API key? (Y/n): ").strip().lower()
        if use_existing != 'n':
            # Use arrow key menu for model selection
            selected_model = select_model_with_arrows(provider_name, provider_models[provider_name])
            if selected_model:
                settings_manager.set_api_key(provider_name, existing_key)
                settings_manager.set_model(provider_name, selected_model)
                return provider_name, selected_model
    
    # Get API key from user
    api_key = get_valid_api_key(f"Enter {provider_name.upper()} API key: ")
    if not api_key:
        return None, None
    
    # Use arrow key menu for model selection
    selected_model = select_model_with_arrows(provider_name, provider_models[provider_name])
    if not selected_model:
        return None  # User cancelled selection
    
    # Save settings
    settings_manager.set_api_key(provider_name, api_key)
    settings_manager.set_model(provider_name, selected_model)
    
    print(f"{Colors.GREEN}✓ {provider_name.upper()} configured successfully!{Colors.RESET}")
    return provider_name, selected_model


def select_model_with_arrows(provider_name: str, models: list) -> Optional[str]:
    """Select model using arrow keys in a curses menu"""
    from ai_agent.utils.curses_menu import get_curses_menu
    
    menu = get_curses_menu(
        f"🤖 {provider_name.upper()} Model Selection",
        "Choose your preferred model using arrow keys:"
    )
    
    # Add models to menu with descriptions
    model_descriptions = {
        "gpt-5.4": "Latest GPT-5.4 • Flagship • Advanced reasoning & coding",
        "gpt-5.4-pro": "GPT-5.4 Pro • Professional tier • Enhanced capabilities",
        "gpt-5-mini": "GPT-5 Mini • Cost-optimized • Fast inference",
        "gpt-4o": "GPT-4 Omni • Multimodal • Strong capabilities",
        "claude-opus-4.6": "Claude Opus 4.6 • Most advanced • Complex reasoning",
        "claude-sonnet-4.6": "Claude Sonnet 4.6 • Latest • Near-Opus performance",
        "claude-sonnet-4.5": "Claude Sonnet 4.5 • Enterprise • B2B workflows",
        "grok-4.20": "Grok 4.20 • Real-time knowledge • Advanced reasoning",
        "grok-4.20-beta": "Grok 4.20 Beta • Preview features",
        "llama-4-scout-17b": "Meta's Llama 4 Scout • 17B • Latest generation",
        "meta-llama-3.1-405b-instruct": "Meta's Llama 3.1 • 405B parameters",
        "llama-3.3-70b-versatile": "Groq's Llama 3.3 • Fast inference",
        "openai/gpt-oss-120b": "OpenAI GPT-OSS 120B • Open source • Powerful",
        "llama-3.1-8b-instant": "Llama 3.1 8B • Fast & efficient",
        "deepseek-r1": "DeepSeek R1 • Advanced reasoning • Uncensored",
        "deepseek-v4": "DeepSeek V4 • Latest • 1T parameters • Efficient",
        "meta-llama/Llama-4-Scout-17B-Instruct": "Together AI's Llama 4 Scout • Latest",
        "meta-llama/Llama-3.3-70B-Instruct-Turbo": "Together AI's Llama 3.3 • Optimized",
        "mistral-large-2411": "Latest Mistral Large • Advanced capabilities",
        "mistral-small-2409": "Mistral Small • Efficient and fast",
        "anthropic.claude-opus-4.6-v1:0": "Claude Opus 4.6 via AWS • Enterprise",
        "anthropic.claude-sonnet-4.6-v1:0": "Claude Sonnet 4.6 via AWS • Latest",
        "command-r-plus-08-2024": "Cohere Command R+ • Advanced reasoning",
        "command-r-08-2024": "Cohere Command R • Efficient"
    }
    
    # Add each model to the menu
    for model in models:
        description = model_descriptions.get(model, f"{model} • Standard model")
        icon = "🚀" if "latest" in description.lower() or "newest" in description.lower() else "🧠"
        menu.add_item(model, description, model, icon)
    
    selected_model = menu.show()
    return selected_model


def get_valid_api_key(prompt):
    """Get and validate API key from user input"""
    from ai_agent.utils.interactive_menu import Colors, warning_message
    
    while True:
        api_key = input(prompt).strip()
        if not api_key:
            return None
        
        if len(api_key) < 10:
            warning_message("API key seems too short. Please check and try again.")
            continue
        
        return api_key


def main():
    """Main entry point"""
    # Check for help flag first
    if "--help" in sys.argv or "-h" in sys.argv:
        show_help()
        sys.exit(0)
    
    # Check for environment check/fix flags (run before venv setup)
    if "--check" in sys.argv or "-c" in sys.argv:
        print("🔍 Running environment check...")
        run_environment_check(fix_mode=False)
        sys.exit(0)
    
    if "--fix" in sys.argv:
        print("🔧 Running environment check with auto-fix...")
        run_environment_check(fix_mode=True)
        sys.exit(0)
    
    # Check if we've already restarted in venv
    if VENV_RESTART_FLAG in sys.argv:
        # Remove the restart flag for clean processing
        sys.argv.remove(VENV_RESTART_FLAG)
        print("✓ Running in virtual environment")
    else:
        # Not in venv or not restarted yet
        if not is_in_virtual_environment():
            print("Not in virtual environment")
            
            # Check if venv exists and is functional
            venv_python = get_venv_python_path()
            if venv_python:
                try:
                    result = subprocess.run([venv_python, "--version"], 
                                          capture_output=True, text=True, timeout=10)
                    if result.returncode == 0:
                        print("Virtual environment found, restarting...")
                        restart_in_venv()
                        return  # This should never execute if restart works
                except Exception:
                    pass
            
            # No working venv found, create one
            if bootstrap_environment():
                print("Restarting in new virtual environment...")
                restart_in_venv()
                return  # This should never execute if restart works
            else:
                print("Failed to bootstrap environment")
                sys.exit(1)
        else:
            print("✓ Already in virtual environment")
    
    # At this point, we're running in a virtual environment
    # Add src to Python path
    current_dir = Path(__file__).parent
    src_dir = current_dir / "src"
    sys.path.insert(0, str(src_dir))
    
    # Validate arguments
    if len(sys.argv) < 2 and not any(flag in sys.argv for flag in ["--install-sdks", "--sdk-status", "--help"]):
        print("Usage: python3 run.py \"your instruction here\"")
        print("Example: python3 run.py \"Take a screenshot\"")
        print("\nOptions:")
        print("  --install-sdks    Install missing AI provider SDKs")
        print("  --sdk-status      Show AI provider SDK installation status")
        print("  --debug           Enable debug mode")
        print("  --no-prompt       Use saved provider preference without prompting")
        print("  --help            Show this help message")
        sys.exit(1)
    
    # Show help
    if "--help" in sys.argv:
        print("VEXIS-CLI - AI-Powered Command Line Assistant")
        print("=" * 50)
        print("\nUsage: python3 run.py \"your instruction here\"")
        print("\nExamples:")
        print("  python3 run.py \"Take a screenshot\"")
        print("  python3 run.py \"Create a new folder called projects\"")
        print("  python3 run.py \"List all files in current directory\"")
        print("\nOptions:")
        print("  --install-sdks    Install missing AI provider SDKs")
        print("  --sdk-status      Show AI provider SDK installation status")
        print("  --debug           Enable debug mode")
        print("  --no-prompt       Use saved provider preference without prompting")
        print("  --help            Show this help message")
        print("\nSDK Management:")
        print("  python3 manage_sdks.py status          # Show SDK status")
        print("  python3 manage_sdks.py install         # Install all missing SDKs")
        print("  python3 manage_sdks.py install google  # Install specific SDK")
        sys.exit(0)
    
    # Filter out flags to get the actual instruction
    instruction_args = [arg for arg in sys.argv[1:] if not arg.startswith("--")]
    instruction = " ".join(instruction_args)
    
    # Allow SDK management commands without instruction
    sdk_only_commands = ["--install-sdks", "--sdk-status"]
    if not instruction and not any(flag in sys.argv for flag in sdk_only_commands + ["--help"]):
        print("No instruction provided")
        print("Usage: python3 run.py \"your instruction here\"")
        print("Use --help for more options")
        sys.exit(1)
    
    # Check for debug mode
    debug_mode = "--debug" in sys.argv
    
    # Check for SDK installation request
    if "--install-sdks" in sys.argv:
        print("🔧 Installing missing AI provider SDKs...")
        try:
            import subprocess
            result = subprocess.run([sys.executable, "manage_sdks.py", "install"], 
                                  capture_output=False, text=True, cwd=current_dir)
            if result.returncode == 0:
                print("✅ SDK installation completed")
            else:
                print("⚠️ Some SDK installations may have failed")
        except Exception as e:
            print(f"❌ Failed to run SDK installation: {e}")
        print()
    
    # Check for SDK status request
    if "--sdk-status" in sys.argv:
        print("🔍 Checking AI provider SDK status...")
        try:
            import subprocess
            subprocess.run([sys.executable, "manage_sdks.py", "status"], 
                         capture_output=False, text=True, cwd=current_dir)
        except Exception as e:
            print(f"❌ Failed to check SDK status: {e}")
        sys.exit(0)
    
    # Model selection - only prompt if not using --no-prompt flag
    if "--no-prompt" not in sys.argv:
        selected_provider = select_model_provider()
        print(f"\nUsing provider: {selected_provider}")
    else:
        from ai_agent.utils.settings_manager import get_settings_manager
        settings_manager = get_settings_manager()
        selected_provider = settings_manager.get_preferred_provider()
        print(f"\nUsing saved provider preference: {selected_provider}")
    
    print(f"\nAI Agent executing: {instruction}")
    
    try:
        from ai_agent.user_interface.two_phase_app import TwoPhaseAIAgent
        
        # Update config with selected provider
        config_path = current_dir / "config.yaml"
        agent = TwoPhaseAIAgent(config_path=str(config_path) if config_path.exists() else None)
        
        # Update the vision client configuration with the selected provider
        if hasattr(agent, 'engine') and hasattr(agent.engine, 'model_runner'):
            model_runner = agent.engine.model_runner
            if hasattr(model_runner, 'vision_client'):
                # Update the vision client config
                from ai_agent.utils.settings_manager import get_settings_manager
                settings_manager = get_settings_manager()
                
                # Reload config with updated provider settings
                updated_config = model_runner.config.copy()
                updated_config['preferred_provider'] = selected_provider
                updated_config['google_api_key'] = settings_manager.get_google_api_key()
                updated_config['google_model'] = settings_manager.get_google_model()
                
                # Reinitialize vision client with updated config
                model_runner.vision_client.config = updated_config
        
        # Run the instruction
        options = {"debug": debug_mode}
        result = agent.run(instruction, options)
        
        if result:
            print("\n✓ Task completed successfully")
        else:
            print("\n✗ Task failed")
            sys.exit(1)
            
    except ImportError as e:
        print(f"Import error: {e}")
        print("This suggests a dependency issue. The virtual environment may not be set up correctly.")
        print("Try deleting the 'venv' directory and running again.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        if debug_mode:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
