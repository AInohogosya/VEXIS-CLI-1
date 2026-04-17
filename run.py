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
    print("VEXIS-CLI-2 AI Agent Runner")
    print("=" * 50)
    print("Usage: python3 run.py \"your instruction here\"")
    print()
    print("This script automatically handles:")
    print("  • Virtual environment creation and management")
    print("  • Dependency installation")
    print("  • Model selection (14 AI providers with model options)")
    print("    - Local: Ollama (privacy-focused)")
    print("    - Cloud: OpenAI, Anthropic, Google, xAI, Meta, Groq, DeepSeek, Together, Microsoft, Mistral, Amazon, Cohere, MiniMax")
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
    print("  🚀 MiniMax: M2-series models for productivity - Beta")
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
            # Check if signed in: returncode 0 AND output is not empty AND doesn't say "not signed in"
            output_combined = (result.stdout or "") + (result.stderr or "")
            is_signed_in = (result.returncode == 0 and
                           output_combined.strip() and
                           "not signed in" not in output_combined.lower())

            if is_signed_in:
                success_message("Ollama is signed in")
                return True, "full"
            else:
                warning_message("Ollama is available but you are not signed in.")
                print(f"{Colors.CYAN}Cloud models require signin. Local models will work.{Colors.RESET}")
                print(f"{Colors.CYAN}Run 'ollama signin' to enable cloud models.{Colors.RESET}")
                return True, "needs_signin"
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
    import tempfile
    
    print(f"{Colors.CYAN}Updating Ollama...{Colors.RESET}")
    try:
        # Create a temporary file for the install script
        with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as tmp_script:
            script_path = tmp_script.name
        
        try:
            # Step 1: Download the install script using curl (without shell)
            download_result = subprocess.run(
                ['curl', '-fsSL', 'https://ollama.com/install.sh'],
                capture_output=True,
                text=True,
                timeout=120
            )
            
            if download_result.returncode != 0:
                error_message(f"Failed to download Ollama install script: {download_result.stderr}")
                return False
            
            # Write script to temp file
            with open(script_path, 'w') as f:
                f.write(download_result.stdout)
            
            # Make script executable
            os.chmod(script_path, 0o755)
            
            # Step 2: Execute the downloaded script with bash (without shell=True)
            result = subprocess.run(
                ['bash', script_path],
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
        finally:
            # Clean up temp file
            try:
                os.unlink(script_path)
            except Exception:
                pass
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
            
            # API keys are not saved anymore - just return the key
            return api_key, False
            
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
        "cohere": ("Cohere Official API", model or settings_manager.get_cohere_model()),
        "minimax": ("MiniMax Official API", model or settings_manager.get_minimax_model())
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
        },
        "minimax": {
            "minimax-m2.7": "MiniMax M2.7 (Latest)",
            "minimax-m2.5": "MiniMax M2.5",
            "minimax-m2": "MiniMax M2 (Legacy)"
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
    
    # Set preferred provider to Ollama
    settings_manager.set_preferred_provider("ollama")
    
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
        "OpenRouter",
        "Access 300+ AI models via OpenRouter • Requires API key",
        "openrouter",
        "🔀"
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
    
    menu.add_item(
        "MiniMax (Beta)",
        "Use MiniMax's M2-series models • Requires API key",
        "minimax",
        "🚀"
    )
    
    menu.add_item(
        "Z.ai/ZhipuAI (Beta)",
        "Use Z.ai's GLM models • Requires API key • https://z.ai",
        "zhipuai",
        "🌐"
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
        return current_provider, settings_manager.get_model(current_provider) if current_provider else None
    
    
    # Handle provider selection
    if selected_provider == "ollama":
        result = configure_ollama_provider()
        if result is None:
            # Failed - show error and let user choose again explicitly
            from ai_agent.utils.interactive_menu import error_message
            error_message("Ollama configuration failed or was cancelled")
            print(f"\n{Colors.YELLOW}Press Enter to return to provider selection...{Colors.RESET}")
            input()
            return select_model_provider()
        ollama_model = settings_manager.get_ollama_model()
        show_config_summary("ollama", ollama_model)
        return "ollama", ollama_model
        
    elif selected_provider == "google":
        provider, model = configure_google_provider()
        if provider is None:
            # User cancelled API key entry - retry
            return select_model_provider()
        show_config_summary(provider, model)
        return provider, model
        
    elif selected_provider == "openrouter":
        provider, model = configure_generic_provider(selected_provider)
        if provider is None:
            # User cancelled API key entry - retry
            return select_model_provider()
        show_config_summary(provider, model)
        return provider, model
        
    elif selected_provider in ["openai", "anthropic", "xai", "meta", "groq", "deepseek", "together", "microsoft", "mistral", "amazon", "cohere", "minimax", "zhipuai"]:
        # Generic handler for all other providers
        provider, model = configure_generic_provider(selected_provider)
        if provider is None:
            # User cancelled API key entry - retry
            return select_model_provider()
        show_config_summary(provider, model)
        return provider, model

def configure_generic_provider(provider_name):
    """Generic configuration for cloud providers with arrow key model selection"""
    from ai_agent.utils.settings_manager import get_settings_manager
    from ai_agent.utils.interactive_menu import Colors, info_message, warning_message
    from ai_agent.utils.curses_menu import get_curses_menu
    
    settings_manager = get_settings_manager()
    
    # Provider-specific model options (verified real models only - 2025-2026 latest)
    provider_models = {
        "openai": [
            "gpt-5.4", "gpt-5.4-mini", "gpt-5.4-nano",
            "gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano",
            "o3", "o3-mini", "o4-mini",
        ],
        "anthropic": [
            "claude-opus-4-6-20260219",
            "claude-sonnet-4-6-20260219",
            "claude-opus-4-5-20251125",
            "claude-sonnet-4-5-20251125",
        ],
        "xai": ["grok-4.1", "grok-4.1-fast", "grok-4.1-thinking"],
        "meta": ["llama-4-scout-17b-16e-instruct", "llama-4-maverick-17b-128e-instruct"],
        "groq": ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"],
        "deepseek": ["deepseek-chat", "deepseek-coder", "deepseek-reasoner"],
        "together": ["meta-llama/Llama-4-Scout-17B-16E-Instruct", "meta-llama/Llama-4-Maverick-17B-128E-Instruct"],
        "microsoft": ["gpt-5.4", "gpt-5.4-mini", "gpt-4.1"],
        "mistral": ["mistral-large-latest", "mistral-medium-latest", "mistral-small-latest"],
        "amazon": ["anthropic.claude-opus-4-6-20260219-v1:0", "anthropic.claude-sonnet-4-6-20260219-v1:0"],
        "cohere": ["command-r-plus", "command-r", "command"],
        "minimax": ["MiniMax-Text-01", "abab6.5s"],
        "zhipuai": ["glm-5", "glm-5.1", "glm-4-plus", "glm-4"],
        "openrouter": [
            "openai/gpt-4o",
            "openai/gpt-4o-mini",
            "anthropic/claude-3.5-sonnet",
            "google/gemini-2.0-flash-exp",
            "meta-llama/llama-3.1-70b-instruct",
            "Other Models"
        ],
        "google": [
            "gemini-3.1-pro-preview",
            "gemini-3-flash-preview",
            "gemini-3.1-flash-lite-preview",
            "gemini-2.5-pro",
            "gemini-2.5-flash",
        ]
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
        "cohere": "COHERE_API_KEY",
        "minimax": "MINIMAX_API_KEY",
        "zhipuai": "ZHIPUAI_API_KEY",
        "openrouter": "OPENROUTER_API_KEY"
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
                # Special handling for OpenRouter "Other Models"
                if provider_name == "openrouter" and selected_model == "Other Models":
                    custom_model = get_custom_model_name()
                    if custom_model:
                        settings_manager.set_api_key(provider_name, existing_key)
                        settings_manager.set_model(provider_name, custom_model)
                        return provider_name, custom_model
                    else:
                        return None, None  # User cancelled custom model entry
                else:
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
        return None, None  # User cancelled selection
    
    # Special handling for OpenRouter "Other Models"
    if provider_name == "openrouter" and selected_model == "Other Models":
        custom_model = get_custom_model_name()
        if not custom_model:
            return None, None  # User cancelled custom model entry
        selected_model = custom_model
    
    # Save settings
    settings_manager.set_api_key(provider_name, api_key)
    settings_manager.set_model(provider_name, selected_model)
    settings_manager.set_preferred_provider(provider_name)

    print(f"{Colors.GREEN}✓ {provider_name.upper()} configured successfully!{Colors.RESET}")
    return provider_name, selected_model


def get_custom_model_name() -> Optional[str]:
    """Get custom model name from user for OpenRouter"""
    from ai_agent.utils.interactive_menu import Colors, info_message
    
    info_message("🔧 Enter Custom OpenRouter Model")
    print(f"{Colors.CYAN}You can use any official OpenRouter model name.{Colors.RESET}")
    print(f"{Colors.CYAN}Examples:{Colors.RESET}")
    print(f"  • openai/gpt-4o")
    print(f"  • anthropic/claude-3.5-sonnet") 
    print(f"  • meta-llama/llama-3.1-70b-instruct")
    print(f"  • google/gemini-2.0-flash-exp")
    print(f"  • deepseek/deepseek-r1")
    print(f"  • openrouter/auto (automatic model selection)")
    print(f"{Colors.YELLOW}Visit https://openrouter.ai/models for all available models{Colors.RESET}")
    print()
    
    while True:
        model_name = input(f"{Colors.WHITE}Enter model name (or 'cancel' to abort): {Colors.RESET}").strip()
        
        if model_name.lower() == 'cancel':
            print(f"{Colors.YELLOW}Cancelled custom model entry.{Colors.RESET}")
            return None
        
        if not model_name:
            print(f"{Colors.RED}Model name cannot be empty. Try again or type 'cancel'.{Colors.RESET}")
            continue
        
        # Basic validation
        if '/' not in model_name and model_name != "openrouter/auto":
            print(f"{Colors.YELLOW}Warning: Model names usually contain a provider prefix (e.g., 'openai/gpt-4o'){Colors.RESET}")
            confirm = input(f"Continue with '{model_name}'? (y/N): ").strip().lower()
            if confirm != 'y':
                continue
        
        print(f"{Colors.GREEN}✓ Using custom model: {model_name}{Colors.RESET}")
        return model_name


def select_model_with_arrows(provider_name: str, models: list) -> Optional[str]:
    """Select model using arrow keys in a curses menu with categorization"""
    from ai_agent.utils.curses_menu import get_curses_menu
    
    # Categorize models for OpenAI provider
    if provider_name.lower() == "openai":
        return select_openai_model_with_categories(models)
    
    menu = get_curses_menu(
        f"🤖 {provider_name.upper()} Model Selection",
        "Choose your preferred model using arrow keys:"
    )
    
    # Add models to menu with descriptions
    model_descriptions = {
        # GPT-5.4 Series (2026 Latest)
        "gpt-5.4": "GPT-5.4 • OpenAI flagship • 1M context • Best reasoning & coding",
        "gpt-5.4-mini": "GPT-5.4 Mini • Strong mini model • Coding & computer use",
        "gpt-5.4-nano": "GPT-5.4 Nano • Cheapest GPT-5.4 • High volume tasks",

        # GPT-4.1 Series
        "gpt-4.1": "GPT-4.1 • 1M context • Smarter & more efficient",
        "gpt-4.1-mini": "GPT-4.1 Mini • Fast & cost-effective",
        "gpt-4.1-nano": "GPT-4.1 Nano • Ultra-fast • Cheapest",

        # Reasoning Models
        "o3": "O3 • Advanced reasoning • STEM & complex tasks • 200K context",
        "o4-mini": "O4 Mini • Fast reasoning • Cost-effective • 200K context",
        "o3-mini": "O3 Mini • Efficient reasoning • 200K context",

        # Anthropic Claude 4.6 (Latest)
        "claude-opus-4-6-20260219": "Claude Opus 4.6 • Most capable • 1M context • Agent teams",
        "claude-sonnet-4-6-20260219": "Claude Sonnet 4.6 • Near-Opus performance • Balanced",
        "claude-opus-4-5-20251125": "Claude Opus 4.5 • Outperforms humans on coding exams",
        "claude-sonnet-4-5-20251125": "Claude Sonnet 4.5 • Efficient & capable",

        # Google Gemini 3.1 (Latest)
        "gemini-3.1-pro-preview": "Gemini 3.1 Pro • 2M context • Advanced agentic coding",
        "gemini-3-flash-preview": "Gemini 3 Flash • Frontier performance • Cost-effective",
        "gemini-3.1-flash-lite-preview": "Gemini 3.1 Flash Lite • Ultra-efficient • New",
        "gemini-2.5-pro": "Gemini 2.5 Pro • 1M context • Advanced reasoning",
        "gemini-2.5-flash": "Gemini 2.5 Flash • Fast & efficient",

        # xAI Grok 4.1 (Latest)
        "grok-4.1": "Grok 4.1 • State-of-the-art • #1 on LMArena • Real-time",
        "grok-4.1-fast": "Grok 4.1 Fast • Quick responses • Dec 2025",
        "grok-4.1-thinking": "Grok 4.1 Thinking • Deep reasoning mode",

        # Meta Llama 4 (Latest)
        "llama-4-scout-17b-16e-instruct": "Llama 4 Scout • 10M context • 17B active • Vision",
        "llama-4-maverick-17b-128e-instruct": "Llama 4 Maverick • 1M context • 128 experts • Vision",

        # Together AI Llama 4
        "meta-llama/Llama-4-Scout-17B-16E-Instruct": "Llama 4 Scout • Together hosted • 10M context",
        "meta-llama/Llama-4-Maverick-17B-128E-Instruct": "Llama 4 Maverick • Together hosted • 1M context",

        # DeepSeek
        "deepseek-chat": "DeepSeek Chat • General conversation",
        "deepseek-coder": "DeepSeek Coder • Code generation specialist",
        "deepseek-reasoner": "DeepSeek Reasoner • Advanced reasoning",

        # Groq
        "llama-3.3-70b-versatile": "Llama 3.3 70B • Groq hosted • Ultra-fast",
        "llama-3.1-8b-instant": "Llama 3.1 8B • Groq hosted • Low latency",
        "mixtral-8x7b-32768": "Mixtral 8x7B • Groq hosted • MoE architecture",

        # Mistral
        "mistral-large-latest": "Mistral Large • Latest version • Strong capabilities",
        "mistral-medium-latest": "Mistral Medium • Balanced performance",
        "mistral-small-latest": "Mistral Small • Fast & efficient",

        # Cohere
        "command-r-plus": "Command R+ • Cohere's best • Long context",
        "command-r": "Command R • Balanced performance",
        "command": "Command • Legacy Cohere model",

        # Zhipu AI (GLM)
        "glm-5": "GLM-5 • Zhipu AI latest • 744B parameters • Advanced coding",
        "glm-5.1": "GLM-5.1 • Zhipu AI enhanced • Feb 2026 release",
        "glm-4-plus": "GLM-4 Plus • Strong general performance",
        "glm-4": "GLM-4 • Base model • Capable generalist",

        # MiniMax
        "MiniMax-Text-01": "MiniMax Text-01 • Latest general model",
        "abab6.5s": "ABAB 6.5S • MiniMax chat model",
    }
    
    # Add each model to the menu
    for model in models:
        description = model_descriptions.get(model, f"{model} • Standard model")
        if "new" in description.lower():
            icon = "✨"  # Special icon for new models
        elif "latest" in description.lower() or "newest" in description.lower():
            icon = "🚀"
        else:
            icon = "🧠"
        menu.add_item(model, description, model, icon)
    
    selected_model = menu.show()
    return selected_model


def select_openai_model_with_categories(models: list) -> Optional[str]:
    """Select OpenAI model using categorized menu"""
    from ai_agent.utils.curses_menu import get_curses_menu
    
    menu = get_curses_menu(
        "🤖 OpenAI Model Selection",
        "Choose your preferred OpenAI model:"
    )
    
    # Separate models by category
    latest_models = []
    legacy_models = []
    
    for model in models:
        if model in ["gpt-5.4", "gpt-5.4-mini (New)", "gpt-5.4-nano (New)", "gpt-5.4-pro", "gpt-5.3-codex", "gpt-oss-20b", "gpt-oss-120b"]:
            latest_models.append(model)
        else:
            legacy_models.append(model)
    
    # Debug: Print model categorization
    print(f"\n[DEBUG] Total models: {len(models)}")
    print(f"[DEBUG] Latest models: {len(latest_models)}")
    print(f"[DEBUG] Legacy models: {len(legacy_models)}")
    
    # Add latest models directly to menu (no category)
    for model in latest_models:
        description = get_model_description(model)
        if "new" in description.lower():
            icon = "✨"  # Special icon for new models
        elif "latest" in description.lower() or "newest" in description.lower():
            icon = "🚀"
        else:
            icon = "🧠"
        menu.add_item(model, description, model, icon)
    
    # Add Legacy Models category (all legacy models in one category)
    if legacy_models:
        menu.add_item(
            "� Legacy Models",
            f"Older models organized by type ({len(legacy_models)} models)",
            "category_legacy",
            "�"
        )
    
    selected_category = menu.show()
    
    if selected_category == "category_legacy":
        return show_models_with_subcategories("Legacy Models", legacy_models, "📚")
    elif selected_category in latest_models:
        return selected_category
    else:
        return None


def get_model_description(model: str) -> str:
    """Get description for a specific model"""
    model_descriptions = {
        # GPT-5.4 Series (2026 Latest)
        "gpt-5.4": "GPT-5.4 • OpenAI flagship • 1M context • Best reasoning & coding",
        "gpt-5.4-mini": "GPT-5.4 Mini • Strong mini model • Coding & computer use",
        "gpt-5.4-nano": "GPT-5.4 Nano • Cheapest GPT-5.4 • High volume tasks",

        # GPT-4.1 Series
        "gpt-4.1": "GPT-4.1 • 1M context • Smarter & more efficient",
        "gpt-4.1-mini": "GPT-4.1 Mini • Fast & cost-effective",
        "gpt-4.1-nano": "GPT-4.1 Nano • Ultra-fast • Cheapest",

        # Reasoning Models
        "o3": "O3 • Advanced reasoning • STEM & complex tasks • 200K context",
        "o4-mini": "O4 Mini • Fast reasoning • Cost-effective • 200K context",
        "o3-mini": "O3 Mini • Efficient reasoning • 200K context",

        # Anthropic Claude 4.6 (Latest)
        "claude-opus-4-6-20260219": "Claude Opus 4.6 • Most capable • 1M context • Agent teams",
        "claude-sonnet-4-6-20260219": "Claude Sonnet 4.6 • Near-Opus performance • Balanced",
        "claude-opus-4-5-20251125": "Claude Opus 4.5 • Outperforms humans on coding exams",
        "claude-sonnet-4-5-20251125": "Claude Sonnet 4.5 • Efficient & capable",

        # Google Gemini 3.1 (Latest)
        "gemini-3.1-pro-preview": "Gemini 3.1 Pro • 2M context • Advanced agentic coding",
        "gemini-3-flash-preview": "Gemini 3 Flash • Frontier performance • Cost-effective",
        "gemini-3.1-flash-lite-preview": "Gemini 3.1 Flash Lite • Ultra-efficient • New",
        "gemini-2.5-pro": "Gemini 2.5 Pro • 1M context • Advanced reasoning",
        "gemini-2.5-flash": "Gemini 2.5 Flash • Fast & efficient",

        # xAI Grok 4.1 (Latest)
        "grok-4.1": "Grok 4.1 • State-of-the-art • #1 on LMArena • Real-time",
        "grok-4.1-fast": "Grok 4.1 Fast • Quick responses • Dec 2025",
        "grok-4.1-thinking": "Grok 4.1 Thinking • Deep reasoning mode",

        # Meta Llama 4 (Latest)
        "llama-4-scout-17b-16e-instruct": "Llama 4 Scout • 10M context • 17B active • Vision",
        "llama-4-maverick-17b-128e-instruct": "Llama 4 Maverick • 1M context • 128 experts • Vision",

        # Together AI Llama 4
        "meta-llama/Llama-4-Scout-17B-16E-Instruct": "Llama 4 Scout • Together hosted • 10M context",
        "meta-llama/Llama-4-Maverick-17B-128E-Instruct": "Llama 4 Maverick • Together hosted • 1M context",

        # DeepSeek
        "deepseek-chat": "DeepSeek Chat • General conversation",
        "deepseek-coder": "DeepSeek Coder • Code generation specialist",
        "deepseek-reasoner": "DeepSeek Reasoner • Advanced reasoning",

        # Groq
        "llama-3.3-70b-versatile": "Llama 3.3 70B • Groq hosted • Ultra-fast",
        "llama-3.1-8b-instant": "Llama 3.1 8B • Groq hosted • Low latency",
        "mixtral-8x7b-32768": "Mixtral 8x7B • Groq hosted • MoE architecture",

        # Mistral
        "mistral-large-latest": "Mistral Large • Latest version • Strong capabilities",
        "mistral-medium-latest": "Mistral Medium • Balanced performance",
        "mistral-small-latest": "Mistral Small • Fast & efficient",

        # Cohere
        "command-r-plus": "Command R+ • Cohere's best • Long context",
        "command-r": "Command R • Balanced performance",
        "command": "Command • Legacy Cohere model",

        # Zhipu AI (GLM)
        "glm-5": "GLM-5 • Zhipu AI latest • 744B parameters • Advanced coding",
        "glm-5.1": "GLM-5.1 • Zhipu AI enhanced • Feb 2026 release",
        "glm-4-plus": "GLM-4 Plus • Strong general performance",
        "glm-4": "GLM-4 • Base model • Capable generalist",

        # MiniMax
        "MiniMax-Text-01": "MiniMax Text-01 • Latest general model",
        "abab6.5s": "ABAB 6.5S • MiniMax chat model",

        # OpenRouter Custom Models
        "Other Models": "🔧 Enter custom model name • Any official OpenRouter model",
    }

    return model_descriptions.get(model, f"{model} • Standard model")


def show_models_in_category(category_name: str, models: list, category_icon: str) -> Optional[str]:
    """Show models within a specific category with sub-categorization"""
    from ai_agent.utils.curses_menu import get_curses_menu
    
    # For legacy categories, further subdivide by generation
    if category_name in ["O Series Models", "GPT Series Models"]:
        return show_models_with_subcategories(category_name, models, category_icon)
    
    menu = get_curses_menu(
        f"{category_icon} {category_name}",
        "Select your preferred model:"
    )
    
    # Model descriptions for OpenAI models
    model_descriptions = {
        # GPT-5.4 Series (2026 Latest)
        "gpt-5.4": "GPT-5.4 • OpenAI flagship • 1M context • Best reasoning & coding",
        "gpt-5.4-mini": "GPT-5.4 Mini • Strong mini model • Coding & computer use",
        "gpt-5.4-nano": "GPT-5.4 Nano • Cheapest GPT-5.4 • High volume tasks",

        # GPT-4.1 Series
        "gpt-4.1": "GPT-4.1 • 1M context • Smarter & more efficient",
        "gpt-4.1-mini": "GPT-4.1 Mini • Fast & cost-effective",
        "gpt-4.1-nano": "GPT-4.1 Nano • Ultra-fast • Cheapest",

        # Reasoning Models
        "o3": "O3 • Advanced reasoning • STEM & complex tasks • 200K context",
        "o4-mini": "O4 Mini • Fast reasoning • Cost-effective • 200K context",
        "o3-mini": "O3 Mini • Efficient reasoning • 200K context",
    }
    
    # Add models to menu
    for model in models:
        description = model_descriptions.get(model, f"{model} • Standard model")
        if "new" in description.lower():
            icon = "✨"  # Special icon for new models
        elif "latest" in description.lower() or "newest" in description.lower():
            icon = "🚀"
        else:
            icon = "🧠"
        menu.add_item(model, description, model, icon)
    
    selected_model = menu.show()
    return selected_model


def show_models_with_subcategories(category_name: str, models: list, category_icon: str) -> Optional[str]:
    """Show models with subcategories for Legacy Models"""
    from ai_agent.utils.curses_menu import get_curses_menu
    
    menu = get_curses_menu(
        f"{category_icon} {category_name}",
        "Choose model type:"
    )
    
    # Subdivide Legacy Models by type
    o_series_models = [m for m in models if m.startswith("o") and not m.startswith("omni")]
    gpt_series_models = [m for m in models if m.startswith("gpt") and not m.startswith("omni")]
    codex_models = [m for m in models if "codex" in m]
    other_models = [m for m in models if not (m.startswith("o") and not m.startswith("omni")) and not m.startswith("gpt") and "codex" not in m]
    
    if o_series_models:
        menu.add_item(
            "🧠 O Series Models",
            f"O1, O3, O4 reasoning models ({len(o_series_models)} models)",
            "subcategory_o_series",
            "🧠"
        )
    if gpt_series_models:
        menu.add_item(
            "💬 GPT Series Models",
            f"GPT-3, GPT-4, GPT-5 legacy models ({len(gpt_series_models)} models)",
            "subcategory_gpt_series",
            "💬"
        )
    if codex_models:
        menu.add_item(
            "💻 Codex Models",
            f"Code generation models ({len(codex_models)} models)",
            "subcategory_codex",
            "💻"
        )
    if other_models:
        menu.add_item(
            "🔧 Other Models",
            f"Specialized and utility models ({len(other_models)} models)",
            "subcategory_other",
            "🔧"
        )
    
    selected_subcategory = menu.show()
    
    if selected_subcategory == "subcategory_o_series":
        return show_o_series_subcategories(o_series_models)
    elif selected_subcategory == "subcategory_gpt_series":
        return show_gpt_series_subcategories(gpt_series_models)
    elif selected_subcategory == "subcategory_codex":
        return show_models_in_category("Codex Models", codex_models, "💻")
    elif selected_subcategory == "subcategory_other":
        return show_models_in_category("Other Models", other_models, "🔧")
    else:
        return None


def show_o_series_subcategories(models: list) -> Optional[str]:
    """Show O Series models subdivided by generation"""
    from ai_agent.utils.curses_menu import get_curses_menu
    
    menu = get_curses_menu(
        "🧠 O Series Models",
        "Choose O Series generation:"
    )
    
    o1_models = [m for m in models if m.startswith("o1")]
    o3_models = [m for m in models if m.startswith("o3")]
    o4_models = [m for m in models if m.startswith("o4")]
    
    if o1_models:
        menu.add_item(
            "🔹 O1 Series",
            f"First generation reasoning models ({len(o1_models)} models)",
            "subcategory_o1",
            "🔹"
        )
    if o3_models:
        menu.add_item(
            "🔹 O3 Series",
            f"Advanced reasoning models ({len(o3_models)} models)",
            "subcategory_o3",
            "🔹"
        )
    if o4_models:
        menu.add_item(
            "🔹 O4 Series",
            f"Next generation reasoning models ({len(o4_models)} models)",
            "subcategory_o4",
            "🔹"
        )
    
    selected_subcategory = menu.show()
    
    if selected_subcategory == "subcategory_o1":
        return show_models_in_category("O1 Series", o1_models, "🔹")
    elif selected_subcategory == "subcategory_o3":
        return show_models_in_category("O3 Series", o3_models, "🔹")
    elif selected_subcategory == "subcategory_o4":
        return show_models_in_category("O4 Series", o4_models, "🔹")
    else:
        return None


def show_gpt_series_subcategories(models: list) -> Optional[str]:
    """Show GPT Series models subdivided by generation"""
    from ai_agent.utils.curses_menu import get_curses_menu
    
    menu = get_curses_menu(
        "💬 GPT Series Models",
        "Choose GPT Series generation:"
    )
    
    gpt3_models = [m for m in models if "gpt-3.5" in m or (m.startswith("gpt-3") and "3.5" not in m)]
    gpt4_models = [m for m in models if "gpt-4" in m]
    gpt5_legacy_models = [m for m in models if "gpt-5" in m and m not in ["gpt-5.4", "gpt-5.4-mini (New)", "gpt-5.4-nano (New)", "gpt-5.4-pro", "gpt-5.3-codex"]]
    
    if gpt3_models:
        menu.add_item(
            "🔹 GPT-3 Series",
            f"Third generation models ({len(gpt3_models)} models)",
            "subcategory_gpt3",
            "🔹"
        )
    if gpt4_models:
        menu.add_item(
            "🔹 GPT-4 Series",
            f"Fourth generation models ({len(gpt4_models)} models)",
            "subcategory_gpt4",
            "🔹"
        )
    if gpt5_legacy_models:
        menu.add_item(
            "🔹 GPT-5 Legacy",
            f"Fifth generation legacy models ({len(gpt5_legacy_models)} models)",
            "subcategory_gpt5",
            "🔹"
        )
    
    selected_subcategory = menu.show()
    
    if selected_subcategory == "subcategory_gpt3":
        return show_models_in_category("GPT-3 Series", gpt3_models, "🔹")
    elif selected_subcategory == "subcategory_gpt4":
        return show_models_in_category("GPT-4 Series", gpt4_models, "🔹")
    elif selected_subcategory == "subcategory_gpt5":
        return show_models_in_category("GPT-5 Legacy", gpt5_legacy_models, "🔹")
    else:
        return None


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
        print("VEXIS-CLI-2 - AI-Powered Command Line Assistant")
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
        print("  --max-iterations  Maximum Phase 2-4 iterations (default: 10)")
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
    selected_provider = None
    selected_model = None
    
    if "--no-prompt" not in sys.argv:
        result = select_model_provider()
        if isinstance(result, tuple) and len(result) == 2:
            selected_provider, selected_model = result
        else:
            selected_provider = result
        print(f"\nUsing provider: {selected_provider}")
        if selected_model:
            print(f"Using model: {selected_model}")
    else:
        from ai_agent.utils.settings_manager import get_settings_manager
        settings_manager = get_settings_manager()
        selected_provider = settings_manager.get_preferred_provider()
        selected_model = settings_manager.get_model(selected_provider) if selected_provider else None
        print(f"\nUsing saved provider preference: {selected_provider}")
        if selected_model:
            print(f"Using saved model: {selected_model}")
    
    print(f"\nAI Agent executing: {instruction}")
    
    max_iterations = 10
    
    # Parse max-iterations if provided
    if "--max-iterations" in sys.argv:
        try:
            idx = sys.argv.index("--max-iterations")
            if idx + 1 < len(sys.argv):
                max_iterations = int(sys.argv[idx + 1])
        except (ValueError, IndexError):
            pass
    
    try:
        from ai_agent.user_interface.five_phase_app import FivePhaseAIAgent
        
        # Create agent with selected provider and model
        config_path = current_dir / "config.yaml"
        agent = FivePhaseAIAgent(
            provider=selected_provider,
            model=selected_model,
            config_path=str(config_path) if config_path.exists() else None
        )
        
        # Run the instruction with 5-phase options
        options = {
            "debug": debug_mode,
            "max_iterations": max_iterations,
            "command_timeout": 30,
            "task_timeout": 300
        }
        result = agent.run(instruction, options)
        
        if result == 0:
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
