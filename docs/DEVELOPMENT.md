# Development Guide

## Overview

This guide covers development setup, coding standards, testing practices, and contribution guidelines for VEXIS-CLI-1.2.

## Development Environment Setup

### Prerequisites

- **Python 3.9+**
- **Git**
- **Ollama** (for local model testing)
- **Google AI API Key** (for Gemini testing)

### Initial Setup

```bash
# Clone the repository
git clone https://github.com/vexis-project/VEXIS-CLI-1.2.git
cd VEXIS-CLI-1.2

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
pip install -e .  # Install in development mode

# Install development dependencies
pip install -r requirements-dev.txt
```

### Development Dependencies

**File**: `requirements-dev.txt`

```txt
# Testing
pytest>=7.0.0
pytest-asyncio>=0.21.0
pytest-cov>=4.0.0
pytest-mock>=3.10.0

# Code quality
black>=23.0.0
flake8>=6.0.0
isort>=5.12.0
mypy>=1.0.0

# Documentation
sphinx>=6.0.0
sphinx-rtd-theme>=1.2.0

# Development tools
pre-commit>=3.0.0
tox>=4.0.0
```

### IDE Configuration

#### VS Code

**File**: `.vscode/settings.json`

```json
{
    "python.defaultInterpreterPath": "./venv/bin/python",
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "python.linting.mypyEnabled": true,
    "python.formatting.provider": "black",
    "python.sortImports.args": ["--profile", "black"],
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
        "source.organizeImports": true
    }
}
```

**File**: `.vscode/launch.json`

```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Debug VEXIS CLI",
            "type": "python",
            "request": "launch",
            "program": "run.py",
            "args": ["test instruction", "--debug"],
            "console": "integratedTerminal",
            "cwd": "${workspaceFolder}"
        }
    ]
}
```

#### PyCharm

- Set Python interpreter to `./venv/bin/python`
- Enable Black formatter
- Configure Flake8 linter
- Set up pytest runner

## Project Structure

```
VEXIS-CLI-1.2/
├── src/
│   └── ai_agent/
│       ├── core_processing/          # Core logic
│       │   ├── two_phase_engine.py
│       │   ├── command_parser.py
│       │   └── task_verifier.py
│       ├── external_integration/     # External APIs
│       │   ├── model_runner.py
│       │   ├── vision_api_client.py
│       │   └── ollama_client.py
│       ├── platform_abstraction/    # Platform-specific code
│       │   ├── macos_handler.py
│       │   ├── linux_handler.py
│       │   └── windows_handler.py
│       ├── utils/                    # Utilities
│       │   ├── logger.py
│       │   ├── config.py
│       │   ├── exceptions.py
│       │   └── ollama_error_handler.py
│       ├── user_interface/          # User interface
│       │   ├── main_app.py
│       │   ├── two_phase_app.py
│       │   └── yellow_selection/
│       └── __init__.py
├── tests/                           # Test suite
│   ├── unit/
│   ├── integration/
│   └── fixtures/
├── docs/                           # Documentation
├── scripts/                        # Utility scripts
├── config.yaml                     # Configuration
├── requirements.txt                # Dependencies
├── pyproject.toml                  # Project metadata
└── run.py                         # Main entry point
```

## Coding Standards

### Python Style Guide

Follow **PEP 8** with these additional rules:

#### Code Formatting

Use **Black** with default settings:

```bash
# Format code
black src/ tests/

# Check formatting
black --check src/ tests/
```

#### Import Sorting

Use **isort** with Black profile:

```bash
# Sort imports
isort --profile black src/ tests/

# Check imports
isort --profile black --check-only src/ tests/
```

#### Linting

Use **Flake8**:

```bash
# Lint code
flake8 src/ tests/

# Configuration in setup.cfg
[flake8]
max-line-length = 88
extend-ignore = E203, W503
exclude = .git,__pycache__,docs/source/conf.py,old,build,dist
```

#### Type Checking

Use **MyPy**:

```bash
# Type check
mypy src/

# Configuration in mypy.ini
[mypy]
python_version = 3.9
warn_return_any = True
warn_unused_configs = True
disallow_untyped_defs = True
```

### Naming Conventions

#### Files and Directories

- **snake_case** for Python files: `command_parser.py`
- **snake_case** for directories: `core_processing/`

#### Classes and Functions

```python
# Classes: PascalCase
class TwoPhaseEngine:
    pass

# Functions: snake_case
def process_instruction(instruction: str) -> Result:
    pass

# Constants: UPPER_SNAKE_CASE
MAX_RETRIES = 3
DEFAULT_TIMEOUT = 120
```

#### Variables

```python
# Local variables: snake_case
user_input = "example"
model_response = "response"

# Private variables: prefix with underscore
self._internal_state = {}
self._helper_method()

# Type hints required
def process_data(data: Dict[str, Any]) -> List[str]:
    return list(data.keys())
```

### Documentation Standards

#### Docstring Format

Use **Google Style** docstrings:

```python
def process_instruction(
    instruction: str, 
    provider: Optional[str] = None,
    debug: bool = False
) -> CommandResult:
    """Process a natural language instruction.
    
    Args:
        instruction: Natural language instruction to process
        provider: AI provider to use (ollama or google)
        debug: Enable debug logging
        
    Returns:
        CommandResult: Result of instruction processing
        
    Raises:
        ValidationError: If instruction is invalid
        ProviderError: If AI provider fails
        
    Example:
        >>> result = process_instruction("list files")
        >>> print(result.success)
        True
    """
    pass
```

#### Module Documentation

Each module should have a module-level docstring:

```python
"""
Command Parser Module

Converts natural language instructions into executable CLI commands.
Provides validation, sanitization, and security checking for all commands.
"""

import re
from typing import List, Dict, Any
from ..utils.exceptions import ValidationError
```

#### Comments

```python
# Single-line comments explain complex logic
if len(commands) > MAX_COMMANDS:
    # Limit command count to prevent resource exhaustion
    commands = commands[:MAX_COMMANDS]

# Multi-line comments for complex algorithms
# This algorithm uses a two-phase approach:
# 1. Parse the natural language into structured commands
# 2. Validate and optimize the command sequence
# This ensures both accuracy and performance
```

## Testing

### Test Structure

```
tests/
├── unit/                           # Unit tests
│   ├── test_command_parser.py
│   ├── test_model_runner.py
│   └── test_error_handler.py
├── integration/                    # Integration tests
│   ├── test_ollama_integration.py
│   ├── test_gemini_integration.py
│   └── test_full_workflow.py
├── fixtures/                       # Test data
│   ├── sample_instructions.yaml
│   └── mock_responses.json
└── conftest.py                     # Pytest configuration
```

### Unit Testing

#### Test Structure

```python
import pytest
from unittest.mock import Mock, patch
from ai_agent.core_processing.command_parser import CommandParser

class TestCommandParser:
    """Test cases for CommandParser class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.parser = CommandParser()
    
    def test_parse_simple_instruction(self):
        """Test parsing simple file listing instruction."""
        instruction = "list files in current directory"
        result = self.parser.parse(instruction)
        
        assert result.success is True
        assert len(result.commands) == 1
        assert result.commands[0].executable == "ls"
    
    def test_parse_invalid_instruction(self):
        """Test parsing invalid instruction raises error."""
        instruction = ""  # Empty instruction
        
        with pytest.raises(ValidationError):
            self.parser.parse(instruction)
    
    @patch('ai_agent.core_processing.command_parser.subprocess.run')
    def test_command_execution(self, mock_run):
        """Test command execution with mocked subprocess."""
        mock_run.return_value = Mock(returncode=0, stdout="file1\nfile2\n")
        
        result = self.parser.execute_command("ls")
        
        assert result.success is True
        assert "file1" in result.output
```

#### Fixtures

```python
# conftest.py
import pytest
from unittest.mock import Mock

@pytest.fixture
def mock_ollama_response():
    """Mock Ollama API response."""
    return {
        "response": "ls -la",
        "done": True,
        "model": "gemini-3-flash-preview"
    }

@pytest.fixture
def sample_config():
    """Sample configuration for testing."""
    return {
        "api": {
            "preferred_provider": "ollama",
            "local_endpoint": "http://localhost:11434"
        },
        "execution": {
            "safety_mode": True,
            "dry_run": False
        }
    }
```

### Integration Testing

#### Provider Integration

```python
import pytest
from ai_agent.external_integration.model_runner import ModelRunner

class TestModelRunnerIntegration:
    """Integration tests for ModelRunner."""
    
    @pytest.mark.slow
    def test_ollama_integration(self):
        """Test actual Ollama integration."""
        runner = ModelRunner()
        
        # Skip if Ollama not available
        if not runner.is_ollama_available():
            pytest.skip("Ollama not available")
        
        result = runner.generate_response(
            task_type="task_generation",
            prompt="Generate a simple ls command",
            provider="ollama"
        )
        
        assert result.success is True
        assert len(result.content) > 0
    
    @pytest.mark.internet
    def test_gemini_integration(self):
        """Test actual Gemini integration."""
        runner = ModelRunner()
        
        # Skip if API key not available
        if not runner.has_gemini_key():
            pytest.skip("Gemini API key not available")
        
        result = runner.generate_response(
            task_type="task_generation",
            prompt="Generate a simple ls command",
            provider="google"
        )
        
        assert result.success is True
        assert len(result.content) > 0
```

### Test Configuration

#### Pytest Configuration

**File**: `pytest.ini`

```ini
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = 
    --verbose
    --tb=short
    --cov=src/ai_agent
    --cov-report=html
    --cov-report=term-missing
    --cov-fail-under=80
markers =
    slow: marks tests as slow (deselect with '-m "not slow"')
    internet: marks tests that require internet
    integration: marks tests as integration tests
```

#### Test Markers

```python
@pytest.mark.slow
def test_large_model_processing():
    """Test that takes a long time."""
    pass

@pytest.mark.internet
def test_api_connectivity():
    """Test that requires internet connection."""
    pass

@pytest.mark.integration
def test_full_workflow():
    """Test complete workflow integration."""
    pass
```

### Running Tests

#### Basic Test Commands

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/unit/test_command_parser.py

# Run with coverage
pytest --cov=src/ai_agent --cov-report=html

# Run only fast tests
pytest -m "not slow"

# Run integration tests
pytest -m integration

# Run with verbose output
pytest -v
```

#### Continuous Integration

**File**: `.github/workflows/test.yml`

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.9, 3.10, 3.11]
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install -r requirements-dev.txt
    
    - name: Run tests
      run: |
        pytest --cov=src/ai_agent --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
```

## Development Workflow

### Pre-commit Hooks

**File**: `.pre-commit-config.yaml`

```yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.1.0
    hooks:
      - id: black
        language_version: python3.9

  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort
        args: [--profile, black]

  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.0.1
    hooks:
      - id: mypy
        additional_dependencies: [types-all]
```

#### Setup Pre-commit

```bash
# Install pre-commit
pip install pre-commit

# Install hooks
pre-commit install

# Run on all files
pre-commit run --all-files
```

### Git Workflow

#### Branch Strategy

- **main**: Stable production code
- **develop**: Integration branch
- **feature/***: Feature development
- **bugfix/***: Bug fixes
- **hotfix/***: Critical fixes

#### Commit Messages

Use **Conventional Commits** format:

```
type(scope): description

[optional body]

[optional footer]
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `style`: Code style
- `refactor`: Code refactoring
- `test`: Test changes
- `chore`: Maintenance

**Examples:**
```
feat(parser): add support for nested commands

fix(ollama): handle connection timeout errors

docs(readme): update installation instructions

test(integration): add Gemini API tests
```

### Development Commands

#### Common Development Tasks

```bash
# Format code
black src/ tests/
isort src/ tests/

# Lint code
flake8 src/ tests/
mypy src/

# Run tests
pytest
pytest --cov=src/ai_agent

# Build package
python -m build

# Install in development mode
pip install -e .

# Run with debug
python run.py "test instruction" --debug

# Check configuration
python run.py --check-config
```

#### Debugging

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
python run.py "instruction" --debug

# Use Python debugger
python -m pdb run.py "instruction"

# Profile performance
python -m cProfile -o profile.stats run.py "instruction"
python -c "import pstats; pstats.Stats('profile.stats').sort_stats('cumulative').print_stats(10)"
```

## Adding New Features

### Feature Development Process

1. **Create Feature Branch**
   ```bash
   git checkout -b feature/new-feature-name
   ```

2. **Implement Feature**
   - Write code following standards
   - Add comprehensive tests
   - Update documentation

3. **Test Feature**
   ```bash
   pytest tests/unit/test_new_feature.py
   pytest -m integration
   ```

4. **Quality Checks**
   ```bash
   black src/ tests/
   flake8 src/ tests/
   mypy src/
   pre-commit run --all-files
   ```

5. **Update Documentation**
   - Add to relevant docs
   - Update README if needed
   - Add examples

6. **Submit Pull Request**
   - Create descriptive PR
   - Request code review
   - Address feedback

### Adding New AI Providers

#### Provider Interface

```python
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

class AIProvider(ABC):
    """Abstract base class for AI providers."""
    
    @abstractmethod
    async def generate_response(
        self, 
        prompt: str, 
        task_type: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate response from AI model."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if provider is available."""
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get provider model information."""
        pass
```

#### Implementation Example

```python
class CustomProvider(AIProvider):
    """Custom AI provider implementation."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.client = self._initialize_client()
    
    async def generate_response(
        self, 
        prompt: str, 
        task_type: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate response using custom API."""
        try:
            response = await self.client.generate(
                prompt=prompt,
                model=self.config.get("model", "default"),
                **kwargs
            )
            
            return {
                "success": True,
                "content": response["text"],
                "model": self.config.get("model"),
                "tokens_used": response.get("tokens")
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__
            }
    
    def is_available(self) -> bool:
        """Check if custom provider is available."""
        try:
            return self.client.health_check()
        except:
            return False
```

### Adding New Commands

#### Command Registration

```python
from ai_agent.core_processing.command_registry import register_command

@register_command(
    name="custom_operation",
    description="Perform custom operation",
    category="custom",
    safety_level="medium"
)
def handle_custom_operation(args: List[str]) -> CommandResult:
    """Handle custom operation command."""
    try:
        # Implement custom logic
        result = perform_custom_operation(args)
        
        return CommandResult(
            success=True,
            output=result,
            execution_time=0.1
        )
        
    except Exception as e:
        return CommandResult(
            success=False,
            error=str(e),
            execution_time=0.1
        )
```

#### Command Validation

```python
def validate_custom_command(command: str) -> ValidationResult:
    """Validate custom command syntax."""
    
    # Check basic syntax
    if not command.startswith("custom"):
        return ValidationResult(False, "Invalid command prefix")
    
    # Parse arguments
    args = command.split()[1:]
    if len(args) < 1:
        return ValidationResult(False, "Missing required argument")
    
    # Validate arguments
    if args[0] not in ["create", "delete", "list"]:
        return ValidationResult(False, "Invalid operation")
    
    return ValidationResult(True, "Command is valid")
```

## Performance Optimization

### Profiling

#### Memory Profiling

```python
import memory_profiler

@memory_profiler.profile
def process_large_instruction(instruction: str):
    """Profile memory usage of instruction processing."""
    # Implementation
    pass

# Run profiling
python -m memory_profiler script.py
```

#### CPU Profiling

```python
import cProfile
import pstats

def profile_function():
    """Profile CPU usage."""
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Function to profile
    process_instruction("test instruction")
    
    profiler.disable()
    
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(10)
```

### Optimization Strategies

#### Caching

```python
from functools import lru_cache
import hashlib

class ModelRunner:
    @lru_cache(maxsize=128)
    def _cached_model_response(
        self, 
        prompt_hash: str, 
        model: str
    ) -> str:
        """Cache model responses."""
        # Implementation
        pass
    
    def generate_response(self, prompt: str, **kwargs):
        """Generate response with caching."""
        prompt_hash = hashlib.md5(prompt.encode()).hexdigest()
        return self._cached_model_response(prompt_hash, kwargs.get("model"))
```

#### Async Operations

```python
import asyncio
import aiohttp

class AsyncModelRunner:
    async def batch_generate(
        self, 
        prompts: List[str], 
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Generate responses for multiple prompts concurrently."""
        
        tasks = [
            self.generate_response(prompt, **kwargs) 
            for prompt in prompts
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return [
            result if not isinstance(result, Exception)
            else {"success": False, "error": str(result)}
            for result in results
        ]
```

## Debugging

### Logging Configuration

#### Debug Logging

```python
import logging
import sys

def setup_debug_logging():
    """Set up detailed debug logging."""
    
    # Create logger
    logger = logging.getLogger("ai_agent")
    logger.setLevel(logging.DEBUG)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    
    # File handler
    file_handler = logging.FileHandler("debug.log")
    file_handler.setLevel(logging.DEBUG)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger
```

#### Debug Utilities

```python
import traceback
from typing import Any

def debug_print(obj: Any, label: str = "Debug"):
    """Print debug information with context."""
    print(f"=== {label} ===")
    print(f"Type: {type(obj)}")
    print(f"Value: {obj}")
    print(f"Repr: {repr(obj)}")
    print("=" * (len(label) + 8))
    print()

def debug_trace():
    """Print current stack trace."""
    print("=== Stack Trace ===")
    traceback.print_stack()
    print("=" * 17)
    print()

def debug_vars():
    """Print all local variables."""
    import inspect
    frame = inspect.currentframe()
    try:
        print("=== Local Variables ===")
        for name, value in frame.f_locals.items():
            if not name.startswith('_'):
                print(f"{name}: {value}")
        print("=" * 22)
    finally:
        del frame
```

### Common Debugging Scenarios

#### Model Runner Issues

```python
def debug_model_runner():
    """Debug model runner configuration and connectivity."""
    
    from ai_agent.external_integration.model_runner import ModelRunner
    
    runner = ModelRunner()
    
    print("=== Model Runner Debug ===")
    print(f"Config: {runner.config}")
    print(f"Ollama Available: {runner.is_ollama_available()}")
    print(f"Gemini Available: {runner.has_gemini_key()}")
    
    # Test Ollama
    if runner.is_ollama_available():
        try:
            result = runner.test_ollama_connection()
            print(f"Ollama Test: {result}")
        except Exception as e:
            print(f"Ollama Error: {e}")
    
    # Test Gemini
    if runner.has_gemini_key():
        try:
            result = runner.test_gemini_connection()
            print(f"Gemini Test: {result}")
        except Exception as e:
            print(f"Gemini Error: {e}")
    
    print("=" * 23)
```

#### Configuration Issues

```python
def debug_config():
    """Debug configuration loading and validation."""
    
    from ai_agent.utils.config import load_config
    
    try:
        config = load_config()
        print("=== Configuration Debug ===")
        print(f"Config loaded successfully")
        print(f"API Provider: {config.get('api', {}).get('preferred_provider')}")
        print(f"Logging Level: {config.get('logging', {}).get('level')}")
        print(f"Safety Mode: {config.get('execution', {}).get('safety_mode')}")
        print("=" * 24)
        
    except Exception as e:
        print(f"Configuration Error: {e}")
        traceback.print_exc()
```

This development guide provides comprehensive information for contributing to VEXIS-CLI-1.2, from initial setup to advanced debugging and optimization techniques.
