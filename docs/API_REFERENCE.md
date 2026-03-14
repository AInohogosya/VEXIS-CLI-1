# API Reference

## Overview

This comprehensive API reference covers all public interfaces, classes, methods, and data structures available in VEXIS-CLI-1.1.

## Core Components

### TwoPhaseEngine

**Location**: `src/ai_agent/core_processing/two_phase_engine.py`

The main orchestration engine that coordinates the two-phase execution process.

#### Class Definition

```python
class TwoPhaseEngine:
    """Two-phase execution engine for processing natural language instructions."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the two-phase engine.
        
        Args:
            config: Configuration dictionary containing engine settings
        """
```

#### Methods

##### `process_instruction`

```python
async def process_instruction(
    self, 
    instruction: str,
    context: Optional[Dict[str, Any]] = None
) -> CommandResult:
    """Process a natural language instruction through two phases.
    
    Args:
        instruction: Natural language instruction to process
        context: Optional context information
        
    Returns:
        CommandResult: Result of instruction processing
        
    Raises:
        ValidationError: If instruction is invalid
        ProcessingError: If processing fails
    """
```

##### `plan_commands`

```python
async def plan_commands(
    self, 
    instruction: str
) -> List[Command]:
    """Phase 1: Plan commands from natural language.
    
    Args:
        instruction: Natural language instruction
        
    Returns:
        List[Command]: Planned command sequence
        
    Raises:
        PlanningError: If command planning fails
    """
```

##### `execute_commands`

```python
async def execute_commands(
    self, 
    commands: List[Command]
) -> ExecutionResult:
    """Phase 2: Execute planned commands.
    
    Args:
        commands: List of commands to execute
        
    Returns:
        ExecutionResult: Results of command execution
        
    Raises:
        ExecutionError: If command execution fails
    """
```

#### Usage Example

```python
from ai_agent.core_processing.two_phase_engine import TwoPhaseEngine
from ai_agent.utils.config import load_config

# Initialize engine
config = load_config()
engine = TwoPhaseEngine(config)

# Process instruction
result = await engine.process_instruction("list files in current directory")

if result.success:
    print(f"Commands executed: {len(result.commands)}")
    print(f"Output: {result.output}")
else:
    print(f"Error: {result.error}")
```

### ModelRunner

**Location**: `src/ai_agent/external_integration/model_runner.py`

Unified AI provider abstraction supporting 13+ providers: Ollama (local), Google Gemini, OpenAI, Anthropic, xAI, Meta, Mistral AI, Microsoft Azure, Amazon Bedrock, Cohere, DeepSeek, Groq, and Together AI.

#### Class Definition

```python
class ModelRunner:
    """Unified AI model runner supporting multiple providers."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize model runner with configuration.
        
        Args:
            config: Configuration dictionary with provider settings
        """
```

#### Methods

##### `generate_response`

```python
async def generate_response(
    self,
    task_type: TaskType,
    prompt: str,
    provider: Optional[str] = None,
    **kwargs
) -> ModelResponse:
    """Generate response from AI model.
    
    Args:
        task_type: Type of task (task_generation or command_parsing)
        prompt: Prompt to send to model
        provider: Specific provider to use (ollama, groq, google, openai, anthropic, xai, meta, mistral, microsoft, amazon, cohere, deepseek, together)
        **kwargs: Additional parameters for the model
        
    Returns:
        ModelResponse: Model's response with metadata
        
    Raises:
        ProviderError: If provider fails
        ModelError: If model generation fails
    """
```

##### `is_provider_available`

```python
def is_provider_available(self, provider: str) -> bool:
    """Check if a specific provider is available.
    
    Args:
        provider: Provider name (ollama, groq, google, openai, anthropic, xai, meta, mistral, microsoft, amazon, cohere, deepseek, together)
        
    Returns:
        bool: True if provider is available
    """
```

##### `get_available_models`

```python
def get_available_models(self, provider: str) -> List[str]:
    """Get list of available models for a provider.
    
    Args:
        provider: Provider name
        
    Returns:
        List[str]: List of available model names
    """
```

#### Usage Example

```python
from ai_agent.external_integration.model_runner import ModelRunner
from ai_agent.external_integration.model_runner import TaskType

# Initialize runner
runner = ModelRunner(config)

# Generate response
response = await runner.generate_response(
    task_type=TaskType.TASK_GENERATION,
    prompt="Generate a command to list files",
    provider="groq"  # or any supported provider
)

if response.success:
    print(f"Response: {response.content}")
    print(f"Model: {response.model}")
    print(f"Tokens used: {response.tokens_used}")
```

### CommandParser

**Location**: `src/ai_agent/core_processing/command_parser.py`

Converts natural language instructions into executable CLI commands.

#### Class Definition

```python
class CommandParser:
    """Parser for converting natural language to CLI commands."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize command parser.
        
        Args:
            config: Configuration dictionary
        """
```

#### Methods

##### `parse_instruction`

```python
def parse_instruction(
    self,
    instruction: str,
    context: Optional[Dict[str, Any]] = None
) -> ParseResult:
    """Parse natural language instruction into commands.
    
    Args:
        instruction: Natural language instruction
        context: Optional context information
        
    Returns:
        ParseResult: Parsed commands and metadata
        
    Raises:
        ParseError: If parsing fails
        ValidationError: If instruction is invalid
    """
```

##### `validate_command`

```python
def validate_command(self, command: Command) -> ValidationResult:
    """Validate a command for safety and correctness.
    
    Args:
        command: Command to validate
        
    Returns:
        ValidationResult: Validation result with details
    """
```

##### `sanitize_command`

```python
def sanitize_command(self, command: str) -> str:
    """Sanitize command string for safe execution.
    
    Args:
        command: Raw command string
        
    Returns:
        str: Sanitized command string
    """
```

#### Usage Example

```python
from ai_agent.core_processing.command_parser import CommandParser

# Initialize parser
parser = CommandParser(config)

# Parse instruction
result = parser.parse_instruction("create a file named test.txt")

if result.success:
    for command in result.commands:
        print(f"Command: {command.executable}")
        print(f"Args: {command.arguments}")
        print(f"Safe: {command.is_safe}")
```

## Data Structures

### Command

**Location**: `src/ai_agent/core_processing/command_parser.py`

```python
@dataclass
class Command:
    """Represents a single executable command."""
    
    executable: str                    # Command executable
    arguments: List[str]               # Command arguments
    working_directory: Optional[str]   # Working directory
    environment: Dict[str, str]       # Environment variables
    timeout: Optional[int]             # Timeout in seconds
    is_safe: bool                      # Safety flag
    risk_level: str                    # Risk level (low/medium/high)
    
    def to_string(self) -> str:
        """Convert command to string representation."""
        pass
    
    def validate(self) -> bool:
        """Validate command structure."""
        pass
```

### CommandResult

**Location**: `src/ai_agent/core_processing/two_phase_engine.py`

```python
@dataclass
class CommandResult:
    """Result of command processing."""
    
    success: bool                      # Success flag
    commands: List[Command]            # Generated commands
    output: Optional[str]              # Output text
    error: Optional[str]               # Error message
    execution_time: float              # Execution time in seconds
    metadata: Dict[str, Any]          # Additional metadata
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        pass
```

### ModelResponse

**Location**: `src/ai_agent/external_integration/model_runner.py`

```python
@dataclass
class ModelResponse:
    """Response from AI model."""
    
    success: bool                      # Success flag
    content: str                       # Response content
    task_type: TaskType               # Type of task
    model: str                        # Model name
    provider: str                     # Provider name
    tokens_used: Optional[int]         # Tokens used
    cost: Optional[float]             # Cost in USD
    latency: Optional[float]          # Response latency
    error: Optional[str]              # Error message
    metadata: Optional[Dict[str, Any]]  # Additional metadata
```

### ValidationError

**Location**: `src/ai_agent/utils/exceptions.py`

```python
class ValidationError(Exception):
    """Raised when input validation fails."""
    
    def __init__(
        self, 
        message: str, 
        field: Optional[str] = None,
        value: Optional[Any] = None
    ):
        """Initialize validation error.
        
        Args:
            message: Error message
            field: Field that failed validation
            value: Value that failed validation
        """
        super().__init__(message)
        self.field = field
        self.value = value
```

## Utilities

### Configuration

**Location**: `src/ai_agent/utils/config.py`

#### `load_config`

```python
def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load configuration from file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Dict[str, Any]: Loaded configuration
        
    Raises:
        ConfigError: If configuration loading fails
    """
```

#### `validate_config`

```python
def validate_config(config: Dict[str, Any]) -> ValidationResult:
    """Validate configuration dictionary.
    
    Args:
        config: Configuration to validate
        
    Returns:
        ValidationResult: Validation result
    """
```

### Logging

**Location**: `src/ai_agent/utils/logger.py`

#### `get_logger`

```python
def get_logger(name: str) -> logging.Logger:
    """Get configured logger instance.
    
    Args:
        name: Logger name
        
    Returns:
        logging.Logger: Configured logger
    """
```

#### `setup_logging`

```python
def setup_logging(config: Dict[str, Any]) -> None:
    """Set up logging configuration.
    
    Args:
        config: Logging configuration
    """
```

### Error Handler

**Location**: `src/ai_agent/utils/ollama_error_handler.py`

#### `handle_ollama_error`

```python
def handle_ollama_error(
    error_message: str,
    context: Optional[Dict[str, Any]] = None,
    display_to_user: bool = True
) -> Tuple[Optional[ErrorInfo], bool]:
    """Handle Ollama-related errors with user guidance.
    
    Args:
        error_message: Error message to handle
        context: Additional context information
        display_to_user: Whether to display guidance to user
        
    Returns:
        Tuple[Optional[ErrorInfo], bool]: Error info and whether to retry
    """
```

## User Interface

### Yellow Selection System

**Location**: `src/ai_agent/user_interface/yellow_selection/`

#### `get_yellow_menu`

```python
def get_yellow_menu(
    title: str,
    description: Optional[str] = None
) -> YellowMenu:
    """Get yellow-themed menu instance.
    
    Args:
        title: Menu title
        description: Optional description
        
    Returns:
        YellowMenu: Menu instance
    """
```

#### `get_yellow_selector`

```python
def get_yellow_selector() -> YellowSelector:
    """Get yellow-themed selector instance.
    
    Returns:
        YellowSelector: Selector instance
    """
```

### Main Application

**Location**: `src/ai_agent/user_interface/main_app.py`

#### `main`

```python
def main() -> None:
    """Main application entry point."""
    pass
```

#### `process_instruction_cli`

```python
def process_instruction_cli(
    instruction: str,
    debug: bool = False,
    no_prompt: bool = False,
    provider: Optional[str] = None
) -> None:
    """Process instruction from command line.
    
    Args:
        instruction: Natural language instruction
        debug: Enable debug mode
        no_prompt: Skip provider selection prompts
        provider: Force specific provider
    """
```

## Integration Examples

### Basic Usage

```python
import asyncio
from ai_agent.core_processing.two_phase_engine import TwoPhaseEngine
from ai_agent.utils.config import load_config

async def main():
    # Load configuration
    config = load_config()
    
    # Initialize engine
    engine = TwoPhaseEngine(config)
    
    # Process instruction
    result = await engine.process_instruction(
        "list files in current directory"
    )
    
    if result.success:
        print("Success!")
        print(f"Commands executed: {len(result.commands)}")
        print(f"Output: {result.output}")
    else:
        print(f"Error: {result.error}")

if __name__ == "__main__":
    asyncio.run(main())
```

### Custom Provider

```python
from ai_agent.external_integration.model_runner import AIProvider, ModelResponse

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
    ) -> ModelResponse:
        """Generate response using custom API."""
        try:
            response = await self.client.generate(prompt=prompt)
            
            return ModelResponse(
                success=True,
                content=response["text"],
                task_type=TaskType(task_type),
                model=self.config.get("model", "custom"),
                provider="custom",
                tokens_used=response.get("tokens"),
                latency=response.get("latency")
            )
            
        except Exception as e:
            return ModelResponse(
                success=False,
                content="",
                task_type=TaskType(task_type),
                model="",
                provider="custom",
                error=str(e)
            )
    
    def is_available(self) -> bool:
        """Check if provider is available."""
        return self.client.health_check()
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get provider model information."""
        return {
            "name": "Custom Provider",
            "models": self.client.list_models(),
            "capabilities": ["text-generation", "code-generation"]
        }
```

### Error Handling

```python
from ai_agent.utils.ollama_error_handler import handle_ollama_error
from ai_agent.utils.exceptions import ValidationError, ProviderError

async def safe_instruction_processing(instruction: str):
    """Process instruction with comprehensive error handling."""
    
    try:
        # Validate instruction
        if not instruction.strip():
            raise ValidationError("Instruction cannot be empty")
        
        # Process with engine
        engine = TwoPhaseEngine(config)
        result = await engine.process_instruction(instruction)
        
        return result
        
    except ValidationError as e:
        print(f"Validation Error: {e}")
        return None
        
    except ProviderError as e:
        # Handle with enhanced error handler
        error_info, should_retry = handle_ollama_error(
            str(e),
            context={"instruction": instruction}
        )
        
        if should_retry:
            print("Retrying with alternative provider...")
            return await safe_instruction_processing(instruction)
        else:
            print(f"Provider Error: {error_info.message}")
            return None
            
    except Exception as e:
        print(f"Unexpected Error: {e}")
        return None
```

### Configuration Management

```python
from ai_agent.utils.config import load_config, validate_config

def setup_environment(config_path: str = "config.yaml"):
    """Set up environment with configuration validation."""
    
    try:
        # Load configuration
        config = load_config(config_path)
        
        # Validate configuration
        validation_result = validate_config(config)
        
        if not validation_result.is_valid:
            print("Configuration validation failed:")
            for error in validation_result.errors:
                print(f"  - {error}")
            return None
        
        # Set up logging
        from ai_agent.utils.logger import setup_logging
        setup_logging(config.get("logging", {}))
        
        return config
        
    except Exception as e:
        print(f"Configuration setup failed: {e}")
        return None
```

## Testing API

### Test Utilities

**Location**: `tests/utils/`

#### `MockModelRunner`

```python
class MockModelRunner:
    """Mock model runner for testing."""
    
    def __init__(self, responses: Dict[str, str]):
        self.responses = responses
    
    async def generate_response(
        self, 
        task_type: TaskType,
        prompt: str,
        **kwargs
    ) -> ModelResponse:
        """Generate mock response."""
        
        response_text = self.responses.get(
            prompt, 
            "Mock response for: " + prompt
        )
        
        return ModelResponse(
            success=True,
            content=response_text,
            task_type=task_type,
            model="mock-model",
            provider="mock",
            tokens_used=len(response_text.split()),
            latency=0.1
        )
```

#### `TestFixtures`

```python
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

@pytest.fixture
def mock_ollama_response():
    """Mock Ollama API response."""
    return {
        "response": "ls -la",
        "done": True,
        "model": "gemini-3-flash-preview"
    }
```

## Migration Guide

### Version Compatibility

This API reference covers VEXIS-CLI-1.1 version 1.0.0.

#### Breaking Changes from 0.x

- `TwoPhaseEngine.process_instruction` is now async
- `ModelRunner` requires explicit provider specification
- Configuration format changed to YAML
- Error handling unified under `ErrorInfo` structure

#### Migration Steps

1. Update imports:
   ```python
   # Old
   from ai_agent.engine import TwoPhaseEngine
   
   # New
   from ai_agent.core_processing.two_phase_engine import TwoPhaseEngine
   ```

2. Make async calls:
   ```python
   # Old
   result = engine.process_instruction("test")
   
   # New
   result = await engine.process_instruction("test")
   ```

3. Update configuration:
   ```python
   # Old JSON format
   config = {"preferred_provider": "ollama"}
   
   # New YAML format
   config = load_config("config.yaml")
   ```

This API reference provides comprehensive documentation for all public interfaces in VEXIS-CLI-1.1, enabling developers to integrate, extend, and test the system effectively.
