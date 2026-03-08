"""
Model Runner for CLI AI Agent System
Simplified: Ollama Cloud Models Only
"""

import time
from typing import Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum

from .vision_api_client import VisionAPIClient, APIRequest, APIProvider
from ..utils.exceptions import ValidationError
from ..utils.logger import get_logger
from ..utils.config import load_config


class TaskType(Enum):
    """Task types for CLI Architecture"""
    TASK_GENERATION = "task_generation"
    COMMAND_PARSING = "command_parsing"


@dataclass
class ModelRequest:
    """Model request structure"""
    task_type: TaskType
    prompt: str
    image_data: Optional[bytes] = None
    image_format: str = "PNG"
    context: Optional[Dict[str, Any]] = None
    parameters: Optional[Dict[str, Any]] = None
    max_tokens: int = 5000
    temperature: float = 1.0
    timeout: int = 30


@dataclass
class ModelResponse:
    """Model response structure"""
    success: bool
    content: str
    task_type: TaskType
    model: str
    provider: str
    tokens_used: Optional[int] = None
    cost: Optional[float] = None
    latency: Optional[float] = None
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class PromptTemplate:
    """Prompt template manager"""

    def __init__(self):
        self.templates = self._load_templates()

    def _load_templates(self) -> Dict[str, str]:
        """Load prompt templates for CLI Architecture"""
        return {
            TaskType.TASK_GENERATION.value: """You are a CLI assistant. Convert this user instruction into specific command-line steps.

User: {instruction}
OS: {os_info}

Generate numbered steps using CLI commands. Only create/navigate directories if the task requires it.

Example:
- "open safari" -> 1. Open safari browser
- "create python project" -> 1. Create project folder, 2. Navigate to folder, 3. Initialize git

CRITICAL: Output ONLY the numbered list. No explanations, no intro text, no markdown formatting.

Output format:
1. [First CLI step]
2. [Second CLI step]
3. [Third CLI step]
...

Each step should be a clear, actionable CLI-oriented task description.""",

            TaskType.COMMAND_PARSING.value: """Convert this task into a single CLI command.

Task: {task_description}
OS: {os_info}
Dir: {current_directory}

Your previous commands:
{previous_terminal_actions}

Last output:
{last_command_output}

IMPORTANT: These are YOUR past actions. Don't repeat them. Build on what's already done.

Available commands: Any CLI command, END (stop), REGENERATE_STEP (retry)

CRITICAL: Do NOT use markdown formatting, code blocks, or any special characters. Output plain text only.

Output format (4 lines):
Reasoning: [Why this command, considering your history]
Target: [What this command targets]
[The CLI command - PLAIN TEXT, no markdown or code blocks]
Terminal Log: [History will show after execution]

Example:
Reasoning: User wants to open safari, no previous actions needed
Target: Safari browser
open -a Safari
Terminal Log: [Will display terminal after execution]

END""",

        }

    def get_template(self, task_type: TaskType) -> str:
        """Get prompt template for task type"""
        return self.templates.get(task_type.value, self.templates[TaskType.TASK_GENERATION.value])

    def format_prompt(self, task_type: TaskType, **kwargs) -> str:
        """Format prompt template with variables"""
        template = self.get_template(task_type)
        return template.format(**kwargs)


class ModelRunner:
    """CLI Architecture Model Runner: Ollama Cloud Models"""

    # Valid Ollama model names
    DEFAULT_OLLAMA_MODEL = "llama3.2:latest"
    DEFAULT_GOOGLE_MODEL = "gemini-2.0-flash-exp"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or load_config().api.__dict__
        self.logger = get_logger("model_runner")

        # Initialize components
        self.vision_client = VisionAPIClient(self.config)
        self.prompt_template = PromptTemplate()

        self.logger.info(
            "Model runner initialized",
            preferred_provider=self.config.get("preferred_provider", "ollama"),
            default_model=self.DEFAULT_OLLAMA_MODEL,
        )

    def run_model(self, request: ModelRequest) -> ModelResponse:
        """Run AI model for CLI Architecture"""
        start_time = time.time()

        try:
            # Validate request
            self._validate_request(request)

            # Format prompt
            prompt = self._format_prompt(request)

            # Determine provider and model
            preferred_provider = self.config.get("preferred_provider", "ollama")

            if preferred_provider == "google" and self.config.get("google_api_key"):
                provider_enum = APIProvider.GOOGLE
                model_name = self.config.get("google_model", self.DEFAULT_GOOGLE_MODEL)
            else:
                provider_enum = APIProvider.OLLAMA
                model_name = self.config.get("local_model", self.DEFAULT_OLLAMA_MODEL)

            # Create API request
            api_request = APIRequest(
                prompt=prompt,
                image_data=request.image_data,
                image_format=request.image_format,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                model=model_name,
                provider=provider_enum,
            )

            # Make API call
            api_response = self.vision_client.analyze_image(api_request)

            # Create model response
            model_response = ModelResponse(
                success=api_response.success,
                content=api_response.content,
                task_type=request.task_type,
                model=api_response.model or model_name,
                provider=api_response.provider or preferred_provider,
                tokens_used=api_response.tokens_used,
                cost=api_response.cost,
                latency=time.time() - start_time,
                error=api_response.error,
            )

            if api_response.success:
                self.logger.info(
                    "Model execution successful",
                    task_type=request.task_type.value,
                    model=model_response.model,
                    latency=model_response.latency,
                )
            else:
                self.logger.error(
                    "Model execution failed",
                    task_type=request.task_type.value,
                    error=model_response.error,
                )
                
                # Enhanced error handling for authentication issues
                if "Authentication required" in model_response.error:
                    try:
                        from ..utils.ollama_error_handler import handle_ollama_error
                        context = {
                            'model_name': model_response.model,
                            'operation': 'model_execution'
                        }
                        handle_ollama_error(model_response.error, context, display_to_user=True)
                        
                        # Prompt user to sign in
                        import sys
                        if sys.stdin.isatty():  # Only prompt if running in terminal
                            try:
                                choice = input("\nWould you like to sign in to Ollama now? (y/n): ").lower().strip()
                                if choice in ['y', 'yes']:
                                    import subprocess
                                    print("\n🔐 Opening Ollama sign-in...")
                                    result = subprocess.run(["ollama", "signin"], capture_output=False, text=True)
                                    if result.returncode == 0:
                                        print("✓ Sign-in initiated. Please complete it in your browser.")
                                        print("Then try running your command again.")
                                    else:
                                        print("✗ Failed to initiate sign-in.")
                            except (KeyboardInterrupt, EOFError):
                                print("\nOperation cancelled.")
                    except ImportError:
                        pass  # Fallback to just logging the error

            return model_response

        except ValidationError:
            raise
        except Exception as e:
            self.logger.error(f"Model execution failed: {e}")
            return ModelResponse(
                success=False,
                content="",
                task_type=request.task_type,
                model="",
                provider="",
                latency=time.time() - start_time,
                error=str(e),
            )

    def _validate_request(self, request: ModelRequest):
        """Validate model request"""
        if not request.prompt:
            raise ValidationError("Prompt cannot be empty", "prompt", request.prompt)

        if request.max_tokens < 1 or request.max_tokens > 7000:
            raise ValidationError("Invalid max_tokens", "max_tokens", request.max_tokens)

        if not (0.0 <= request.temperature <= 2.0):
            raise ValidationError("Invalid temperature", "temperature", request.temperature)

        if request.task_type not in TaskType:
            raise ValidationError("Invalid task type", "task_type", request.task_type)

    def _format_prompt(self, request: ModelRequest) -> str:
        """Format prompt based on task type and context"""
        template = self.prompt_template.get_template(request.task_type)

        format_vars = {
            "instruction": request.prompt,
            "task_description": request.prompt,
        }

        if request.context:
            format_vars.update(request.context)
            if request.task_type == TaskType.COMMAND_PARSING:
                format_vars.setdefault("previous_terminal_actions", "No previous actions")
                format_vars.setdefault("last_command_output", "No previous output")
                format_vars.setdefault("current_directory", "Unknown")

        format_vars.setdefault("os_info", "Unknown OS")

        try:
            return template.format(**format_vars)
        except KeyError as e:
            self.logger.warning(f"Template variable missing: {e}")
            return request.prompt
        except Exception as e:
            self.logger.error(f"Template formatting error: {e}")
            return request.prompt

    def generate_tasks(self, instruction: str, os_info: str, context: Optional[Dict[str, Any]] = None) -> ModelResponse:
        """Phase 1: Generate task list from instruction and OS environment"""
        enhanced_context = context or {}
        enhanced_context["os_info"] = os_info

        request = ModelRequest(
            task_type=TaskType.TASK_GENERATION,
            prompt=instruction,
            context=enhanced_context,
        )

        return self.run_model(request)

    def parse_command(self, task_description: str, os_info: str, context: Optional[Dict[str, Any]] = None) -> ModelResponse:
        """Phase 2: Parse task description into CLI command"""
        enhanced_context = context or {}
        enhanced_context["os_info"] = os_info

        request = ModelRequest(
            task_type=TaskType.COMMAND_PARSING,
            prompt=task_description,
            context=enhanced_context,
        )

        return self.run_model(request)


def get_model_runner() -> ModelRunner:
    """Get model runner instance (always fresh to use latest settings)"""
    # Always create a new instance to ensure latest settings are loaded
    return ModelRunner()
