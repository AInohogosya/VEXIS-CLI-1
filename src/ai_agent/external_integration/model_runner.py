"""
Model Runner for VEXIS-CLI AI Agent System
Multi-Provider Support: 13+ AI providers available
"""

import time
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum

from .multi_provider_vision_client import MultiProviderVisionAPIClient, APIRequest, APIProvider
from ..utils.exceptions import ValidationError
from ..utils.logger import get_logger
from ..utils.config import load_config


class TaskType(Enum):
		"""Task types for 5-Phase CLI Architecture"""
		PHASE1_COMMAND_SUGGESTION = "phase1_command_suggestion"
		INPUT_SUMMARIZATION = "input_summarization"
		PHASE2_COMMAND_EXTRACTION = "phase2_command_extraction"
		PHASE4_LOG_EVALUATION = "phase4_log_evaluation"
		PHASE5_SUMMARY_GENERATION = "phase5_summary_generation"


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
				"""Load prompt templates for 5-Phase CLI Architecture"""
				return {
						TaskType.PHASE1_COMMAND_SUGGESTION.value: '''I have received the instruction: "{user_prompt}". What commands should I run to carry this out? Please tell me. I can only use terminal commands, so do not suggest GUI operations. The OS I am using is {os_info}.

CRITICAL: Plan for success by considering:
1. The primary approach to accomplish this task
2. At least 2-3 alternative approaches if the primary method fails
3. Common failure points and how to avoid them
4. Verification steps to confirm the task succeeded (not just completed)

Remember: The task is only successful when the goal is FULLY ACHIEVED. Provide comprehensive command suggestions with built-in fallback strategies.{conversation_history}''',

						TaskType.INPUT_SUMMARIZATION.value: '''Please summarize the following input into a single sentence. This is critical - you must provide exactly one sentence that captures the essence of the input. Do not use multiple sentences. Do not add explanations. Just provide the summary in a single sentence.

Input: {user_prompt}

Summary (one sentence only):''',

						TaskType.PHASE2_COMMAND_EXTRACTION.value: '''Please look at this: {phase_1_output}. This is a relatively long text with many explanations, but please put all the necessary commands into a single code block. You may use only one code block.

IMPORTANT FOR LONG-RUNNING SERVICES: If the requested task starts a server, bot, watcher, scheduler, tunnel, or any process that should keep running after this pipeline response is sent, run it as an explicit background command by ending the command with `&` and include a separate verification command (for example `ps`, `curl`, or log inspection). Do not leave long-running services in the foreground, because foreground commands are subject to execution timeouts.{conversation_history}''',

						TaskType.PHASE4_LOG_EVALUATION.value: '''I executed the commands to carry out the instruction {user_prompt}. This resulted in the following log: {full_terminal_log_so_far} However, since I am a beginner, I do not know if it succeeded or failed.

CRITICAL EVALUATION RULES:
1. **Success vs Failure Definition** - The task is ONLY successful if the goal was FULLY ACHIEVED. Partial completion or workarounds that don't meet the original objective count as FAILURE.
2. **Failure Response** - If the task failed, you MUST:
	 - Output a code block containing the corrected/alternative commands
	 - Analyze what went wrong and why
	 - Propose a COMPLETELY DIFFERENT alternative approach (Method B, C, etc.)
3. **Success Response** - If the task succeeded, you MUST:
	 - Confirm success without including any code blocks
	 - Do NOT output code blocks under any circumstances when the task succeeds
4. **Never Accept Defeat** - If the task failed, do not stop at analysis. Continue proposing alternative approaches until success is achieved.
5. **Code Block Re-output** - On failure, you MUST re-output corrected code blocks. Success is determined by the absence of code blocks.{conversation_history}''',

						TaskType.PHASE5_SUMMARY_GENERATION.value: '''I received the instruction "{user_prompt}" and have been executing commands. Here is the terminal log: {full_terminal_log}

Your task is to write a HUMAN-READABLE SUMMARY in plain English explaining what was done and the result.

ABSOLUTELY FORBIDDEN:
- Do NOT output any code blocks (```)
- Do NOT output any shell commands, scripts, or bash code
- Do NOT output any programming code
- Do NOT use markdown formatting
- Do NOT include variable assignments like PROJECT_ROOT=...
- Do NOT include if/then/else logic or conditionals
- Do NOT include any technical implementation details

REQUIRED FORMAT:
- Write in plain English sentences
- Explain what the task was
- Explain what commands were run (in plain English, not the actual commands)
- Explain whether the task succeeded or failed
- Explain the final result
- Keep it concise and readable

Example of GOOD summary:
"The task was to fix the database connection. I updated the configuration file and restarted the service. The database connection now works correctly. Task succeeded."

Example of BAD summary (DO NOT DO THIS):
"```bash
PROJECT_ROOT=/home/user
cd $PROJECT_ROOT
sed -i 's/old/new/' config.py
```"

Write your summary now in plain English only:{conversation_history}''',
				}

		def get_template(self, task_type: TaskType) -> str:
				"""Get template for task type"""
				return self.templates.get(task_type.value, "")


class ModelRunner:
		"""CLI Architecture Model Runner: Ollama Cloud Models"""

		# Valid Ollama model names
		DEFAULT_OLLAMA_MODEL = "llama3.2:latest"
		DEFAULT_GOOGLE_MODEL = "gemini-3.1-pro-preview"
		MAX_RETRIES = 3

		def __init__(self, provider: str = None, model: str = None, config: Optional[Dict[str, Any]] = None, auto_install_sdks: bool = False):
				# Direct provider and model from runtime arguments
				self.provider = provider
				self.model = model

				# Fallback to config if not provided
				self.config = config or load_config().api.__dict__
				self.logger = get_logger(__name__)

				# Initialize multi-provider vision client with SDK installation support
				self.vision_client = MultiProviderVisionAPIClient(self.config, auto_install_sdks=auto_install_sdks)
				self.prompt_template = PromptTemplate()

				self.logger.info(
						"Model runner initialized",
						provider=self.provider,
						model=self.model,
				)

		def run_model(self, request: ModelRequest) -> ModelResponse:
				"""Run AI model for CLI Architecture with retry on validation failure"""
				start_time = time.time()

				try:
						# Validate request
						self._validate_request(request)

						# Use runtime provider and model if provided, otherwise fallback to settings
						if self.provider and self.model:
								provider_name = self.provider
								model_name = self.model
						else:
								# Fallback to settings for backward compatibility
								from ..utils.settings_manager import get_settings_manager
								settings = get_settings_manager()
								provider_name = settings.get_preferred_provider()
								model_name = settings.get_model(provider_name)

						if not provider_name:
								raise ValidationError("No provider configured. Please select a provider first.")

						if not model_name:
								raise ValidationError(f"No model configured for provider '{provider_name}'. Please select a model first.")

						# Retry loop for validation failures
						for attempt in range(self.MAX_RETRIES):
								# Format prompt
								prompt = self._format_prompt(request)

								# Get system instructions for API request
								system_instructions = self._get_system_instructions(request.task_type)

								# Add retry instruction if not first attempt
								if attempt > 0:
										system_instructions += f"\n\n## RETRY ATTEMPT {attempt + 1}/{self.MAX_RETRIES}\nYour previous output did not meet the expected format. Please carefully follow the format requirements and provide a valid response."

								# Create API request with user's exact selection
								api_request = APIRequest(
										prompt=prompt,
										image_data=request.image_data,
										image_format=request.image_format,
										max_tokens=request.max_tokens,
										temperature=request.temperature,
										model=model_name,
										provider=provider_name,
										system_instruction=system_instructions
								)

								# Make API call
								api_response = self.vision_client.generate_response(api_request)

								if not api_response.success:
										# API call failed, don't retry on API errors
										model_response = ModelResponse(
												success=api_response.success,
												content=api_response.content,
												task_type=request.task_type,
												model=api_response.model or model_name,
												provider=api_response.provider or provider_name,
												tokens_used=api_response.tokens_used,
												cost=api_response.cost,
												latency=time.time() - start_time,
												error=api_response.error,
										)

										self.logger.error(
												"Model execution failed",
												task_type=request.task_type.value,
												error=model_response.error,
										)

										# Enhanced error handling for authentication issues
										auth_error_keywords = ['authentication', 'unauthorized', '401', '403', 'api key', 'credential']
										error_lower = (model_response.error or '').lower()
										if any(keyword in error_lower for keyword in auth_error_keywords):
												try:
														from ..utils.ollama_error_handler import handle_ollama_error
														context = {
																'model_name': model_response.model,
																'operation': 'model_execution'
														}
														handle_ollama_error(model_response.error, context, display_to_user=True)

														# Prompt user to sign in (Ollama-specific, only in Normal mode - NEVER in Telegram mode)
														if model_response.provider == 'ollama':
																import sys
																import os
																# Check if running in Telegram mode via environment variable
																is_telegram_mode = os.getenv('VEXIS_TELEGRAM_MODE', '').lower() in ('true', '1', 'yes')
																if sys.stdin.isatty() and not is_telegram_mode:	# Only prompt if in terminal AND not in Telegram mode
																		try:
																				choice = input("\nWould you like to sign in to Ollama now? (y/n): ").lower().strip()
																				if choice in ['y', 'yes']:
																						import subprocess
																						print("\n🔐 Opening Ollama sign-in...")
																						try:
																								result = subprocess.run(["ollama", "signin"], capture_output=False, text=True)
																								if result.returncode == 0:
																										print("✓ Sign-in initiated. Please complete it in your browser.")
																										print("Then try running your command again.")
																								else:
																										print("✗ Failed to initiate sign-in.")
																						except FileNotFoundError:
																								print("✗ Ollama command not found. Please ensure Ollama is installed.")
																		except (KeyboardInterrupt, EOFError):
																				print("\nOperation cancelled.")
																elif is_telegram_mode:
																		# In Telegram mode, log the issue but don't block execution
																		self.logger.info("Ollama authentication required but running in Telegram mode - skipping interactive sign-in prompt")
												except ImportError:
														pass	# Fallback to just logging the error

										return model_response

								# API call succeeded, validate output format
								is_valid, validation_error = self._validate_output_format(
										api_response.content,
										request.task_type
								)

								if is_valid:
										# Output is valid, return success
										model_response = ModelResponse(
												success=True,
												content=api_response.content,
												task_type=request.task_type,
												model=api_response.model or model_name,
												provider=api_response.provider or provider_name,
												tokens_used=api_response.tokens_used,
												cost=api_response.cost,
												latency=time.time() - start_time,
												error=None,
										)

										self.logger.info(
												"Model execution successful",
												task_type=request.task_type.value,
												model=model_response.model,
												latency=model_response.latency,
												attempt=attempt + 1,
										)

										return model_response
								else:
										# Output validation failed, log and retry
										self.logger.warning(
												"Output validation failed, retrying",
												task_type=request.task_type.value,
												attempt=attempt + 1,
												max_retries=self.MAX_RETRIES,
												validation_error=validation_error,
										)

										if attempt == self.MAX_RETRIES - 1:
												# Last attempt failed, return the last response with validation error
												model_response = ModelResponse(
														success=False,
														content=api_response.content,
														task_type=request.task_type,
														model=api_response.model or model_name,
														provider=api_response.provider or provider_name,
														tokens_used=api_response.tokens_used,
														cost=api_response.cost,
														latency=time.time() - start_time,
														error=f"Output validation failed after {self.MAX_RETRIES} attempts: {validation_error}",
												)
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

				if request.timeout < 1 or request.timeout > 300:
						raise ValidationError("Invalid timeout (must be 1-300 seconds)", "timeout", request.timeout)

		def _format_prompt(self, request: ModelRequest) -> str:
				"""Format prompt based on task type and context"""
				template = self.prompt_template.get_template(request.task_type)

				format_vars = {
						"instruction": request.prompt,
						"task_description": request.prompt,
						"user_prompt": request.prompt,
				}

				# Add context variables if available (e.g., phase_1_output for Phase 2)
				if request.context:
						format_vars.update(request.context)

				format_vars.setdefault("os_info", "Unknown OS")
				format_vars.setdefault("conversation_history", "")

				try:
						formatted_prompt = template.format(**format_vars)

						return formatted_prompt
				except KeyError as e:
						self.logger.warning(f"Template variable missing: {e}")
						return request.prompt
				except Exception as e:
						self.logger.error(f"Template formatting error: {e}")
						return request.prompt

		def _validate_output_format(self, content: str, task_type: TaskType) -> tuple[bool, Optional[str]]:
				"""Validate that the output matches the expected format for the task type"""
				if not content or not content.strip():
						return False, "Output is empty"

				if task_type == TaskType.INPUT_SUMMARIZATION:
						# Must be exactly one sentence
						sentences = [s.strip() for s in content.split('.') if s.strip()]
						# Check if content ends with period and has no other sentence-ending punctuation
						if content.count('.') > 1 or content.count('!') > 0 or content.count('?') > 0:
								return False, "Summary must be exactly one sentence"
						# Check for code blocks
						if '```' in content:
								return False, "Summary must not contain code blocks"
						return True, None

				elif task_type == TaskType.PHASE2_COMMAND_EXTRACTION:
						# Must contain at least one code block
						if '```' not in content:
								return False, "Command extraction must contain at least one code block"
						return True, None

				elif task_type == TaskType.PHASE4_LOG_EVALUATION:
						# Either success (no code blocks) or failure (with code blocks)
						# Both formats are valid as long as they're consistent
						has_code_block = '```' in content
						# Just check that it's not empty
						return True, None

				elif task_type == TaskType.PHASE5_SUMMARY_GENERATION:
						# Must NOT contain code blocks
						if '```' in content:
								return False, "Summary must not contain code blocks"
						# Check for shell command patterns
						suspicious_patterns = ['$', '#!', 'sudo ', 'apt ', 'npm ', 'pip ']
						if any(pattern in content for pattern in suspicious_patterns):
								return False, "Summary must not contain shell commands"
						return True, None

				elif task_type == TaskType.PHASE1_COMMAND_SUGGESTION:
						# Should contain some substantive content
						if len(content.strip()) < 50:
								return False, "Command suggestion is too short"
						return True, None

				return True, None

		def _get_system_instructions(self, task_type: TaskType) -> str:
				"""Get system instructions for better AI behavior"""
				base_instructions = """# VEXIS-CLI AI Agent System Instructions

You are operating as part of the VEXIS-CLI automation system. Your responses directly impact system execution and user experience.

## Behavioral Guidelines
1. **Precision Over Verbosity** - Be exact and concise
2. **Context Awareness** - Always consider previous actions and current state
3. **Error Resilience** - Handle failures gracefully and suggest alternatives
4. **Safety First** - Never suggest destructive commands without clear warnings
5. **User Intent Focus** - Stay focused on accomplishing the user's original goal

## Response Standards
- Use clear, unambiguous language
- Provide specific, actionable outputs
- Avoid conversational filler or unnecessary explanations
- Maintain consistency with previous interactions
- Respect the established workflow and command patterns

## Quality Assurance
- Double-check command syntax before outputting
- Verify file paths and parameters are valid
- Consider edge cases and potential failure points
- Ensure outputs match the expected format exactly

## Autonomous Task Completion & Persistence (CRITICAL)
1. **Success vs Completion** - A task is finished ONLY when it SUCCEEDS, not merely when it completes. Even if a completion report is issued, the task is NOT finished if the result is a failure.
2. **Never Give Up** - Do NOT terminate the task while any failure remains. Persistently and autonomously continue until success is achieved.
3. **Flexible Alternative Methods** - When Method A fails, immediately pivot to a completely different approach (Method B, C, D...). Be creative and resourceful in finding alternative solutions.
4. **Code Block Re-output** - Since success is determined by the presence of properly formatted code blocks, you MUST re-output code blocks as many times as necessary until the task succeeds. Never stop at "analysis" or "explanation" - always provide executable code blocks.
5. **Proactive Error Recovery** - Anticipate failures before they occur and prepare fallback strategies. If an error occurs, immediately propose and execute the next best alternative without waiting for user input."""

				# Add custom system prompt for Phase 1 only (Amore configuration)
				if task_type == TaskType.PHASE1_COMMAND_SUGGESTION:
						try:
								config = load_config()
								custom_prompt = config.custom_system_prompt
								if custom_prompt and custom_prompt.strip():
										# Append custom prompt to base instructions
										base_instructions += f"\n\n## Custom System Prompt (User Configured)\n{custom_prompt.strip()}"
										self.logger.info("Custom system prompt injected into Phase 1")
						except Exception as e:
								self.logger.warning(f"Failed to load custom system prompt: {e}")

				return base_instructions

		def install_missing_sdks(self, providers: Optional[List[str]] = None, interactive: bool = True) -> Dict[str, bool]:
				"""Install missing SDKs for specified providers"""
				return self.vision_client.install_missing_sdks(providers, interactive)

		def show_sdk_status(self, providers: Optional[List[str]] = None):
				"""Show SDK installation status"""
				self.vision_client.show_sdk_status(providers)


def get_model_runner(provider: str = None, model: str = None) -> ModelRunner:
		"""Get model runner instance with optional provider and model"""
		# Create instance with runtime provider and model
		return ModelRunner(provider=provider, model=model)
