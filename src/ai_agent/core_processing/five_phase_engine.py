"""
5-Phase Pipeline Execution Engine for CLI AI Agent System

Phase 1: Command Suggestion (User Prompt → Base Model)
Phase 2: Command Extraction (Phase 1 Output → Single Code Block)
Phase 3: Command Execution (Terminal Injection)
Phase 4: Log Evaluation and Re-execution Decision
Phase 5: Summary Generation and Display
"""

import re
import time
import platform
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from ..external_integration.model_runner import ModelRunner, TaskType, ModelRequest, ModelResponse
from ..utils.exceptions import ExecutionError, ValidationError
from ..utils.logger import get_logger
from .terminal_history import TerminalHistory, get_terminal_history, TerminalEntryType


class PipelinePhase(Enum):
    """5-Phase Pipeline phases"""
    PHASE1_COMMAND_SUGGESTION = "phase1_command_suggestion"
    PHASE2_COMMAND_EXTRACTION = "phase2_command_extraction"
    PHASE3_COMMAND_EXECUTION = "phase3_command_execution"
    PHASE4_LOG_EVALUATION = "phase4_log_evaluation"
    PHASE5_SUMMARY_GENERATION = "phase5_summary_generation"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class PipelineContext:
    """Context for tracking 5-phase pipeline execution"""
    user_prompt: str
    phase1_output: Optional[str] = None
    extracted_commands: Optional[str] = None
    terminal_log: str = ""
    phase4_output: Optional[str] = None
    final_summary: Optional[str] = None
    current_phase: PipelinePhase = PipelinePhase.PHASE1_COMMAND_SUGGESTION
    iteration_count: int = 0
    max_iterations: int = 10
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class FivePhaseEngine:
    """
    5-Phase Pipeline Execution Engine
    
    Implements the complete 5-phase architecture:
    - Phase 1: Command Suggestion
    - Phase 2: Command Extraction
    - Phase 3: Command Execution
    - Phase 4: Log Evaluation
    - Phase 5: Summary Generation
    """
    
    def __init__(self, provider: str = None, model: str = None, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = get_logger("five_phase_engine")
        
        # Initialize terminal history system
        self.terminal_history = get_terminal_history()
        
        # Initialize model runner with runtime provider and model
        self.model_runner = ModelRunner(provider=provider, model=model, config=self.config)
        
        # Configuration
        self.max_iterations = self.config.get("max_iterations", 300)
        self.command_timeout = self.config.get("command_timeout", 30)
        self.task_timeout = self.config.get("task_timeout", 300)
        
        self.logger.info("5-Phase Pipeline Engine initialized")
    
    def execute_instruction(self, user_prompt: str) -> PipelineContext:
        """
        Execute user instruction through the 5-phase pipeline
        
        Args:
            user_prompt: Natural language instruction from user
            
        Returns:
            PipelineContext with complete execution results
        """
        self.logger.info("Starting 5-Phase Pipeline execution", user_prompt=user_prompt)
        
        # Initialize context
        context = PipelineContext(
            user_prompt=user_prompt,
            max_iterations=self.max_iterations,
            metadata={"os_info": self._get_os_info()}
        )
        
        try:
            # Phase 1: Command Suggestion
            if not self._run_phase1(context):
                context.current_phase = PipelinePhase.FAILED
                context.error = "Phase 1 (Command Suggestion) failed"
                return context
            
            # Phase 2-4 Loop: Extract, Execute, Evaluate until complete
            while context.iteration_count < context.max_iterations:
                context.iteration_count += 1
                self.logger.info(f"Starting iteration {context.iteration_count}", 
                               phase=context.current_phase.value)
                
                # Phase 2: Command Extraction
                if not self._run_phase2(context):
                    context.current_phase = PipelinePhase.FAILED
                    context.error = "Phase 2 (Command Extraction) failed"
                    return context
                
                # Phase 3: Command Execution
                if not self._run_phase3(context):
                    context.current_phase = PipelinePhase.FAILED
                    context.error = "Phase 3 (Command Execution) failed"
                    return context
                
                # Phase 4: Log Evaluation
                should_continue = self._run_phase4(context)
                
                if not should_continue:
                    # No "failure" in Phase 4 output - task is successful
                    break
                
                # Continue to next iteration (Phase 2-4 loop)
                self.logger.info(f"Phase 4 detected failure indicator, continuing to iteration {context.iteration_count + 1}")
            
            if context.iteration_count >= context.max_iterations:
                self.logger.warning("Maximum iterations reached, forcing completion")
            
            # Phase 5: Summary Generation
            if not self._run_phase5(context):
                context.current_phase = PipelinePhase.FAILED
                context.error = "Phase 5 (Summary Generation) failed"
                return context
            
            context.current_phase = PipelinePhase.COMPLETED
            context.end_time = time.time()
            
            self.logger.info("5-Phase Pipeline execution completed successfully",
                           duration=context.end_time - context.start_time,
                           iterations=context.iteration_count)
            
            return context
            
        except Exception as e:
            self.logger.error(f"5-Phase Pipeline execution failed: {e}")
            context.current_phase = PipelinePhase.FAILED
            context.error = str(e)
            context.end_time = time.time()
            return context
    
    def _run_phase1(self, context: PipelineContext) -> bool:
        """
        Phase 1: Command Suggestion
        
        Send user prompt to base model with system prompt to get natural-language
        description of what commands should be run.
        
        Returns:
            True if successful, False otherwise
        """
        self.logger.info("Phase 1: Command Suggestion started")
        context.current_phase = PipelinePhase.PHASE1_COMMAND_SUGGESTION
        
        try:
            os_info = context.metadata.get("os_info", self._get_os_info())
            
            # Create request for Phase 1
            request = ModelRequest(
                task_type=TaskType.PHASE1_COMMAND_SUGGESTION,
                prompt=context.user_prompt,
                context={
                    "user_prompt": context.user_prompt,
                    "os_info": os_info,
                },
                max_tokens=4000,
                temperature=0.7
            )
            
            # Run model
            response = self.model_runner.run_model(request)
            
            if not response.success:
                self.logger.error(f"Phase 1 model execution failed: {response.error}")
                return False
            
            context.phase1_output = response.content
            self.logger.info("Phase 1 completed successfully", 
                           output_length=len(response.content) if response.content else 0)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Phase 1 failed: {e}")
            return False
    
    def _run_phase2(self, context: PipelineContext) -> bool:
        """
        Phase 2: Command Extraction
        
        Send Phase 1 output to base model to compress all necessary commands
        into a single code block.
        
        If no code block found, re-run Phase 2 (up to 3 retries).
        
        Returns:
            True if successful, False otherwise
        """
        self.logger.info("Phase 2: Command Extraction started")
        context.current_phase = PipelinePhase.PHASE2_COMMAND_EXTRACTION
        
        # Determine what to use as input for Phase 2
        # On first iteration, use Phase 1 output
        # On subsequent iterations, use Phase 4 output
        phase_input = context.phase4_output if context.phase4_output else context.phase1_output
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Create request for Phase 2
                request = ModelRequest(
                    task_type=TaskType.PHASE2_COMMAND_EXTRACTION,
                    prompt=phase_input,
                    context={
                        "phase_1_output": phase_input,
                    },
                    max_tokens=3000,
                    temperature=0.3
                )
                
                # Run model
                response = self.model_runner.run_model(request)
                
                if not response.success:
                    self.logger.error(f"Phase 2 model execution failed: {response.error}")
                    if attempt < max_retries - 1:
                        continue
                    return False
                
                # Extract code block from response
                commands = self._extract_code_block(response.content)
                
                if commands:
                    context.extracted_commands = commands
                    self.logger.info("Phase 2 completed successfully",
                                   commands_length=len(commands))
                    return True
                else:
                    self.logger.warning(f"Phase 2: No code block found in attempt {attempt + 1}")
                    if attempt < max_retries - 1:
                        continue
                    return False
                    
            except Exception as e:
                self.logger.error(f"Phase 2 failed on attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    continue
                return False
        
        return False
    
    def _run_phase3(self, context: PipelineContext) -> bool:
        """
        Phase 3: Command Execution
        
        Execute the extracted commands by injecting all lines into the terminal at once.
        Reuse the same terminal session throughout the entire task.
        Terminal logs accumulate continuously.
        
        Returns:
            True if successful, False otherwise
        """
        self.logger.info("Phase 3: Command Execution started")
        context.current_phase = PipelinePhase.PHASE3_COMMAND_EXECUTION
        
        try:
            if not context.extracted_commands:
                self.logger.error("Phase 3: No commands to execute")
                return False
            
            # Parse commands from the extracted code block
            commands = self._parse_commands(context.extracted_commands)
            
            if not commands:
                self.logger.error("Phase 3: No valid commands parsed")
                return False
            
            self.logger.info(f"Phase 3: Executing {len(commands)} commands in batch")
            
            # Execute all commands in a single batch (maintaining same terminal session)
            result = self.terminal_history.execute_commands_batch(
                commands, 
                timeout=self.command_timeout
            )
            
            # Log result
            if result.get("success"):
                self.logger.info("Phase 3: Batch execution succeeded")
            else:
                self.logger.warning(f"Phase 3: Batch execution failed: {result.get('stderr', 'Unknown')}")
            
            # Update terminal log in context
            context.terminal_log = self.terminal_history.display_terminal_log(max_entries=1000)
            
            self.logger.info("Phase 3 completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Phase 3 failed: {e}")
            return False
    
    def _run_phase4(self, context: PipelineContext) -> bool:
        """
        Phase 4: Log Evaluation and Re-execution Decision
        
        Send terminal logs to base model for evaluation.
        
        Returns:
            True if the Phase 4 output contains the word "failure" (continue loop),
            False if "failure" is not present (proceed to Phase 5)
        """
        self.logger.info("Phase 4: Log Evaluation started")
        context.current_phase = PipelinePhase.PHASE4_LOG_EVALUATION
        
        try:
            # Get full terminal log
            full_terminal_log = self.terminal_history.display_terminal_log(max_entries=1000)
            
            # Create request for Phase 4
            request = ModelRequest(
                task_type=TaskType.PHASE4_LOG_EVALUATION,
                prompt=full_terminal_log,
                context={
                    "user_prompt": context.user_prompt,
                    "full_terminal_log_so_far": full_terminal_log,
                },
                max_tokens=4000,
                temperature=0.5
            )
            
            # Run model
            response = self.model_runner.run_model(request)
            
            if not response.success:
                self.logger.error(f"Phase 4 model execution failed: {response.error}")
                # Assume success to proceed to Phase 5
                return False
            
            context.phase4_output = response.content
            
            # Check if output contains the word "failure" (case-insensitive)
            has_failure = self._has_failure_indicator(response.content)
            
            self.logger.info("Phase 4 completed",
                           has_failure=has_failure,
                           output_length=len(response.content))
            
            # If "failure" is present, the task failed - continue Phase 2-4 loop
            # If "failure" is NOT present, the task succeeded - proceed to Phase 5
            return has_failure
            
        except Exception as e:
            self.logger.error(f"Phase 4 failed: {e}")
            # On error, proceed to Phase 5
            return False
    
    def _run_phase5(self, context: PipelineContext) -> bool:
        """
        Phase 5: Summary Generation and Display
        
        Generate human-readable summary of what was done and the result.
        
        Returns:
            True if successful, False otherwise
        """
        self.logger.info("Phase 5: Summary Generation started")
        context.current_phase = PipelinePhase.PHASE5_SUMMARY_GENERATION
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Get full terminal log
                full_terminal_log = self.terminal_history.display_terminal_log(max_entries=1000)
                
                # Create request for Phase 5
                request = ModelRequest(
                    task_type=TaskType.PHASE5_SUMMARY_GENERATION,
                    prompt=full_terminal_log,
                    context={
                        "user_prompt": context.user_prompt,
                        "full_terminal_log": full_terminal_log,
                    },
                    max_tokens=4000,
                    temperature=0.7
                )
                
                # Run model
                response = self.model_runner.run_model(request)
                
                if not response.success:
                    self.logger.error(f"Phase 5 model execution failed: {response.error}")
                    if attempt < max_retries - 1:
                        continue
                    return False
                
                # Extract and concatenate all code blocks
                summary = self._extract_all_code_blocks(response.content)
                
                if summary:
                    context.final_summary = summary
                    self.logger.info("Phase 5 completed successfully",
                                   summary_length=len(summary))
                    
                    # Display the summary
                    print("")
                    print(context.final_summary)
                    
                    return True
                else:
                    self.logger.warning(f"Phase 5: No code block found in attempt {attempt + 1}")
                    if attempt < max_retries - 1:
                        continue
                    return False
                    
            except Exception as e:
                self.logger.error(f"Phase 5 failed on attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    continue
                return False
        
        return False
    
    def _extract_code_block(self, text: str) -> Optional[str]:
        """
        Extract code block from text.
        If multiple code blocks are present, use the last one.
        Supports any language specifier (bash, python, json, etc.) or no specifier.
        
        Args:
            text: Text containing code blocks
            
        Returns:
            Extracted code block content or None if not found
        """
        # Pattern to match code blocks with any language specifier
        pattern = r'```(?:\w+)?\s*\n?(.*?)```'
        
        # Find all code blocks (non-greedy match)
        matches = re.findall(pattern, text, re.DOTALL)
        
        if matches:
            # Use the last code block if multiple exist
            return matches[-1].strip()
        
        return None
    
    def _has_code_block(self, text: str) -> bool:
        """Check if text contains a code block"""
        pattern = r'```(?:\w+)?\s*\n?.*?```'
        return bool(re.search(pattern, text, re.DOTALL))
    
    def _has_failure_indicator(self, text: str) -> bool:
        """
        Check if text indicates command failure using multiple indicators.
        
        Uses weighted scoring based on:
        - Explicit failure keywords
        - Error patterns in terminal output
        - Command exit status indicators
        - Negative sentiment words
        
        Returns True only if failure score exceeds threshold.
        """
        if not text:
            return False
        
        text_lower = text.lower()
        
        # Explicit failure indicators (weight: 3)
        failure_keywords = [
            'failure', 'failed', 'error occurred', 'unsuccessful',
            'command failed', 'execution failed', 'task failed'
        ]
        
        # Terminal error patterns (weight: 2)
        error_patterns = [
            'exit code:', 'returned non-zero', 'command not found',
            'permission denied', 'no such file', 'syntax error',
            'segmentation fault', 'core dumped', 'killed', 'terminated'
        ]
        
        # Negative indicators (weight: 1)
        negative_indicators = [
            'error', 'warning', 'abort', 'fatal', 'critical',
            'cannot', 'unable', 'not found', 'invalid'
        ]
        
        # Success indicators that override failure (weight: -5)
        success_indicators = [
            'success', 'completed successfully', 'done', 'finished',
            'task completed', 'execution successful', 'all commands executed'
        ]
        
        score = 0
        
        # Check failure keywords
        for keyword in failure_keywords:
            if keyword in text_lower:
                score += 3
                self.logger.debug(f"Failure indicator found: '{keyword}' (+3)")
        
        # Check error patterns
        for pattern in error_patterns:
            if pattern in text_lower:
                score += 2
                self.logger.debug(f"Error pattern found: '{pattern}' (+2)")
        
        # Check negative indicators
        for indicator in negative_indicators:
            if indicator in text_lower:
                score += 1
                self.logger.debug(f"Negative indicator found: '{indicator}' (+1)")
        
        # Check success indicators (these reduce score)
        for indicator in success_indicators:
            if indicator in text_lower:
                score -= 5
                self.logger.debug(f"Success indicator found: '{indicator}' (-5)")
        
        # Threshold: score >= 3 indicates failure
        is_failure = score >= 3
        
        self.logger.info(f"Failure analysis complete",
                        score=score,
                        threshold=3,
                        is_failure=is_failure,
                        text_preview=text[:100] + "..." if len(text) > 100 else text)
        
        return is_failure
    
    def _extract_all_code_blocks(self, text: str) -> Optional[str]:
        """
        Extract and concatenate all code blocks from text.
        
        Args:
            text: Text containing code blocks
            
        Returns:
            Concatenated code block content or None if not found
        """
        pattern = r'```(?:\w+)?\s*\n?(.*?)```'
        matches = re.findall(pattern, text, re.DOTALL)
        
        if matches:
            # Concatenate all code blocks
            return "\n\n".join(block.strip() for block in matches)
        
        return None
    
    def _parse_commands(self, code_block: str) -> List[str]:
        """
        Parse commands from a code block.
        
        Args:
            code_block: Code block content
            
        Returns:
            List of individual commands
        """
        commands = []
        
        # Split by newlines and filter
        lines = code_block.split('\n')
        
        for line in lines:
            line = line.strip()
            
            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue
            
            # Skip markdown formatting remnants
            if line.startswith('```'):
                continue
            
            commands.append(line)
        
        return commands
    
    def _get_os_info(self) -> str:
        """Get OS information for CLI context"""
        try:
            import os
            
            system = platform.system()
            release = platform.release()
            version = platform.version()
            machine = platform.machine()
            
            # Get shell info for Unix-like systems
            shell = os.environ.get('SHELL', 'Unknown')
            
            if system == "Linux":
                try:
                    with open('/etc/os-release', 'r') as f:
                        lines = f.readlines()
                    distro_info = {}
                    for line in lines:
                        if '=' in line:
                            key, value = line.strip().split('=', 1)
                            distro_info[key] = value.strip('"')
                    
                    distro_name = distro_info.get('NAME', 'Unknown Linux')
                    distro_version = distro_info.get('VERSION', '')
                    os_info = f"{distro_name} {distro_version} ({system} {release} {machine})"
                except:
                    os_info = f"Linux {release} {machine}"
            
            elif system == "Darwin":
                os_info = f"macOS {release} {machine}"
            
            elif system == "Windows":
                os_info = f"Windows {release} {machine}"
            
            else:
                os_info = f"{system} {release} {machine}"
            
            if system in ["Linux", "Darwin"]:
                os_info += f" (Shell: {shell})"
            
            return os_info
            
        except Exception as e:
            self.logger.warning(f"Failed to get OS info: {e}")
            return "Unknown OS"


def get_five_phase_engine(config: Optional[Dict[str, Any]] = None) -> FivePhaseEngine:
    """Get 5-Phase Pipeline Engine instance"""
    return FivePhaseEngine(config)
