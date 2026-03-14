"""
Two-Phase Execution Engine for CLI AI Agent System
Implements the revised architecture: Task List Generation + Sequential Task Execution
Zero-defect policy: robust execution with context preservation
"""

import time
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
from enum import Enum

from ..external_integration.model_runner import ModelRunner, TaskType
from ..utils.exceptions import ExecutionError, ValidationError, TaskGenerationError
from ..utils.logger import get_logger
from .command_parser import ParsedCommand, CommandType
from .terminal_history import TerminalHistory, get_terminal_history
from .command_output import get_command_formatter, format_command_output
from .task_robustness_manager import get_task_robustness_manager, TaskRobustnessManager, RobustnessConfig, TaskCompletionStatus


@dataclass
class AutomationResult:
    """Result of automation operation"""
    success: bool
    action: str
    duration: float
    method: str
    error: Optional[str] = None
    coordinates: Optional[Tuple[float, float]] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class Task:
    """Individual task structure"""
    description: str


@dataclass
class TaskList:
    """Task list structure"""
    tasks: List[str]  # Simple list of task descriptions
    instruction: str
    generation_time: float = 0.0


class ExecutionPhase(Enum):
    """Execution phases"""
    TASK_GENERATION = "task_generation"
    TASK_EXECUTION = "task_execution"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ExecutionContext:
    """Execution context for CLI Architecture"""
    phase: ExecutionPhase
    current_task_index: int = 0
    previous_command_output: Optional[str] = None
    previous_command: Optional[str] = None
    task_list: Optional[TaskList] = None
    executed_commands: List[str] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class TwoPhaseEngine:
    """Two-phase execution engine implementing the revised architecture"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = get_logger("two_phase_engine")
        
        # Initialize terminal history system using global instance
        self.terminal_history = get_terminal_history()
        
        # Initialize command output formatter
        self.command_formatter = get_command_formatter()
        
        # Initialize components
        auto_install_sdks = self.config.pop('auto_install_sdks', False)
        self.model_runner = ModelRunner(auto_install_sdks=auto_install_sdks)
        
        
        # Initialize task robustness manager
        robustness_config = RobustnessConfig(
            min_commands_per_task=self.config.get("min_commands_per_task", 0),
            max_commands_per_task=self.config.get("max_commands_per_task", 0),
            require_completion_validation=self.config.get("require_completion_validation", True),
            progress_check_interval=self.config.get("progress_check_interval", 3),
            completion_confidence_threshold=self.config.get("completion_confidence_threshold", 0.8),
            allow_early_completion=self.config.get("allow_early_completion", False),
            force_full_execution=self.config.get("force_full_execution", True)
        )
        self.robustness_manager = get_task_robustness_manager(robustness_config)
        
        
        # Execution settings
        self.max_task_retries = self.config.get("max_task_retries", 3)
        self.max_command_retries = self.config.get("max_command_retries", 3)
        self.command_timeout = self.config.get("command_timeout", 30)
        self.task_timeout = self.config.get("task_timeout", 300)  # 5 minutes per task
        
        self.logger.info("Two-phase execution engine initialized with terminal history system, command output formatter, and task robustness manager")
    
    def execute_instruction(self, instruction: str) -> ExecutionContext:
        """Execute user instruction using two-phase approach"""
        self.logger.info(f"Starting two-phase execution for instruction: {instruction}")
        
        # Initialize execution context
        context = ExecutionContext(
            phase=ExecutionPhase.TASK_GENERATION,
            metadata={"instruction": instruction}
        )
        
        try:
            # Phase 1: Task List Generation
            context = self._execute_phase_1(context, instruction)
            
            # Phase 2: Sequential Task Execution
            context = self._execute_phase_2(context)
            
            # Mark as completed
            context.phase = ExecutionPhase.COMPLETED
            context.end_time = time.time()
            
            # End terminal history session
            self.terminal_history.end_session()
            
            self.logger.info(
                "Two-phase execution completed successfully",
                total_tasks=len(context.task_list.tasks) if context.task_list else 0,
                executed_commands=len(context.executed_commands),
                duration=context.end_time - context.start_time
            )
            
            return context
            
        except Exception as e:
            context.phase = ExecutionPhase.FAILED
            context.error = str(e)
            context.end_time = time.time()
            
            # End terminal history session even on failure
            self.terminal_history.end_session()
            
            self.logger.error(f"Two-phase execution failed: {e}")
            raise ExecutionError(f"Two-phase execution failed: {e}")
    
    def _execute_phase_1(self, context: ExecutionContext, instruction: str) -> ExecutionContext:
        """Phase 1: Task List Generation"""
        self.logger.info("Starting Phase 1: Task List Generation")
        
        try:
            # Get OS information for CLI context
            os_info = self._get_os_info()
            context.metadata["os_info"] = os_info
            
            # Generate task list using AI
            self.logger.info("Generating task list with AI")
            try:
                response = self.model_runner.generate_tasks(instruction, os_info)
            except Exception as e:
                self.logger.error(f"Model runner failed in Phase 1: {type(e).__name__}: {e}")
                raise TaskGenerationError(f"Model runner error: {str(e)}")
            
            if not response.success:
                raise TaskGenerationError(f"Task generation failed: {response.error}")
            
            # Parse task list from AI response
            task_list = self._parse_task_list_response(response.content, instruction)
            context.task_list = task_list
            
            self.logger.info(
                "Phase 1 completed successfully",
                task_count=len(task_list.tasks),
                generation_time=task_list.generation_time,
            )
            
            # Transition to Phase 2
            context.phase = ExecutionPhase.TASK_EXECUTION
            context.current_task_index = 0
            
            return context
            
        except Exception as e:
            self.logger.error(f"Phase 1 failed: {e}")
            raise ExecutionError(f"Task generation phase failed: {e}")
        
        return context
    
    def _execute_phase_2(self, context: ExecutionContext) -> ExecutionContext:
        """Phase 2: Sequential Task Execution with context preservation"""
        self.logger.info("Starting Phase 2: Sequential Task Execution")
        
        if not context.task_list or not context.task_list.tasks:
            raise ValidationError("No tasks to execute", "task_list", context.task_list)
        
        try:
            # Execute each task sequentially
            for task_index, task_description in enumerate(context.task_list.tasks):
                task_start_time = time.time()
                self.logger.info(f"Executing task {task_index + 1}: {task_description}")
                
                # Update context
                context.current_task_index = task_index
                
                # Execute task with command generation loop
                success = self._execute_task_with_command_loop(context, task_description)
                
                # Log task execution completion
                task_duration = time.time() - task_start_time
                commands_executed = len([cmd for cmd in context.executed_commands if cmd])
                self.logger.log_task_execution(
                    task_index=task_index + 1,
                    task_description=task_description,
                    success=success,
                    commands_executed=commands_executed,
                    duration=task_duration
                )
                
                if not success:
                    self.logger.warning(f"Task {task_index + 1} failed, continuing to next task")
                
                # Small delay between tasks
                time.sleep(0.5)
            
            self.logger.info("Phase 2 completed successfully")
            return context
            
        except Exception as e:
            self.logger.error(f"Phase 2 failed: {e}")
            raise ExecutionError(f"Task execution phase failed: {e}")
    
    def _execute_task_with_command_loop(self, context: ExecutionContext, task_description: str) -> bool:
        """Execute a single task using command generation loop with reflection and robustness management"""
        task_start_time = time.time()
        command_count = 0
        
        # Start robustness tracking for this task
        task_id = self.robustness_manager.start_task_execution(task_description, estimated_steps=5)
        context.metadata["current_task_id"] = task_id
        
        # Get dynamic command limits from robustness manager
        max_commands_per_task = self.robustness_manager.config.max_commands_per_task
        
        self.logger.info(f"Starting robust task execution for: {task_description}",
                        task_id=task_id, max_commands=max_commands_per_task)
        
        while max_commands_per_task == 0 or command_count < max_commands_per_task:
            # Check if robustness manager allows continuation
            should_continue, continue_reason = self.robustness_manager.should_continue_task_execution(task_id, command_count)
            if not should_continue:
                self.logger.info(f"Robustness manager stopping execution: {continue_reason}",
                               task_id=task_id)
                break
            
            
            # Get terminal history for context
            last_command_output = self.terminal_history.get_last_command_output()
            recent_terminal_entries = self.terminal_history.get_recent_output(5)
            
            # Format previous terminal actions for better context
            formatted_previous_actions = ""
            if recent_terminal_entries:
                action_lines = []
                for entry in recent_terminal_entries:
                    if entry.entry_type.value == "command":
                        status = ""
                        if entry.return_code is not None:
                            if entry.return_code == 0:
                                status = " (SUCCESS)"
                            else:
                                status = f" (FAILED: return code {entry.return_code})"
                        elif entry.duration is None:
                            status = " (EXECUTION PENDING)"
                        else:
                            status = " (STATUS UNKNOWN)"
                        
                        action_lines.append(f"- EXECUTED: {entry.content}{status}")
                    elif entry.entry_type.value == "output":
                        action_lines.append(f"- OUTPUT: {entry.content.strip()}")
                    elif entry.entry_type.value == "error":
                        action_lines.append(f"- ERROR: {entry.content.strip()}")
                
                formatted_previous_actions = "\n".join(action_lines)
            else:
                formatted_previous_actions = "No previous terminal actions recorded"
            
            # Get current working directory
            current_directory = self.terminal_history.get_current_working_directory()
            
            # Get current OS info and previous command output
            os_info = context.metadata.get("os_info", "Unknown")
            previous_output = context.previous_command_output
            
            # Generate command using AI with context preservation and reflection
            self.logger.debug(f"Generating command for task: {task_description}")
            
            response = self.model_runner.parse_command(
                task_description,
                os_info,
                context={
                    "task_description": task_description,
                    "previous_terminal_actions": formatted_previous_actions,
                    "last_command_output": last_command_output,
                    "current_directory": current_directory,
                    "current_task_id": task_id,
                    "command_count": command_count,
                    "robustness_enabled": True,
                    "previous_output": previous_output
                }
            )
            
            if not response.success:
                self.logger.error(f"Command generation failed: {response.error}")
                return False
            
            # Parse command
            command_text = response.content.strip()
            
            # Log command generation with enhanced details
            self.logger.log_command_generation(
                task_description=task_description,
                command=command_text,
                success=True,
                model=response.model,
                latency=response.latency
            )
            
            # Parse command using command parser
            from .command_parser import CommandParser
            parser = CommandParser()
            parsed_command = parser.parse_command(command_text, context.previous_command_output)
            
            # Check for END command with robustness validation
            if parsed_command.type == CommandType.END:
                # Validate with robustness manager before allowing completion
                allow_completion, completion_reason = self.robustness_manager.should_allow_task_completion(task_id, "END")
                
                if allow_completion:
                    self.logger.info(f"Task completion approved by robustness manager: {completion_reason}",
                                   task_id=task_id, commands_executed=command_count + 1)
                    # Update progress for final step
                    self.robustness_manager.update_task_progress(task_id, "END command executed")
                    break
                else:
                    self.logger.warning(f"Task completion blocked by robustness manager: {completion_reason}",
                                      task_id=task_id, commands_executed=command_count)
                    # Continue execution instead of breaking
                    command_count += 1
                    continue
            
            
            # Execute the command
            success = self._execute_single_command(parsed_command, context)
            
            # Update robustness progress with completion analysis
            completion_indicators = []
            missing_indicators = []
            
            if success:
                completion_indicators.append(f"Command executed: {command_text}")
                # Analyze if this command moved the task forward
                if self._analyze_command_progress(command_text, task_description):
                    completion_indicators.append("Progress made toward task goal")
                else:
                    missing_indicators.append("Command may not have advanced task")
            else:
                missing_indicators.append(f"Failed command: {command_text}")
            
            # Update task progress in robustness manager
            self.robustness_manager.update_task_progress(
                task_id, 
                command_text,
                completion_indicators=completion_indicators,
                missing_indicators=missing_indicators
            )
            
            if not success:
                self.logger.warning(f"Command execution failed: {command_text}")
                # Continue trying with next command
            
            # Update context for next iteration
            context.previous_command_output = self._capture_command_output(command_text)
            context.previous_command = command_text
            context.executed_commands.append(command_text)
            command_count += 1
            
            # Check timeout
            if time.time() - task_start_time > self.task_timeout:
                self.logger.error(f"Task timeout after {self.task_timeout} seconds")
                # End task with timeout status
                self.robustness_manager.end_task_execution(task_id, TaskCompletionStatus.FAILED)
                return False
        
        # Determine final task status and end tracking
        final_status = self.robustness_manager.get_task_status(task_id)
        task_summary = self.robustness_manager.end_task_execution(task_id, final_status)
        
        # Store summary in context
        context.metadata["task_robustness_summary"] = task_summary
        
        self.logger.info(f"Task execution completed with robustness management",
                        task_id=task_id, final_status=final_status.value,
                        commands_executed=command_count,
                        confidence_score=task_summary.get("confidence_score", 0.0))
        
        return command_count > 0
    
    def _execute_single_command(self, parsed_command: ParsedCommand, context: ExecutionContext) -> bool:
        """Execute a single command with terminal history integration and formatted output"""
        try:
            # Extract the actual command text based on type
            if parsed_command.type == CommandType.CLI_COMMAND:
                command_text = parsed_command.parameters.get("command", "")
            elif parsed_command.type == CommandType.END:
                command_text = "END"
            elif parsed_command.type == CommandType.REGENERATE_STEP:
                command_text = "REGENERATE_STEP"
            else:
                command_text = ""
            
            self.logger.info(f"Executing command: {command_text}")
            
            # Execute parsed command
            result = self._execute_parsed_command(parsed_command)
            
            # Generate reasoning and target based on command type and result
            reasoning, target = self._generate_reasoning_and_target(parsed_command, result)
            
            # Format output with reasoning and target
            if result.success:
                terminal_content = f"Command executed successfully: {command_text}"
                
                try:
                    # Use command formatter for proper output format
                    formatted_output = self.command_formatter.format_command_output(
                        reasoning=reasoning,
                        target=target,
                        command=command_text,
                        terminal_content=terminal_content,
                        coordinates=result.coordinates,
                        metadata={
                            "duration": result.duration,
                            "method": result.method
                        }
                    )
                    
                    # Print formatted output
                    print(formatted_output)
                    
                except Exception as e:
                    self.logger.error(f"Error formatting command output: {e}")
                    # Fallback to simple terminal log display
                    terminal_log = self.terminal_history.display_terminal_log(max_entries=5)
                    print(f"Reasoning: {reasoning}")
                    print(f"Target: {target}")
                    print(f"{command_text}")
                    print(f"Terminal Log:")
                    print(terminal_log)
                
                self.logger.info(f"Command executed successfully: {command_text}")
                return True
            else:
                # Save failure information with formatted output
                terminal_content = f"Command execution failed: {command_text}"
                
                try:
                    # Use command formatter for failure output
                    formatted_output = self.command_formatter.format_failure_output(
                        reasoning=f"Failed to execute {parsed_command.type.value}: {result.error}",
                        target=target,
                        command=command_text,
                        error_message=result.error or "Unknown error",
                        coordinates=result.coordinates
                    )
                    
                    # Print formatted output
                    print(formatted_output)
                    
                except Exception as e:
                    self.logger.error(f"Error formatting failure output: {e}")
                    # Fallback to simple terminal log display
                    terminal_log = self.terminal_history.display_terminal_log(max_entries=5)
                    print(f"Reasoning: {reasoning}")
                    print(f"Target: {target}")
                    print(f"{command_text}")
                    print(f"Terminal Log:")
                    print(terminal_log)
                
                self.logger.error(f"Command execution failed: {command_text}, error: {result.error}")
                return False
                
        except Exception as e:
            # Save exception information with formatted output
            terminal_content = f"Command execution error: {command_text}"
            
            try:
                # Use command formatter for exception output
                formatted_output = self.command_formatter.format_failure_output(
                    reasoning=f"Exception during command execution: {str(e)}",
                    target="unknown target",
                    command=command_text,
                    error_message=str(e)
                )
                
                # Print formatted output
                print(formatted_output)
                
            except Exception as format_error:
                self.logger.error(f"Error formatting exception output: {format_error}")
                # Fallback to simple terminal log display
                terminal_log = self.terminal_history.display_terminal_log(max_entries=5)
                print(f"Reasoning: Exception during command execution")
                print(f"Target: unknown target")
                print(f"{command_text}")
                print(f"Terminal Log:")
                print(terminal_log)
            
            self.logger.error(f"Command execution error: {command_text}, error: {e}")
            return False
    
    def _generate_reasoning_and_target(self, command: ParsedCommand, result: AutomationResult) -> tuple[str, str]:
        """Generate reasoning and target descriptions based on command type and result"""
        command_type = command.type.value
        
        if command_type == "cli_command":
            cli_command = command.parameters["command"]
            reasoning = f"Need to execute CLI command: '{cli_command}' to perform system operation"
            target = f"System command execution: {cli_command}"
            
        elif command_type == "end":
            reasoning = "Need to end the current task execution"
            target = "Task completion signal"
            
        elif command_type == "regenerate_step":
            reasoning = "Need to regenerate the current step with new approach"
            target = "Step regeneration request"
            
        else:
            reasoning = f"Need to execute {command_type} command"
            target = f"Command target for {command_type}"
        
        # Adjust reasoning based on result
        if not result.success:
            reasoning = f"Attempted to {reasoning.lower()} but failed"
            target = f"Failed target: {target}"
        
        return reasoning, target
    
    def _analyze_command_progress(self, command_text: str, task_description: str) -> bool:
        """Analyze if a command made meaningful progress toward the task goal"""
        # Simple heuristic-based progress analysis
        command_lower = command_text.lower()
        task_lower = task_description.lower()
        
        # Check if command contains task-relevant keywords
        task_keywords = set(task_lower.split())
        command_keywords = set(command_lower.split())
        
        # High relevance if command shares keywords with task
        relevance_score = len(task_keywords.intersection(command_keywords)) / max(1, len(task_keywords))
        
        # Check for action verbs that typically indicate progress
        progress_actions = ['click', 'type', 'enter', 'select', 'choose', 'open', 'close', 'save', 'submit', 'confirm']
        has_progress_action = any(action in command_lower for action in progress_actions)
        
        # Check for navigation actions
        navigation_actions = ['scroll', 'drag', 'move', 'go', 'navigate']
        has_navigation_action = any(action in command_lower for action in navigation_actions)
        
        # Determine if command likely made progress
        made_progress = (
            relevance_score > 0.2 or  # Keyword relevance
            has_progress_action or   # Progress action
            has_navigation_action     # Navigation action
        )
        
        self.logger.debug("Command progress analysis",
                         command=command_text,
                         task=task_description,
                         relevance_score=relevance_score,
                         has_progress_action=has_progress_action,
                         has_navigation_action=has_navigation_action,
                         made_progress=made_progress)
        
        return made_progress
    
    def _execute_parsed_command(self, command: ParsedCommand) -> AutomationResult:
        """Execute a parsed command"""
        try:
            if command.type == CommandType.CLI_COMMAND:
                cli_command = command.parameters["command"]
                return self._execute_cli_command(cli_command)
                
            elif command.type == CommandType.END:
                # END command - no action needed
                return AutomationResult(
                    success=True,
                    action="end",
                    duration=0.0,
                    method="builtin",
                )
                
            elif command.type == CommandType.REGENERATE_STEP:
                # REGENERATE_STEP command - no action needed
                return AutomationResult(
                    success=True,
                    action="regenerate_step",
                    duration=0.0,
                    method="builtin",
                )
                
            else:
                raise ExecutionError(f"Unsupported command type: {command.type}")
                
        except Exception as e:
            return AutomationResult(
                success=False,
                action="command_execution",
                duration=0.0,
                method="error",
                error=str(e),
            )
    
    def _execute_cli_command(self, cli_command: str) -> AutomationResult:
        """Execute a CLI command using terminal history system"""
        start_time = time.time()
        try:
            self.logger.info(f"Executing CLI command: {cli_command}")
            
            # Execute command through terminal history system
            result = self.terminal_history.execute_command(cli_command)
            
            duration = time.time() - start_time
            
            # Validate execution results
            if result.get("success", False):
                self.logger.info(f"Command succeeded: {cli_command}")
                return AutomationResult(
                    success=True,
                    action="cli_command",
                    duration=duration,
                    method="subprocess",
                    coordinates=None,
                    metadata={
                        "command": cli_command,
                        "stdout": result.get("stdout", ""),
                        "stderr": result.get("stderr", ""),
                        "return_code": result.get("return_code", 0),
                        "working_directory": result.get("working_directory", "")
                    }
                )
            else:
                error_msg = result.get("stderr", "Unknown error")
                self.logger.error(f"Command failed: {cli_command}, error: {error_msg}")
                return AutomationResult(
                    success=False,
                    action="cli_command",
                    duration=duration,
                    method="subprocess",
                    error=error_msg,
                    metadata={
                        "command": cli_command,
                        "stdout": result.get("stdout", ""),
                        "stderr": error_msg,
                        "return_code": result.get("return_code", 1),
                        "working_directory": result.get("working_directory", "")
                    }
                )
            
        except Exception as e:
            self.logger.error(f"Exception executing command '{cli_command}': {e}")
            return AutomationResult(
                success=False,
                action="cli_command",
                duration=time.time() - start_time,
                method="error",
                error=str(e),
                metadata={"command": cli_command}
            )
    
    
    def _parse_task_list_response(self, response_content: str, instruction: str) -> TaskList:
        """Parse AI response into TaskList"""
        try:
            import re
            
            # Extract numbered tasks
            tasks = []
            lines = response_content.strip().split('\n')
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Match numbered patterns (1., 1), etc.)
                match = re.match(r'^\d+[\.\)]\s*(.+)$', line)
                if match:
                    task_text = match.group(1).strip()
                    if task_text:
                        tasks.append(task_text)
            
            if not tasks:
                raise TaskGenerationError("No valid tasks found in AI response")
            
            return TaskList(
                tasks=tasks,
                instruction=instruction,
                generation_time=time.time(),
            )
            
        except Exception as e:
            raise TaskGenerationError(f"Failed to parse task list response: {e}")
    
    def _get_os_info(self) -> str:
        """Get OS information for CLI context"""
        try:
            import platform
            import os
            
            # Get basic system info
            system = platform.system()
            release = platform.release()
            version = platform.version()
            machine = platform.machine()
            processor = platform.processor()
            
            # Get shell info for Unix-like systems
            shell = os.environ.get('SHELL', 'Unknown')
            
            # Format OS information
            if system == "Linux":
                # Try to get distribution info
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
            
            elif system == "Darwin":  # macOS
                os_info = f"macOS {release} {machine}"
            
            elif system == "Windows":
                os_info = f"Windows {release} {machine}"
            
            else:
                os_info = f"{system} {release} {machine}"
            
            # Add shell info for Unix-like systems
            if system in ["Linux", "Darwin"]:
                os_info += f" (Shell: {shell})"
            
            return os_info
            
        except Exception as e:
            self.logger.warning(f"Failed to get OS info: {e}")
            return "Unknown OS"
    
    def _capture_command_output(self, command_text: str) -> str:
        """Capture the output of the previous command using terminal history"""
        try:
            # Use the terminal history system to get the actual command output
            return self.terminal_history.get_last_command_output()
            
        except Exception as e:
            self.logger.warning(f"Failed to capture command output: {e}")
            return f"No output available for command: {command_text}"
    
    
