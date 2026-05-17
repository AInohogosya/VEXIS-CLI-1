"""
Command Output System for CLI Execution Engine
Handles reasoning output and command formatting according to specifications:
1. Output reasoning first
2. Output specific target for command execution
3. Output CLI command (second-to-last line)
4. Output save command (final line)
"""

import time
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass

from ..utils.logger import get_logger
from .terminal_history import get_terminal_history, TerminalEntryType


@dataclass
class CommandOutput:
		"""Structure for formatted command output"""
		reasoning: str
		target: str
		command: str
		terminal_content: str


class CommandOutputFormatter:
		"""
		Formats command output according to specifications:
		1. Reasoning about the action
		2. Specific target for command execution
		3. CLI command (second-to-last line)
		4. Terminal log display (final line)
		"""

		def __init__(self):
				self.logger = get_logger("command_output")
				self.terminal_history = get_terminal_history()

		def format_command_output(
				self,
				reasoning: str,
				target: str,
				command: str,
				terminal_content: str,
				coordinates: Optional[Tuple[float, float]] = None,
				**kwargs
		) -> str:
				"""
				Format command output according to specifications

				Args:
						reasoning: Reasoning about why this action is being taken
						target: Specific target for clicking (e.g., "search bar at top right")
						command: The actual command to execute (e.g., "click(0.5, 0.2)")
						terminal_content: Content for terminal log display
						coordinates: Coordinates for metadata
						**kwargs: Additional parameters

				Returns:
						str: Formatted multi-line output
				"""
				# Get recent terminal log to display
				recent_terminal_log = self.terminal_history.display_terminal_log(max_entries=10)

				# Create the formatted output
				output_lines = [
						f"Reasoning: {reasoning}",
						f"Target: {target}",
						f"{command}",
						f"Terminal Log:",
						recent_terminal_log
				]

				# Join with newlines
				formatted_output = "\n".join(output_lines)

				self.logger.debug(f"Formatted command output with terminal log")
				return formatted_output

		def format_failure_output(
				self,
				reasoning: str,
				target: str,
				command: str,
				error_message: str,
				coordinates: Optional[Tuple[float, float]] = None,
				**kwargs
		) -> str:
				"""
				Format a failure output with reasoning and target

				Args:
						reasoning: Reasoning about why the action failed
						target: Target that was attempted
						command: The command that failed
						error_message: Description of the failure
						coordinates: Coordinates that failed
						**kwargs: Additional parameters

				Returns:
						str: Formatted multi-line output
				"""
				terminal_content = f"Failed to {target}: {error_message}"

				# Get recent terminal log to display
				recent_terminal_log = self.terminal_history.display_terminal_log(max_entries=10)

				# Create the formatted output
				output_lines = [
						f"Reasoning: {reasoning}",
						f"Target: {target}",
						f"{command}",
						f"Terminal Log:",
						recent_terminal_log
				]

				# Join with newlines
				formatted_output = "\n".join(output_lines)

				self.logger.debug(f"Formatted failure output with terminal log")
				return formatted_output

		def format_extraction_output(
				self,
				reasoning: str,
				target: str,
				extracted_info: Dict[str, Any],
				terminal_content: Optional[str] = None,
				**kwargs
		) -> str:
				"""
				Format an extraction output with reasoning and target

				Args:
						reasoning: Reasoning about the extraction
						target: Target of extraction (e.g., "error message dialog")
						extracted_info: Dictionary of extracted information
						terminal_content: Content for terminal log display (auto-generated if None)
						**kwargs: Additional parameters

				Returns:
						str: Formatted multi-line output
				"""
				if terminal_content is None:
						info_summary = ", ".join([f"{k}: {v}" for k, v in extracted_info.items()])
						terminal_content = f"Extracted from {target}: {info_summary}"

				# Get recent terminal log to display
				recent_terminal_log = self.terminal_history.display_terminal_log(max_entries=10)

				# For extraction, we might not have a click command
				command = kwargs.get('operation_command', 'extract()')

				# Create the formatted output
				output_lines = [
						f"Reasoning: {reasoning}",
						f"Target: {target}",
						f"{command}",
						f"Terminal Log:",
						recent_terminal_log
				]

				# Join with newlines
				formatted_output = "\n".join(output_lines)

				self.logger.debug(f"Formatted extraction output with terminal log")
				return formatted_output


# Global formatter instance
_global_formatter: Optional[CommandOutputFormatter] = None


def get_command_formatter() -> CommandOutputFormatter:
		"""Get global command formatter instance"""
		global _global_formatter
		if _global_formatter is None:
				_global_formatter = CommandOutputFormatter()
		return _global_formatter


def format_command_output(
		reasoning: str,
		target: str,
		command: str,
		terminal_content: str,
		**kwargs
) -> str:
		"""
		Global function to format any command with reasoning and target
		"""
		return get_command_formatter().format_command_output(
				reasoning, target, command, terminal_content, **kwargs
		)
