"""
Command Parser for CLI AI Agent System
CLI Architecture: Standard Linux/Unix Command Processing
"""

import re
import subprocess
from typing import Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum

from ..external_integration.model_runner import get_model_runner

# Simple exception classes
class CommandParsingError(Exception):
		"""Simple command parsing error"""
		pass

class ValidationError(Exception):
		"""Simple validation error"""
		pass


class CommandType(Enum):
		"""Command set for CLI Architecture"""
		CLI_COMMAND = "cli_command"
		END = "end"
		REGENERATE_STEP = "regenerate_step"


@dataclass
class ParsedCommand:
		"""Parsed command structure"""
		type: CommandType
		parameters: Dict[str, Any]
		raw_text: str


class CommandParser:
		"""Command parser for CLI Architecture with standard Linux/Unix command processing"""

		def __init__(self):
				# CLI command patterns
				self.command_patterns = self._initialize_cli_command_patterns()

		def _initialize_cli_command_patterns(self) -> Dict[CommandType, list]:
				"""Initialize CLI command patterns"""
				return {
						CommandType.CLI_COMMAND: [
								# Match any line that doesn't start with administrative commands
								re.compile(r'^(?!end\s*$|regenerate_step\s*$).+', re.IGNORECASE),
						],
						CommandType.END: [
								re.compile(r'^end\s*$', re.IGNORECASE),
						],
						CommandType.REGENERATE_STEP: [
								re.compile(r'^regenerate_step\s*$', re.IGNORECASE),
						],
				}


		def parse_command(self, command_text: str, previous_output: Optional[str] = None, context: Optional[Dict[str, Any]] = None) -> ParsedCommand:
				"""Parse command text into structured command"""
				# Validate input first
				if command_text is None:
						raise ValidationError("Command text cannot be None", "command_text", command_text)

				if not command_text or not command_text.strip():
						raise ValidationError("Command text cannot be empty", "command_text", command_text)

				try:
						# Apply more robust parsing that can handle different AI response formats
						lines = command_text.strip().split('\n')

						# Try multiple parsing strategies

						# Strategy 1: Standard 4-line format (lines 0,1,2,3)
						if len(lines) >= 3:
								command_line = lines[2].strip()

								# Check if this looks like a valid command
								if (command_line and
										not command_line.startswith("Reasoning:") and
										not command_line.startswith("Target:") and
										not command_line.startswith("Terminal Log:") and
										not command_line.startswith("save(")):

										cleaned_command = self._clean_command_text(command_line)
										parsed = self._parse_with_patterns(cleaned_command)

										if parsed:
												return parsed

						# Strategy 2: Look for any line that looks like a command
						for i, line in enumerate(lines):
								line_stripped = line.strip()
								if (line_stripped and
										not line_stripped.startswith("Reasoning:") and
										not line_stripped.startswith("Target:") and
										not line_stripped.startswith("Terminal Log:") and
										not line_stripped.startswith("save(") and
										not line_stripped.startswith("END") and
										not line_stripped.startswith("REGENERATE_STEP") and
										len(line_stripped) > 0):

										cleaned_command = self._clean_command_text(line_stripped)
										parsed = self._parse_with_patterns(cleaned_command)

										if parsed:
												return parsed

						# Strategy 3: Fallback - try to parse any non-empty line
						for i, line in enumerate(lines):
								line_stripped = line.strip()
								if line_stripped and len(line_stripped) > 0:

										cleaned_command = self._clean_command_text(line_stripped)
										parsed = self._parse_with_patterns(cleaned_command)

										if parsed:
												return parsed

						# If all strategies fail, raise validation error
						raise ValidationError(f"Invalid command format: {command_text}", "command_text", command_text)

				except ValidationError:
						# Re-raise validation errors - these are legitimate failures
						raise
				except Exception as e:
						# For other exceptions, fail the command
						raise CommandParsingError(f"Failed to parse command: {command_text}")

		def _clean_command_text(self, command_text: str) -> str:
				"""Clean and normalize command text"""
				if not command_text:
						raise ValidationError("Command text cannot be empty", "command_text", command_text)

				# Remove extra whitespace but preserve structure for validation
				cleaned = ' '.join(command_text.split())

				# Remove markdown code blocks if present
				if cleaned.startswith('```bash'):
						cleaned = cleaned[7:].strip()
				elif cleaned.startswith('```'):
						cleaned = cleaned[3:].strip()
				if cleaned.endswith('```'):
						cleaned = cleaned[:-3].strip()

				# Remove line number prefixes like "Line 3:", "Line 1:", etc.
				import re
				line_number_pattern = r'^Line \d+:\s*'
				if re.match(line_number_pattern, cleaned, re.IGNORECASE):
						cleaned = re.sub(line_number_pattern, '', cleaned, flags=re.IGNORECASE)

				# Remove common prefixes
				prefixes_to_remove = ["please ", "can you ", "i want you to ", "now "]
				for prefix in prefixes_to_remove:
						if cleaned.lower().startswith(prefix):
								cleaned = cleaned[len(prefix):]

				# Filter out markdown explanations and documentation (after stripping code blocks)
				markdown_patterns = [
						r'^\*.*`.*`.*$',	# Lines starting with * and containing backticks
						r'^\*\*.*\*\*.*$',	# Bold text
						r'^```.*$',	# Code blocks (should be stripped above, but check again)
						r'^#.*$',	# Headers
						r'^Note:.*$',	# Notes
						r'^\*\*Note:\*\*.*$',	# Bold notes
				]

				for pattern in markdown_patterns:
						if re.match(pattern, cleaned, re.IGNORECASE):
								raise ValidationError(f"Command appears to be markdown/documentation: {command_text}", "command_text", command_text)

				# Filter out explanations that contain common documentation keywords
				doc_keywords = ["shortcut", "represents", "home directory", "command prompt", "power shell", "windows", "note:"]
				if any(keyword.lower() in cleaned.lower() for keyword in doc_keywords):
						raise ValidationError(f"Command appears to be documentation: {command_text}", "command_text", command_text)

				final_cleaned = cleaned.strip()

				# Final validation
				if not final_cleaned:
						raise ValidationError("Command text cannot be empty after cleaning", "command_text", command_text)

				return final_cleaned

		def _parse_with_patterns(self, command_text: str) -> Optional[ParsedCommand]:
				"""Parse command using regex patterns"""
				command_lower = command_text.lower()

				# Check administrative commands first (more specific)
				for command_type in [CommandType.END, CommandType.REGENERATE_STEP]:
						patterns = self.command_patterns[command_type]
						for pattern in patterns:
								match = pattern.match(command_text)
								if match:
										return self._create_command_from_match(command_type, match, command_text)

				# Check CLI commands (most general)
				patterns = self.command_patterns[CommandType.CLI_COMMAND]
				for pattern in patterns:
						match = pattern.match(command_text)
						if match:
								return self._create_command_from_match(CommandType.CLI_COMMAND, match, command_text)

				return None

		def _create_command_from_match(self, command_type: CommandType, match: re.Match, raw_text: str) -> ParsedCommand:
				"""Create parsed command from regex match"""
				try:
						parameters = {}

						if command_type == CommandType.CLI_COMMAND:
								# CLI command: the entire command text
								command_content = raw_text.strip()
								if not command_content:
										raise ValidationError("CLI command content cannot be empty")
								parameters = {"command": command_content}

						elif command_type == CommandType.END:
								# END command: no parameters
								parameters = {}

						elif command_type == CommandType.REGENERATE_STEP:
								# REGENERATE_STEP command: no parameters
								parameters = {}

						return ParsedCommand(
								type=command_type,
								parameters=parameters,
								raw_text=raw_text,
						)

				except Exception as e:
						return None
