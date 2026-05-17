#!/usr/bin/env python3
"""
Test script for improved VEXIS-CLI prompts
Validates that the enhanced prompt engineering improvements work correctly
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from ai_agent.external_integration.model_runner import ModelRunner, TaskType, ModelRequest

def test_task_generation_prompt():
		"""Test the improved TASK_GENERATION prompt"""
		print("🧪 Testing TASK_GENERATION prompt improvements...")

		# Create a model runner instance
		runner = ModelRunner()

		# Create a test request
		request = ModelRequest(
				task_type=TaskType.TASK_GENERATION,
				prompt="create a Python web app with Flask",
				context={
						"os_info": "macOS 14.0",
						"current_directory": "/Users/test/projects"
				}
		)

		# Format the prompt
		formatted_prompt = runner._format_prompt(request)

		# Check for improvements
		improvements_found = []

		# Check for structured sections
		if "## Mission" in formatted_prompt:
				improvements_found.append("✅ Clear mission statement")

		if "## Core Principles" in formatted_prompt:
				improvements_found.append("✅ Core principles defined")

		if "## Command Generation Rules" in formatted_prompt:
				improvements_found.append("✅ Explicit generation rules")

		if "## OS-Specific Guidelines" in formatted_prompt:
				improvements_found.append("✅ OS-specific guidance")

		if "## Quality Examples" in formatted_prompt:
				improvements_found.append("✅ Quality examples included")

		if "## Critical Constraint" in formatted_prompt:
				improvements_found.append("✅ Critical constraints specified")

		# Check for system instructions
		try:
				system_instructions = runner._get_system_instructions(TaskType.TASK_GENERATION)
				if system_instructions and "VEXIS-CLI AI Agent System Instructions" in system_instructions:
						improvements_found.append("✅ System instructions present")

				if "Behavioral Guidelines" in system_instructions:
						improvements_found.append("✅ Behavioral guidelines defined")

				if "Task Generation Specific Guidelines" in system_instructions:
						improvements_found.append("✅ Task-specific guidelines")
		except AttributeError:
				improvements_found.append("❌ System instructions method not found")

		print(f"Found {len(improvements_found)} improvements:")
		for improvement in improvements_found:
				print(f"	{improvement}")

		return len(improvements_found) >= 8

def test_command_parsing_prompt():
		"""Test the improved COMMAND_PARSING prompt"""
		print("\n🧪 Testing COMMAND_PARSING prompt improvements...")

		# Create a model runner instance
		runner = ModelRunner()

		# Create a test request
		request = ModelRequest(
				task_type=TaskType.COMMAND_PARSING,
				prompt="navigate to project directory and create main.py",
				context={
						"os_info": "Linux Ubuntu 22.04",
						"current_directory": "/home/user/projects",
						"previous_terminal_actions": "- EXECUTED: mkdir my_project (SUCCESS)\n- OUTPUT: Directory created",
						"last_command_output": "Directory created successfully"
				}
		)

		# Format the prompt
		formatted_prompt = runner._format_prompt(request)

		# Check for improvements
		improvements_found = []

		# Check for structured sections
		if "## Current Situation" in formatted_prompt:
				improvements_found.append("✅ Clear situation analysis")

		if "## Decision Framework" in formatted_prompt:
				improvements_found.append("✅ Decision framework included")

		if "## Command Selection Guidelines" in formatted_prompt:
				improvements_found.append("✅ Command selection guidelines")

		if "## Contextual Examples" in formatted_prompt:
				improvements_found.append("✅ Contextual examples provided")

		if "## Critical Instructions" in formatted_prompt:
				improvements_found.append("✅ Critical instructions specified")

		# Check for proper context handling
		if "previous_terminal_actions" in formatted_prompt and "No previous actions" not in formatted_prompt:
				improvements_found.append("✅ Context properly integrated")

		if "last_command_output" in formatted_prompt and "No previous output" not in formatted_prompt:
				improvements_found.append("✅ Command output context used")

		# Check for system instructions
		try:
				system_instructions = runner._get_system_instructions(TaskType.COMMAND_PARSING)
				if system_instructions and "Command Parsing Specific Guidelines" in system_instructions:
						improvements_found.append("✅ Command-specific system instructions")

				if "Decision Making" in system_instructions:
						improvements_found.append("✅ Decision making guidelines")
		except AttributeError:
				improvements_found.append("❌ System instructions method not found")

		print(f"Found {len(improvements_found)} improvements:")
		for improvement in improvements_found:
				print(f"	{improvement}")

		return len(improvements_found) >= 9

def test_system_instructions():
		"""Test system instructions for both task types"""
		print("\n🧪 Testing system instructions...")

		runner = ModelRunner()

		improvements_found = []

		# Test task generation system instructions
		try:
				task_sys_instructions = runner._get_system_instructions(TaskType.TASK_GENERATION)
				command_sys_instructions = runner._get_system_instructions(TaskType.COMMAND_PARSING)

				# Check base instructions are present in both
				if "VEXIS-CLI AI Agent System Instructions" in task_sys_instructions:
						improvements_found.append("✅ Task generation has base instructions")

				if "VEXIS-CLI AI Agent System Instructions" in command_sys_instructions:
						improvements_found.append("✅ Command parsing has base instructions")

				# Check for task-specific instructions
				if "Task Generation Specific Guidelines" in task_sys_instructions:
						improvements_found.append("✅ Task generation specific guidelines")

				if "Command Parsing Specific Guidelines" in command_sys_instructions:
						improvements_found.append("✅ Command parsing specific guidelines")

				# Check for behavioral guidelines
				if "Precision Over Verbosity" in task_sys_instructions:
						improvements_found.append("✅ Precision guideline in task generation")

				if "Precision Over Verbosity" in command_sys_instructions:
						improvements_found.append("✅ Precision guideline in command parsing")

				# Check for quality assurance
				if "Quality Assurance" in task_sys_instructions:
						improvements_found.append("✅ Quality assurance in task generation")

				if "Quality Assurance" in command_sys_instructions:
						improvements_found.append("✅ Quality assurance in command parsing")
		except AttributeError:
				improvements_found.append("❌ System instructions method not found")

		print(f"Found {len(improvements_found)} improvements:")
		for improvement in improvements_found:
				print(f"	{improvement}")

		return len(improvements_found) >= 7

def main():
		"""Run all tests"""
		print("🚀 Testing VEXIS-CLI Prompt Engineering Improvements\n")

		all_tests_passed = True

		# Test task generation prompt
		if not test_task_generation_prompt():
				print("❌ TASK_GENERATION prompt test failed")
				all_tests_passed = False
		else:
				print("✅ TASK_GENERATION prompt test passed")

		# Test command parsing prompt
		if not test_command_parsing_prompt():
				print("❌ COMMAND_PARSING prompt test failed")
				all_tests_passed = False
		else:
				print("✅ COMMAND_PARSING prompt test passed")

		# Test system instructions
		if not test_system_instructions():
				print("❌ System instructions test failed")
				all_tests_passed = False
		else:
				print("✅ System instructions test passed")

		# Summary
		print("\n" + "="*60)
		if all_tests_passed:
				print("🎉 All prompt engineering tests PASSED!")
				print("The improved prompts should provide better AI behavior:")
				print("- More structured and clear instructions")
				print("- Better context awareness")
				print("- Enhanced error handling")
				print("- Improved decision making")
				print("- Consistent behavioral guidelines")
		else:
				print("❌ Some tests FAILED. Please review the prompt improvements.")

		print("="*60)

		return 0 if all_tests_passed else 1

if __name__ == "__main__":
		exit(main())
