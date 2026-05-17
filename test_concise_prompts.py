#!/usr/bin/env python3
"""
Simple test for the improved concise prompts
"""

import sys
sys.path.append('src')

from ai_agent.external_integration.model_runner import ModelRunner, TaskType, ModelRequest

def test_concise_prompts():
		"""Test the new concise prompts"""
		print("🧪 Testing concise prompts...")

		runner = ModelRunner()

		# Test task generation
		task_request = ModelRequest(
				task_type=TaskType.TASK_GENERATION,
				prompt="create a Python web app with Flask",
				context={
						"os_info": "macOS 14.0",
						"current_directory": "/Users/test/projects"
				}
		)

		task_prompt = runner._format_prompt(task_request)
		print("✅ TASK_GENERATION prompt formatted successfully")
		print(f"Length: {len(task_prompt)} characters")
		print("Contains key elements:")
		print(f"	- User instruction: {'User:' in task_prompt}")
		print(f"	- Rules: {'RULES:' in task_prompt}")
		print(f"	- Examples: {'EXAMPLES:' in task_prompt}")
		print(f"	- Clear call to action: {'Now generate' in task_prompt}")

		# Test command parsing
		cmd_request = ModelRequest(
				task_type=TaskType.COMMAND_PARSING,
				prompt="set up the Flask app structure",
				context={
						"os_info": "macOS 14.0",
						"current_directory": "/Users/test/projects/webapp",
						"previous_terminal_actions": "- EXECUTED: mkdir webapp (SUCCESS)\n- EXECUTED: cd webapp (SUCCESS)",
						"last_command_output": "Changed to webapp directory"
				}
		)

		cmd_prompt = runner._format_prompt(cmd_request)
		print("\n✅ COMMAND_PARSING prompt formatted successfully")
		print(f"Length: {len(cmd_prompt)} characters")
		print("Contains key elements:")
		print(f"	- Task context: {'Task:' in cmd_prompt}")
		print(f"	- Previous actions: {'Previous Actions:' in cmd_prompt}")
		print(f"	- Options: {'OPTIONS:' in cmd_prompt}")
		print(f"	- Response format: {'RESPONSE FORMAT:' in cmd_prompt}")
		print(f"	- Examples: {'EXAMPLES:' in cmd_prompt}")

		# Test system instructions
		try:
				task_sys = runner._get_system_instructions(TaskType.TASK_GENERATION)
				cmd_sys = runner._get_system_instructions(TaskType.COMMAND_PARSING)

				print("\n✅ System instructions working")
				print(f"Task generation sys instructions: {len(task_sys)} chars")
				print(f"Command parsing sys instructions: {len(cmd_sys)} chars")
				print("Concise and focused:")
				print(f"	- Task: {'expert' in task_sys.lower() and len(task_sys) < 200}")
				print(f"	- Command: {'expert' in cmd_sys.lower() and len(cmd_sys) < 200}")

		except AttributeError as e:
				print(f"\n❌ System instructions not accessible: {e}")

		print("\n🎉 Concise prompt test completed!")

if __name__ == "__main__":
		test_concise_prompts()
