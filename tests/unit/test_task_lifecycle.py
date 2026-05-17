"""Tests for task lifecycle and long-running command handling."""

import os
import time

from ai_agent.core_processing.terminal_history import TerminalHistory
from ai_agent.user_interface.five_phase_app import FivePhaseAIAgent
from ai_agent.utils.config import load_config


def test_background_batch_command_is_detached_and_survives_shell_exit(tmp_path):
		marker = tmp_path / "background_marker.txt"
		history = TerminalHistory(session_id="test_background", history_dir=tmp_path / "history")

		with history.temporary_directory(tmp_path):
				result = history.execute_commands_batch(
						[
								(
										"python -c \"import time, pathlib; "
										f"time.sleep(2); pathlib.Path({str(marker)!r}).write_text('done')\" &"
								)
						],
						timeout=1,
				)

		assert result["success"] is True
		assert "Started background process PID" in result["stdout"]

		deadline = time.time() + 5
		while time.time() < deadline and not marker.exists():
				time.sleep(0.1)

		assert marker.read_text() == "done"


def test_foreground_batch_command_respects_timeout(tmp_path):
		history = TerminalHistory(session_id="test_timeout", history_dir=tmp_path / "history")

		with history.temporary_directory(tmp_path):
				result = history.execute_commands_batch(
						["python -c \"import time; time.sleep(5)\""],
						timeout=1,
				)

		assert result["success"] is False
		assert "TIMEOUT" in result["stderr"]


def test_execution_config_timeout_values_are_loaded_from_example_config():
		config = load_config("config.example.yaml", force_reload=True)

		assert config.execution.command_timeout == 30
		assert config.execution.task_timeout == 7200
		assert config.execution.max_iterations == 500


def test_runtime_options_are_applied_to_engine(monkeypatch):
		agent = FivePhaseAIAgent(provider="dummy", model="dummy")

		captured = {}

		def fake_execute_instruction(instruction, conversation_history=None, telegram_mode=False):
				captured["command_timeout"] = agent.engine.command_timeout
				captured["task_timeout"] = agent.engine.task_timeout
				captured["max_iterations"] = agent.engine.max_iterations

				class Context:
						current_phase = None
						iteration_count = 0
						start_time = time.time()
						end_time = time.time()
						error = None
						final_summary = None

				from ai_agent.core_processing.five_phase_engine import PipelinePhase

				Context.current_phase = PipelinePhase.COMPLETED
				return Context()

		monkeypatch.setattr(agent.engine, "execute_instruction", fake_execute_instruction)

		assert agent.run(
				"test instruction",
				{
						"quiet": True,
						"command_timeout": 123,
						"task_timeout": 456,
						"max_iterations": 7,
				},
		) == 0
		assert captured == {
				"command_timeout": 123,
				"task_timeout": 456,
				"max_iterations": 7,
		}
