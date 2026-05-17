"""
5-Phase Pipeline Application Entry Point for CLI AI Agent System
Implements the 5-phase pipeline architecture with CLI interface
"""

import sys
import argparse
import time
import signal
from typing import Optional, Dict, Any
from pathlib import Path

from ..core_processing.five_phase_engine import FivePhaseEngine, PipelinePhase
from ..utils.exceptions import AIAgentException
from ..utils.logger import get_logger, setup_logging
from ..utils.config import load_config


class FivePhaseAIAgent:
		"""5-Phase Pipeline AI Agent implementing the complete architecture"""

		def __init__(self, provider: str = None, model: str = None, config_path: Optional[str] = None,
								 telegram_bot=None):
				self.config = load_config(config_path, force_reload=bool(config_path)) if config_path else load_config()
				self.logger = get_logger("five_phase_app")

				# Initialize 5-phase engine with provider and model
				engine_config = self._build_engine_config()

				self.engine = FivePhaseEngine(provider=provider, model=model, config=engine_config,
																		 telegram_bot=telegram_bot)

				# Setup signal handlers for graceful shutdown
				signal.signal(signal.SIGINT, self._signal_handler)
				signal.signal(signal.SIGTERM, self._signal_handler)

				self.logger.info("5-Phase Pipeline AI Agent initialized")

		def _build_engine_config(self) -> Dict[str, Any]:
				"""Build engine config from both legacy and current config sections."""
				execution = getattr(self.config, "execution", None)
				engine = getattr(self.config, "engine", None)

				engine_command_timeout = getattr(engine, "command_timeout", 600)
				execution_command_timeout = getattr(execution, "command_timeout", 600)
				engine_task_timeout = getattr(engine, "task_timeout", 5400)
				execution_task_timeout = getattr(execution, "task_timeout", 5400)

				return {
						# Prefer explicit engine values, but honor the legacy execution
						# section when the engine section is still at dataclass defaults.
						"command_timeout": (
								engine_command_timeout
								if engine_command_timeout != 600
								else execution_command_timeout
						),
						"task_timeout": (
								engine_task_timeout
								if engine_task_timeout != 5400
								else execution_task_timeout
						),
						"max_iterations": getattr(
								engine,
								"max_iterations",
								getattr(execution, "max_iterations", 500),
						),
						"enable_phase2_summarization": getattr(
								engine,
								"enable_phase2_summarization",
								getattr(execution, "enable_phase2_summarization", True),
						),
				}

		def _apply_runtime_options(self, options: Dict[str, Any]) -> None:
				"""Apply CLI/runtime options to the already-created engine."""
				for option_name in ("command_timeout", "task_timeout", "max_iterations"):
						if option_name in options and options[option_name] is not None:
								setattr(self.engine, option_name, options[option_name])

		def run(self, instruction: str, options: Dict[str, Any], conversation_history=None, cancel_event=None) -> int:
				"""Run AI Agent with instruction using 5-phase pipeline"""
				try:
						self.logger.info(
								"Starting 5-Phase Pipeline AI Agent execution",
								instruction=instruction,
								options=options,
						)

						# Setup logging if requested
						if options.get("verbose"):
								setup_logging(level="DEBUG")
						elif options.get("log_file"):
								setup_logging(file_path=options["log_file"])

						# Validate instruction
						if not instruction or not instruction.strip():
								self.logger.error("Instruction cannot be empty")
								return 1

						self._apply_runtime_options(options)

						# Execute instruction using 5-phase engine with conversation history
						execute_kwargs = {
								"conversation_history": conversation_history,
								"telegram_mode": False,
						}
						if cancel_event is not None:
								execute_kwargs["cancel_event"] = cancel_event

						context = self.engine.execute_instruction(instruction, **execute_kwargs)

						# Determine success based on final phase
						success = context.current_phase == PipelinePhase.COMPLETED

						# Print results if not quiet mode
						if not options.get("quiet") and not getattr(context, "cancelled", False):
								self._print_results(context, instruction, success)

						# Save results if requested
						if options.get("output"):
								self._save_results(context, options["output"])

						# Return exit code based on success
						return 0 if success else 1

				except AIAgentException as e:
						self.logger.error(f"5-Phase AI Agent error: {e}")
						print(f"Error: {e}", file=sys.stderr)
						return 3
				except Exception as e:
						self.logger.error(f"Unexpected error: {e}")
						print(f"Unexpected error: {e}", file=sys.stderr)
						return 4

		def _print_results(self, context, instruction: str, success: bool):
				"""Print execution results to console"""
				print(f"\n{'='*60}")
				print("5-PHASE PIPELINE EXECUTION SUMMARY")
				print(f"{'='*60}")
				print(f"Instruction: {instruction}")
				print(f"Success: {success}")
				print(f"Iterations: {context.iteration_count}")

				if context.end_time and context.start_time:
						duration = context.end_time - context.start_time
						print(f"Duration: {duration:.2f} seconds")

				if context.error:
						print(f"Error: {context.error}")

				# Print final phase
				print(f"Final Phase: {context.current_phase.value}")

				# Print final summary if available
				if context.final_summary:
						print(f"\n{'='*60}")
						print("FINAL SUMMARY")
						print(f"{'='*60}")
						print(context.final_summary)

				print(f"{'='*60}")

		def _save_results(self, context, output_file: str):
				"""Save execution results to file"""
				try:
						import json

						output_path = Path(output_file)
						output_path.parent.mkdir(parents=True, exist_ok=True)

						results = {
								"instruction": context.user_prompt,
								"success": context.current_phase == PipelinePhase.COMPLETED,
								"final_phase": context.current_phase.value,
								"iterations": context.iteration_count,
								"error": context.error,
								"phase1_output": context.phase1_output,
								"phase4_output": context.phase4_output,
								"final_summary": context.final_summary,
								"terminal_log": context.terminal_log,
						}

						# Add timing info if available
						if context.end_time and context.start_time:
								results["duration_seconds"] = context.end_time - context.start_time

						with open(output_path, 'w') as f:
								json.dump(results, f, indent=2, default=str)

						self.logger.info(f"Results saved to: {output_file}")

				except Exception as e:
						self.logger.error(f"Failed to save results: {e}")
						print(f"Warning: Failed to save results to {output_file}: {e}", file=sys.stderr)

		def _signal_handler(self, signum, frame):
				"""Handle shutdown signals"""
				self.logger.info(f"Received signal {signum}, shutting down gracefully...")
				try:
						if hasattr(self, 'engine') and self.engine:
								self.engine.cleanup()
				except Exception as e:
						self.logger.error(f"Error during cleanup: {e}")
				sys.exit(0)	# Standard exit code for SIGINT

		def shutdown(self):
				"""Shutdown 5-Phase AI Agent"""
				self.logger.info("Shutting down 5-Phase AI Agent...")
				self.logger.info("5-Phase AI Agent shutdown complete")


def create_five_phase_argument_parser() -> argparse.ArgumentParser:
		"""Create 5-phase command line argument parser"""
		parser = argparse.ArgumentParser(
				description="AI Agent - 5-Phase Pipeline CLI automation (Command Suggestion → Extraction → Execution → Evaluation → Summary)",
				formatter_class=argparse.RawDescriptionHelpFormatter,
				epilog="""
Examples:
	%(prog)s "Create a new project folder"
	%(prog)s "List all files in the current directory"
	%(prog)s --verbose "Install dependencies using pip"
	%(prog)s --output results.json "Set up a development environment"
	%(prog)s --max-iterations 5 "Run a complex build process"
				"""
		)

		# Positional arguments
		parser.add_argument(
				"instruction",
				type=str,
				help="Natural language instruction for the AI agent"
		)

		# Configuration options
		parser.add_argument(
				"--config",
				type=str,
				help="Path to configuration file"
		)

		# Output options
		parser.add_argument(
				"--output", "-o",
				type=str,
				help="Save execution results to file (JSON format)"
		)

		parser.add_argument(
				"--quiet", "-q",
				action="store_true",
				help="Suppress output except errors"
		)

		# Logging options
		parser.add_argument(
				"--verbose", "-v",
				action="store_true",
				help="Enable verbose logging"
		)

		parser.add_argument(
				"--log-file",
				type=str,
				help="Log to specified file"
		)

		# 5-Phase Pipeline options
		parser.add_argument(
				"--max-iterations",
				type=int,
				default=10,
				help="Maximum number of Phase 2-4 iterations (default: 10)"
		)

		# Execution options
		parser.add_argument(
				"--command-timeout",
				type=int,
				default=600,
				help="Timeout for individual commands in seconds (default: 600)"
		)

		parser.add_argument(
				"--task-timeout",
				type=int,
				default=5400,
				help="Timeout for tasks in seconds (default: 5400)"
		)

		# Testing options
		parser.add_argument(
				"--validate-only",
				action="store_true",
				help="Validate configuration and exit"
		)

		return parser


def validate_arguments(args: argparse.Namespace) -> bool:
		"""Validate command line arguments"""
		# Check instruction
		if not args.instruction or not args.instruction.strip():
				print("Error: Instruction cannot be empty", file=sys.stderr)
				return False

		# Check config file
		if args.config and not Path(args.config).exists():
				print(f"Error: Configuration file not found: {args.config}", file=sys.stderr)
				return False

		# Check log file directory
		if args.log_file:
				log_path = Path(args.log_file)
				if not log_path.parent.exists():
						try:
								log_path.parent.mkdir(parents=True)
						except Exception as e:
								print(f"Error: Cannot create log directory: {e}", file=sys.stderr)
								return False

		# Check output file directory
		if args.output:
				output_path = Path(args.output)
				if not output_path.parent.exists():
						try:
								output_path.parent.mkdir(parents=True)
						except Exception as e:
								print(f"Error: Cannot create output directory: {e}", file=sys.stderr)
								return False

		# Validate timeout values
		if args.command_timeout <= 0:
				print("Error: Command timeout must be positive", file=sys.stderr)
				return False

		if args.task_timeout <= 0:
				print("Error: Task timeout must be positive", file=sys.stderr)
				return False

		if args.max_iterations < 1:
				print("Error: Max iterations must be at least 1", file=sys.stderr)
				return False

		return True


def main():
		"""Main entry point for 5-Phase Pipeline AI Agent"""
		# Parse arguments
		parser = create_five_phase_argument_parser()
		args = parser.parse_args()

		# Validate arguments
		if not validate_arguments(args):
				sys.exit(1)

		# Handle validate-only mode
		if args.validate_only:
				try:
						config = load_config(args.config)
						print("Configuration validation passed")
						return 0
				except Exception as e:
						print(f"Configuration validation failed: {e}", file=sys.stderr)
						return 1

		# Create 5-Phase AI Agent
		try:
				agent = FivePhaseAIAgent(config_path=args.config)
		except Exception as e:
				print(f"Failed to initialize 5-Phase AI Agent: {e}", file=sys.stderr)
				return 1

		# Prepare options
		options = {
				"verbose": args.verbose,
				"quiet": args.quiet,
				"output": args.output,
				"log_file": args.log_file,
				"max_iterations": args.max_iterations,
				"command_timeout": args.command_timeout,
				"task_timeout": args.task_timeout,
		}

		# Run 5-Phase AI Agent
		start_time = time.time()
		exit_code = agent.run(args.instruction, options)
		execution_time = time.time() - start_time

		# Print final summary
		if not args.quiet:
				print(f"\nTotal execution time: {execution_time:.2f} seconds")
				print(f"Exit code: {exit_code}")

		return exit_code


if __name__ == "__main__":
		sys.exit(main())
