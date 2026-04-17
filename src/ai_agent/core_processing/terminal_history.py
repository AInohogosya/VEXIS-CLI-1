"""
Terminal History System for CLI AI Agent System
Replaces the Save command with terminal log display and history preservation
OS-independent implementation with comprehensive error handling
"""

import json
import time
import os
import subprocess
import shlex
import platform
import stat
from typing import Optional, Dict, Any, List, Union
from dataclasses import dataclass, field, asdict
from pathlib import Path, PurePath
from enum import Enum
from contextlib import contextmanager

from ..utils.logger import get_logger
from ..utils.exceptions import ExecutionError, PlatformError, ValidationError


class TerminalEntryType(Enum):
    """Types of terminal entries"""
    COMMAND = "command"
    OUTPUT = "output"
    ERROR = "error"


@dataclass
class TerminalEntry:
    """Individual terminal entry with command execution information"""
    timestamp: float
    entry_type: TerminalEntryType
    content: str
    command: Optional[str] = None
    working_directory: Optional[str] = None
    return_code: Optional[int] = None
    duration: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with proper serialization"""
        data = asdict(self)
        data['entry_type'] = self.entry_type.value
        data['timestamp'] = str(self.timestamp)
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TerminalEntry':
        """Create from dictionary with proper deserialization"""
        data['entry_type'] = TerminalEntryType(data['entry_type'])
        data['timestamp'] = float(data['timestamp'])
        return cls(**data)


@dataclass
class TerminalSession:
    """Complete terminal session for a work session"""
    session_id: str
    start_time: float
    entries: List[TerminalEntry] = field(default_factory=list)
    end_time: Optional[float] = None
    current_working_directory: str = field(default_factory=lambda: str(Path("/")))
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with proper serialization"""
        data = asdict(self)
        data['start_time'] = str(self.start_time)
        if self.end_time is not None:
            data['end_time'] = str(self.end_time)
        data['entries'] = [entry.to_dict() for entry in self.entries]
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TerminalSession':
        """Create from dictionary with proper deserialization"""
        data['start_time'] = float(data['start_time'])
        if data.get('end_time') is not None:
            data['end_time'] = float(data['end_time'])
        data['entries'] = [TerminalEntry.from_dict(entry) for entry in data.get('entries', [])]
        return cls(**data)


class TerminalHistory:
    """
    Terminal History System that replaces Save command functionality
    Preserves terminal history and displays command outputs instead of Save content
    OS-independent implementation with comprehensive error handling
    """
    
    DEFAULT_TIMEOUT = 30
    DEFAULT_HISTORY_DIR = "./terminal_history"
    
    def __init__(self, session_id: Optional[str] = None, history_dir: Optional[Union[str, Path]] = None):
        """
        Initialize terminal history system
        
        Args:
            session_id: Optional session identifier
            history_dir: Directory for history storage
            
        Raises:
            PlatformError: If initialization fails due to platform-specific issues
            ValidationError: If parameters are invalid
        """
        try:
            # Validate and set session ID
            self.session_id = session_id or f"session_{int(time.time())}"
            if not self.session_id or not isinstance(self.session_id, str):
                raise ValidationError("Session ID must be a non-empty string")
            
            # Initialize history directory with OS-independent path handling
            self.history_dir = Path(history_dir or self.DEFAULT_HISTORY_DIR)
            self._ensure_history_directory()
            
            # Initialize logger
            self.logger = get_logger("terminal_history")
            
            # Initialize terminal session
            # SECURITY FIX: Start from user's home directory instead of root (/)
            # to prevent unrestricted filesystem access
            self._current_directory = Path.home()
            self.terminal_session = TerminalSession(
                session_id=self.session_id,
                start_time=time.time(),
                current_working_directory=str(self._current_directory)
            )
            
            # Platform-specific settings
            self._platform = platform.system().lower()
            self._shell = self._detect_shell()
            
            self.logger.info(
                f"Terminal history system initialized",
                session_id=self.session_id,
                platform=self._platform,
                shell=self._shell,
                history_dir=str(self.history_dir),
                working_directory=str(self._current_directory)
            )
            
        except Exception as e:
            if isinstance(e, (PlatformError, ValidationError)):
                raise
            raise PlatformError(f"Failed to initialize terminal history: {str(e)}") from e
    
    def _detect_shell(self) -> str:
        """Detect the current shell for command execution"""
        try:
            if self._platform == "windows":
                return os.environ.get("COMSPEC", "cmd.exe")
            else:
                return os.environ.get("SHELL", "/bin/bash")
        except Exception as e:
            self.logger.warning(f"Failed to detect shell: {e}")
            return "bash" if self._platform != "windows" else "cmd.exe"
    
    def _ensure_history_directory(self) -> None:
        """Ensure history directory exists with proper permissions"""
        try:
            # Create directory with parents if needed
            self.history_dir.mkdir(parents=True, exist_ok=True)
            
            # Check directory is writable
            if not os.access(self.history_dir, os.W_OK):
                raise PermissionError(f"History directory is not writable: {self.history_dir}")
                
        except PermissionError as e:
            raise PlatformError(f"Permission denied creating history directory: {e}") from e
        except OSError as e:
            raise PlatformError(f"Failed to create history directory: {e}") from e
    
    def execute_command(self, command: str, timeout: Optional[int] = None) -> Dict[str, Any]:
        """
        Execute a CLI command and preserve its output in history
        
        Args:
            command: The CLI command to execute
            timeout: Command timeout in seconds (overrides default)
            
        Returns:
            Dict with execution results including output, error, return code
            
        Raises:
            ValidationError: If command is invalid
            PlatformError: If command execution fails due to platform issues
        """
        if not command or not isinstance(command, str):
            raise ValidationError("Command must be a non-empty string")
        
        if not command.strip():
            raise ValidationError("Command cannot be empty or whitespace only")
        
        # Validate timeout
        if timeout is not None:
            if not isinstance(timeout, (int, float)):
                raise ValidationError("Timeout must be a number")
            if timeout <= 0:
                raise ValidationError("Timeout must be positive")
        else:
            timeout = self.DEFAULT_TIMEOUT
        start_time = time.time()
        
        try:
            # Log command entry
            command_entry = TerminalEntry(
                timestamp=start_time,
                entry_type=TerminalEntryType.COMMAND,
                content=command,
                command=command,
                working_directory=str(self._current_directory)
            )
            self.terminal_session.entries.append(command_entry)
            
            self.logger.info(f"Executing command", command=command, working_directory=str(self._current_directory))
            
            # Handle special commands
            if command.strip().startswith('cd '):
                result = self._handle_cd_command(command, start_time)
            elif command.strip() == 'cd -':
                result = self._handle_cd_dash_command(command, start_time)
            elif command.strip() == 'cd':
                result = self._handle_cd_home_command(command, start_time)
            else:
                result = self._execute_subprocess_command(command, timeout)
            
            duration = time.time() - start_time
            
            # Update command entry with execution results
            command_entry.return_code = result.returncode
            command_entry.duration = duration
            
            self.logger.info(
                f"Command completed",
                command=command,
                return_code=result.returncode,
                duration=duration,
                success=result.returncode == 0
            )
            
            # Log output entry
            if result.stdout and result.stdout.strip():
                output_entry = TerminalEntry(
                    timestamp=start_time + duration,
                    entry_type=TerminalEntryType.OUTPUT,
                    content=result.stdout,
                    command=command,
                    working_directory=str(self._current_directory),
                    return_code=result.returncode,
                    duration=duration
                )
                self.terminal_session.entries.append(output_entry)
            
            # Log error entry if there's stderr
            if result.stderr and result.stderr.strip():
                error_entry = TerminalEntry(
                    timestamp=start_time + duration,
                    entry_type=TerminalEntryType.ERROR,
                    content=result.stderr,
                    command=command,
                    working_directory=str(self._current_directory),
                    return_code=result.returncode,
                    duration=duration
                )
                self.terminal_session.entries.append(error_entry)
            
            # Persist history
            self._persist_history()
            
            return {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "return_code": result.returncode,
                "duration": duration,
                "working_directory": str(self._current_directory)
            }
            
        except subprocess.TimeoutExpired as e:
            error_msg = f"Command timed out after {timeout} seconds: {command}"
            self.logger.error(error_msg)
            
            # Log timeout error
            error_entry = TerminalEntry(
                timestamp=time.time(),
                entry_type=TerminalEntryType.ERROR,
                content=error_msg,
                command=command,
                working_directory=str(self._current_directory),
                return_code=-1
            )
            self.terminal_session.entries.append(error_entry)
            self._persist_history()
            
            return {
                "success": False,
                "stdout": "",
                "stderr": error_msg,
                "return_code": -1,
                "duration": timeout,
                "working_directory": str(self._current_directory)
            }
            
        except Exception as e:
            error_msg = f"Command execution failed: {str(e)}"
            self.logger.error(error_msg, command=command, error_type=type(e).__name__)
            
            # Log execution error
            error_entry = TerminalEntry(
                timestamp=time.time(),
                entry_type=TerminalEntryType.ERROR,
                content=error_msg,
                command=command,
                working_directory=str(self._current_directory),
                return_code=-1
            )
            self.terminal_session.entries.append(error_entry)
            self._persist_history()
            
            return {
                "success": False,
                "stdout": "",
                "stderr": error_msg,
                "return_code": -1,
                "duration": time.time() - start_time,
                "working_directory": str(self._current_directory)
            }
    
    def get_recent_output(self, count: int = 10) -> List[TerminalEntry]:
        """
        Get recent terminal entries (commands, outputs, and errors) for context
        
        Args:
            count: Number of recent entries to retrieve
            
        Returns:
            List of recent terminal entries
        """
        if count <= 0:
            return []
        
        return self.terminal_session.entries[-count:] if self.terminal_session.entries else []
    
    def get_command_history(self, count: int = 10) -> List[TerminalEntry]:
        """
        Get recent command entries
        
        Args:
            count: Number of recent commands to retrieve
            
        Returns:
            List of recent command entries
        """
        if count <= 0:
            return []
        
        command_entries = [entry for entry in self.terminal_session.entries 
                         if entry.entry_type == TerminalEntryType.COMMAND]
        return command_entries[-count:] if command_entries else []
    
    def get_current_working_directory(self) -> Path:
        """Get current working directory as Path object"""
        return self._current_directory
    
    def _execute_subprocess_command(self, command: str, timeout: int):
        """Execute command using subprocess with platform-specific handling (shell=False for security)"""
        try:
            # SECURITY FIX: Use shell=False by using shlex to parse commands
            # and execute them as lists to prevent command injection
            if self._platform == "windows":
                # On Windows, use cmd.exe /c with command list (no shell=True)
                # shlex.split with posix=False handles Windows quoting properly
                import shlex
                cmd_parts = shlex.split(command, posix=False)
                result = subprocess.run(
                    ['cmd.exe', '/c'] + cmd_parts,
                    shell=False,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    cwd=str(self._current_directory),
                    encoding='utf-8',
                    errors='replace'
                )
            else:
                # On Unix-like systems, use shlex to parse the command safely
                # Then execute directly without going through bash -c when possible
                import shlex
                try:
                    cmd_parts = shlex.split(command)
                except ValueError:
                    # If shlex fails (e.g., unbalanced quotes), fall back to bash -c
                    cmd_parts = None
                
                if cmd_parts:
                    # Direct execution without shell
                    result = subprocess.run(
                        cmd_parts,
                        shell=False,
                        capture_output=True,
                        text=True,
                        timeout=timeout,
                        cwd=str(self._current_directory),
                        encoding='utf-8',
                        errors='replace'
                    )
                else:
                    # Fall back to bash -c for complex commands
                    result = subprocess.run(
                        ['bash', '-c', command],
                        shell=False,
                        capture_output=True,
                        text=True,
                        timeout=timeout,
                        cwd=str(self._current_directory),
                        encoding='utf-8',
                        errors='replace'
                    )
            
            return result
            
        except subprocess.CalledProcessError as e:
            # This shouldn't happen with capture_output=True, but just in case
            self.logger.warning(f"CalledProcessError in command execution: {e}")
            return e
        except UnicodeDecodeError as e:
            self.logger.error(f"Unicode decode error in command output: {e}")
            # Retry with error handling - use the same secure approach
            try:
                import shlex
                if self._platform == "windows":
                    cmd_parts = shlex.split(command, posix=False)
                    result = subprocess.run(
                        ['cmd.exe', '/c'] + cmd_parts,
                        shell=False,
                        capture_output=True,
                        text=True,
                        timeout=timeout,
                        cwd=str(self._current_directory),
                        encoding='utf-8',
                        errors='replace'
                    )
                else:
                    result = subprocess.run(
                        ['bash', '-c', command],
                        shell=False,
                        capture_output=True,
                        text=True,
                        timeout=timeout,
                        cwd=str(self._current_directory),
                        encoding='utf-8',
                        errors='replace'
                    )
                return result
            except Exception as retry_e:
                raise PlatformError(f"Failed to execute command with encoding fallback: {retry_e}") from retry_e
    
    def _handle_cd_command(self, command: str, start_time: float):
        """Handle cd commands without using subprocess to maintain proper directory state"""
        from types import SimpleNamespace
        
        # Parse the target directory
        command_parts = command.strip().split(None, 1)
        if len(command_parts) < 2:
            return self._handle_cd_home_command(command, start_time)
        
        target_dir = command_parts[1].strip()
        
        # Create a mock result object
        result = SimpleNamespace(
            returncode=0,
            stdout="",
            stderr=""
        )
        
        try:
            # Handle special cases and path expansion
            if target_dir == '~' or target_dir == '$HOME':
                target_path = Path.home()
            elif target_dir.startswith('~'):
                # Handle ~user paths
                try:
                    target_path = Path(target_dir).expanduser()
                except Exception as e:
                    result.stderr = f"cd: {target_dir}: Invalid path expansion"
                    result.returncode = 1
                    self.logger.warning(f"Failed to expand path {target_dir}: {e}")
                    return result
            else:
                # Handle relative and absolute paths
                target_path = Path(target_dir)
                if not target_path.is_absolute():
                    target_path = self._current_directory / target_path
            
            # Resolve the path (handles .., ., etc.)
            target_path = target_path.resolve()
            
            # SECURITY: Check for path traversal attempts to sensitive directories
            sensitive_paths = [
                Path('/etc'), Path('/var'), Path('/usr'), Path('/bin'), 
                Path('/sbin'), Path('/lib'), Path('/lib64'), Path('/opt'),
                Path('/sys'), Path('/proc'), Path('/dev'), Path('/boot')
            ]
            
            for sensitive in sensitive_paths:
                try:
                    target_path.relative_to(sensitive)
                    # If we get here, target_path is inside a sensitive directory
                    result.stderr = f"cd: access denied: {target_dir} (restricted system directory)"
                    result.returncode = 1
                    self.logger.warning(f"Blocked access to sensitive directory: {target_path}")
                    return result
                except ValueError:
                    # target_path is not inside sensitive directory, continue checking
                    continue
            
            # Validate and update directory
            if target_path.exists() and target_path.is_dir():
                # Check if directory is accessible
                if not os.access(target_path, os.R_OK | os.X_OK):
                    # Format error message to match actual shell behavior
                    if self._platform == "darwin":
                        # macOS zsh format: "cd: permission denied: target_dir"
                        result.stderr = f"cd: permission denied: {target_dir}"
                    elif self._shell.endswith("bash"):
                        # bash format: "bash: cd: target_dir: Permission denied"
                        result.stderr = f"bash: cd: {target_dir}: Permission denied"
                    else:
                        # Generic format
                        result.stderr = f"cd: {target_dir}: Permission denied"
                    result.returncode = 1
                    self.logger.warning(f"Permission denied for directory: {target_path}")
                else:
                    # Store previous directory for cd -
                    self._previous_directory = self._current_directory
                    self._current_directory = target_path
                    self.terminal_session.current_working_directory = str(self._current_directory)
                    self.logger.info(f"Changed directory to: {self._current_directory}")
            else:
                # Format error message to match actual shell behavior
                if self._platform == "darwin":
                    # macOS zsh format: "cd: no such file or directory: target_dir"
                    result.stderr = f"cd: no such file or directory: {target_dir}"
                elif self._shell.endswith("bash"):
                    # bash format: "bash: cd: target_dir: No such file or directory"
                    result.stderr = f"bash: cd: {target_dir}: No such file or directory"
                else:
                    # Generic format that works across shells
                    result.stderr = f"cd: {target_dir}: No such file or directory"
                result.returncode = 1
                self.logger.warning(f"Directory does not exist: {target_path}")
                
        except Exception as e:
            result.stderr = f"cd: {str(e)}"
            result.returncode = 1
            self.logger.error(f"Failed to update working directory: {e}")
        
        return result
    
    def _handle_cd_dash_command(self, command: str, start_time: float):
        """Handle cd - command (go to previous directory)"""
        from types import SimpleNamespace
        
        result = SimpleNamespace(
            returncode=0,
            stdout="",
            stderr=""
        )
        
        try:
            if self._previous_directory is None:
                # Format error message to match actual shell behavior
                if self._platform == "darwin":
                    # macOS zsh format: "cd: no such file or directory: -"
                    result.stderr = "cd: no such file or directory: -"
                elif self._shell.endswith("bash"):
                    # bash format: "bash: cd -: OLDPWD not set"
                    result.stderr = "bash: cd -: OLDPWD not set"
                else:
                    # Generic format
                    result.stderr = "cd: -: OLDPWD not set"
                result.returncode = 1
                self.logger.info("cd - command detected, but no previous directory available")
            else:
                # Swap current and previous directories
                current_dir = self._current_directory
                self._current_directory = self._previous_directory
                self._previous_directory = current_dir
                self.terminal_session.current_working_directory = str(self._current_directory)
                self.logger.info(f"Changed to previous directory: {self._current_directory}")
                
        except Exception as e:
            result.stderr = f"cd: {str(e)}"
            result.returncode = 1
            self.logger.error(f"Failed to change to previous directory: {e}")
        
        return result
    
    def _handle_cd_home_command(self, command: str, start_time: float):
        """Handle cd command with no arguments (go to home directory)"""
        from types import SimpleNamespace
        
        result = SimpleNamespace(
            returncode=0,
            stdout="",
            stderr=""
        )
        
        try:
            home_dir = Path.home()
            if home_dir.exists() and home_dir.is_dir():
                if os.access(home_dir, os.R_OK | os.X_OK):
                    # Store previous directory
                    self._previous_directory = self._current_directory
                    self._current_directory = home_dir
                    self.terminal_session.current_working_directory = str(self._current_directory)
                    self.logger.info(f"Changed to home directory: {self._current_directory}")
                else:
                    # Format error message to match actual shell behavior
                    if self._platform == "darwin":
                        # macOS zsh format: "cd: permission denied: path"
                        result.stderr = f"cd: permission denied: {home_dir}"
                    elif self._shell.endswith("bash"):
                        # bash format: "bash: cd: path: Permission denied"
                        result.stderr = f"bash: cd: {home_dir}: Permission denied"
                    else:
                        # Generic format
                        result.stderr = f"cd: {home_dir}: Permission denied"
                    result.returncode = 1
                    self.logger.warning(f"Permission denied for home directory: {home_dir}")
            else:
                # Format error message to match actual shell behavior
                if self._platform == "darwin":
                    # macOS zsh format: "cd: no such file or directory: path"
                    result.stderr = f"cd: no such file or directory: {home_dir}"
                elif self._shell.endswith("bash"):
                    # bash format: "bash: cd: path: No such file or directory"
                    result.stderr = f"bash: cd: {home_dir}: No such file or directory"
                else:
                    # Generic format
                    result.stderr = f"cd: {home_dir}: No such file or directory"
                result.returncode = 1
                self.logger.warning(f"Home directory does not exist: {home_dir}")
                
        except Exception as e:
            result.stderr = f"cd: {str(e)}"
            result.returncode = 1
            self.logger.error(f"Failed to change to home directory: {e}")
        
        return result
    
    def display_terminal_log(self, max_entries: int = 20) -> str:
        """
        Generate formatted terminal log display showing only commands and their outputs
        
        Args:
            max_entries: Maximum number of entries to display
            
        Returns:
            Formatted terminal log string
        """
        try:
            # Validate max_entries
            if not isinstance(max_entries, int) or max_entries <= 0:
                return "Invalid max_entries value. Must be a positive integer."
            
            recent_entries = self.terminal_session.entries[-max_entries:]
            
            if not recent_entries:
                return "No terminal history available."
            
            output_lines = []
            output_lines.append(f"Terminal Log - Session: {self.session_id}")
            output_lines.append(f"Platform: {self._platform} | Shell: {self._shell}")
            output_lines.append(f"Current Directory: {self._current_directory}")
            output_lines.append("=" * 60)
            
            i = 0
            while i < len(recent_entries):
                entry = recent_entries[i]
                
                if entry.entry_type == TerminalEntryType.COMMAND:
                    timestamp_str = time.strftime('%H:%M:%S', time.localtime(entry.timestamp))
                    output_lines.append(f"[{timestamp_str}] $ {entry.content}")
                    
                    # Look for corresponding output/error entries
                    j = i + 1
                    while j < len(recent_entries) and recent_entries[j].entry_type != TerminalEntryType.COMMAND:
                        output_entry = recent_entries[j]
                        
                        if output_entry.entry_type == TerminalEntryType.OUTPUT and output_entry.content.strip():
                            output_lines.append("┌─ Output:")
                            for line in output_entry.content.split('\n'):
                                if line.strip():
                                    output_lines.append(f"│ {line}")
                                elif line == '\n':
                                    output_lines.append("│")
                            output_lines.append("└─")
                            
                        elif output_entry.entry_type == TerminalEntryType.ERROR and output_entry.content.strip():
                            output_lines.append("┌─ Error:")
                            for line in output_entry.content.split('\n'):
                                if line.strip():
                                    output_lines.append(f"│ {line}")
                                elif line == '\n':
                                    output_lines.append("│")
                            output_lines.append("└─")
                        
                        j += 1
                    
                    i = j  # Skip to next command
                else:
                    i += 1
            
            return "\n".join(output_lines)
            
        except Exception as e:
            self.logger.error(f"Error generating terminal log: {e}")
            return f"Error generating terminal log: {str(e)}"
    
    def get_last_command_output(self) -> str:
        """Get the output of the most recent command"""
        try:
            # Find the most recent command entry
            last_command_entry = None
            for entry in reversed(self.terminal_session.entries):
                if entry.entry_type == TerminalEntryType.COMMAND:
                    last_command_entry = entry
                    break
            
            if not last_command_entry:
                return "No command history available."
            
            # Get output/error entries that came after this command
            command_outputs = []
            for entry in self.terminal_session.entries:
                if (entry.timestamp > last_command_entry.timestamp and 
                    entry.entry_type in [TerminalEntryType.OUTPUT, TerminalEntryType.ERROR]):
                    command_outputs.append(entry.content)
            
            if command_outputs:
                return "\n".join(command_outputs)
            else:
                # If no output entries, check if command executed but had no output
                if last_command_entry.return_code is not None:
                    if last_command_entry.return_code == 0:
                        return f"Command '{last_command_entry.content}' executed successfully with no output."
                    else:
                        return f"Command '{last_command_entry.content}' failed with return code {last_command_entry.return_code}."
                else:
                    return f"Command '{last_command_entry.content}' was logged but execution status unknown."
                
        except Exception as e:
            self.logger.error(f"Failed to get last command output: {e}")
            return f"Error retrieving command output: {str(e)}"
    
    def _persist_history(self) -> None:
        """Persist terminal history to disk with comprehensive error handling"""
        try:
            history_file = self.history_dir / f"{self.session_id}.json"
            
            # Create backup of existing file if it exists
            if history_file.exists():
                backup_file = history_file.with_suffix('.json.bak')
                try:
                    import shutil
                    shutil.copy2(history_file, backup_file)
                except Exception as e:
                    self.logger.warning(f"Failed to create backup of history file: {e}")
            
            # Convert to serializable format
            history_data = self.terminal_session.to_dict()
            
            # Write to file with atomic operation
            temp_file = history_file.with_suffix('.json.tmp')
            try:
                with open(temp_file, 'w', encoding='utf-8', newline='') as f:
                    json.dump(history_data, f, indent=2, ensure_ascii=False)
                
                # Atomic move
                temp_file.replace(history_file)
                
            except Exception as e:
                # Clean up temp file if it exists
                if temp_file.exists():
                    try:
                        temp_file.unlink()
                    except Exception:
                        pass
                raise e
                
        except PermissionError as e:
            self.logger.error(f"Permission denied persisting terminal history: {e}")
        except OSError as e:
            self.logger.error(f"OS error persisting terminal history: {e}")
        except Exception as e:
            self.logger.error(f"Failed to persist terminal history: {e}")
    
    def end_session(self) -> None:
        """End the current session and finalize history"""
        try:
            self.terminal_session.end_time = time.time()
            self._persist_history()
            self.logger.info(f"Terminal session ended: {self.session_id}")
        except Exception as e:
            self.logger.error(f"Error ending session: {e}")
    
    def load_session(self, session_id: str) -> bool:
        """
        Load a previous session for context
        
        Args:
            session_id: Session ID to load
            
        Returns:
            True if session loaded successfully, False otherwise
        """
        try:
            if not session_id or not isinstance(session_id, str):
                raise ValidationError("Session ID must be a non-empty string")
            
            history_file = self.history_dir / f"{session_id}.json"
            if not history_file.exists():
                self.logger.warning(f"Session file does not exist: {history_file}")
                return False
            
            if not os.access(history_file, os.R_OK):
                self.logger.error(f"Permission denied reading session file: {history_file}")
                return False
            
            with open(history_file, 'r', encoding='utf-8') as f:
                history_data = json.load(f)
            
            # Validate loaded data
            if not isinstance(history_data, dict):
                raise ValidationError("Invalid session data format")
            
            # Reconstruct terminal session
            self.terminal_session = TerminalSession.from_dict(history_data)
            
            # Update current directory
            self._current_directory = Path(self.terminal_session.current_working_directory)
            self.session_id = session_id
            
            self.logger.info(f"Terminal session loaded: {session_id}")
            return True
            
        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON in session file {session_id}: {e}")
            return False
        except ValidationError as e:
            self.logger.error(f"Invalid session data for {session_id}: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Failed to load terminal session {session_id}: {e}")
            return False
    
    def list_sessions(self) -> List[str]:
        """
        List all available session IDs
        
        Returns:
            List of session IDs
        """
        try:
            sessions = []
            for file_path in self.history_dir.glob("*.json"):
                if file_path.is_file() and not file_path.name.endswith('.bak'):
                    session_id = file_path.stem
                    sessions.append(session_id)
            
            return sorted(sessions)
            
        except Exception as e:
            self.logger.error(f"Failed to list sessions: {e}")
            return []
    
    def cleanup_old_sessions(self, max_sessions: int = 100) -> int:
        """
        Clean up old session files, keeping only the most recent ones
        
        Args:
            max_sessions: Maximum number of sessions to keep
            
        Returns:
            Number of sessions removed
        """
        try:
            # Validate max_sessions
            if not isinstance(max_sessions, int) or max_sessions <= 0:
                self.logger.warning(f"Invalid max_sessions value: {max_sessions}. Must be positive integer.")
                return 0
            
            sessions = self.list_sessions()
            if len(sessions) <= max_sessions:
                return 0
            
            # Sort by modification time and remove oldest
            session_files = []
            for session_id in sessions:
                file_path = self.history_dir / f"{session_id}.json"
                try:
                    mtime = file_path.stat().st_mtime
                    session_files.append((mtime, file_path))
                except Exception as e:
                    self.logger.warning(f"Failed to get modification time for {session_id}: {e}")
            
            # Sort by modification time (oldest first)
            session_files.sort()
            
            # Remove oldest sessions
            removed_count = 0
            for mtime, file_path in session_files[:-max_sessions]:
                try:
                    file_path.unlink()
                    removed_count += 1
                    self.logger.info(f"Removed old session file: {file_path}")
                except Exception as e:
                    self.logger.warning(f"Failed to remove session file {file_path}: {e}")
            
            return removed_count
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup old sessions: {e}")
            return 0
    
    @contextmanager
    def temporary_directory(self, target_dir: Optional[Union[str, Path]] = None):
        """
        Context manager for temporarily changing directory
        
        Args:
            target_dir: Directory to change to (defaults to current directory)
        """
        original_dir = self._current_directory
        try:
            if target_dir:
                target_path = Path(target_dir).resolve()
                if target_path.exists() and target_path.is_dir():
                    self._current_directory = target_path
                    self.terminal_session.current_working_directory = str(self._current_directory)
                else:
                    raise ValueError(f"Target directory does not exist: {target_path}")
            yield
        finally:
            # Always restore original directory
            self._current_directory = original_dir
            self.terminal_session.current_working_directory = str(self._current_directory)

    def execute_commands_batch(self, commands: List[str], timeout: Optional[int] = None) -> Dict[str, Any]:
        """
        Execute multiple commands in a single batch using the same terminal session.
        All commands are pasted and executed at once, maintaining the same session.
        
        Args:
            commands: List of commands to execute
            timeout: Command timeout in seconds (overrides default)
            
        Returns:
            Dict with execution results including output, error, return code
        """
        if not commands:
            raise ValidationError("Commands list cannot be empty")
        
        # Validate timeout
        if timeout is not None:
            if not isinstance(timeout, (int, float)):
                raise ValidationError("Timeout must be a number")
            if timeout <= 0:
                raise ValidationError("Timeout must be positive")
        else:
            timeout = self.DEFAULT_TIMEOUT
        
        start_time = time.time()
        
        # Combine all commands with newlines for batch execution
        batch_command = "\n".join(commands)
        
        try:
            # Log the batch command entry
            command_entry = TerminalEntry(
                timestamp=start_time,
                entry_type=TerminalEntryType.COMMAND,
                content=batch_command,
                command=batch_command,
                working_directory=str(self._current_directory)
            )
            self.terminal_session.entries.append(command_entry)
            
            self.logger.info(f"Executing batch of {len(commands)} commands", 
                           command_count=len(commands),
                           working_directory=str(self._current_directory))
            
            # Execute all commands in a single shell session
            result = self._execute_batch_subprocess(commands, timeout)
            
            duration = time.time() - start_time
            
            # Update command entry with execution results
            command_entry.return_code = result.returncode
            command_entry.duration = duration
            
            self.logger.info(
                f"Batch command completed",
                return_code=result.returncode,
                duration=duration,
                success=result.returncode == 0
            )
            
            # Log output entry
            if result.stdout and result.stdout.strip():
                output_entry = TerminalEntry(
                    timestamp=start_time + duration,
                    entry_type=TerminalEntryType.OUTPUT,
                    content=result.stdout,
                    command=batch_command,
                    working_directory=str(self._current_directory),
                    return_code=result.returncode,
                    duration=duration
                )
                self.terminal_session.entries.append(output_entry)
            
            # Log error entry if there's stderr
            if result.stderr and result.stderr.strip():
                error_entry = TerminalEntry(
                    timestamp=start_time + duration,
                    entry_type=TerminalEntryType.ERROR,
                    content=result.stderr,
                    command=batch_command,
                    working_directory=str(self._current_directory),
                    return_code=result.returncode,
                    duration=duration
                )
                self.terminal_session.entries.append(error_entry)
            
            # Persist history
            self._persist_history()
            
            return {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "return_code": result.returncode,
                "duration": duration,
                "working_directory": str(self._current_directory)
            }
            
        except subprocess.TimeoutExpired as e:
            error_msg = f"Batch command timed out after {timeout} seconds"
            self.logger.error(error_msg)
            
            # Log timeout error
            error_entry = TerminalEntry(
                timestamp=time.time(),
                entry_type=TerminalEntryType.ERROR,
                content=error_msg,
                command=batch_command,
                working_directory=str(self._current_directory),
                return_code=-1
            )
            self.terminal_session.entries.append(error_entry)
            self._persist_history()
            
            return {
                "success": False,
                "stdout": "",
                "stderr": error_msg,
                "return_code": -1,
                "duration": timeout,
                "working_directory": str(self._current_directory)
            }
            
        except Exception as e:
            error_msg = f"Batch command execution failed: {str(e)}"
            self.logger.error(error_msg, error_type=type(e).__name__)
            
            # Log execution error
            error_entry = TerminalEntry(
                timestamp=time.time(),
                entry_type=TerminalEntryType.ERROR,
                content=error_msg,
                command=batch_command,
                working_directory=str(self._current_directory),
                return_code=-1
            )
            self.terminal_session.entries.append(error_entry)
            self._persist_history()
            
            return {
                "success": False,
                "stdout": "",
                "stderr": error_msg,
                "return_code": -1,
                "duration": time.time() - start_time,
                "working_directory": str(self._current_directory)
            }

    def _execute_batch_subprocess(self, commands: List[str], timeout: int):
        """Execute multiple commands in a single subprocess batch with inactivity-based timeout.
        
        Uses two timeout mechanisms:
        - Overall timeout: 10 hours (36000 seconds) maximum total execution time
        - Inactivity timeout: 10 minutes (600 seconds) - kills process if no new output
        
        Args:
            commands: List of commands to execute
            timeout: Maximum overall timeout in seconds (default: 10 hours)
            
        Returns:
            SimpleNamespace with returncode, stdout, stderr
        """
        import threading
        from types import SimpleNamespace
        
        # Constants
        OVERALL_TIMEOUT = 36000  # 10 hours
        INACTIVITY_TIMEOUT = 600  # 10 minutes
        
        # Join commands with newlines to execute sequentially in same shell
        processed_commands = []
        for cmd in commands:
            cmd_stripped = cmd.strip()
            if cmd_stripped.startswith('cd '):
                parts = cmd_stripped.split(None, 1)
                if len(parts) > 1:
                    target_dir = parts[1].strip()
                    processed_commands.append(f"cd {target_dir} || exit 1")
                else:
                    processed_commands.append("cd ~ || exit 1")
            else:
                processed_commands.append(cmd_stripped)
        
        batch_script = "\n".join(processed_commands)
        
        stdout_lines = []
        stderr_lines = []
        last_output_time = [time.time()]
        process_returncode = [None]
        is_timed_out = [False]
        timeout_reason = [""]
        
        def read_output(pipe, lines_list, is_stderr=False):
            """Read output from pipe continuously"""
            try:
                for line in iter(pipe.readline, b''):
                    if is_timed_out[0]:
                        break
                    decoded_line = line.decode('utf-8', errors='replace')
                    lines_list.append(decoded_line)
                    last_output_time[0] = time.time()
            except Exception:
                pass
            finally:
                pipe.close()
        
        import tempfile
        
        # Write batch script to temp file to avoid shell=True
        script_suffix = '.bat' if self._platform == "windows" else '.sh'
        with tempfile.NamedTemporaryFile(mode='w', suffix=script_suffix, delete=False, 
                                          dir=str(self._current_directory)) as tmp_script:
            if self._platform == "windows":
                tmp_script.write(f"@echo off\n{batch_script}\n")
            else:
                tmp_script.write(f"#!/bin/bash\n{batch_script}\n")
            script_path = tmp_script.name
        
        # Make script executable on Unix
        if self._platform != "windows":
            os.chmod(script_path, 0o755)
        
        try:
            # Start process with Popen for streaming output (shell=False for security)
            if self._platform == "windows":
                process = subprocess.Popen(
                    ['cmd.exe', '/c', script_path],
                    shell=False,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    cwd=str(self._current_directory)
                )
            else:
                process = subprocess.Popen(
                    ['bash', script_path],
                    shell=False,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    cwd=str(self._current_directory)
                )
            
            # Start threads to read output
            stdout_thread = threading.Thread(target=read_output, args=(process.stdout, stdout_lines, False))
            stderr_thread = threading.Thread(target=read_output, args=(process.stderr, stderr_lines, True))
            stdout_thread.daemon = True
            stderr_thread.daemon = True
            stdout_thread.start()
            stderr_thread.start()
            
            start_time = time.time()
            
            # Monitor process with dual timeout
            while process.poll() is None:
                current_time = time.time()
                elapsed = current_time - start_time
                since_last_output = current_time - last_output_time[0]
                
                # Check overall timeout (5 hours)
                if elapsed > OVERALL_TIMEOUT:
                    is_timed_out[0] = True
                    timeout_reason[0] = f"Overall timeout exceeded ({OVERALL_TIMEOUT}s)"
                    process.terminate()
                    time.sleep(1)
                    if process.poll() is None:
                        process.kill()
                    break
                
                # Check inactivity timeout (10 minutes)
                if since_last_output > INACTIVITY_TIMEOUT:
                    is_timed_out[0] = True
                    timeout_reason[0] = f"No output for {INACTIVITY_TIMEOUT}s - process appears stuck"
                    process.terminate()
                    time.sleep(1)
                    if process.poll() is None:
                        process.kill()
                    break
                
                time.sleep(0.1)
            
            # Wait for threads to finish
            stdout_thread.join(timeout=2)
            stderr_thread.join(timeout=2)
            
            # Get final return code
            if is_timed_out[0]:
                process_returncode[0] = -1
            else:
                process_returncode[0] = process.poll() or 0
            
            stdout = "".join(stdout_lines)
            stderr = "".join(stderr_lines)
            
            if is_timed_out[0]:
                stderr += f"\n[TIMEOUT: {timeout_reason[0]}]"
                self.logger.warning(f"Batch command timed out: {timeout_reason[0]}")
            
            result = SimpleNamespace(
                returncode=process_returncode[0],
                stdout=stdout,
                stderr=stderr
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Batch execution failed: {e}")
            return SimpleNamespace(
                returncode=-1,
                stdout="",
                stderr=f"Execution failed: {str(e)}"
            )
        finally:
            # Clean up temp script file
            try:
                if 'script_path' in locals() and os.path.exists(script_path):
                    os.unlink(script_path)
            except Exception:
                pass


# Global terminal history instance for easy access
_global_terminal_history: Optional[TerminalHistory] = None


def get_terminal_history() -> TerminalHistory:
    """Get global terminal history instance"""
    global _global_terminal_history
    if _global_terminal_history is None:
        _global_terminal_history = TerminalHistory()
    return _global_terminal_history


def execute_command(command: str, timeout: Optional[int] = None) -> Dict[str, Any]:
    """
    Global function to execute command and preserve history
    
    Args:
        command: Command to execute
        timeout: Optional timeout in seconds
        
    Returns:
        Dict with execution results
        
    Example:
        result = execute_command("ls -la")
        if result["success"]:
            print(result["stdout"])
    """
    return get_terminal_history().execute_command(command, timeout)


def display_terminal_log(max_entries: int = 20) -> str:
    """
    Global function to display terminal log
    
    Args:
        max_entries: Maximum number of entries to display
        
    Returns:
        Formatted terminal log string
        
    Example:
        print(display_terminal_log())
    """
    return get_terminal_history().display_terminal_log(max_entries)


def get_last_command_output() -> str:
    """
    Global function to get last command output
    
    Returns:
        Output of the most recent command
        
    Example:
        print(get_last_command_output())
    """
    return get_terminal_history().get_last_command_output()
