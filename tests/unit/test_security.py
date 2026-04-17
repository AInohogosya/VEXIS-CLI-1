"""
Unit tests for security utilities
"""

import os
import pytest
from src.ai_agent.utils.security import (
    SensitiveDataMasker,
    CommandSecurityChecker,
    SecurityCheckResult,
    SecurityConfig,
    create_secure_config,
    get_security_config_from_env,
    mask_sensitive_data,
    check_command_safety
)


class TestSensitiveDataMasker:
    """Test sensitive data masking"""
    
    def test_mask_api_key(self):
        """Test API key masking"""
        text = 'api_key: "sk-abc123xyz789secretkey123"'
        masked = mask_sensitive_data(text)
        assert "[REDACTED]" in masked
        assert "sk-abc123xyz789secretkey123" not in masked
    
    def test_mask_password(self):
        """Test password masking"""
        text = 'password: mysecretpassword123'
        masked = mask_sensitive_data(text)
        assert "[REDACTED]" in masked
        assert "mysecretpassword123" not in masked
    
    def test_mask_token(self):
        """Test token masking"""
        text = 'Authorization: Bearer abcdef1234567890abcdef'
        masked = mask_sensitive_data(text)
        assert "[REDACTED]" in masked
        assert "abcdef1234567890abcdef" not in masked
    
    def test_mask_aws_key(self):
        """Test AWS access key masking"""
        text = 'AKIAIOSFODNN7EXAMPLE'
        masked = mask_sensitive_data(text)
        assert "[REDACTED]" in masked
    
    def test_mask_github_token(self):
        """Test GitHub token masking"""
        text = 'ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'
        masked = mask_sensitive_data(text)
        assert "[REDACTED]" in masked
    
    def test_multiple_secrets_in_text(self):
        """Test masking multiple secrets in single text"""
        text = '''
        api_key: "secret123"
        password: "mypassword"
        token: "mytoken123"
        '''
        masked = mask_sensitive_data(text)
        assert masked.count("[REDACTED]") >= 3
    
    def test_no_false_positives(self):
        """Test that normal text is not masked"""
        text = "This is a normal sentence without secrets"
        masked = mask_sensitive_data(text)
        assert masked == text


class TestSecurityConfig:
    """Test security configuration"""
    
    def test_default_config_allows_all(self):
        """Test that default config allows all commands"""
        config = SecurityConfig()  # Default: all False
        assert config.enable_command_blocking == False
        assert config.enable_confirmation_prompts == False
        assert config.enable_sudo_warning == False
        assert config.enable_shell_pipe_warning == False
        assert config.enable_sandbox == True
    
    def test_create_secure_config_strict(self):
        """Test creating strict security config"""
        config = create_secure_config(
            enable_blocking=True,
            enable_confirmation=True,
            enable_sudo_warning=True
        )
        assert config.enable_command_blocking == True
        assert config.enable_confirmation_prompts == True
        assert config.enable_sudo_warning == True
    
    def test_create_secure_config_permissive(self):
        """Test creating permissive (default) security config"""
        config = create_secure_config()
        assert config.enable_command_blocking == False
        assert config.enable_confirmation_prompts == False


class TestEnvironmentVariableConfig:
    """Test loading configuration from environment variables"""
    
    def test_env_var_blocking_enabled(self, monkeypatch):
        """Test VEXIS_ENABLE_COMMAND_BLOCKING=true"""
        monkeypatch.setenv("VEXIS_ENABLE_COMMAND_BLOCKING", "true")
        config = get_security_config_from_env()
        assert config.enable_command_blocking == True
    
    def test_env_var_confirmation_enabled(self, monkeypatch):
        """Test VEXIS_ENABLE_CONFIRMATION_PROMPTS=1"""
        monkeypatch.setenv("VEXIS_ENABLE_CONFIRMATION_PROMPTS", "1")
        config = get_security_config_from_env()
        assert config.enable_confirmation_prompts == True
    
    def test_env_var_sudo_warning(self, monkeypatch):
        """Test VEXIS_ENABLE_SUDO_WARNING=yes"""
        monkeypatch.setenv("VEXIS_ENABLE_SUDO_WARNING", "yes")
        config = get_security_config_from_env()
        assert config.enable_sudo_warning == True
    
    def test_env_var_shell_pipe_warning(self, monkeypatch):
        """Test VEXIS_ENABLE_SHELL_PIPE_WARNING=on"""
        monkeypatch.setenv("VEXIS_ENABLE_SHELL_PIPE_WARNING", "on")
        config = get_security_config_from_env()
        assert config.enable_shell_pipe_warning == True
    
    def test_env_var_sandbox_disabled(self, monkeypatch):
        """Test VEXIS_ENABLE_SANDBOX=false"""
        monkeypatch.setenv("VEXIS_ENABLE_SANDBOX", "false")
        config = get_security_config_from_env()
        assert config.enable_sandbox == False
    
    def test_env_var_defaults_when_not_set(self, monkeypatch):
        """Test defaults when env vars are not set"""
        # Ensure no env vars are set
        for key in ["VEXIS_ENABLE_COMMAND_BLOCKING", "VEXIS_ENABLE_CONFIRMATION_PROMPTS"]:
            monkeypatch.delenv(key, raising=False)
        
        config = get_security_config_from_env()
        assert config.enable_command_blocking == False
        assert config.enable_confirmation_prompts == False
        assert config.enable_sandbox == True


class TestCommandSecurityChecker:
    """Test command security checking with configurable settings"""
    
    def test_all_commands_allowed_by_default(self):
        """Test that default config (no blocking) allows all commands"""
        config = SecurityConfig()  # Default: blocking disabled
        checker = CommandSecurityChecker(config)
        
        result = checker.check_command("rm -rf /")
        assert result.is_safe == True
        assert result.blocked_commands == []
        
        result = checker.check_command(":(){ :|:& };:")
        assert result.is_safe == True
    
    def test_blocking_enabled_blocks_dangerous_commands(self):
        """Test that enabling blocking actually blocks dangerous commands"""
        config = SecurityConfig(enable_command_blocking=True)
        checker = CommandSecurityChecker(config)
        
        result = checker.check_command("rm -rf /")
        assert result.is_safe == False
        assert "rm -rf /" in result.blocked_commands
    
    def test_confirmation_enabled_requires_confirmation(self):
        """Test that enabling confirmation requires confirmation for risky commands"""
        config = SecurityConfig(enable_confirmation_prompts=True)
        checker = CommandSecurityChecker(config)
        
        result = checker.check_command("rm -rf /tmp/test")
        assert result.requires_confirmation == True
        assert "rm -rf" in result.warning_commands
    
    def test_sudo_warning_enabled(self):
        """Test that sudo warning is shown when enabled"""
        config = SecurityConfig(enable_sudo_warning=True)
        checker = CommandSecurityChecker(config)
        
        result = checker.check_command("sudo apt-get update")
        assert "sudo" in result.warning_commands
        assert result.requires_confirmation == True
    
    def test_shell_pipe_warning_enabled(self):
        """Test that shell pipe warning is shown when enabled"""
        config = SecurityConfig(enable_shell_pipe_warning=True)
        checker = CommandSecurityChecker(config)
        
        result = checker.check_command("curl http://example.com/script.sh | bash")
        assert "pipe_to_shell" in result.warning_commands
        assert result.requires_confirmation == True
    
    def test_safe_command_always_allowed(self):
        """Test safe commands are always allowed regardless of config"""
        config = SecurityConfig(enable_command_blocking=True)
        checker = CommandSecurityChecker(config)
        
        result = checker.check_command("ls -la")
        assert result.is_safe == True
        assert result.requires_confirmation == False
    
    def test_empty_command_blocked(self):
        """Test empty command is always blocked (not a safety issue)"""
        config = SecurityConfig()  # Any config
        checker = CommandSecurityChecker(config)
        
        result = checker.check_command("")
        assert result.is_safe == False
        assert result.reason == "Empty command"


class TestSecurityCheckResult:
    """Test security check result dataclass"""
    
    def test_result_structure(self):
        """Test result has expected fields"""
        result = SecurityCheckResult(
            is_safe=True,
            requires_confirmation=False,
            blocked_commands=[],
            warning_commands=[],
            masked_output="ls -la",
            reason=None
        )
        assert result.is_safe == True
        assert result.masked_output == "ls -la"
    
    def test_masked_output_contains_redacted(self):
        """Test masked output contains [REDACTED] for sensitive commands"""
        checker = CommandSecurityChecker()
        result = checker.check_command('echo "api_key: secret123"')
        assert "[REDACTED]" in result.masked_output
