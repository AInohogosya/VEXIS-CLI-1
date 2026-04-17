"""
Unit tests for enhanced exception handling
"""

import pytest
from src.ai_agent.utils.exceptions import (
    ErrorCategory,
    ErrorContext,
    ErrorHandler,
    APIError,
    ValidationError,
    ExecutionError,
    TimeoutError,
    ResourceExhaustedError
)


class TestErrorCategory:
    """Test error category definitions"""
    
    def test_category_values(self):
        """Test that all categories have correct string values"""
        assert ErrorCategory.TRANSIENT.value == "transient"
        assert ErrorCategory.PERMANENT.value == "permanent"
        assert ErrorCategory.AUTHENTICATION.value == "authentication"
        assert ErrorCategory.RATE_LIMIT.value == "rate_limit"


class TestAPIError:
    """Test APIError with automatic categorization"""
    
    def test_auth_error_categorization(self):
        """Test 401/403 errors are categorized as authentication"""
        error = APIError("Unauthorized", status_code=401)
        assert error.context.category == ErrorCategory.AUTHENTICATION
        assert error.context.retryable == False
    
    def test_rate_limit_categorization(self):
        """Test 429 errors are categorized as rate limit"""
        error = APIError("Too Many Requests", status_code=429)
        assert error.context.category == ErrorCategory.RATE_LIMIT
        assert error.context.retryable == True
        assert error.context.backoff_seconds == 60.0
    
    def test_server_error_categorization(self):
        """Test 5xx errors are categorized as external and retryable"""
        error = APIError("Server Error", status_code=500)
        assert error.context.category == ErrorCategory.EXTERNAL
        assert error.context.retryable == True
    
    def test_validation_error_categorization(self):
        """Test 4xx errors (except 429) are validation errors"""
        error = APIError("Bad Request", status_code=400)
        assert error.context.category == ErrorCategory.VALIDATION
        assert error.context.retryable == False


class TestErrorHandler:
    """Test centralized error handling"""
    
    def test_should_retry_for_retryable_error(self):
        """Test that retryable errors are retried within limit"""
        error = APIError("Server Error", status_code=500)
        assert ErrorHandler.should_retry(error, attempt=0) == True
        assert ErrorHandler.should_retry(error, attempt=1) == True
        assert ErrorHandler.should_retry(error, attempt=2) == True
        assert ErrorHandler.should_retry(error, attempt=3) == False  # max_retries=3
    
    def test_should_not_retry_for_non_retryable_error(self):
        """Test that non-retryable errors are never retried"""
        error = ValidationError("Invalid input")
        assert ErrorHandler.should_retry(error, attempt=0) == False
    
    def test_retry_delay_with_exponential_backoff(self):
        """Test exponential backoff calculation"""
        error = APIError("Server Error", status_code=500)
        
        # Base backoff is 2.0 seconds for 5xx errors
        assert ErrorHandler.get_retry_delay(error, attempt=0) == 2.0
        assert ErrorHandler.get_retry_delay(error, attempt=1) == 4.0
        assert ErrorHandler.get_retry_delay(error, attempt=2) == 8.0
        assert ErrorHandler.get_retry_delay(error, attempt=5) == 30.0  # capped at 30s
    
    def test_rate_limit_delay_is_longer(self):
        """Test rate limit errors have longer delays"""
        error = APIError("Too Many Requests", status_code=429)
        
        # Base backoff is 60.0 seconds for rate limits
        assert ErrorHandler.get_retry_delay(error, attempt=0) == 60.0
        assert ErrorHandler.get_retry_delay(error, attempt=1) == 120.0
        # Capped at 300s (5 minutes) for rate limits
        assert ErrorHandler.get_retry_delay(error, attempt=5) == 300.0


class TestExecutionError:
    """Test ExecutionError categorization based on exit code"""
    
    def test_sigint_not_retryable(self):
        """Test SIGINT (130) is not retryable"""
        error = ExecutionError("Interrupted", exit_code=130)
        assert error.context.category == ErrorCategory.PERMANENT
        assert error.context.retryable == False
    
    def test_other_signals_are_retryable(self):
        """Test other signal terminations are retryable"""
        error = ExecutionError("Terminated", exit_code=137)  # SIGKILL
        assert error.context.category == ErrorCategory.TRANSIENT
        assert error.context.retryable == True
    
    def test_regular_exit_code_retryable(self):
        """Test regular exit codes are retryable"""
        error = ExecutionError("Command failed", exit_code=1)
        assert error.context.category == ErrorCategory.EXTERNAL
        assert error.context.retryable == True


class TestSpecializedErrors:
    """Test other specialized error types"""
    
    def test_timeout_error_is_retryable(self):
        """Test timeout errors are retryable"""
        error = TimeoutError("Connection timeout", timeout_seconds=30.0)
        assert error.context.category == ErrorCategory.TIMEOUT
        assert error.context.retryable == True
        assert error.context.max_retries == 3
    
    def test_resource_exhausted_is_retryable(self):
        """Test resource exhausted errors are retryable with longer backoff"""
        error = ResourceExhaustedError("Out of memory", resource_type="memory")
        assert error.context.category == ErrorCategory.RESOURCE
        assert error.context.retryable == True
        assert error.context.backoff_seconds == 30.0
    
    def test_validation_error_is_not_retryable(self):
        """Test validation errors are never retryable"""
        error = ValidationError("Invalid field", field="name", value="")
        assert error.context.category == ErrorCategory.VALIDATION
        assert error.context.retryable == False
        assert error.context.max_retries == 0
