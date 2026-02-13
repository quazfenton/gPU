"""
Tests for error handling utilities.

This module tests the error handling utilities including error response formatting,
error message generation, and sanitization.
"""

import pytest
from datetime import datetime

from gui.error_handling import (
    ErrorType,
    ErrorResponse,
    format_validation_error,
    format_backend_error,
    format_auth_error,
    format_system_error,
    format_workflow_error,
    format_generic_error,
    sanitize_error_message,
    create_success_message,
    create_loading_message,
    create_warning_message
)


class TestErrorResponse:
    """Tests for ErrorResponse dataclass."""
    
    def test_error_response_creation(self):
        """Test creating an error response."""
        error = ErrorResponse(
            error_type=ErrorType.VALIDATION,
            message="Test error message",
            details={"field": "test_field"},
            suggestions=["Try again"]
        )
        
        assert error.error_type == ErrorType.VALIDATION
        assert error.message == "Test error message"
        assert error.details == {"field": "test_field"}
        assert error.suggestions == ["Try again"]
        assert error.timestamp is not None
    
    def test_error_response_to_dict(self):
        """Test converting error response to dictionary."""
        error = ErrorResponse(
            error_type=ErrorType.BACKEND,
            message="Backend error",
            details={"backend": "test_backend"}
        )
        
        error_dict = error.to_dict()
        
        assert error_dict['error_type'] == "backend"
        assert error_dict['message'] == "Backend error"
        assert error_dict['details'] == {"backend": "test_backend"}
        assert 'timestamp' in error_dict
    
    def test_error_response_to_markdown(self):
        """Test converting error response to markdown."""
        error = ErrorResponse(
            error_type=ErrorType.VALIDATION,
            message="Validation failed",
            details={"field": "email"},
            suggestions=["Check email format", "Try again"]
        )
        
        markdown = error.to_markdown()
        
        assert "❌ **Error:** Validation failed" in markdown
        assert "**Details:**" in markdown
        assert "field: email" in markdown
        assert "**Suggestions:**" in markdown
        assert "Check email format" in markdown
        assert "Try again" in markdown


class TestErrorFormatters:
    """Tests for error formatting functions."""
    
    def test_format_validation_error(self):
        """Test formatting validation errors."""
        error = format_validation_error(
            "username",
            "Username is required",
            "invalid_value"
        )
        
        assert error.error_type == ErrorType.VALIDATION
        assert "username" in error.message
        assert "Username is required" in error.message
        assert error.details['field'] == "username"
        assert error.details['value'] == "invalid_value"
        assert len(error.suggestions) > 0
    
    def test_format_backend_error(self):
        """Test formatting backend errors."""
        error = format_backend_error(
            "colab",
            "Connection timeout",
            ["kaggle", "modal"]
        )
        
        assert error.error_type == ErrorType.BACKEND
        assert "colab" in error.message
        assert "Connection timeout" in error.message
        assert error.details['backend'] == "colab"
        assert any("kaggle" in s for s in error.suggestions)
    
    def test_format_auth_error_login(self):
        """Test formatting authentication errors for login."""
        error = format_auth_error(
            "Invalid credentials",
            "login"
        )
        
        assert error.error_type == ErrorType.AUTH
        assert "Invalid credentials" in error.message
        assert error.details['auth_type'] == "login"
        assert any("username" in s.lower() or "password" in s.lower() for s in error.suggestions)
    
    def test_format_auth_error_session(self):
        """Test formatting authentication errors for session."""
        error = format_auth_error(
            "Session expired",
            "session"
        )
        
        assert error.error_type == ErrorType.AUTH
        assert "Session expired" in error.message
        assert error.details['auth_type'] == "session"
        assert any("log in" in s.lower() for s in error.suggestions)
    
    def test_format_system_error_recoverable(self):
        """Test formatting recoverable system errors."""
        error = format_system_error(
            "database",
            "Connection lost",
            is_recoverable=True
        )
        
        assert error.error_type == ErrorType.SYSTEM
        assert "database" in error.message
        assert "Connection lost" in error.message
        assert error.details['component'] == "database"
        assert error.details['recoverable'] is True
        assert any("recover" in s.lower() for s in error.suggestions)
    
    def test_format_system_error_non_recoverable(self):
        """Test formatting non-recoverable system errors."""
        error = format_system_error(
            "core",
            "Critical failure",
            is_recoverable=False
        )
        
        assert error.error_type == ErrorType.SYSTEM
        assert error.details['recoverable'] is False
        assert any("administrator" in s.lower() for s in error.suggestions)
    
    def test_format_workflow_error(self):
        """Test formatting workflow errors."""
        error = format_workflow_error(
            "My Workflow",
            "Circular dependency detected",
            "step2",
            ["Step1 -> Step2 -> Step1"]
        )
        
        assert error.error_type == ErrorType.WORKFLOW
        assert "My Workflow" in error.message
        assert "Circular dependency detected" in error.message
        assert error.details['workflow'] == "My Workflow"
        assert error.details['step'] == "step2"
        assert error.details['validation_errors'] == ["Step1 -> Step2 -> Step1"]
    
    def test_format_generic_error_validation(self):
        """Test formatting generic ValueError as validation error."""
        exception = ValueError("Invalid input")
        error = format_generic_error(exception, "Processing failed")
        
        assert error.error_type == ErrorType.VALIDATION
        assert "Processing failed" in error.message
        assert "Invalid input" in error.message
        assert error.details['exception_type'] == "ValueError"
    
    def test_format_generic_error_backend(self):
        """Test formatting generic backend-related error."""
        exception = ConnectionError("Backend unreachable")
        error = format_generic_error(exception)
        
        assert error.error_type == ErrorType.BACKEND
        assert "Backend unreachable" in error.message
        assert error.details['exception_type'] == "ConnectionError"


class TestSanitization:
    """Tests for error message sanitization."""
    
    def test_sanitize_file_paths_windows(self):
        """Test sanitizing Windows file paths."""
        message = "Error in C:\\Users\\test\\file.py at line 10"
        sanitized = sanitize_error_message(message)
        
        assert "C:\\Users\\test\\file.py" not in sanitized
        assert "[path]" in sanitized
    
    def test_sanitize_file_paths_unix(self):
        """Test sanitizing Unix file paths."""
        message = "Error in /home/user/project/file.py at line 10"
        sanitized = sanitize_error_message(message)
        
        assert "/home/user/project/file.py" not in sanitized
        assert "[path]" in sanitized
    
    def test_sanitize_stack_trace(self):
        """Test sanitizing stack traces."""
        message = """Traceback (most recent call last):
  File "test.py", line 10, in <module>
    raise ValueError("Test error")
ValueError: Test error"""
        
        sanitized = sanitize_error_message(message)
        
        assert "Traceback" not in sanitized
        assert "File \"test.py\"" not in sanitized
        assert "ValueError: Test error" in sanitized
    
    def test_sanitize_credentials(self):
        """Test sanitizing credentials from error messages."""
        message = "Connection failed: password=secret123 token=abc123"
        sanitized = sanitize_error_message(message)
        
        assert "secret123" not in sanitized
        assert "abc123" not in sanitized
        assert "[REDACTED]" in sanitized
    
    def test_sanitize_long_message(self):
        """Test truncating long error messages."""
        message = "Error: " + "x" * 600
        sanitized = sanitize_error_message(message)
        
        assert len(sanitized) <= 550  # 500 + "... (message truncated)"
        assert "truncated" in sanitized


class TestMessageCreators:
    """Tests for message creation utilities."""
    
    def test_create_success_message(self):
        """Test creating success messages."""
        message = create_success_message(
            "Job submitted",
            {"Job ID": "123", "Status": "queued"}
        )
        
        assert "✅" in message
        assert "Success" in message
        assert "Job submitted" in message
        assert "Job ID" in message
        assert "123" in message
    
    def test_create_success_message_no_details(self):
        """Test creating success message without details."""
        message = create_success_message("Operation completed")
        
        assert "✅" in message
        assert "Success" in message
        assert "Operation completed" in message
        assert "**Details:**" not in message
    
    def test_create_loading_message(self):
        """Test creating loading messages."""
        message = create_loading_message("Processing data")
        
        assert "⏳" in message
        assert "Loading" in message
        assert "Processing data" in message
    
    def test_create_warning_message(self):
        """Test creating warning messages."""
        message = create_warning_message(
            "Backend is slow",
            ["Try a different backend", "Wait for completion"]
        )
        
        assert "⚠️" in message
        assert "Warning" in message
        assert "Backend is slow" in message
        assert "**Suggestions:**" in message
        assert "Try a different backend" in message
    
    def test_create_warning_message_no_suggestions(self):
        """Test creating warning message without suggestions."""
        message = create_warning_message("Low disk space")
        
        assert "⚠️" in message
        assert "Warning" in message
        assert "Low disk space" in message
        assert "**Suggestions:**" not in message


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
