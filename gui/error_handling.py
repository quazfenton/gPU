"""
Error handling utilities for GUI interface.

This module provides standardized error handling, error message formatting,
and user-friendly error message generation for the GUI interface.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional
from enum import Enum


class ErrorType(Enum):
    """Error type categories."""
    VALIDATION = "validation"
    BACKEND = "backend"
    AUTH = "auth"
    SYSTEM = "system"
    WORKFLOW = "workflow"


@dataclass
class ErrorResponse:
    """Standard error response format."""
    error_type: ErrorType
    message: str
    details: Optional[Dict[str, Any]] = None
    suggestions: Optional[List[str]] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        """Set timestamp if not provided."""
        if self.timestamp is None:
            self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert error response to dictionary.
        
        Returns:
            Dictionary representation of error response
        """
        return {
            'error_type': self.error_type.value,
            'message': self.message,
            'details': self.details,
            'suggestions': self.suggestions,
            'timestamp': self.timestamp.isoformat()
        }
    
    def to_markdown(self) -> str:
        """
        Convert error response to user-friendly markdown format.
        
        Returns:
            Markdown formatted error message
        """
        lines = [f"❌ **Error:** {self.message}"]
        
        if self.details:
            lines.append("\n**Details:**")
            for key, value in self.details.items():
                lines.append(f"- {key}: {value}")
        
        if self.suggestions:
            lines.append("\n**Suggestions:**")
            for suggestion in self.suggestions:
                lines.append(f"- {suggestion}")
        
        return "\n".join(lines)


def format_validation_error(
    field_name: str,
    error_message: str,
    field_value: Any = None
) -> ErrorResponse:
    """
    Format a validation error with user-friendly message.
    
    Args:
        field_name: Name of the field that failed validation
        error_message: Validation error message
        field_value: Optional field value that failed validation
        
    Returns:
        ErrorResponse with validation error details
    """
    details = {'field': field_name}
    if field_value is not None:
        details['value'] = str(field_value)
    
    suggestions = [
        f"Check the '{field_name}' field and ensure it meets the requirements",
        "Review the template documentation for field specifications"
    ]
    
    return ErrorResponse(
        error_type=ErrorType.VALIDATION,
        message=f"Validation failed for field '{field_name}': {error_message}",
        details=details,
        suggestions=suggestions
    )


def format_backend_error(
    backend_name: str,
    error_message: str,
    available_backends: Optional[List[str]] = None
) -> ErrorResponse:
    """
    Format a backend error with user-friendly message.
    
    Args:
        backend_name: Name of the backend that failed
        error_message: Backend error message
        available_backends: Optional list of available alternative backends
        
    Returns:
        ErrorResponse with backend error details
    """
    details = {'backend': backend_name}
    
    suggestions = []
    if available_backends:
        suggestions.append(
            f"Try using an alternative backend: {', '.join(available_backends)}"
        )
    suggestions.extend([
        "Check backend status in the Backend Status tab",
        "Verify backend credentials and configuration",
        "Try again later if the backend is temporarily unavailable"
    ])
    
    return ErrorResponse(
        error_type=ErrorType.BACKEND,
        message=f"Backend '{backend_name}' error: {error_message}",
        details=details,
        suggestions=suggestions
    )


def format_auth_error(
    error_message: str,
    auth_type: str = "general"
) -> ErrorResponse:
    """
    Format an authentication error with user-friendly message.
    
    Args:
        error_message: Authentication error message
        auth_type: Type of authentication error (login, session, permission)
        
    Returns:
        ErrorResponse with authentication error details
    """
    details = {'auth_type': auth_type}
    
    suggestions = []
    if auth_type == "login":
        suggestions.extend([
            "Verify your username and password",
            "Check if your account is active",
            "Contact your administrator if you continue to have issues"
        ])
    elif auth_type == "session":
        suggestions.extend([
            "Your session may have expired - please log in again",
            "Check your network connection"
        ])
    elif auth_type == "permission":
        suggestions.extend([
            "You may not have permission to perform this action",
            "Contact your administrator to request access"
        ])
    else:
        suggestions.append("Please authenticate and try again")
    
    return ErrorResponse(
        error_type=ErrorType.AUTH,
        message=f"Authentication error: {error_message}",
        details=details,
        suggestions=suggestions
    )


def format_system_error(
    component: str,
    error_message: str,
    is_recoverable: bool = True
) -> ErrorResponse:
    """
    Format a system error with user-friendly message.
    
    Args:
        component: System component that failed (database, websocket, etc.)
        error_message: System error message
        is_recoverable: Whether the error is recoverable
        
    Returns:
        ErrorResponse with system error details
    """
    details = {
        'component': component,
        'recoverable': is_recoverable
    }
    
    suggestions = []
    if is_recoverable:
        suggestions.extend([
            "The system will attempt to recover automatically",
            "Try refreshing the page if the issue persists",
            "Check the system status in the Backend Status tab"
        ])
    else:
        suggestions.extend([
            "This is a critical system error",
            "Please contact your system administrator",
            "Check the system logs for more details"
        ])
    
    return ErrorResponse(
        error_type=ErrorType.SYSTEM,
        message=f"System error in {component}: {error_message}",
        details=details,
        suggestions=suggestions
    )


def format_workflow_error(
    workflow_name: str,
    error_message: str,
    step_id: Optional[str] = None,
    validation_errors: Optional[List[str]] = None
) -> ErrorResponse:
    """
    Format a workflow error with user-friendly message.
    
    Args:
        workflow_name: Name of the workflow that failed
        error_message: Workflow error message
        step_id: Optional step ID where error occurred
        validation_errors: Optional list of validation errors
        
    Returns:
        ErrorResponse with workflow error details
    """
    details = {'workflow': workflow_name}
    if step_id:
        details['step'] = step_id
    if validation_errors:
        details['validation_errors'] = validation_errors
    
    suggestions = [
        "Review the workflow structure in the Workflow Builder",
        "Check that all steps are properly connected",
        "Verify that output types match input types for connections",
        "Ensure all required inputs are provided"
    ]
    
    if step_id:
        suggestions.insert(0, f"Check the configuration of step '{step_id}'")
    
    return ErrorResponse(
        error_type=ErrorType.WORKFLOW,
        message=f"Workflow '{workflow_name}' error: {error_message}",
        details=details,
        suggestions=suggestions
    )


def format_generic_error(
    error: Exception,
    context: Optional[str] = None
) -> ErrorResponse:
    """
    Format a generic exception into a user-friendly error response.
    
    Args:
        error: Exception that occurred
        context: Optional context about where the error occurred
        
    Returns:
        ErrorResponse with generic error details
    """
    error_type = ErrorType.SYSTEM
    error_class = error.__class__.__name__
    
    # Determine error type based on exception class
    if 'Validation' in error_class or 'ValueError' in error_class:
        error_type = ErrorType.VALIDATION
    elif 'Backend' in error_class or 'Connection' in error_class:
        error_type = ErrorType.BACKEND
    elif 'Auth' in error_class or 'Permission' in error_class:
        error_type = ErrorType.AUTH
    elif 'Workflow' in error_class:
        error_type = ErrorType.WORKFLOW
    
    message = str(error)
    if context:
        message = f"{context}: {message}"
    
    details = {
        'exception_type': error_class
    }
    
    suggestions = [
        "Please try again",
        "If the problem persists, contact support",
        "Check the system logs for more details"
    ]
    
    return ErrorResponse(
        error_type=error_type,
        message=message,
        details=details,
        suggestions=suggestions
    )


def sanitize_error_message(error_message: str) -> str:
    """
    Sanitize error message to remove sensitive information.
    
    This function removes:
    - File paths that might expose system structure
    - Stack traces
    - Internal implementation details
    - Credentials or tokens
    
    Args:
        error_message: Raw error message
        
    Returns:
        Sanitized error message safe for user display
    """
    # Remove file paths (common patterns)
    import re
    
    # Remove absolute file paths
    sanitized = re.sub(r'[A-Za-z]:\\[^\s]+', '[path]', error_message)
    sanitized = re.sub(r'/[^\s]+/[^\s]+', '[path]', sanitized)
    
    # Remove stack trace lines
    lines = sanitized.split('\n')
    filtered_lines = []
    in_stack_trace = False
    
    for line in lines:
        # Detect stack trace patterns
        if 'Traceback' in line or 'File "' in line or line.strip().startswith('at '):
            in_stack_trace = True
            continue
        
        # Skip lines that look like stack trace entries
        if in_stack_trace and (line.startswith('  ') or line.startswith('\t')):
            continue
        
        in_stack_trace = False
        filtered_lines.append(line)
    
    sanitized = '\n'.join(filtered_lines)
    
    # Remove potential credentials (basic patterns)
    sanitized = re.sub(r'(password|token|key|secret)[\s:=]+[^\s]+', r'\1=[REDACTED]', sanitized, flags=re.IGNORECASE)
    
    # Limit message length
    if len(sanitized) > 500:
        sanitized = sanitized[:500] + '... (message truncated)'
    
    return sanitized.strip()


def create_success_message(
    operation: str,
    details: Optional[Dict[str, Any]] = None
) -> str:
    """
    Create a user-friendly success message.
    
    Args:
        operation: Description of the successful operation
        details: Optional additional details to include
        
    Returns:
        Markdown formatted success message
    """
    lines = [f"✅ **Success!** {operation}"]
    
    if details:
        lines.append("\n**Details:**")
        for key, value in details.items():
            lines.append(f"- {key}: {value}")
    
    return "\n".join(lines)


def create_loading_message(operation: str) -> str:
    """
    Create a loading message for long-running operations.
    
    Args:
        operation: Description of the operation in progress
        
    Returns:
        Markdown formatted loading message
    """
    return f"⏳ **Loading...** {operation}"


def create_warning_message(
    warning: str,
    suggestions: Optional[List[str]] = None
) -> str:
    """
    Create a user-friendly warning message.
    
    Args:
        warning: Warning message
        suggestions: Optional suggestions for the user
        
    Returns:
        Markdown formatted warning message
    """
    lines = [f"⚠️ **Warning:** {warning}"]
    
    if suggestions:
        lines.append("\n**Suggestions:**")
        for suggestion in suggestions:
            lines.append(f"- {suggestion}")
    
    return "\n".join(lines)
