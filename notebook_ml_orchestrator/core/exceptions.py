"""
Custom exceptions and error handling for the Notebook ML Orchestrator.

This module defines custom exception classes and error handling utilities
for different types of failures that can occur in the orchestration system.
"""

from typing import Any, Dict, Optional


class OrchestratorError(Exception):
    """Base exception class for all orchestrator errors."""
    
    def __init__(self, message: str, error_code: str = None, details: Dict[str, Any] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.details = details or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for serialization."""
        return {
            'error_type': self.__class__.__name__,
            'error_code': self.error_code,
            'message': self.message,
            'details': self.details
        }


class JobError(OrchestratorError):
    """Base class for job-related errors."""
    pass


class JobValidationError(JobError):
    """Raised when job input validation fails."""
    
    def __init__(self, message: str, validation_errors: Dict[str, str] = None):
        super().__init__(message, "JOB_VALIDATION_ERROR")
        self.validation_errors = validation_errors or {}
        self.details['validation_errors'] = self.validation_errors


class JobExecutionError(JobError):
    """Raised when job execution fails."""
    
    def __init__(self, message: str, job_id: str = None, backend_id: str = None):
        super().__init__(message, "JOB_EXECUTION_ERROR")
        self.job_id = job_id
        self.backend_id = backend_id
        self.details.update({
            'job_id': job_id,
            'backend_id': backend_id
        })


class JobTimeoutError(JobError):
    """Raised when job execution times out."""
    
    def __init__(self, message: str, job_id: str = None, timeout_seconds: int = None):
        super().__init__(message, "JOB_TIMEOUT_ERROR")
        self.job_id = job_id
        self.timeout_seconds = timeout_seconds
        self.details.update({
            'job_id': job_id,
            'timeout_seconds': timeout_seconds
        })


class TemplateError(OrchestratorError):
    """Base class for template-related errors."""
    pass


class TemplateNotFoundError(TemplateError):
    """Raised when a requested template is not found."""
    
    def __init__(self, template_name: str):
        super().__init__(f"Template '{template_name}' not found", "TEMPLATE_NOT_FOUND")
        self.template_name = template_name
        self.details['template_name'] = template_name


class TemplateValidationError(TemplateError):
    """Raised when template validation fails."""
    
    def __init__(self, message: str, template_name: str = None):
        super().__init__(message, "TEMPLATE_VALIDATION_ERROR")
        self.template_name = template_name
        self.details['template_name'] = template_name


class BackendError(OrchestratorError):
    """Base class for backend-related errors."""
    pass


class BackendNotAvailableError(BackendError):
    """Raised when no suitable backend is available."""
    
    def __init__(self, message: str, required_capabilities: list = None):
        super().__init__(message, "BACKEND_NOT_AVAILABLE")
        self.required_capabilities = required_capabilities or []
        self.details['required_capabilities'] = self.required_capabilities


class BackendConnectionError(BackendError):
    """Raised when backend connection fails."""
    
    def __init__(self, message: str, backend_id: str = None):
        super().__init__(message, "BACKEND_CONNECTION_ERROR")
        self.backend_id = backend_id
        self.details['backend_id'] = backend_id


class BackendResourceError(BackendError):
    """Raised when backend resources are insufficient."""
    
    def __init__(self, message: str, backend_id: str = None, resource_type: str = None):
        super().__init__(message, "BACKEND_RESOURCE_ERROR")
        self.backend_id = backend_id
        self.resource_type = resource_type
        self.details.update({
            'backend_id': backend_id,
            'resource_type': resource_type
        })


class WorkflowError(OrchestratorError):
    """Base class for workflow-related errors."""
    pass


class WorkflowValidationError(WorkflowError):
    """Raised when workflow definition validation fails."""
    
    def __init__(self, message: str, workflow_id: str = None):
        super().__init__(message, "WORKFLOW_VALIDATION_ERROR")
        self.workflow_id = workflow_id
        self.details['workflow_id'] = workflow_id


class WorkflowExecutionError(WorkflowError):
    """Raised when workflow execution fails."""
    
    def __init__(self, message: str, workflow_id: str = None, step_name: str = None):
        super().__init__(message, "WORKFLOW_EXECUTION_ERROR")
        self.workflow_id = workflow_id
        self.step_name = step_name
        self.details.update({
            'workflow_id': workflow_id,
            'step_name': step_name
        })


class BatchError(OrchestratorError):
    """Base class for batch processing errors."""
    pass


class BatchValidationError(BatchError):
    """Raised when batch job validation fails."""
    
    def __init__(self, message: str, batch_id: str = None):
        super().__init__(message, "BATCH_VALIDATION_ERROR")
        self.batch_id = batch_id
        self.details['batch_id'] = batch_id


class DatabaseError(OrchestratorError):
    """Base class for database-related errors."""
    pass


class DatabaseConnectionError(DatabaseError):
    """Raised when database connection fails."""
    
    def __init__(self, message: str, db_path: str = None):
        super().__init__(message, "DATABASE_CONNECTION_ERROR")
        self.db_path = db_path
        self.details['db_path'] = db_path


class DatabaseOperationError(DatabaseError):
    """Raised when database operation fails."""
    
    def __init__(self, message: str, operation: str = None):
        super().__init__(message, "DATABASE_OPERATION_ERROR")
        self.operation = operation
        self.details['operation'] = operation


class ConfigurationError(OrchestratorError):
    """Raised when configuration is invalid or missing."""
    
    def __init__(self, message: str, config_key: str = None):
        super().__init__(message, "CONFIGURATION_ERROR")
        self.config_key = config_key
        self.details['config_key'] = config_key


class SecurityError(OrchestratorError):
    """Raised when security validation fails."""
    
    def __init__(self, message: str, security_check: str = None):
        super().__init__(message, "SECURITY_ERROR")
        self.security_check = security_check
        self.details['security_check'] = security_check


def handle_exception(func):
    """
    Decorator to handle exceptions and convert them to OrchestratorError.
    
    Args:
        func: Function to wrap
        
    Returns:
        Wrapped function
    """
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except OrchestratorError:
            # Re-raise orchestrator errors as-is
            raise
        except Exception as e:
            # Convert other exceptions to OrchestratorError
            raise OrchestratorError(
                f"Unexpected error in {func.__name__}: {str(e)}",
                "UNEXPECTED_ERROR",
                {'original_exception': str(e), 'function': func.__name__}
            ) from e
    return wrapper


class ErrorHandler:
    """Centralized error handling and reporting."""
    
    def __init__(self, logger=None):
        self.logger = logger
    
    def handle_error(self, error: Exception, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Handle an error and return structured error information.
        
        Args:
            error: Exception to handle
            context: Additional context information
            
        Returns:
            Structured error information
        """
        if isinstance(error, OrchestratorError):
            error_info = error.to_dict()
        else:
            error_info = {
                'error_type': type(error).__name__,
                'error_code': 'UNKNOWN_ERROR',
                'message': str(error),
                'details': {}
            }
        
        if context:
            error_info['context'] = context
        
        if self.logger:
            self.logger.error(f"Error handled: {error_info}")
        
        return error_info
    
    def create_error_response(self, error: Exception, request_id: str = None) -> Dict[str, Any]:
        """
        Create a standardized error response.
        
        Args:
            error: Exception to create response for
            request_id: Optional request ID for tracking
            
        Returns:
            Standardized error response
        """
        error_info = self.handle_error(error)
        
        response = {
            'success': False,
            'error': error_info,
            'timestamp': str(datetime.now())
        }
        
        if request_id:
            response['request_id'] = request_id
        
        return response