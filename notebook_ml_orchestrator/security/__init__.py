"""
Security package for Notebook ML Orchestrator.

This package provides comprehensive security features including:
- Credential encryption (AES-256-GCM)
- JWT authentication
- OAuth 2.0 support
- Security logging and monitoring
- XSS prevention
- Input sanitization
- Security middleware for web frameworks

Example usage:
    from notebook_ml_orchestrator.security import (
        CredentialStore, AuthManager, SecurityLogger,
        ContentSanitizer, SecurityMiddleware, create_security_middleware
    )
    
    # Initialize security components
    store = CredentialStore(master_key=os.environ['MASTER_KEY'])
    auth = AuthManager(secret_key=os.environ['JWT_SECRET'])
    logger = SecurityLogger(log_file='security.log')
    
    # Or use the factory function
    middleware = create_security_middleware(
        enable_auth=True,
        enable_rate_limit=True,
        enable_audit_logging=True
    )
"""

from .credential_store import CredentialStore, CredentialEncryptionError
from .auth_manager import AuthManager, AuthenticationError, TokenValidationError, Role
from .security_logger import SecurityLogger, SecurityEvent
from .xss_prevention import ContentSanitizer, XSSPreventionError
from .middleware import (
    SecurityMiddleware,
    SecurityContext,
    GradioSecurityMiddleware,
    require_auth,
    rate_limit,
    validate_request,
    create_security_middleware
)

__version__ = '1.0.0'

__all__ = [
    # Core security
    'CredentialStore',
    'CredentialEncryptionError',
    'AuthManager',
    'AuthenticationError',
    'TokenValidationError',
    'Role',
    'SecurityLogger',
    'SecurityEvent',
    'ContentSanitizer',
    'XSSPreventionError',
    
    # Middleware
    'SecurityMiddleware',
    'SecurityContext',
    'GradioSecurityMiddleware',
    
    # Decorators
    'require_auth',
    'rate_limit',
    'validate_request',
    
    # Factory
    'create_security_middleware',
]
