"""
Security middleware and utilities for integration with web frameworks.

This module provides:
- FastAPI/Gradio security middleware
- Authentication decorators
- Request validation utilities
- Security header injection
- Rate limiting integration
"""

import functools
import hashlib
import os
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class SecurityContext:
    """Security context for current request."""
    user_id: Optional[str] = None
    username: Optional[str] = None
    role: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    session_id: Optional[str] = None
    authenticated: bool = False
    permissions: List[str] = None
    
    def __post_init__(self):
        if self.permissions is None:
            self.permissions = []


class SecurityMiddleware:
    """
    Security middleware for web applications.
    
    Provides:
    - Authentication verification
    - Authorization checks
    - Security header injection
    - Request logging
    - Rate limiting integration
    """
    
    def __init__(
        self,
        auth_manager=None,
        credential_store=None,
        security_logger=None,
        enabled: bool = True
    ):
        """
        Initialize security middleware.
        
        Args:
            auth_manager: AuthManager instance
            credential_store: CredentialStore instance
            security_logger: SecurityLogger instance
            enabled: Enable/disable middleware
        """
        self.auth_manager = auth_manager
        self.credential_store = credential_store
        self.security_logger = security_logger
        self.enabled = enabled
        
        # Security headers
        self.security_headers = {
            'X-Content-Type-Options': 'nosniff',
            'X-Frame-Options': 'DENY',
            'X-XSS-Protection': '1; mode=block',
            'Referrer-Policy': 'strict-origin-when-cross-origin',
            'Permissions-Policy': 'geolocation=(), microphone=(), camera=()'
        }
        
        # Rate limiting
        self._rate_limit_store: Dict[str, List[float]] = {}
        self._rate_limit_requests = 60  # per minute
        self._rate_limit_window = 60  # seconds
        
        logger.info("SecurityMiddleware initialized")
    
    def get_security_context(
        self,
        request=None,
        token: Optional[str] = None,
        api_key: Optional[str] = None
    ) -> SecurityContext:
        """
        Extract security context from request.
        
        Args:
            request: Request object
            token: JWT token (alternative to request header)
            api_key: API key (alternative to request header)
            
        Returns:
            SecurityContext instance
        """
        ctx = SecurityContext()
        
        # Extract IP address and user agent
        if request:
            client = getattr(request, 'client', None)
            if isinstance(client, dict):
                ctx.ip_address = client.get('host')
            else:
                ctx.ip_address = getattr(client, 'host', None)

            headers = getattr(request, 'headers', None)
            if hasattr(headers, 'get'):
                ctx.user_agent = headers.get('user-agent')
        # Try token authentication
        if token:
            try:
                payload = self.auth_manager.validate_token(token)
                ctx.user_id = payload.user_id
                ctx.username = payload.username
                ctx.role = payload.role
                ctx.authenticated = True
            except Exception as e:
                logger.warning(f"Token validation failed: {e}")
        
        # Try API key authentication
        elif api_key:
            user = self.auth_manager.authenticate_api_key(api_key)
            if user:
                ctx.user_id = user.id
                ctx.username = user.username
                ctx.role = user.role.value
                ctx.authenticated = True
        
        return ctx
    
    def check_rate_limit(self, identifier: str) -> Tuple[bool, int]:
        """
        Check rate limit for identifier.
        
        Args:
            identifier: Rate limit identifier (user ID, IP, etc.)
            
        Returns:
            Tuple of (allowed, retry_after_seconds)
        """
        now = time.time()
        window_start = now - self._rate_limit_window
        
        # Clean old entries
        if identifier in self._rate_limit_store:
            self._rate_limit_store[identifier] = [
                t for t in self._rate_limit_store[identifier] if t > window_start
            ]
        else:
            self._rate_limit_store[identifier] = []
        
        # Check limit
        if len(self._rate_limit_store[identifier]) >= self._rate_limit_requests:
            retry_after = int(self._rate_limit_store[identifier][0] + self._rate_limit_window - now)
            return False, max(1, retry_after)
        
        # Record request
        self._rate_limit_store[identifier].append(now)
        return True, 0
    
    def add_security_headers(self, response) -> Any:
        """
        Add security headers to response.
        
        Args:
            response: Response object
            
        Returns:
            Response with security headers
        """
        if hasattr(response, 'headers'):
            for header, value in self.security_headers.items():
                response.headers[header] = value
        return response
    
    def log_request(
        self,
        method: str,
        path: str,
        status_code: int,
        duration_ms: float,
        context: SecurityContext
    ) -> None:
        """
        Log request for security auditing.
        
        Args:
            method: HTTP method
            path: Request path
            status_code: Response status code
            duration_ms: Request duration in milliseconds
            context: Security context
        """
        if not self.enabled or not self.security_logger:
            return
        
        # Log suspicious activity
        if status_code >= 400:
            if status_code == 401:
                self.security_logger.log_auth_failure(
                    context.username or 'unknown',
                    ip_address=context.ip_address,
                    reason='unauthorized_access'
                )
            elif status_code == 403:
                self.security_logger.log_authz_failure(
                    context.username or 'unknown',
                    path,
                    'unknown',
                    ip_address=context.ip_address
                )
    
    def sanitize_input(self, value: Any, context: str = '') -> Any:
        """
        Sanitize input value.
        
        Args:
            value: Input value to sanitize
            context: Context description for logging
            
        Returns:
            Sanitized value
        """
        from .xss_prevention import ContentSanitizer, detect_xss
        
        sanitizer = ContentSanitizer()
        
        if isinstance(value, str):
            # Check for XSS
            is_malicious, patterns = detect_xss(value)
            if is_malicious:
                logger.warning(f"XSS attempt detected in {context}: {patterns}")
                if self.security_logger:
                    self.security_logger.log_xss_attempt(value)
            
            # Sanitize HTML
            return sanitizer.sanitize_html(value).content
        
        elif isinstance(value, dict):
            return {k: self.sanitize_input(v, f"{context}.{k}") for k, v in value.items()}
        
        elif isinstance(value, list):
            return [self.sanitize_input(item, f"{context}[{i}]") for i, item in enumerate(value)]
        
        return value


def require_auth(auth_manager=None, required_role: Optional[str] = None):
    """
    Decorator to require authentication for endpoint.
    
    Args:
        auth_manager: AuthManager instance
        required_role: Required role (None = any authenticated user)
        
    Usage:
        @app.get('/admin')
        @require_auth(auth_manager, required_role='admin')
        def admin_endpoint():
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get token from request (framework-specific)
            token = kwargs.get('token') or kwargs.get('authorization')
            
            if not token:
                return {'error': 'Authentication required'}, 401
            
            # Validate token
            try:
                payload = auth_manager.validate_token(token)
                
                # Check role
                if required_role and payload.role != required_role:
                    return {'error': 'Insufficient permissions'}, 403
                
                # Add user info to kwargs
                kwargs['current_user'] = {
                    'id': payload.user_id,
                    'username': payload.username,
                    'role': payload.role
                }
                
                return func(*args, **kwargs)
                
            except Exception as e:
                return {'error': f'Authentication failed: {str(e)}'}, 401
        
        return wrapper
    return decorator


def rate_limit(requests_per_minute: int = 60):
    """
    Decorator to apply rate limiting.
    
    Args:
        requests_per_minute: Maximum requests per minute
        
    Usage:
        @app.post('/api/jobs')
        @rate_limit(requests_per_minute=100)
        def create_job():
            ...
    """
    def decorator(func: Callable) -> Callable:
        _rate_store: Dict[str, List[float]] = {}
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            identifier = kwargs.get('ip_address') or kwargs.get('user_id', 'unknown')
            now = time.time()
            window_start = now - 60
            
            # Clean old entries
            if identifier in _rate_store:
                _rate_store[identifier] = [t for t in _rate_store[identifier] if t > window_start]
            else:
                _rate_store[identifier] = []
            
            # Check limit
            if len(_rate_store[identifier]) >= requests_per_minute:
                return {'error': 'Rate limit exceeded'}, 429
            
            # Record request
            _rate_store[identifier].append(now)
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


def validate_request(schema: Dict[str, Any]):
    """
    Decorator to validate request data against schema.
    
    Args:
        schema: Validation schema
        
    Usage:
        @app.post('/api/jobs')
        @validate_request({
            'template': {'type': str, 'required': True},
            'inputs': {'type': dict, 'required': True}
        })
        def create_job(template, inputs):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            request_data = kwargs.get('request_data', {})
            errors = []
            
            for field, rules in schema.items():
                value = request_data.get(field)
                
                # Check required
                if rules.get('required') and value is None:
                    errors.append(f"Field '{field}' is required")
                    continue
                
                # Check type
                if value is not None and 'type' in rules:
                    if not isinstance(value, rules['type']):
                        errors.append(f"Field '{field}' must be {rules['type'].__name__}")
                
                # Check pattern
                if value is not None and 'pattern' in rules and isinstance(value, str):
                    import re
                    if not re.match(rules['pattern'], value):
                        errors.append(f"Field '{field}' does not match pattern")
            
            if errors:
                return {'error': 'Validation failed', 'details': errors}, 400
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


class GradioSecurityMiddleware:
    """
    Security middleware specifically for Gradio applications.
    
    Integrates with Gradio's event system to add security checks.
    """
    
    def __init__(self, middleware: SecurityMiddleware):
        """
        Initialize Gradio security middleware.
        
        Args:
            middleware: SecurityMiddleware instance
        """
        self.middleware = middleware
    
    def secure_function(
        self,
        func: Callable,
        require_auth: bool = True,
        required_role: Optional[str] = None,
        sanitize_inputs: bool = True
    ) -> Callable:
        """
        Wrap a function with security checks.
        
        Args:
            func: Function to wrap
            require_auth: Require authentication
            required_role: Required role
            sanitize_inputs: Sanitize inputs
            
        Returns:
            Wrapped function
        """
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get security context from kwargs or global state
            ctx = kwargs.pop('security_context', None)
            
            # Check authentication
            if require_auth and (not ctx or not ctx.authenticated):
                raise PermissionError("Authentication required")
            
            # Check role
            if required_role and ctx and ctx.role != required_role:
                if self.middleware.security_logger:
                    self.middleware.security_logger.log_authz_failure(
                        ctx.username,
                        func.__name__,
                        required_role
                    )
                raise PermissionError(f"Role '{ctx.role}' insufficient, requires '{required_role}'")
            
            # Sanitize inputs
            if sanitize_inputs:
                for key, value in kwargs.items():
                    kwargs[key] = self.middleware.sanitize_input(value, f"{func.__name__}.{key}")
            
            return func(*args, **kwargs)
        
        return wrapper


def create_security_middleware(
    enable_auth: bool = True,
    enable_rate_limit: bool = True,
    enable_audit_logging: bool = True
) -> SecurityMiddleware:
    """
    Factory function to create configured security middleware.
    
    Args:
        enable_auth: Enable authentication checks
        enable_rate_limit: Enable rate limiting
        enable_audit_logging: Enable audit logging
        
    Returns:
        Configured SecurityMiddleware instance
    """
    from .auth_manager import AuthManager
    from .credential_store import CredentialStore
    from .security_logger import SecurityLogger
    
    # Initialize components
    auth_manager = None
    credential_store = None
    security_logger = None
    
    if enable_auth:
        auth_manager = AuthManager(
            secret_key=os.environ.get('JWT_SECRET'),
            enable_audit_logging=enable_audit_logging
        )
    
    if enable_auth or enable_audit_logging:
        credential_store = CredentialStore(
            master_key=os.environ.get('MASTER_KEY'),
            enable_audit_logging=enable_audit_logging
        )
    
    if enable_audit_logging:
        security_logger = SecurityLogger(
            log_file='security.log',
            include_console=False
        )
    
    middleware = SecurityMiddleware(
        auth_manager=auth_manager,
        credential_store=credential_store,
        security_logger=security_logger,
        enabled=True
    )
    
    if enable_rate_limit:
        middleware._rate_limit_requests = 60
        middleware._rate_limit_window = 60
    
    logger.info("Security middleware created successfully")
    return middleware
