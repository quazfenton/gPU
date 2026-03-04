"""
Security Utilities for Notebook ML Orchestrator.

This module provides comprehensive security utilities including:
- Input validation and sanitization
- Error handling with security awareness
- Rate limiting
- Security headers
- Request validation

These utilities should be used throughout the application to ensure
consistent security practices.
"""

import re
import html
import time
import hashlib
import secrets
import socket
import ipaddress
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict
from urllib.parse import urlparse
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# Input Validation
# =============================================================================

class InputValidator:
    """Comprehensive input validation utilities."""

    # Dangerous characters that could indicate injection attempts
    DANGEROUS_CHARS = ['<', '>', ';', '|', '&', '$', '`', '(', ')', '{', '}', '[', ']']

    # SQL injection patterns
    SQL_INJECTION_PATTERNS = [
        r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|UNION|ALTER|CREATE|TRUNCATE)\b)",
        r"(--|#|/\*|\*/)",
        r"(\bOR\b\s+\d+\s*=\s*\d+)",
        r"(\bAND\b\s+\d+\s*=\s*\d+)",
        r"(\bOR\b\s+['\"]*\w+['\"]*\s*=\s*['\"]*\w+['\"]*)",
    ]

    # XSS patterns
    XSS_PATTERNS = [
        r"<\s*script",
        r"javascript\s*:",
        r"on\w+\s*=",
        r"<\s*img[^>]+onerror",
        r"<\s*iframe",
        r"<\s*object",
        r"<\s*embed",
    ]

    # Path traversal patterns
    PATH_TRAVERSAL_PATTERNS = [
        r"\.\./",
        r"\.\.\\",
        r"%2e%2e%2f",
        r"%2e%2e/",
        r"\.\.%2f",
        r"%2e%2e\\",
    ]

    @classmethod
    def validate_string(cls, value: str, max_length: int = 1000,
                       allow_empty: bool = False,
                       pattern: str = None) -> Tuple[bool, str]:
        """
        Validate a string input.

        Args:
            value: String to validate
            max_length: Maximum allowed length
            allow_empty: Whether empty string is allowed
            pattern: Optional regex pattern to match

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not isinstance(value, str):
            return False, "Input must be a string"

        if not allow_empty and not value.strip():
            return False, "Input cannot be empty"

        if len(value) > max_length:
            return False, f"Input exceeds maximum length of {max_length}"

        # Check for dangerous characters
        for char in cls.DANGEROUS_CHARS:
            if char in value:
                return False, f"Input contains invalid character: {char}"

        # Check for SQL injection patterns
        for pattern in cls.SQL_INJECTION_PATTERNS:
            if re.search(pattern, value, re.IGNORECASE):
                return False, "Input contains potentially dangerous SQL patterns"

        # Check for XSS patterns
        for pattern in cls.XSS_PATTERNS:
            if re.search(pattern, value, re.IGNORECASE):
                return False, "Input contains potentially dangerous XSS patterns"

        # Check custom pattern if provided
        if pattern and not re.match(pattern, value):
            return False, f"Input does not match required pattern"

        return True, ""

    @classmethod
    def validate_integer(cls, value: Any, min_value: int = None,
                        max_value: int = None) -> Tuple[bool, str]:
        """
        Validate an integer input.

        Args:
            value: Value to validate
            min_value: Minimum allowed value
            max_value: Maximum allowed value

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            int_value = int(value)
        except (ValueError, TypeError):
            return False, "Input must be an integer"

        if min_value is not None and int_value < min_value:
            return False, f"Value must be at least {min_value}"

        if max_value is not None and int_value > max_value:
            return False, f"Value must be at most {max_value}"

        return True, ""

    @classmethod
    def validate_email(cls, email: str) -> Tuple[bool, str]:
        """
        Validate an email address.

        Args:
            email: Email address to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not isinstance(email, str):
            return False, "Email must be a string"

        # Basic email pattern
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(pattern, email):
            return False, "Invalid email format"

        if len(email) > 254:
            return False, "Email address too long"

        return True, ""

    @classmethod
    def validate_url(cls, url: str, allowed_schemes: List[str] = None) -> Tuple[bool, str]:
        """
        Validate a URL.

        Args:
            url: URL to validate
            allowed_schemes: List of allowed URL schemes (default: ['http', 'https'])

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not isinstance(url, str):
            return False, "URL must be a string"

        if allowed_schemes is None:
            allowed_schemes = ['http', 'https']

        try:
            parsed = urlparse(url)

            # Check scheme
            if parsed.scheme not in allowed_schemes:
                return False, f"URL scheme must be one of: {', '.join(allowed_schemes)}"

            # Check hostname
            if not parsed.hostname:
                return False, "URL must have a hostname"

            # Check for path traversal
            for pattern in cls.PATH_TRAVERSAL_PATTERNS:
                if re.search(pattern, url, re.IGNORECASE):
                    return False, "URL contains path traversal patterns"

            return True, ""

        except Exception:
            return False, "Invalid URL format"

    @classmethod
    def validate_json_dict(cls, value: Any, max_depth: int = 10,
                          max_keys: int = 100) -> Tuple[bool, str]:
        """
        Validate a JSON/dict structure.

        Args:
            value: Value to validate
            max_depth: Maximum nesting depth
            max_keys: Maximum number of keys per object

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not isinstance(value, dict):
            return False, "Input must be a dictionary"

        def check_depth(obj, current_depth):
            if current_depth > max_depth:
                return False, f"JSON structure exceeds maximum depth of {max_depth}"

            if isinstance(obj, dict):
                if len(obj) > max_keys:
                    return False, f"Object exceeds maximum keys of {max_keys}"

                for key, val in obj.items():
                    if not isinstance(key, str):
                        return False, "All keys must be strings"

                    result = check_depth(val, current_depth + 1)
                    if not result[0]:
                        return result

            elif isinstance(obj, list):
                if len(obj) > max_keys:
                    return False, f"Array exceeds maximum size of {max_keys}"

                for item in obj:
                    result = check_depth(item, current_depth + 1)
                    if not result[0]:
                        return result

            return True, ""

        return check_depth(value, 0)


# =============================================================================
# Input Sanitization
# =============================================================================

class InputSanitizer:
    """Input sanitization utilities."""

    @staticmethod
    def sanitize_string(value: str, max_length: int = 1000) -> str:
        """
        Sanitize a string by removing dangerous characters.

        Args:
            value: String to sanitize
            max_length: Maximum length after sanitization

        Returns:
            Sanitized string
        """
        if not isinstance(value, str):
            value = str(value)

        # Remove null bytes
        value = value.replace('\x00', '')

        # Trim whitespace
        value = value.strip()

        # Remove dangerous characters
        for char in ['<', '>', ';', '|', '&', '$', '`']:
            value = value.replace(char, '')

        # Truncate to max length
        if len(value) > max_length:
            value = value[:max_length]

        return value

    @staticmethod
    def escape_html(value: str) -> str:
        """
        Escape HTML special characters.

        Args:
            value: String to escape

        Returns:
            HTML-escaped string
        """
        if not isinstance(value, str):
            value = str(value)

        return html.escape(value, quote=True)

    @staticmethod
    def sanitize_filename(filename: str, allow_extensions: List[str] = None) -> str:
        """
        Sanitize a filename to prevent path traversal.

        Args:
            filename: Filename to sanitize
            allow_extensions: List of allowed extensions

        Returns:
            Sanitized filename
        """
        if not isinstance(filename, str):
            return ""

        # Get just the filename without path
        filename = filename.split('/')[-1].split('\\')[-1]

        # Remove null bytes
        filename = filename.replace('\x00', '')

        # Remove path traversal sequences
        filename = filename.replace('../', '').replace('..\\', '')
        filename = filename.replace('.', '', filename.count('.') - 1) if '.' in filename else filename

        # Remove dangerous characters
        for char in ['<', '>', ':', '"', '|', '?', '*', ';', '&', '$', '`']:
            filename = filename.replace(char, '_')

        # Check extension if allowed extensions specified
        if allow_extensions:
            ext = filename.split('.')[-1].lower() if '.' in filename else ''
            if ext not in allow_extensions:
                return ""

        # Limit length
        if len(filename) > 255:
            name, ext = filename.rsplit('.', 1) if '.' in filename else (filename, '')
            filename = f"{name[:250-len(ext)]}.{ext}" if ext else filename[:255]

        return filename


# =============================================================================
# SSRF Protection
# =============================================================================

class SSRFProtection:
    """Server-Side Request Forgery (SSRF) protection."""

    # Private IP ranges
    PRIVATE_IP_RANGES = [
        '10.0.0.0/8',
        '172.16.0.0/12',
        '192.168.0.0/16',
        '127.0.0.0/8',
        '169.254.0.0/16',
        '0.0.0.0/8',
    ]

    # Cloud metadata endpoints
    METADATA_ENDPOINTS = [
        '169.254.169.254',  # AWS, GCP, Azure
        'metadata.google.internal',  # GCP
        '169.254.170.2',  # AWS ECS
    ]

    @classmethod
    def is_safe_url(cls, url: str) -> Tuple[bool, str]:
        """
        Check if a URL is safe to access (not internal/private).

        Args:
            url: URL to check

        Returns:
            Tuple of (is_safe, error_message)
        """
        try:
            parsed = urlparse(url)

            # Only allow HTTP and HTTPS
            if parsed.scheme not in ['http', 'https']:
                return False, "Only HTTP and HTTPS URLs are allowed"

            # Check hostname
            hostname = parsed.hostname
            if not hostname:
                return False, "URL must have a hostname"

            # Check for metadata endpoints
            if hostname in cls.METADATA_ENDPOINTS:
                return False, "Access to cloud metadata endpoints is forbidden"

            # Resolve hostname
            try:
                ip = socket.gethostbyname(hostname)
                ip_obj = ipaddress.ip_address(ip)
            except socket.gaierror:
                return False, f"Could not resolve hostname: {hostname}"

            # Check for private IP addresses
            if ip_obj.is_private:
                return False, "Access to private IP addresses is forbidden"

            if ip_obj.is_loopback:
                return False, "Access to localhost is forbidden"

            if ip_obj.is_link_local:
                return False, "Access to link-local addresses is forbidden"

            if ip_obj.is_multicast:
                return False, "Access to multicast addresses is forbidden"

            if ip_obj.is_unspecified:
                return False, "Access to unspecified addresses is forbidden"

            return True, ""

        except Exception as e:
            return False, f"URL validation failed: {str(e)}"


# =============================================================================
# Rate Limiting
# =============================================================================

@dataclass
class RateLimitConfig:
    """Rate limit configuration."""
    requests_per_minute: int = 60
    requests_per_hour: int = 1000
    requests_per_day: int = 10000
    burst_size: int = 10
    enabled: bool = True


class RateLimiter:
    """Token bucket rate limiter with multiple time windows."""

    def __init__(self, config: RateLimitConfig = None):
        """
        Initialize rate limiter.

        Args:
            config: Rate limit configuration
        """
        self.config = config or RateLimitConfig()
        self._requests = defaultdict(list)  # identifier -> list of timestamps
        self._lock = None

        try:
            import threading
            self._lock = threading.RLock()
        except ImportError:
            pass

    def check_rate_limit(self, identifier: str) -> Tuple[bool, Dict[str, Any]]:
        """
        Check if request is within rate limits.

        Args:
            identifier: Unique identifier (user ID, IP, API key, etc.)

        Returns:
            Tuple of (allowed, info_dict)
        """
        if not self.config.enabled:
            return True, {"allowed": True, "reason": "Rate limiting disabled"}

        now = time.time()

        with self._lock if self._lock else threading.RLock():
            # Get request history for this identifier
            history = self._requests[identifier]

            # Clean old entries (older than 1 day)
            cutoff = now - 86400
            history = [t for t in history if t > cutoff]
            self._requests[identifier] = history

            # Check daily limit
            daily_count = len(history)
            if daily_count >= self.config.requests_per_day:
                return False, {
                    "allowed": False,
                    "reason": "Daily rate limit exceeded",
                    "retry_after": 86400,
                    "limit": self.config.requests_per_day,
                    "remaining": 0,
                }

            # Check hourly limit
            hour_cutoff = now - 3600
            hour_count = len([t for t in history if t > hour_cutoff])
            if hour_count >= self.config.requests_per_hour:
                return False, {
                    "allowed": False,
                    "reason": "Hourly rate limit exceeded",
                    "retry_after": 3600,
                    "limit": self.config.requests_per_hour,
                    "remaining": 0,
                }

            # Check minute limit
            minute_cutoff = now - 60
            minute_count = len([t for t in history if t > minute_cutoff])
            if minute_count >= self.config.requests_per_minute:
                return False, {
                    "allowed": False,
                    "reason": "Per-minute rate limit exceeded",
                    "retry_after": 60,
                    "limit": self.config.requests_per_minute,
                    "remaining": 0,
                }

            # Check burst limit
            burst_cutoff = now - 1
            burst_count = len([t for t in history if t > burst_cutoff])
            if burst_count >= self.config.burst_size:
                return False, {
                    "allowed": False,
                    "reason": "Burst rate limit exceeded",
                    "retry_after": 1,
                    "limit": self.config.burst_size,
                    "remaining": 0,
                }

            # Record this request
            history.append(now)
            self._requests[identifier] = history

            # Calculate remaining requests
            remaining_minute = self.config.requests_per_minute - minute_count - 1
            remaining_hour = self.config.requests_per_hour - hour_count - 1
            remaining_day = self.config.requests_per_day - daily_count - 1

            return True, {
                "allowed": True,
                "remaining_minute": max(0, remaining_minute),
                "remaining_hour": max(0, remaining_hour),
                "remaining_day": max(0, remaining_day),
                "limit_minute": self.config.requests_per_minute,
                "limit_hour": self.config.requests_per_hour,
                "limit_day": self.config.requests_per_day,
            }

    def reset(self, identifier: str):
        """
        Reset rate limit for an identifier.

        Args:
            identifier: Identifier to reset
        """
        with self._lock if self._lock else threading.RLock():
            if identifier in self._requests:
                del self._requests[identifier]


# =============================================================================
# Security Headers
# =============================================================================

class SecurityHeaders:
    """Security header management."""

    DEFAULT_HEADERS = {
        'X-Content-Type-Options': 'nosniff',
        'X-Frame-Options': 'DENY',
        'X-XSS-Protection': '1; mode=block',
        'Referrer-Policy': 'strict-origin-when-cross-origin',
        'Permissions-Policy': 'geolocation=(), microphone=(), camera=()',
        'Cache-Control': 'no-store, no-cache, must-revalidate',
        'Pragma': 'no-cache',
    }

    CSP_DEFAULT = (
        "default-src 'self'; "
        "script-src 'self'; "
        "style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data: https:; "
        "font-src 'self' data:; "
        "connect-src 'self'; "
        "frame-ancestors 'none'; "
        "base-uri 'self'; "
        "form-action 'self';"
    )

    @classmethod
    def get_all_headers(cls, include_csp: bool = True) -> Dict[str, str]:
        """
        Get all security headers.

        Args:
            include_csp: Whether to include Content-Security-Policy

        Returns:
            Dictionary of headers
        """
        headers = cls.DEFAULT_HEADERS.copy()

        if include_csp:
            headers['Content-Security-Policy'] = cls.CSP_DEFAULT

        return headers

    @classmethod
    def apply_to_response(cls, response, include_csp: bool = True):
        """
        Apply security headers to a response object.

        Args:
            response: Response object with headers attribute
            include_csp: Whether to include CSP
        """
        headers = cls.get_all_headers(include_csp)

        for header, value in headers.items():
            if hasattr(response, 'headers'):
                response.headers[header] = value

        return response


# =============================================================================
# Error Handling
# =============================================================================

@dataclass
class SecureError:
    """Secure error response."""
    success: bool = False
    error_code: str = "ERROR"
    message: str = "An error occurred"
    details: Dict[str, Any] = field(default_factory=dict)
    request_id: str = field(default_factory=lambda: secrets.token_hex(8))
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "error_code": self.error_code,
            "message": self.message,
            "request_id": self.request_id,
            "timestamp": self.timestamp,
            **self.details
        }


def create_error_response(error: Exception, include_details: bool = False,
                         request_id: str = None) -> SecureError:
    """
    Create a secure error response.

    Args:
        error: Exception that occurred
        include_details: Whether to include error details (for development)
        request_id: Optional request ID

    Returns:
        SecureError instance
    """
    # Map exception types to error codes
    error_mapping = {
        ValueError: ("BAD_REQUEST", 400),
        PermissionError: ("FORBIDDEN", 403),
        FileNotFoundError: ("NOT_FOUND", 404),
        TimeoutError: ("TIMEOUT", 408),
    }

    error_class = type(error)
    error_code, status_code = error_mapping.get(error_class, ("INTERNAL_ERROR", 500))

    # Create base response
    response = SecureError(
        error_code=error_code,
        message=get_error_message(error_code),
        request_id=request_id or secrets.token_hex(8),
    )

    # Add details only in development mode
    if include_details:
        import traceback
        response.details = {
            "exception_type": error_class.__name__,
            "exception_message": str(error),
            "traceback": traceback.format_exc(),
            "status_code": status_code,
        }

    return response


def get_error_message(error_code: str) -> str:
    """Get user-friendly error message for error code."""
    messages = {
        "BAD_REQUEST": "The request was invalid or malformed",
        "FORBIDDEN": "You do not have permission to access this resource",
        "NOT_FOUND": "The requested resource was not found",
        "TIMEOUT": "The request timed out",
        "INTERNAL_ERROR": "An internal error occurred",
        "RATE_LIMITED": "Too many requests. Please try again later",
        "UNAUTHORIZED": "Authentication is required",
        "VALIDATION_ERROR": "Input validation failed",
    }
    return messages.get(error_code, "An error occurred")


# =============================================================================
# Export Public API
# =============================================================================

__all__ = [
    # Validation
    "InputValidator",
    "InputSanitizer",
    # SSRF Protection
    "SSRFProtection",
    # Rate Limiting
    "RateLimitConfig",
    "RateLimiter",
    # Security Headers
    "SecurityHeaders",
    # Error Handling
    "SecureError",
    "create_error_response",
    "get_error_message",
]
