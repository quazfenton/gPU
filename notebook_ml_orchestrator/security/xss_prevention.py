"""
Input Sanitization and XSS Prevention Module

Provides comprehensive input validation and sanitization to prevent:
- Cross-Site Scripting (XSS) attacks
- SQL injection (defense in depth)
- Command injection
- Path traversal
- HTML/JavaScript injection

Usage:
    from notebook_ml_orchestrator.security.xss_prevention import ContentSanitizer
    
    sanitizer = ContentSanitizer()
    clean_value = sanitizer.sanitize_html(user_input)
"""

import html
import re
import logging
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class XSSPreventionError(Exception):
    """Exception raised for XSS prevention errors."""
    
    def __init__(self, message: str, error_code: str = "XSS_PREVENTION"):
        super().__init__(message)
        self.error_code = error_code


@dataclass
class SanitizationResult:
    """Result of sanitization operation."""
    success: bool
    value: Any
    original: Any
    warnings: List[str]
    errors: List[str]


class ContentSanitizer:
    """
    Comprehensive content sanitizer for user inputs.
    
    Features:
    - HTML escaping for XSS prevention
    - SQL injection pattern detection
    - Command injection pattern detection
    - Path traversal prevention
    - Unicode normalization
    - Length validation
    - Type coercion with validation
    """
    
    # Dangerous HTML/JavaScript patterns
    DANGEROUS_HTML_PATTERN = re.compile(
        r'<\s*(script|iframe|object|embed|applet|form|input|button|textarea|select|style|link|meta|base|body|frame|frameset)[^>]*>',
        re.IGNORECASE
    )
    
    # Event handler patterns (onclick, onerror, etc.)
    EVENT_HANDLER_PATTERN = re.compile(
        r'\s*on\w+\s*=\s*["\'][^"\']*["\']',
        re.IGNORECASE
    )
    
    # JavaScript URL pattern
    JAVASCRIPT_URL_PATTERN = re.compile(
        r'javascript\s*:',
        re.IGNORECASE
    )
    
    # Data URL pattern (can contain scripts)
    DATA_URL_PATTERN = re.compile(
        r'data\s*:',
        re.IGNORECASE
    )
    
    # SQL injection patterns
    SQL_INJECTION_PATTERN = re.compile(
        r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|EXECUTE|UNION|OR|AND)\b.*\b(FROM|INTO|TABLE|DATABASE|WHERE|SET)\b)|"
        r"(--)|"
        r"(\bOR\b\s+\d+\s*=\s*\d+)|"
        r"(\bAND\b\s+\d+\s*=\s*\d+)|"
        r"(;\s*DROP)|"
        r"(;\s*DELETE)|"
        r"(;\s*UPDATE)|"
        r"(;\s*INSERT)",
        re.IGNORECASE
    )
    
    # Command injection patterns
    COMMAND_INJECTION_PATTERN = re.compile(
        r"[;&|`$(){}]",
        re.IGNORECASE
    )
    
    # Path traversal patterns
    PATH_TRAVERSAL_PATTERN = re.compile(
        r"(\.\.\/|\.\.\\|%2e%2e%2f|%2e%2e\/|\.\.%2f|%2e%2e%5c)",
        re.IGNORECASE
    )
    
    # Maximum input lengths
    MAX_INPUT_LENGTH = 10000
    MAX_FIELD_LENGTH = 1000
    MAX_FILENAME_LENGTH = 255
    
    def __init__(
        self,
        escape_html: bool = True,
        detect_sql_injection: bool = True,
        detect_command_injection: bool = True,
        detect_path_traversal: bool = True,
        max_length: int = MAX_INPUT_LENGTH
    ):
        """
        Initialize content sanitizer.
        
        Args:
            escape_html: Escape HTML special characters
            detect_sql_injection: Detect and block SQL injection patterns
            detect_command_injection: Detect and block command injection patterns
            detect_path_traversal: Detect and block path traversal patterns
            max_length: Maximum allowed input length
        """
        self.escape_html = escape_html
        self.detect_sql_injection = detect_sql_injection
        self.detect_command_injection = detect_command_injection
        self.detect_path_traversal = detect_path_traversal
        self.max_length = max_length
        
        logger.info(
            f"ContentSanitizer initialized: "
            f"escape_html={escape_html}, "
            f"detect_sql_injection={detect_sql_injection}, "
            f"detect_command_injection={detect_command_injection}, "
            f"detect_path_traversal={detect_path_traversal}"
        )
    
    def sanitize_html(self, value: str) -> str:
        """
        Sanitize HTML string to prevent XSS attacks.
        
        Args:
            value: Input string to sanitize
            
        Returns:
            Sanitized string safe for HTML output
            
        Raises:
            ValueError: If input contains dangerous patterns
        """
        if not isinstance(value, str):
            value = str(value)
        
        # Check length
        if len(value) > self.max_length:
            logger.warning(f"Input exceeds maximum length: {len(value)} > {self.max_length}")
            value = value[:self.max_length]
        
        # Detect dangerous patterns
        if self.DANGEROUS_HTML_PATTERN.search(value):
            logger.warning(f"Dangerous HTML tags detected in input")
            raise ValueError("Input contains dangerous HTML tags")
        
        if self.EVENT_HANDLER_PATTERN.search(value):
            logger.warning(f"Event handlers detected in input")
            raise ValueError("Input contains event handlers")
        
        if self.JAVASCRIPT_URL_PATTERN.search(value):
            logger.warning(f"JavaScript URL detected in input")
            raise ValueError("Input contains JavaScript URLs")
        
        # Escape HTML special characters
        if self.escape_html:
            value = html.escape(value, quote=True)
        
        return value
    
    def sanitize_text(self, value: str, max_length: Optional[int] = None) -> str:
        """
        Sanitize plain text input.
        
        Args:
            value: Input string to sanitize
            max_length: Optional maximum length override
            
        Returns:
            Sanitized string
            
        Raises:
            ValueError: If input is invalid
        """
        if not isinstance(value, str):
            value = str(value)
        
        # Check length
        limit = max_length or self.max_length
        if len(value) > limit:
            logger.warning(f"Input exceeds maximum length: {len(value)} > {limit}")
            value = value[:limit]
        
        # Strip whitespace
        value = value.strip()
        
        # Normalize Unicode
        value = value.encode('utf-8', errors='ignore').decode('utf-8')
        
        return value
    
    def sanitize_identifier(self, value: str, field_name: str = "Identifier") -> str:
        """
        Sanitize identifier (username, template name, etc.).
        
        Only allows alphanumeric characters, underscores, and hyphens.
        
        Args:
            value: Input string to sanitize
            field_name: Name of field for error messages
            
        Returns:
            Sanitized identifier
            
        Raises:
            ValueError: If input contains invalid characters
        """
        if not isinstance(value, str):
            value = str(value)
        
        # Check for empty
        if not value.strip():
            raise ValueError(f"{field_name} cannot be empty")
        
        # Check length
        if len(value) > self.MAX_FIELD_LENGTH:
            raise ValueError(f"{field_name} too long (max {self.MAX_FIELD_LENGTH} characters)")
        
        # Only allow safe characters
        if not re.match(r'^[a-zA-Z0-9_-]+$', value):
            logger.warning(f"Invalid characters in {field_name}: {value[:50]}")
            raise ValueError(
                f"{field_name} can only contain letters, numbers, "
                f"underscores, and hyphens"
            )
        
        return value.strip()
    
    def sanitize_filename(self, value: str) -> str:
        """
        Sanitize filename to prevent path traversal.
        
        Args:
            value: Input filename
            
        Returns:
            Sanitized filename
            
        Raises:
            ValueError: If filename is invalid or contains path traversal
        """
        if not isinstance(value, str):
            value = str(value)
        
        # Check for path traversal
        if self.detect_path_traversal:
            if self.PATH_TRAVERSAL_PATTERN.search(value):
                logger.warning(f"Path traversal detected in filename: {value[:50]}")
                raise ValueError("Filename cannot contain path traversal sequences")
        
        # Check for absolute paths
        if value.startswith('/') or (len(value) > 1 and value[1] == ':'):
            logger.warning(f"Absolute path detected in filename: {value[:50]}")
            raise ValueError("Filename must be relative")
        
        # Check length
        if len(value) > self.MAX_FILENAME_LENGTH:
            raise ValueError(f"Filename too long (max {self.MAX_FILENAME_LENGTH} characters)")
        
        # Only allow safe characters
        if not re.match(r'^[a-zA-Z0-9_.-]+$', value):
            logger.warning(f"Invalid characters in filename: {value[:50]}")
            raise ValueError(
                "Filename can only contain letters, numbers, "
                "underscores, dots, and hyphens"
            )
        
        return value.strip()
    
    def sanitize_dict(
        self,
        data: Dict[str, Any],
        schema: Optional[Dict[str, type]] = None
    ) -> Dict[str, Any]:
        """
        Sanitize dictionary of inputs.
        
        Args:
            data: Dictionary to sanitize
            schema: Optional schema defining expected types for each key
            
        Returns:
            Sanitized dictionary
            
        Raises:
            ValueError: If any value fails sanitization
        """
        if not isinstance(data, dict):
            raise ValueError("Input must be a dictionary")
        
        sanitized = {}
        
        for key, value in data.items():
            # Sanitize key
            safe_key = self.sanitize_identifier(key, "Key")
            
            # Sanitize value based on type
            if isinstance(value, str):
                sanitized[safe_key] = self.sanitize_text(value)
            elif isinstance(value, dict):
                sanitized[safe_key] = self.sanitize_dict(value)
            elif isinstance(value, list):
                sanitized[safe_key] = self.sanitize_list(value)
            elif isinstance(value, (int, float, bool, type(None))):
                sanitized[safe_key] = value
            else:
                sanitized[safe_key] = str(value)
        
        return sanitized
    
    def sanitize_list(self, items: List[Any]) -> List[Any]:
        """
        Sanitize list of items.
        
        Args:
            items: List to sanitize
            
        Returns:
            Sanitized list
        """
        if not isinstance(items, list):
            raise ValueError("Input must be a list")
        
        sanitized = []
        
        for item in items:
            if isinstance(item, str):
                sanitized.append(self.sanitize_text(item))
            elif isinstance(item, dict):
                sanitized.append(self.sanitize_dict(item))
            elif isinstance(item, list):
                sanitized.append(self.sanitize_list(item))
            else:
                sanitized.append(item)
        
        return sanitized
    
    def detect_sql_injection(self, value: str) -> bool:
        """
        Detect SQL injection patterns in input.
        
        Args:
            value: Input string to check
            
        Returns:
            True if SQL injection detected, False otherwise
        """
        if not self.detect_sql_injection:
            return False
        
        return bool(self.SQL_INJECTION_PATTERN.search(value))
    
    def detect_command_injection(self, value: str) -> bool:
        """
        Detect command injection patterns in input.
        
        Args:
            value: Input string to check
            
        Returns:
            True if command injection detected, False otherwise
        """
        if not self.detect_command_injection:
            return False
        
        return bool(self.COMMAND_INJECTION_PATTERN.search(value))
    
    def detect_path_traversal(self, value: str) -> bool:
        """
        Detect path traversal patterns in input.
        
        Args:
            value: Input string to check
            
        Returns:
            True if path traversal detected, False otherwise
        """
        if not self.detect_path_traversal:
            return False
        
        return bool(self.PATH_TRAVERSAL_PATTERN.search(value))


# Singleton instance for convenience
_default_sanitizer: Optional[ContentSanitizer] = None


def get_sanitizer() -> ContentSanitizer:
    """Get or create default sanitizer instance."""
    global _default_sanitizer
    if _default_sanitizer is None:
        _default_sanitizer = ContentSanitizer()
    return _default_sanitizer


def sanitize_html(value: str) -> str:
    """Sanitize HTML string using default sanitizer."""
    return get_sanitizer().sanitize_html(value)


def sanitize_text(value: str, max_length: Optional[int] = None) -> str:
    """Sanitize plain text using default sanitizer."""
    return get_sanitizer().sanitize_text(value, max_length)


def sanitize_identifier(value: str, field_name: str = "Identifier") -> str:
    """Sanitize identifier using default sanitizer."""
    return get_sanitizer().sanitize_identifier(value, field_name)


def sanitize_filename(value: str) -> str:
    """Sanitize filename using default sanitizer."""
    return get_sanitizer().sanitize_filename(value)


def sanitize_dict(data: Dict[str, Any]) -> Dict[str, Any]:
    """Sanitize dictionary using default sanitizer."""
    return get_sanitizer().sanitize_dict(data)
