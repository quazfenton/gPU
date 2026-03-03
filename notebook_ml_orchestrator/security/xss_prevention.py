"""
XSS Prevention and Content Sanitization utilities.

This module provides protection against Cross-Site Scripting (XSS) attacks through:
- HTML escaping
- Content sanitization with allowlists
- CSP header generation
- Input validation
- Safe JSON serialization

Security Features:
- Defense in depth with multiple sanitization layers
- Allowlist-based approach (block unknown by default)
- Context-aware escaping
- Protection against DOM XSS
- Safe handling of user-generated content
"""

import html
import json
import re
import threading
from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


class XSSPreventionError(Exception):
    """Exception raised for XSS prevention errors."""
    
    def __init__(self, message: str, is_malicious: bool = False):
        super().__init__(message)
        self.is_malicious = is_malicious


@dataclass
class SanitizationResult:
    """Result of content sanitization."""
    content: str
    is_safe: bool
    removed_elements: List[str]
    removed_attributes: List[str]
    warnings: List[str]


# HTML tag allowlist for different contexts
ALLOWED_TAGS = {
    'basic': {'p', 'br', 'strong', 'em', 'u', 's', 'code', 'pre'},
    'formatting': {'p', 'br', 'strong', 'em', 'u', 's', 'code', 'pre', 
                   'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'blockquote', 'hr'},
    'links': {'p', 'br', 'strong', 'em', 'u', 's', 'code', 'pre', 'a'},
    'images': {'p', 'br', 'strong', 'em', 'u', 's', 'code', 'pre', 'img'},
    'tables': {'p', 'br', 'strong', 'em', 'u', 's', 'code', 'pre', 
               'table', 'thead', 'tbody', 'tr', 'th', 'td'},
    'lists': {'p', 'br', 'strong', 'em', 'u', 's', 'code', 'pre',
              'ul', 'ol', 'li'},
    'all': {'p', 'br', 'strong', 'em', 'u', 's', 'code', 'pre',
            'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'blockquote', 'hr',
            'a', 'img', 'table', 'thead', 'tbody', 'tr', 'th', 'td',
            'ul', 'ol', 'li', 'div', 'span'}
}

# HTML attribute allowlist
ALLOWED_ATTRIBUTES = {
    'a': {'href', 'title', 'rel', 'target'},
    'img': {'src', 'alt', 'title', 'width', 'height'},
    'td': {'colspan', 'rowspan'},
    'th': {'colspan', 'rowspan'},
    '*': {'class', 'id', 'style'}  # Global attributes
}

# Dangerous URL schemes
DANGEROUS_URL_SCHEMES = {
    'javascript', 'vbscript', 'data', 'blob'
}

# Dangerous HTML patterns
DANGEROUS_PATTERNS = [
    # Script tags
    (r'<\s*script[^>]*>.*?<\s*/\s*script\s*>', 'script tag'),
    (r'<\s*script[^>]*>', 'script open tag'),
    
    # Event handlers
    (r'\bon\w+\s*=\s*["\'][^"\']*["\']', 'event handler'),
    (r'\bon\w+\s*=\s*[^\s>]+', 'event handler'),
    
    # JavaScript URLs
    (r'javascript\s*:', 'javascript URL'),
    (r'vbscript\s*:', 'vbscript URL'),
    (r'data\s*:', 'data URL'),
    
    # Expression/behavior (IE)
    (r'expression\s*\(', 'CSS expression'),
    (r'behavior\s*:', 'CSS behavior'),
    
    # Embedded objects
    (r'<\s*embed[^>]*>', 'embed tag'),
    (r'<\s*object[^>]*>', 'object tag'),
    (r'<\s*iframe[^>]*>', 'iframe tag'),
    (r'<\s*frame[^>]*>', 'frame tag'),
    (r'<\s*applet[^>]*>', 'applet tag'),
    (r'<\s*meta[^>]*>', 'meta tag'),
    (r'<\s*link[^>]*>', 'link tag'),
    (r'<\s*style[^>]*>', 'style tag'),
    (r'<\s*form[^>]*>', 'form tag'),
    (r'<\s*input[^>]*>', 'input tag'),
    (r'<\s*button[^>]*>', 'button tag'),
    (r'<\s*select[^>]*>', 'select tag'),
    (r'<\s*textarea[^>]*>', 'textarea tag'),
]


class ContentSanitizer:
    """
    HTML content sanitizer for XSS prevention.
    
    This class provides methods for sanitizing HTML content using an
    allowlist-based approach to prevent XSS attacks.
    
    Example:
        sanitizer = ContentSanitizer()
        
        # Sanitize HTML content
        result = sanitizer.sanitize_html(user_input, allowed_tags='basic')
        
        # Escape HTML
        escaped = sanitizer.escape_html(user_input)
        
        # Validate URL
        is_safe = sanitizer.is_safe_url(user_url)
    """
    
    def __init__(
        self,
        allowed_tags: Set[str] = None,
        allowed_attributes: Dict[str, Set[str]] = None,
        remove_comments: bool = True,
        remove_styles: bool = True
    ):
        """
        Initialize content sanitizer.
        
        Args:
            allowed_tags: Set of allowed HTML tags (None for default 'basic')
            allowed_attributes: Dict mapping tags to allowed attributes
            remove_comments: Whether to remove HTML comments
            remove_styles: Whether to remove style attributes
        """
        self._lock = threading.RLock()
        self.allowed_tags = allowed_tags or ALLOWED_TAGS['basic']
        self.allowed_attributes = allowed_attributes or ALLOWED_ATTRIBUTES
        self.remove_comments = remove_comments
        self.remove_styles = remove_styles
        
        # Compile regex patterns
        self._dangerous_patterns = [
            (re.compile(pattern, re.IGNORECASE | re.DOTALL), name)
            for pattern, name in DANGEROUS_PATTERNS
        ]
        
        logger.info(f"ContentSanitizer initialized with tags: {self.allowed_tags}")
    
    def escape_html(self, content: str) -> str:
        """
        Escape HTML special characters.
        
        This is the safest approach - escape all HTML characters.
        
        Args:
            content: Content to escape
            
        Returns:
            Escaped content
        """
        if not isinstance(content, str):
            content = str(content)
        
        return html.escape(content, quote=True)
    
    def sanitize_html(
        self,
        content: str,
        allowed_tags: Optional[Set[str]] = None
    ) -> SanitizationResult:
        """
        Sanitize HTML content by removing dangerous elements.
    
        Args:
            content: HTML content to sanitize
            allowed_tags: Override allowed tags set (may be a set of tag names
                          or a string key like 'basic' referring to ALLOWED_TAGS)
        
        Returns:
            SanitizationResult with sanitized content and metadata
        """
        if not isinstance(content, str):
            content = str(content)
    
        with self._lock:
            # Normalize allowed_tags so both set-of-tags and named profiles work
            if isinstance(allowed_tags, str):
                tags = ALLOWED_TAGS.get(allowed_tags, self.allowed_tags)
            elif allowed_tags is not None:
                tags = allowed_tags
            else:
                tags = self.allowed_tags
        
            removed_elements = []
            removed_attributes = []
            warnings = []
        
            # Check for dangerous patterns
            for pattern, name in self._dangerous_patterns:
                matches = pattern.findall(content)
                if matches:
                    warnings.append(f"Detected {name}: {len(matches)} occurrences")
                    content = pattern.sub('', content)
                    removed_elements.append(name)
        
            # Remove HTML comments
            if self.remove_comments:
                comment_pattern = re.compile(r'<!--.*?-->', re.DOTALL)
                if comment_pattern.search(content):
                    warnings.append("Removed HTML comments")
                content = comment_pattern.sub('', content)
        
            # Remove style attributes if configured
            if self.remove_styles:
                style_pattern = re.compile(r'\s*style\s*=\s*["\'][^"\']*["\']', re.IGNORECASE)
                if style_pattern.search(content):
                    warnings.append("Removed style attributes")
                content = style_pattern.sub('', content)
        
            # Remove disallowed tags
            tag_pattern = re.compile(r'<\s*(/\s*)?(\w+)([^>]*)>', re.IGNORECASE)
        
            def replace_tag(match):
                closing = match.group(1) or ''
                tag_name = match.group(2).lower()
                attributes = match.group(3)
            
                if tag_name not in tags:
                    removed_elements.append(f"<{tag_name}>")
                    return ''  # Remove tag but keep content
            
                # Sanitize attributes
                if attributes:
                    sanitized_attrs = self._sanitize_attributes(tag_name, attributes)
                    removed_attributes.extend(sanitized_attrs['removed'])
                    attributes = sanitized_attrs['attributes']
            
                return f'<{closing}{tag_name}{attributes}>'
        
            content = tag_pattern.sub(replace_tag, content)
        
            is_safe = len(removed_elements) == 0 and len(warnings) == 0
        
            return SanitizationResult(
                content=content,
                is_safe=is_safe,
                removed_elements=removed_elements,
                removed_attributes=removed_attributes,
                warnings=warnings
            )
            removed_attributes = []
            warnings = []
            
            # Check for dangerous patterns
            for pattern, name in self._dangerous_patterns:
                matches = pattern.findall(content)
                if matches:
                    warnings.append(f"Detected {name}: {len(matches)} occurrences")
                    content = pattern.sub('', content)
                    removed_elements.append(name)
            
            # Remove HTML comments
            if self.remove_comments:
                comment_pattern = re.compile(r'<!--.*?-->', re.DOTALL)
                if comment_pattern.search(content):
                    warnings.append("Removed HTML comments")
                content = comment_pattern.sub('', content)
            
            # Remove style attributes if configured
            if self.remove_styles:
                style_pattern = re.compile(r'\s*style\s*=\s*["\'][^"\']*["\']', re.IGNORECASE)
                if style_pattern.search(content):
                    warnings.append("Removed style attributes")
                content = style_pattern.sub('', content)
            
            # Remove disallowed tags
            tag_pattern = re.compile(r'<\s*(/\s*)?(\w+)([^>]*)>', re.IGNORECASE)
            
            def replace_tag(match):
                closing = match.group(1) or ''
                tag_name = match.group(2).lower()
                attributes = match.group(3)
                
                if tag_name not in tags:
                    removed_elements.append(f"<{tag_name}>")
                    return ''  # Remove tag but keep content
                
                # Sanitize attributes
                if attributes:
                    sanitized_attrs = self._sanitize_attributes(tag_name, attributes)
                    removed_attributes.extend(sanitized_attrs['removed'])
                    attributes = sanitized_attrs['attributes']
                
                return f'<{closing}{tag_name}{attributes}>'
            
            content = tag_pattern.sub(replace_tag, content)
            
            is_safe = len(removed_elements) == 0 and len(warnings) == 0
            
            return SanitizationResult(
                content=content,
                is_safe=is_safe,
                removed_elements=removed_elements,
                removed_attributes=removed_attributes,
                warnings=warnings
            )
    
    def _sanitize_attributes(
        self,
        tag_name: str,
        attributes: str
    ) -> Dict[str, Any]:
        """
        Sanitize HTML attributes.
        
        Args:
            tag_name: HTML tag name
            attributes: Attribute string
            
        decoded = re.sub(r'[\x00-\x20]+', '', html.unescape(url))
            Dictionary with sanitized attributes and removed list
        """
        removed = []
        allowed = self.allowed_attributes.get(tag_name, set())
        global_allowed = self.allowed_attributes.get('*', set())
        all_allowed = allowed | global_allowed
        
        # Parse attributes
        attr_pattern = re.compile(r'(\w+)\s*=\s*["\']([^"\']*)["\']', re.IGNORECASE)
        
        def replace_attr(match):
            attr_name = match.group(1).lower()
            attr_value = match.group(2)
            
            if attr_name not in all_allowed:
                removed.append(attr_name)
                return ''
            
            # Check URL attributes for dangerous schemes
            if attr_name in {'href', 'src', 'action'}:
                if not self.is_safe_url(attr_value):
                    removed.append(f"{attr_name}={attr_value[:50]}")
                    return ''
            
            return f' {attr_name}="{html.escape(attr_value, quote=True)}"'
        
        sanitized = attr_pattern.sub(replace_attr, attributes)
        
        # Also remove bare attributes (without values)
        bare_attr_pattern = re.compile(r'\s+(\w+)(?=\s|>|/>)', re.IGNORECASE)
        sanitized = bare_attr_pattern.sub(
            lambda m: f' {m.group(1)}' if m.group(1).lower() in all_allowed else '',
            sanitized
        )
        
        return {
            'attributes': sanitized,
            'removed': removed
        }
    
    def is_safe_url(self, url: str) -> bool:
        """
        Check if URL is safe (no dangerous schemes).
        
        Args:
            url: URL to check
            
        Returns:
            True if safe, False otherwise
        """
        if not isinstance(url, str):
            url = str(url)
        
        url = url.strip().lower()
        
        # Check for dangerous schemes
        for scheme in DANGEROUS_URL_SCHEMES:
            if url.startswith(f'{scheme}:'):
                return False
            if url.startswith(f'{scheme} '):
                return False
        
        # Check for encoded dangerous schemes
        decoded = html.unescape(url)
        for scheme in DANGEROUS_URL_SCHEMES:
            if decoded.startswith(f'{scheme}:'):
                return False
        
        return True
    
    def sanitize_text(self, content: str) -> str:
        """
        Sanitize plain text (escape all HTML).
        
        Args:
            content: Text content to sanitize
            
        Returns:
            Sanitized text
        """
        return self.escape_html(content)
    
    def sanitize_json_value(self, value: Any) -> Any:
        """
        Sanitize value for safe JSON serialization.
        
        Args:
            value: Value to sanitize
            
        Returns:
            Sanitized value
        """
        if isinstance(value, str):
            # Escape HTML in strings
            return self.escape_html(value)
        elif isinstance(value, dict):
            return {k: self.sanitize_json_value(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [self.sanitize_json_value(item) for item in value]
        else:
            return value
    
    def detect_xss_attempt(self, content: str) -> Tuple[bool, List[str]]:
        """
        Detect potential XSS attack patterns.
        
        Args:
            content: Content to analyze
            
        Returns:
            Tuple of (is_malicious, detected_patterns)
        """
        if not isinstance(content, str):
            content = str(content)
        
        detected = []
        
        for pattern, name in self._dangerous_patterns:
            if pattern.search(content):
                detected.append(name)
        
        # Check for encoded attacks
        try:
            decoded = html.unescape(content)
            for pattern, name in self._dangerous_patterns:
                if pattern.search(decoded) and name not in detected:
                    detected.append(f"{name} (encoded)")
        except Exception:
            pass
        
        # Check for Unicode escapes
        unicode_pattern = re.compile(r'\\u[0-9a-fA-F]{4}')
        if unicode_pattern.search(content):
            try:
                decoded_unicode = content.encode().decode('unicode_escape')
                for pattern, name in self._dangerous_patterns:
                    if pattern.search(decoded_unicode) and name not in detected:
                        detected.append(f"{name} (unicode encoded)")
            except Exception:
                pass
        
        return len(detected) > 0, detected


class CSPHeaderGenerator:
    """
    Content-Security-Policy header generator.
    
    This class generates CSP headers for XSS prevention.
    
    Example:
        csp = CSPHeaderGenerator()
        
        # Generate strict CSP
        header = csp.generate_strict_csp()
        
        # Generate custom CSP
        header = csp.generate_csp(
            default_src="'self'",
            script_src="'self' https://cdn.example.com",
            style_src="'self' 'unsafe-inline'"
        )
    """
    
    def __init__(self):
        """Initialize CSP header generator."""
        self.default_policy = {
            'default-src': "'self'",
            'script-src': "'self'",
            'style-src': "'self'",
            'img-src': "'self' data: https:",
            'font-src': "'self'",
            'connect-src': "'self'",
            'frame-src': "'none'",
            'object-src': "'none'",
            'base-uri': "'self'",
            'form-action': "'self'",
            'frame-ancestors': "'none'",
            'upgrade-insecure-requests': None
        }
    
    def generate_csp(self, **policies) -> str:
        """
        Generate CSP header string.
        
        Args:
            **policies: CSP directives
            
        Returns:
            CSP header string
        """
        parts = []
        
        for directive, value in policies.items():
            if value is not None:
                parts.append(f"{directive} {value}")
            else:
                parts.append(directive)
        
        return '; '.join(parts)
    
    def generate_strict_csp(self) -> str:
        """
        Generate strict CSP for maximum security.
        
        Returns:
            Strict CSP header
        """
        return self.generate_csp(**self.default_policy)
    
    def generate_development_csp(self) -> str:
        """
        Generate CSP for development (less restrictive).
        
        Returns:
            Development CSP header
        """
        dev_policy = {
            'default-src': "'self' 'unsafe-inline' 'unsafe-eval'",
            'script-src': "'self' 'unsafe-inline' 'unsafe-eval'",
            'style-src': "'self' 'unsafe-inline'",
            'img-src': "'self' data: https: http:",
            'connect-src': "'self' http: https: ws: wss:"
        }
        return self.generate_csp(**dev_policy)
    
    def generate_gradio_csp(self) -> str:
        """
        Generate CSP compatible with Gradio applications.
        
        Returns:
            Gradio-compatible CSP header
        """
        gradio_policy = {
            'default-src': "'self'",
            'script-src': "'self' 'unsafe-eval' blob:",
            'style-src': "'self' 'unsafe-inline'",
            'img-src': "'self' data: blob: https:",
            'font-src': "'self' data:",
            'connect-src': "'self' blob: https: wss:",
            'worker-src': "'self' blob:",
            'child-src': "'self' blob:",
            'frame-src': "'self' blob:",
            'object-src': "'none'",
            'base-uri': "'self'",
            'form-action': "'self'"
        }
        return self.generate_csp(**gradio_policy)
    
    def get_headers_dict(self, csp: str = None) -> Dict[str, str]:
        """
        Get all security headers as dictionary.
        
        Args:
            csp: CSP header value (uses strict CSP if not provided)
            
        Returns:
            Dictionary of security headers
        """
        headers = {
            'Content-Security-Policy': csp or self.generate_strict_csp(),
            'X-Content-Type-Options': 'nosniff',
            'X-Frame-Options': 'DENY',
            'X-XSS-Protection': '1; mode=block',
            'Referrer-Policy': 'strict-origin-when-cross-origin',
            'Permissions-Policy': 'geolocation=(), microphone=(), camera=()'
        }
        return headers


# Module-level sanitizer instance
_sanitizer: Optional[ContentSanitizer] = None
_csp_generator: Optional[CSPHeaderGenerator] = None


def get_sanitizer() -> ContentSanitizer:
    """Get or create module-level sanitizer."""
    global _sanitizer
    if _sanitizer is None:
        _sanitizer = ContentSanitizer()
    return _sanitizer


def get_csp_generator() -> CSPHeaderGenerator:
    """Get or create module-level CSP generator."""
    global _csp_generator
    if _csp_generator is None:
        _csp_generator = CSPHeaderGenerator()
    return _csp_generator


# Convenience functions
def escape_html(content: str) -> str:
    """Escape HTML special characters."""
    return get_sanitizer().escape_html(content)


def sanitize_html(content: str, **kwargs) -> str:
    """Sanitize HTML content."""
    result = get_sanitizer().sanitize_html(content, **kwargs)
    return result.content


def detect_xss(content: str) -> Tuple[bool, List[str]]:
    """Detect XSS attempt in content."""
    return get_sanitizer().detect_xss_attempt(content)


def is_safe_url(url: str) -> bool:
    """Check if URL is safe."""
    return get_sanitizer().is_safe_url(url)


def generate_csp_header(**kwargs) -> str:
    """Generate CSP header."""
    return get_csp_generator().generate_csp(**kwargs)


def get_security_headers() -> Dict[str, str]:
    """Get all security headers."""
    return get_csp_generator().get_headers_dict()
