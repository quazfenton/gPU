"""
Security Logger for audit and monitoring.

This module provides comprehensive security event logging for:
- Authentication attempts
- Authorization failures
- Rate limit violations
- Input validation failures
- Credential access
- Suspicious activity detection

Features:
- Structured JSON logging for SIEM integration
- Configurable log levels per event type
- Log rotation and retention
- Alert generation for critical events
- IP address and user tracking
"""

import json
import logging
import os
import threading
from datetime import datetime
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional
import hashlib
from collections import defaultdict
import requests


class SecurityEventType(Enum):
    """Security event type enumeration."""
    # Authentication events
    AUTH_SUCCESS = "auth.success"
    AUTH_FAILURE = "auth.failure"
    AUTH_LOCKOUT = "auth.lockout"
    TOKEN_GENERATED = "token.generated"
    TOKEN_REVOKED = "token.revoked"
    TOKEN_EXPIRED = "token.expired"
    
    # Authorization events
    AUTHZ_SUCCESS = "authz.success"
    AUTHZ_FAILURE = "authz.failure"
    PERMISSION_DENIED = "permission.denied"
    ROLE_CHANGED = "role.changed"
    
    # Rate limiting events
    RATE_LIMIT_EXCEEDED = "rate_limit.exceeded"
    RATE_LIMIT_WARNING = "rate_limit.warning"
    
    # Input validation events
    VALIDATION_FAILURE = "validation.failure"
    SQL_INJECTION_ATTEMPT = "security.sql_injection"
    XSS_ATTEMPT = "security.xss"
    PATH_TRAVERSAL_ATTEMPT = "security.path_traversal"
    
    # Credential events
    CREDENTIAL_ACCESS = "credential.access"
    CREDENTIAL_CREATED = "credential.created"
    CREDENTIAL_UPDATED = "credential.updated"
    CREDENTIAL_DELETED = "credential.deleted"
    CREDENTIAL_ROTATED = "credential.rotated"
    
    # Session events
    SESSION_CREATED = "session.created"
    SESSION_INVALIDATED = "session.invalidated"
    SESSION_HIJACK_ATTEMPT = "session.hijack_attempt"
    
    # System events
    SECURITY_CONFIG_CHANGED = "security.config_changed"
    BACKEND_UNHEALTHY = "backend.unhealthy"
    SERVICE_STARTED = "service.started"
    SERVICE_STOPPED = "service.stopped"


class SeverityLevel(Enum):
    """Event severity level."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class SecurityEvent:
    """Security event data structure."""
    event_type: str
    timestamp: str
    severity: str
    message: str
    user_id: Optional[str] = None
    username: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    resource: Optional[str] = None
    action: Optional[str] = None
    status_code: Optional[int] = None
    details: Dict[str, Any] = field(default_factory=dict)
    event_id: str = field(default_factory=lambda: hashlib.sha256(
        f"{datetime.now().isoformat()}{os.urandom(16).hex()}".encode()
    ).hexdigest()[:16])
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert to JSON string for logging."""
        return json.dumps(self.to_dict(), indent=2)


class SecurityLogger:
    """
    Security event logger for audit and monitoring.
    
    This class provides methods for logging security events with
    structured output suitable for SIEM integration.
    
    Example:
        logger = SecurityLogger(log_file='security.log')
        
        # Log authentication success
        logger.log_auth_success('user123', ip_address='192.168.1.1')
        
        # Log authorization failure
        logger.log_authz_failure('user123', 'admin_panel', 'VIEWER')
        
        # Log rate limit exceeded
        logger.log_rate_limit_exceeded('user123', '/api/jobs', 100)
        
        # Log suspicious activity
        logger.log_suspicious_activity('user123', 'Multiple failed logins', 
                                        {'attempts': 5})
    """
    
    # Event type to severity mapping
    SEVERITY_MAP = {
        SecurityEventType.AUTH_SUCCESS: SeverityLevel.INFO,
        SecurityEventType.AUTH_FAILURE: SeverityLevel.WARNING,
        SecurityEventType.AUTH_LOCKOUT: SeverityLevel.ERROR,
        SecurityEventType.TOKEN_GENERATED: SeverityLevel.INFO,
        SecurityEventType.TOKEN_REVOKED: SeverityLevel.INFO,
        SecurityEventType.TOKEN_EXPIRED: SeverityLevel.DEBUG,
        SecurityEventType.AUTHZ_SUCCESS: SeverityLevel.DEBUG,
        SecurityEventType.AUTHZ_FAILURE: SeverityLevel.WARNING,
        SecurityEventType.PERMISSION_DENIED: SeverityLevel.WARNING,
        SecurityEventType.ROLE_CHANGED: SeverityLevel.INFO,
        SecurityEventType.RATE_LIMIT_EXCEEDED: SeverityLevel.WARNING,
        SecurityEventType.RATE_LIMIT_WARNING: SeverityLevel.DEBUG,
        SecurityEventType.VALIDATION_FAILURE: SeverityLevel.DEBUG,
        SecurityEventType.SQL_INJECTION_ATTEMPT: SeverityLevel.CRITICAL,
        SecurityEventType.XSS_ATTEMPT: SeverityLevel.CRITICAL,
        SecurityEventType.PATH_TRAVERSAL_ATTEMPT: SeverityLevel.ERROR,
        SecurityEventType.CREDENTIAL_ACCESS: SeverityLevel.INFO,
        SecurityEventType.CREDENTIAL_CREATED: SeverityLevel.INFO,
        SecurityEventType.CREDENTIAL_UPDATED: SeverityLevel.INFO,
        SecurityEventType.CREDENTIAL_DELETED: SeverityLevel.INFO,
        SecurityEventType.CREDENTIAL_ROTATED: SeverityLevel.INFO,
        SecurityEventType.SESSION_CREATED: SeverityLevel.INFO,
        SecurityEventType.SESSION_INVALIDATED: SeverityLevel.INFO,
        SecurityEventType.SESSION_HIJACK_ATTEMPT: SeverityLevel.CRITICAL,
        SecurityEventType.SECURITY_CONFIG_CHANGED: SeverityLevel.WARNING,
        SecurityEventType.BACKEND_UNHEALTHY: SeverityLevel.ERROR,
        SecurityEventType.SERVICE_STARTED: SeverityLevel.INFO,
        SecurityEventType.SERVICE_STOPPED: SeverityLevel.INFO,
    }
    
    def __init__(
        self,
        log_file: Optional[str] = None,
        log_level: int = logging.INFO,
        include_console: bool = True,
        max_bytes: int = 10 * 1024 * 1024,  # 10 MB
        backup_count: int = 5,
        include_details: bool = True
    ):
        """
        Initialize security logger.
        
        Args:
            log_file: Path to log file (optional)
            log_level: Logging level
            include_console: Whether to log to console
            max_bytes: Max log file size before rotation
            backup_count: Number of backup files to keep
            include_details: Whether to include event details in logs
        """
        self._lock = threading.RLock()
        self.include_details = include_details
        self._event_handlers: Dict[SecurityEventType, List[callable]] = {}
        
        # Create logger
        self.logger = logging.getLogger('security')
        self.logger.setLevel(log_level)
        self.logger.handlers = []  # Clear existing handlers
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Console handler
        if include_console:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(log_level)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        
        # File handler with rotation
        if log_file:
            try:
                from logging.handlers import RotatingFileHandler
                file_handler = RotatingFileHandler(
                    log_file,
                    maxBytes=max_bytes,
                    backupCount=backup_count
                )
                file_handler.setLevel(log_level)
                file_handler.setFormatter(formatter)
                self.logger.addHandler(file_handler)
                self.logger.info(f"Security logging to file: {log_file}")
            except Exception as e:
                self.logger.warning(f"Failed to create file handler: {e}")

        # Event counters for statistics
        self._event_counts: Dict[str, int] = {}
        self._hourly_counts: Dict[str, Dict[str, int]] = {}

        self.logger.info("SecurityLogger initialized")
    
    def _get_severity(self, event_type: SecurityEventType) -> SeverityLevel:
        """Get severity level for event type."""
        return self.SEVERITY_MAP.get(event_type, SeverityLevel.INFO)
    
    def _log_event(self, event: SecurityEvent) -> None:
        """Log security event."""
        with self._lock:
            # Update counters
            hour_key = datetime.now().strftime('%Y-%m-%d-%H')
            
            if event.event_type not in self._event_counts:
                self._event_counts[event.event_type] = 0
            self._event_counts[event.event_type] += 1
            
            if hour_key not in self._hourly_counts:
                self._hourly_counts[hour_key] = {}
            if event.event_type not in self._hourly_counts[hour_key]:
                self._hourly_counts[hour_key][event.event_type] = 0
            self._hourly_counts[hour_key][event.event_type] += 1
            
            # Get log level
            severity = self._get_severity(SecurityEventType(event.event_type))
            log_level = getattr(logging, severity.value.upper())
            
            # Create log message
            if self.include_details and event.details:
                message = (
                    f"[{event.event_type}] {event.message} | "
                    f"user={event.username or event.user_id or 'unknown'} | "
                    f"ip={event.ip_address or 'unknown'} | "
                    f"details={json.dumps(event.details)}"
                )
            else:
                message = (
                    f"[{event.event_type}] {event.message} | "
                    f"user={event.username or event.user_id or 'unknown'} | "
                    f"ip={event.ip_address or 'unknown'}"
                )
            
            # Log the event
            self.logger.log(log_level, message)
            
            # Call event handlers
            handlers = self._event_handlers.get(SecurityEventType(event.event_type), [])
            for handler in handlers:
                try:
                    handler(event)
                except Exception as e:
                    self.logger.error(f"Event handler failed: {e}")
            
            # Check for alert conditions
            self._check_alerts(event)

    def _check_alerts(self, event: SecurityEvent) -> None:
        """Check if event should trigger an alert."""
        severity = self._get_severity(SecurityEventType(event.event_type))

        # Alert on critical events
        if severity == SeverityLevel.CRITICAL:
            self.logger.critical(
                f"[ALERT] CRITICAL SECURITY EVENT: {event.event_type} - {event.message}"
            )
            # Send webhook alerts
            self._send_webhook_alerts(event)

    def register_handler(
        self,
        event_type: SecurityEventType,
        handler: callable
    ) -> None:
        """
        Register event handler for specific event type.
        
        Args:
            event_type: Event type to handle
            handler: Handler function
        """
        with self._lock:
            if event_type not in self._event_handlers:
                self._event_handlers[event_type] = []
            self._event_handlers[event_type].append(handler)
            logger.info(f"Registered handler for {event_type.value}")
    
    def log_event(
        self,
        event_type: SecurityEventType,
        message: str,
        user_id: Optional[str] = None,
        username: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        resource: Optional[str] = None,
        action: Optional[str] = None,
        status_code: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None
    ) -> SecurityEvent:
        """
        Log a security event.
        
        Args:
            event_type: Type of event
            message: Event message
            user_id: User ID
            username: Username
            ip_address: IP address
            user_agent: User agent
            resource: Resource involved
            action: Action performed
            status_code: HTTP status code
            details: Additional details
            
        Returns:
            Created security event
        """
        event = SecurityEvent(
            event_type=event_type.value,
            timestamp=datetime.now().isoformat(),
            severity=self._get_severity(event_type).value,
            message=message,
            user_id=user_id,
            username=username,
            ip_address=ip_address,
            user_agent=user_agent,
            resource=resource,
            action=action,
            status_code=status_code,
            details=details or {}
        )
        
        self._log_event(event)
        return event
    
    # Authentication events
    def log_auth_success(
        self,
        username: str,
        ip_address: Optional[str] = None,
        method: str = 'password',
        details: Optional[Dict[str, Any]] = None
    ) -> SecurityEvent:
        """Log successful authentication."""
        return self.log_event(
            SecurityEventType.AUTH_SUCCESS,
            f"Successful authentication for user '{username}'",
            username=username,
            ip_address=ip_address,
            action=f"login:{method}",
            details=details
        )
    
    def log_auth_failure(
        self,
        username: str,
        ip_address: Optional[str] = None,
        reason: str = 'invalid_credentials',
        details: Optional[Dict[str, Any]] = None
    ) -> SecurityEvent:
        """Log failed authentication."""
        return self.log_event(
            SecurityEventType.AUTH_FAILURE,
            f"Failed authentication for user '{username}': {reason}",
            username=username,
            ip_address=ip_address,
            action='login:failed',
            details={**details, 'reason': reason} if details else {'reason': reason}
        )
    
    def log_auth_lockout(
        self,
        username: str,
        ip_address: Optional[str] = None,
        failed_attempts: int = 0,
        lockout_duration_minutes: int = 30
    ) -> SecurityEvent:
        """Log account lockout."""
        return self.log_event(
            SecurityEventType.AUTH_LOCKOUT,
            f"Account locked for user '{username}' after {failed_attempts} failed attempts",
            username=username,
            ip_address=ip_address,
            details={
                'failed_attempts': failed_attempts,
                'lockout_duration_minutes': lockout_duration_minutes
            }
        )
    
    # Authorization events
    def log_authz_failure(
        self,
        username: str,
        resource: str,
        required_role: str,
        user_role: Optional[str] = None,
        ip_address: Optional[str] = None
    ) -> SecurityEvent:
        """Log authorization failure."""
        return self.log_event(
            SecurityEventType.AUTHZ_FAILURE,
            f"Authorization denied for user '{username}' accessing '{resource}'",
            username=username,
            ip_address=ip_address,
            resource=resource,
            action='access:denied',
            details={
                'required_role': required_role,
                'user_role': user_role
            }
        )
    
    # Rate limiting events
    def log_rate_limit_exceeded(
        self,
        username: str,
        endpoint: str,
        limit: int,
        ip_address: Optional[str] = None
    ) -> SecurityEvent:
        """Log rate limit exceeded."""
        return self.log_event(
            SecurityEventType.RATE_LIMIT_EXCEEDED,
            f"Rate limit exceeded for user '{username}' on endpoint '{endpoint}'",
            username=username,
            ip_address=ip_address,
            resource=endpoint,
            details={'limit': limit}
        )
    
    # Security threat events
    def log_sql_injection_attempt(
        self,
        input_value: str,
        ip_address: Optional[str] = None,
        username: Optional[str] = None
    ) -> SecurityEvent:
        """Log SQL injection attempt."""
        return self.log_event(
            SecurityEventType.SQL_INJECTION_ATTEMPT,
            f"SQL injection attempt detected",
            username=username,
            ip_address=ip_address,
            action='attack:sql_injection',
            details={'input_preview': input_value[:100] if input_value else None}
        )
    
    def log_xss_attempt(
        self,
        input_value: str,
        ip_address: Optional[str] = None,
        username: Optional[str] = None
    ) -> SecurityEvent:
        """Log XSS attempt."""
        return self.log_event(
            SecurityEventType.XSS_ATTEMPT,
            f"XSS attempt detected",
            username=username,
            ip_address=ip_address,
            action='attack:xss',
            details={'input_preview': input_value[:100] if input_value else None}
        )
    
    def log_path_traversal_attempt(
        self,
        path: str,
        ip_address: Optional[str] = None,
        username: Optional[str] = None
    ) -> SecurityEvent:
        """Log path traversal attempt."""
        return self.log_event(
            SecurityEventType.PATH_TRAVERSAL_ATTEMPT,
            f"Path traversal attempt detected: {path}",
            username=username,
            ip_address=ip_address,
            action='attack:path_traversal',
            details={'path': path}
        )
    
    # Credential events
    def log_credential_access(
        self,
        service: str,
        key: str,
        username: str,
        ip_address: Optional[str] = None
    ) -> SecurityEvent:
        """Log credential access."""
        return self.log_event(
            SecurityEventType.CREDENTIAL_ACCESS,
            f"Credential accessed: {service}:{key}",
            username=username,
            ip_address=ip_address,
            resource=f"{service}:{key}",
            action='credential:access'
        )
    
    def log_credential_rotated(
        self,
        service: str,
        key: str,
        username: str,
        ip_address: Optional[str] = None
    ) -> SecurityEvent:
        """Log credential rotation."""
        return self.log_event(
            SecurityEventType.CREDENTIAL_ROTATED,
            f"Credential rotated: {service}:{key}",
            username=username,
            ip_address=ip_address,
            resource=f"{service}:{key}",
            action='credential:rotate'
        )
    
    # Session events
    def log_session_hijack_attempt(
        self,
        session_id: str,
        original_ip: str,
        new_ip: str,
        username: Optional[str] = None
    ) -> SecurityEvent:
        """Log session hijack attempt."""
        return self.log_event(
            SecurityEventType.SESSION_HIJACK_ATTEMPT,
            f"Session hijack attempt detected for session {session_id}",
            username=username,
            ip_address=new_ip,
            resource=session_id,
            action='attack:session_hijack',
            details={
                'original_ip': original_ip,
                'new_ip': new_ip
            }
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get security event statistics.

        Returns:
            Dictionary with event statistics
        """
        with self._lock:
            return {
                'total_events': sum(self._event_counts.values()),
                'events_by_type': dict(self._event_counts),
                'hourly_counts': dict(self._hourly_counts),
                'handlers_registered': sum(
                    len(handlers) for handlers in self._event_handlers.values()
                )
            }

    # ==================== ENHANCEMENT METHODS ====================

    def export_events(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        event_types: Optional[List[str]] = None,
        output_format: str = 'json'
    ) -> str:
        """
        Export security events for external analysis.

        Args:
            start_time: Start time filter
            end_time: End time filter
            event_types: List of event types to include
            output_format: Output format ('json', 'csv', 'cef', 'leef')

        Returns:
            Formatted event data
        """
        # This would require storing events in memory or database
        # For now, return a placeholder
        return json.dumps({'status': 'export_not_implemented', 'note': 'Requires event storage'})

    def search_events(
        self,
        query: str,
        limit: int = 100,
        event_types: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search security events.

        Args:
            query: Search query
            limit: Maximum results
            event_types: Filter by event types

        Returns:
            List of matching events
        """
        # Placeholder - would require event storage
        return []

    def set_retention_policy(
        self,
        max_age_days: int = 90,
        max_events: int = 100000,
        archive_path: Optional[str] = None
    ) -> None:
        """
        Set log retention policy.

        Args:
            max_age_days: Maximum age of events to keep
            max_events: Maximum number of events to keep
            archive_path: Path to archive old events
        """
        self._retention_max_age_days = max_age_days
        self._retention_max_events = max_events
        self._retention_archive_path = archive_path
        logger.info(f"Retention policy set: {max_age_days} days, {max_events} events")

    def add_alert_webhook(self, url: str, event_types: Optional[List[SecurityEventType]] = None,
                          headers: Optional[Dict[str, str]] = None) -> None:
        """
        Add a webhook for real-time alerting.

        Args:
            url: Webhook URL
            event_types: Event types to trigger webhook (None = all critical events)
            headers: Optional HTTP headers
        """
        if not hasattr(self, '_alert_webhooks'):
            self._alert_webhooks: List[Dict[str, Any]] = []

        self._alert_webhooks.append({
            'url': url,
            'event_types': event_types,
            'headers': headers or {},
            'enabled': True
        })
        logger.info(f"Added alert webhook: {url}")

    def _send_webhook_alerts(self, event: SecurityEvent) -> None:
        """Send webhook alerts for critical events."""
        if not hasattr(self, '_alert_webhooks'):
            return

        severity = self._get_severity(SecurityEventType(event.event_type))
        if severity != SeverityLevel.CRITICAL:
            return

        for webhook in self._alert_webhooks:
            if not webhook['enabled']:
                continue

            # Check if event type matches
            if webhook['event_types']:
                try:
                    event_type = SecurityEventType(event.event_type)
                    if event_type not in webhook['event_types']:
                        continue
                except ValueError:
                    continue

            # Send webhook
            try:
                response = requests.post(
                    webhook['url'],
                    json=event.to_dict(),
                    headers=webhook['headers'],
                    timeout=10
                )
                if response.status_code != 200:
                    logger.warning(f"Webhook alert failed: {response.status_code}")
            except Exception as e:
                logger.error(f"Webhook alert error: {e}")

    def get_security_summary(self, hours: int = 24) -> Dict[str, Any]:
        """
        Get security summary for the specified time period.

        Args:
            hours: Number of hours to summarize

        Returns:
            Security summary dictionary
        """
        with self._lock:
            now = datetime.now()
            cutoff = now - timedelta(hours=hours)
            hour_key = now.strftime('%Y-%m-%d-%H')

            # Get recent events
            recent_counts = defaultdict(int)
            for h_key, events in self._hourly_counts.items():
                try:
                    h_time = datetime.strptime(h_key, '%Y-%m-%d-%H')
                    if h_time >= cutoff:
                        for event_type, count in events.items():
                            recent_counts[event_type] += count
                except ValueError:
                    continue

            # Calculate risk score
            risk_score = 0
            critical_events = [
                SecurityEventType.SQL_INJECTION_ATTEMPT.value,
                SecurityEventType.XSS_ATTEMPT.value,
                SecurityEventType.SESSION_HIJACK_ATTEMPT.value
            ]
            for event_type in critical_events:
                risk_score += recent_counts.get(event_type, 0) * 10

            risk_score += recent_counts.get(SecurityEventType.AUTH_FAILURE.value, 0) * 2
            risk_score += recent_counts.get(SecurityEventType.AUTH_LOCKOUT.value, 0) * 5

            return {
                'period_hours': hours,
                'total_events': sum(recent_counts.values()),
                'events_by_type': dict(recent_counts),
                'risk_score': risk_score,
                'risk_level': 'HIGH' if risk_score > 50 else 'MEDIUM' if risk_score > 20 else 'LOW',
                'generated_at': now.isoformat()
            }


# Module-level logger instance
_security_logger: Optional[SecurityLogger] = None


def get_security_logger() -> SecurityLogger:
    """Get or create module-level security logger."""
    global _security_logger
    if _security_logger is None:
        _security_logger = SecurityLogger()
    return _security_logger


def log_auth_success(username: str, **kwargs) -> SecurityEvent:
    """Log authentication success."""
    return get_security_logger().log_auth_success(username, **kwargs)


def log_auth_failure(username: str, **kwargs) -> SecurityEvent:
    """Log authentication failure."""
    return get_security_logger().log_auth_failure(username, **kwargs)


def log_authz_failure(username: str, resource: str, **kwargs) -> SecurityEvent:
    """Log authorization failure."""
    return get_security_logger().log_authz_failure(username, resource, **kwargs)


def log_rate_limit_exceeded(username: str, endpoint: str, limit: int, **kwargs) -> SecurityEvent:
    """Log rate limit exceeded."""
    return get_security_logger().log_rate_limit_exceeded(username, endpoint, limit, **kwargs)


def log_sql_injection_attempt(input_value: str, **kwargs) -> SecurityEvent:
    """Log SQL injection attempt."""
    return get_security_logger().log_sql_injection_attempt(input_value, **kwargs)


def log_xss_attempt(input_value: str, **kwargs) -> SecurityEvent:
    """Log XSS attempt."""
    return get_security_logger().log_xss_attempt(input_value, **kwargs)
