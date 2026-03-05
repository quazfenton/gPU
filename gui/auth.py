"""Authentication and authorization module for GUI application.

This module provides:
- Authentication middleware
- Configurable authentication providers
- Session management with timeout
- Role-based access control (RBAC)
- Rate limiting for authentication endpoints

Requirements validated: 8.1, 8.2, 8.3, 8.5, 8.6, 8.7

SECURITY ENHANCED: Rate limiting added to prevent brute force attacks.
"""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Callable, Dict, Optional, Set
from uuid import uuid4
from collections import defaultdict
import hmac

from gui.rate_limiter import RateLimiter, RateLimitConfig, RateLimitError


class Role(Enum):
    """User roles for role-based access control."""
    ADMIN = "admin"
    USER = "user"
    VIEWER = "viewer"


class Permission(Enum):
    """Permissions for different operations."""
    SUBMIT_JOB = "submit_job"
    VIEW_ALL_JOBS = "view_all_jobs"
    VIEW_OWN_JOBS = "view_own_jobs"
    EXECUTE_WORKFLOW = "execute_workflow"
    MANAGE_BACKENDS = "manage_backends"
    VIEW_BACKEND_STATUS = "view_backend_status"


# Role to permissions mapping
ROLE_PERMISSIONS: Dict[Role, Set[Permission]] = {
    Role.ADMIN: {
        Permission.SUBMIT_JOB,
        Permission.VIEW_ALL_JOBS,
        Permission.VIEW_OWN_JOBS,
        Permission.EXECUTE_WORKFLOW,
        Permission.MANAGE_BACKENDS,
        Permission.VIEW_BACKEND_STATUS,
    },
    Role.USER: {
        Permission.SUBMIT_JOB,
        Permission.VIEW_OWN_JOBS,
        Permission.EXECUTE_WORKFLOW,
        Permission.VIEW_BACKEND_STATUS,
    },
    Role.VIEWER: {
        Permission.VIEW_OWN_JOBS,
        Permission.VIEW_BACKEND_STATUS,
    },
}


@dataclass
class User:
    """Represents an authenticated user."""
    username: str
    role: Role
    user_id: str = field(default_factory=lambda: str(uuid4()))
    metadata: Dict[str, str] = field(default_factory=dict)
    
    def has_permission(self, permission: Permission) -> bool:
        """Check if user has a specific permission.
        
        Args:
            permission: The permission to check
            
        Returns:
            True if user has the permission, False otherwise
        """
        return permission in ROLE_PERMISSIONS.get(self.role, set())


@dataclass
class Session:
    """Represents an authenticated session."""
    session_id: str
    user: User
    created_at: datetime
    last_activity: datetime
    timeout_seconds: int
    
    def is_expired(self) -> bool:
        """Check if session has expired.
        
        Returns:
            True if session has exceeded timeout, False otherwise
        """
        elapsed = datetime.now() - self.last_activity
        return elapsed.total_seconds() > self.timeout_seconds
    
    def refresh(self) -> None:
        """Update last activity timestamp to current time."""
        self.last_activity = datetime.now()


class AuthenticationProvider(ABC):
    """Abstract base class for authentication providers.
    
    Subclasses must implement the authenticate method to validate
    credentials against their specific authentication backend.
    """
    
    @abstractmethod
    def authenticate(self, username: str, password: str) -> Optional[User]:
        """Authenticate user credentials.
        
        Args:
            username: Username to authenticate
            password: Password to validate
            
        Returns:
            User object if authentication succeeds, None otherwise
        """
        pass


class SimpleAuthProvider(AuthenticationProvider):
    """Simple in-memory authentication provider for testing/development.
    
    SECURITY ENHANCED: Rate limiting and account lockout added to prevent brute force attacks.

    Stores username/password pairs in memory. Not suitable for production.
    """

    def __init__(
        self,
        users: Optional[Dict[str, tuple[str, Role]]] = None,
        max_failed_attempts: int = 5,
        lockout_duration_minutes: int = 30
    ):
        """Initialize with user credentials.

        Args:
            users: Dictionary mapping username to (password, role) tuples
            max_failed_attempts: Maximum failed login attempts before lockout
            lockout_duration_minutes: Duration to lock account after max failures
        """
        self.users = users or {}
        self.max_failed_attempts = max_failed_attempts
        self.lockout_duration = timedelta(minutes=lockout_duration_minutes)
        
        # Rate limiting for authentication endpoints
        self.auth_rate_limiter = RateLimiter(RateLimitConfig(
            requests_per_minute=10,  # Max 10 auth attempts per minute
            requests_per_hour=60,    # Max 60 auth attempts per hour
        ))
        
        # Track failed attempts per user
        self._failed_attempts: Dict[str, list] = defaultdict(list)
        self._locked_accounts: Dict[str, datetime] = {}

    def authenticate(
        self,
        username: str,
        password: str,
        ip_address: Optional[str] = None
    ) -> Optional[User]:
        """Authenticate against in-memory user store.
        
        SECURITY: Rate limiting and account lockout enforced.

        Args:
            username: Username to authenticate
            password: Password to validate
            ip_address: Optional IP address for rate limiting

        Returns:
            User object if credentials match, None otherwise
            
        Raises:
            RateLimitError: If rate limit exceeded
            AuthenticationError: If account locked or credentials invalid
        """
        client_id = ip_address or username
        
        # SECURITY: Check rate limit
        try:
            self.auth_rate_limiter.check_rate_limit(client_id)
        except RateLimitError as e:
            raise AuthenticationError(
                f"Too many authentication attempts. Please wait {e.retry_after} seconds."
            )
        
        # SECURITY: Check if account is locked
        if username in self._locked_accounts:
            lockout_until = self._locked_accounts[username]
            if datetime.now() < lockout_until:
                remaining = (lockout_until - datetime.now()).seconds // 60
                raise AuthenticationError(
                    f"Account locked due to multiple failed attempts. Try again in {remaining} minutes."
                )
            else:
                # Lockout expired, remove it
                del self._locked_accounts[username]
                self._failed_attempts[username] = []
        
        # Validate credentials
        if username in self.users:
            stored_password, role = self.users[username]
            
            # Use constant-time comparison to prevent timing attacks
            if hmac.compare_digest(stored_password.encode(), password.encode()):
                # Success - reset failed attempts
                self._failed_attempts[username] = []
                return User(username=username, role=role)
        
        # Failed attempt - track it
        self._failed_attempts[username].append(datetime.now())
        
        # Check if max attempts exceeded
        if len(self._failed_attempts[username]) >= self.max_failed_attempts:
            self._locked_accounts[username] = datetime.now() + self.lockout_duration
            raise AuthenticationError(
                f"Account locked due to {self.max_failed_attempts} failed attempts. "
                f"Try again in {self.lockout_duration.seconds // 60} minutes."
            )
        
        raise AuthenticationError("Invalid credentials")
    
    def get_failed_attempts(self, username: str) -> int:
        """Get number of failed attempts for a user."""
        return len(self._failed_attempts.get(username, []))
    
    def is_locked(self, username: str) -> bool:
        """Check if account is locked."""
        if username not in self._locked_accounts:
            return False
        
        if datetime.now() >= self._locked_accounts[username]:
            # Lockout expired
            del self._locked_accounts[username]
            self._failed_attempts[username] = []
            return False
        
        return True
    
    def unlock_account(self, username: str) -> bool:
        """Manually unlock a locked account."""
        if username in self._locked_accounts:
            del self._locked_accounts[username]
            self._failed_attempts[username] = []
            return True
        return False


class SessionManager:
    """Manages user sessions with timeout support.
    
    Handles session creation, validation, and cleanup of expired sessions.
    """
    
    def __init__(self, timeout_seconds: int = 3600):
        """Initialize session manager.
        
        Args:
            timeout_seconds: Session timeout in seconds (default: 3600)
        """
        self.timeout_seconds = timeout_seconds
        self.sessions: Dict[str, Session] = {}
    
    def create_session(self, user: User) -> Session:
        """Create a new session for authenticated user.
        
        Args:
            user: Authenticated user
            
        Returns:
            New session object
        """
        session_id = str(uuid4())
        now = datetime.now()
        session = Session(
            session_id=session_id,
            user=user,
            created_at=now,
            last_activity=now,
            timeout_seconds=self.timeout_seconds
        )
        self.sessions[session_id] = session
        return session
    
    def get_session(self, session_id: str) -> Optional[Session]:
        """Retrieve session by ID.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Session object if valid and not expired, None otherwise
        """
        session = self.sessions.get(session_id)
        if session is None:
            return None
        
        if session.is_expired():
            self.invalidate_session(session_id)
            return None
        
        session.refresh()
        return session
    
    def invalidate_session(self, session_id: str) -> None:
        """Invalidate and remove a session.
        
        Args:
            session_id: Session identifier to invalidate
        """
        self.sessions.pop(session_id, None)
    
    def cleanup_expired_sessions(self) -> int:
        """Remove all expired sessions.
        
        Returns:
            Number of sessions removed
        """
        expired = [
            sid for sid, session in self.sessions.items()
            if session.is_expired()
        ]
        for sid in expired:
            self.invalidate_session(sid)
        return len(expired)


class AuthenticationMiddleware:
    """Middleware for enforcing authentication and authorization.
    
    Provides decorators and utilities for protecting endpoints with
    authentication and permission checks.
    """
    
    def __init__(
        self,
        provider: AuthenticationProvider,
        session_manager: SessionManager,
        enabled: bool = True
    ):
        """Initialize authentication middleware.
        
        Args:
            provider: Authentication provider for credential validation
            session_manager: Session manager for session handling
            enabled: Whether authentication is enabled (default: True)
        """
        self.provider = provider
        self.session_manager = session_manager
        self.enabled = enabled
    
    def authenticate(self, username: str, password: str) -> Optional[Session]:
        """Authenticate user and create session.
        
        Args:
            username: Username to authenticate
            password: Password to validate
            
        Returns:
            Session object if authentication succeeds, None otherwise
        """
        if not self.enabled:
            # When auth is disabled, create a default admin session
            default_user = User(username="anonymous", role=Role.ADMIN)
            return self.session_manager.create_session(default_user)
        
        user = self.provider.authenticate(username, password)
        if user is None:
            return None
        
        return self.session_manager.create_session(user)
    
    def validate_session(self, session_id: str) -> Optional[Session]:
        """Validate session ID and return session if valid.
        
        Args:
            session_id: Session identifier to validate
            
        Returns:
            Session object if valid, None otherwise
        """
        if not self.enabled:
            # When auth is disabled, return a default admin session
            default_user = User(username="anonymous", role=Role.ADMIN)
            return Session(
                session_id="anonymous",
                user=default_user,
                created_at=datetime.now(),
                last_activity=datetime.now(),
                timeout_seconds=self.session_manager.timeout_seconds
            )
        
        return self.session_manager.get_session(session_id)
    
    def check_permission(
        self,
        session: Session,
        permission: Permission
    ) -> bool:
        """Check if session user has required permission.
        
        Args:
            session: User session
            permission: Required permission
            
        Returns:
            True if user has permission, False otherwise
        """
        return session.user.has_permission(permission)
    
    def require_auth(self, func: Callable) -> Callable:
        """Decorator to require authentication for a function.
        
        Args:
            func: Function to protect
            
        Returns:
            Wrapped function that checks authentication
        """
        def wrapper(session_id: str, *args, **kwargs):
            if self.enabled:
                session = self.validate_session(session_id)
                if session is None:
                    raise PermissionError("Authentication required")
            return func(*args, **kwargs)
        return wrapper
    
    def require_permission(self, permission: Permission) -> Callable:
        """Decorator to require specific permission for a function.
        
        Args:
            permission: Required permission
            
        Returns:
            Decorator function
        """
        def decorator(func: Callable) -> Callable:
            def wrapper(session_id: str, *args, **kwargs):
                if self.enabled:
                    session = self.validate_session(session_id)
                    if session is None:
                        raise PermissionError("Authentication required")
                    if not self.check_permission(session, permission):
                        raise PermissionError(
                            f"Permission denied: {permission.value} required"
                        )
                return func(*args, **kwargs)
            return wrapper
        return decorator
    
    def logout(self, session_id: str) -> None:
        """Logout user by invalidating session.
        
        Args:
            session_id: Session identifier to invalidate
        """
        self.session_manager.invalidate_session(session_id)
