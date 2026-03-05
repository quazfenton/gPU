"""
Authentication Manager with JWT and OAuth 2.0 support.

This module provides comprehensive authentication capabilities including:
- JWT token generation and validation
- OAuth 2.0 flow support
- API key authentication
- Password hashing with bcrypt
- Session management
- Role-based access control
- Two-factor authentication (TOTP)
- Brute force protection
- Password policy enforcement
- Login history tracking

Security Features:
- Secure token generation with cryptographic randomness
- Token expiration and refresh
- Password hashing with bcrypt (cost factor 12)
- Protection against timing attacks
- Audit logging for authentication events
- Account lockout after failed attempts
- TOTP-based two-factor authentication
"""

import hashlib
import hmac
import json
import os
import secrets
import threading
import time
import base64
import struct
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Callable, Tuple
from collections import defaultdict
import logging
import re

logger = logging.getLogger(__name__)


class AuthenticationError(Exception):
    """Exception raised for authentication errors."""
    
    def __init__(self, message: str, error_code: str = "AUTH_ERROR"):
        super().__init__(message)
        self.error_code = error_code


class TokenValidationError(Exception):
    """Exception raised for token validation errors."""
    
    def __init__(self, message: str, error_code: str = "TOKEN_INVALID"):
        super().__init__(message)
        self.error_code = error_code


class Role(Enum):
    """User role enumeration."""
    ADMIN = "admin"
    USER = "user"
    VIEWER = "viewer"
    SERVICE = "service"


@dataclass
class User:
    """User data structure."""
    id: str
    username: str
    email: Optional[str] = None
    role: Role = Role.USER
    created_at: datetime = field(default_factory=datetime.now)
    last_login: Optional[datetime] = None
    is_active: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TokenPayload:
    """JWT token payload."""
    user_id: str
    username: str
    role: str
    issued_at: datetime
    expires_at: datetime
    token_type: str  # 'access' or 'refresh'
    jti: str  # Unique token ID
    scopes: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Session:
    """User session data."""
    id: str
    user_id: str
    username: str
    role: Role
    created_at: datetime
    expires_at: datetime
    last_activity: datetime
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    is_valid: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LoginAttempt:
    """Login attempt record for security tracking."""
    timestamp: datetime
    username: str
    ip_address: Optional[str]
    user_agent: Optional[str]
    success: bool
    failure_reason: Optional[str] = None


@dataclass
class TwoFactorConfig:
    """Two-factor authentication configuration."""
    enabled: bool = False
    secret: Optional[str] = None  # Base32-encoded TOTP secret
    backup_codes: List[str] = field(default_factory=list)
    verified: bool = False


@dataclass
class PasswordPolicy:
    """Password policy configuration."""
    min_length: int = 8
    require_uppercase: bool = True
    require_lowercase: bool = True
    require_digit: bool = True
    require_special: bool = False
    min_special_chars: int = 1
    prevent_common_passwords: bool = True
    max_age_days: Optional[int] = None  # None = no expiration
    history_count: int = 5  # Number of previous passwords to remember


@dataclass
class OAuthConfig:
    """OAuth provider configuration."""
    provider: str  # 'google', 'github', 'azure', etc.
    client_id: str
    client_secret: str
    redirect_uri: str
    scopes: List[str] = field(default_factory=list)
    authorization_url: str = ""
    token_url: str = ""
    userinfo_url: str = ""


# Common weak passwords to prevent
COMMON_WEAK_PASSWORDS = {
    'password', 'password123', '123456', '12345678', 'qwerty',
    'abc123', 'monkey', 'master', 'dragon', 'letmein', 'login',
    'admin', 'welcome', 'password1', 'p@ssw0rd', 'pass123',
}

# OAuth provider configurations
OAUTH_PROVIDERS = {
    'google': {
        'authorization_url': 'https://accounts.google.com/o/oauth2/v2/auth',
        'token_url': 'https://oauth2.googleapis.com/token',
        'userinfo_url': 'https://www.googleapis.com/oauth2/v2/userinfo',
        'default_scopes': ['openid', 'email', 'profile'],
    },
    'github': {
        'authorization_url': 'https://github.com/login/oauth/authorize',
        'token_url': 'https://github.com/login/oauth/access_token',
        'userinfo_url': 'https://api.github.com/user',
        'default_scopes': ['user:email'],
    },
    'azure': {
        'authorization_url': 'https://login.microsoftonline.com/common/oauth2/v2.0/authorize',
        'token_url': 'https://login.microsoftonline.com/common/oauth2/v2.0/token',
        'userinfo_url': 'https://graph.microsoft.com/v1.0/me',
        'default_scopes': ['openid', 'email', 'profile'],
    },
}


class AuthManager:
    """
    Authentication manager with JWT and OAuth 2.0 support.

    This class provides methods for:
    - User authentication with username/password
    - JWT token generation and validation
    - API key authentication
    - Session management
    - Password hashing and verification
    - Two-factor authentication (TOTP)
    - Brute force protection
    - OAuth 2.0 integration

    Example:
        auth = AuthManager(secret_key=os.environ['JWT_SECRET'])

        # Register a user
        auth.register_user('admin', 'admin@example.com', 'password123', Role.ADMIN)

        # Authenticate user
        tokens = auth.authenticate('admin', 'password123')

        # Validate token
        payload = auth.validate_token(tokens['access_token'])

        # Generate API key
        api_key = auth.generate_api_key('admin')
    """

    # Token configuration
    ACCESS_TOKEN_EXPIRY = timedelta(minutes=30)
    REFRESH_TOKEN_EXPIRY = timedelta(days=7)
    API_KEY_PREFIX = 'nml_'

    # Brute force protection configuration
    MAX_FAILED_ATTEMPTS = 5
    LOCKOUT_DURATION_MINUTES = 30
    FAILED_ATTEMPT_WINDOW_MINUTES = 15

    def __init__(
        self,
        secret_key: Optional[str] = None,
        token_expiry: Optional[timedelta] = None,
        refresh_token_expiry: Optional[timedelta] = None,
        allow_refresh: bool = True,
        password_policy: Optional[PasswordPolicy] = None,
        enable_audit_logging: bool = True,
        audit_logger: Optional[Callable] = None,
        max_concurrent_sessions: int = 5,
        enable_2fa: bool = False
    ):
        """
        Initialize authentication manager.

        Args:
            secret_key: Secret key for JWT signing
                       If not provided, will try to load from JWT_SECRET env var
            token_expiry: Access token expiry duration (default: 30 minutes)
            refresh_token_expiry: Refresh token expiry duration (default: 7 days)
            allow_refresh: Whether to allow token refresh
            password_policy: Password policy configuration
            enable_audit_logging: Enable audit logging for auth events
            audit_logger: Custom audit logger function
            max_concurrent_sessions: Maximum concurrent sessions per user
            enable_2fa: Enable two-factor authentication support

        Raises:
            AuthenticationError: If secret key cannot be obtained
        """
        self._lock = threading.RLock()
        self._users: Dict[str, User] = {}
        self._sessions: Dict[str, Session] = {}
        self._api_keys: Dict[str, str] = {}  # key -> user_id mapping
        self._token_blacklist: set = set()
        self._password_hashes: Dict[str, str] = {}
        
        # Brute force protection
        self._failed_attempts: Dict[str, List[datetime]] = defaultdict(list)
        self._locked_accounts: Dict[str, datetime] = {}
        
        # Login history
        self._login_history: Dict[str, List[LoginAttempt]] = defaultdict(list)
        
        # Password history (for preventing reuse)
        self._password_history: Dict[str, List[str]] = defaultdict(list)
        
        # Two-factor authentication configs
        self._2fa_configs: Dict[str, TwoFactorConfig] = {}
        
        # Audit logging
        self._enable_audit_logging = enable_audit_logging
        self._audit_logger = audit_logger
        
        # Configuration
        self.password_policy = password_policy or PasswordPolicy()
        self.max_concurrent_sessions = max_concurrent_sessions
        self.enable_2fa = enable_2fa
        
        # OAuth configurations
        self._oauth_configs: Dict[str, OAuthConfig] = {}
        self._oauth_states: Dict[str, Dict[str, Any]] = {}  # State -> config

        # Get secret key
        self._secret_key = self._get_secret_key(secret_key)

        # Token expiry settings
        self.access_token_expiry = token_expiry or self.ACCESS_TOKEN_EXPIRY
        self.refresh_token_expiry = refresh_token_expiry or self.REFRESH_TOKEN_EXPIRY
        self.allow_refresh = allow_refresh

        logger.info("AuthManager initialized with JWT support and enhanced security features")

    def _audit_log(self, action: str, username: Optional[str] = None, details: Optional[Dict[str, Any]] = None) -> None:
        """Log an audit event for authentication actions."""
        if not self._enable_audit_logging:
            return
        
        audit_data = {
            'timestamp': datetime.now().isoformat(),
            'action': action,
            'username': username,
            'details': details or {}
        }
        
        if self._audit_logger:
            try:
                self._audit_logger(audit_data)
            except Exception as e:
                logger.error(f"Audit logger failed: {e}")
        else:
            logger.info(f"[AUDIT] {action}: {username or 'unknown'} - {details}")
    
    def _get_secret_key(self, secret_key: Optional[str]) -> bytes:
        """Get or generate secret key.
        
        SECURITY: In production mode, fails closed if no secret key is configured.
        Random keys are only allowed in development/test environments.
        
        Raises:
            AuthenticationError: If secret key cannot be obtained in production
        """
        # Try provided key first
        if secret_key:
            if isinstance(secret_key, str):
                return secret_key.encode('utf-8')
            return secret_key

        # Try environment variable
        env_key = os.environ.get('JWT_SECRET')
        if env_key:
            logger.info("Loaded JWT secret from JWT_SECRET environment variable")
            return env_key.encode('utf-8')

        # Check if running in production
        is_production = os.environ.get('ENVIRONMENT', 'development').lower() in [
            'production', 'prod', 'live'
        ]
        is_testing = os.environ.get('PYTEST_CURRENT_TEST') is not None
        
        if is_production:
            # SECURITY: Fail closed in production - do not start without proper secret
            error_msg = (
                "CRITICAL: JWT_SECRET not configured in production environment. "
                "This is a security requirement - the application cannot start without a secure JWT secret. "
                "Set JWT_SECRET environment variable with a cryptographically secure value. "
                "Generate one with: python -c \"import secrets; print(secrets.token_hex(32))\""
            )
            logger.error(error_msg)
            raise AuthenticationError(error_msg, "JWT_SECRET_NOT_CONFIGURED")
        
        if is_testing:
            # Allow random key for testing
            logger.warning("Using random JWT secret for testing (tokens will not persist)")
            return secrets.token_bytes(32)

        # Development mode - allow random key with strong warning
        logger.warning(
            "DEVELOPMENT MODE: No JWT secret provided. Using random key. "
            "WARNING: Tokens will NOT persist across application restarts. "
            "Set JWT_SECRET environment variable for consistent behavior. "
            "Generate secure key: python -c \"import secrets; print(secrets.token_hex(32))\""
        )
        return secrets.token_bytes(32)
    
    def _hash_password(self, password: str) -> str:
        """
        Hash password using bcrypt.
        
        Args:
            password: Plain text password
            
        Returns:
            Hashed password
        """
        try:
            import bcrypt
            salt = bcrypt.gensalt(rounds=12)
            hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
            return hashed.decode('utf-8')
        except ImportError:
            # Fallback to PBKDF2 if bcrypt not available
            logger.warning("bcrypt not available, using PBKDF2 fallback")
            salt = secrets.token_hex(16)
            hash_bytes = hashlib.pbkdf2_hmac(
                'sha256',
                password.encode('utf-8'),
                salt.encode('utf-8'),
                100000
            )
            return f"pbkdf2:{salt}:{hash_bytes.hex()}"
    
    def _verify_password(self, password: str, hashed: str) -> bool:
        """
        Verify password against hash.
        
        Args:
            password: Plain text password
            hashed: Hashed password
            
        Returns:
            True if password matches
        """
        try:
            import bcrypt
            return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))
        except ImportError:
            # Fallback to PBKDF2 verification
            try:
                parts = hashed.split(':')
                if len(parts) != 3 or parts[0] != 'pbkdf2':
                    return False
                salt = parts[1]
                stored_hash = parts[2]
                hash_bytes = hashlib.pbkdf2_hmac(
                    'sha256',
                    password.encode('utf-8'),
                    salt.encode('utf-8'),
                    100000
                )
                return hmac.compare_digest(hash_bytes.hex(), stored_hash)
            except Exception:
                return False
    
    def _generate_token_id(self) -> str:
        """Generate unique token ID."""
        return secrets.token_hex(16)
    
    def _create_jwt(self, payload: TokenPayload) -> str:
        """
        Create JWT token from payload.
        
        Args:
            payload: Token payload
            
        Returns:
            JWT token string
        """
        try:
            import jwt
            
            payload_dict = {
                'user_id': payload.user_id,
                'username': payload.username,
                'role': payload.role,
                'iat': payload.issued_at.timestamp(),
                'exp': payload.expires_at.timestamp(),
                'token_type': payload.token_type,
                'jti': payload.jti,
                'scopes': payload.scopes,
            }
            
            if payload.metadata:
                payload_dict['metadata'] = payload.metadata
            
            token = jwt.encode(
                payload_dict,
                self._secret_key,
                algorithm='HS256'
            )
            
            return token
            
        except ImportError:
            # Fallback to simple base64 encoding if PyJWT not available
            logger.warning("PyJWT not available, using base64 fallback (NOT SECURE FOR PRODUCTION)")
            payload_dict = {
                'user_id': payload.user_id,
                'username': payload.username,
                'role': payload.role,
                'iat': payload.issued_at.timestamp(),
                'exp': payload.expires_at.timestamp(),
                'token_type': payload.token_type,
                'jti': payload.jti,
            }
            payload_json = json.dumps(payload_dict).encode('utf-8')
            signature = hmac.new(
                self._secret_key,
                payload_json,
                hashlib.sha256
            ).digest()
            return base64.b64encode(payload_json + signature).decode('utf-8')
    
    def _parse_jwt(self, token: str) -> TokenPayload:
        """
        Parse JWT token to payload.
        
        Args:
            token: JWT token string
            
        Returns:
            Token payload
            
        Raises:
            TokenValidationError: If token is invalid
        """
        try:
            import jwt
            
            payload_dict = jwt.decode(
                token,
                self._secret_key,
                algorithms=['HS256']
            )
            
            return TokenPayload(
                user_id=payload_dict['user_id'],
                username=payload_dict['username'],
                role=payload_dict['role'],
                issued_at=datetime.fromtimestamp(payload_dict['iat']),
                expires_at=datetime.fromtimestamp(payload_dict['exp']),
                token_type=payload_dict['token_type'],
                jti=payload_dict['jti'],
                scopes=payload_dict.get('scopes', []),
                metadata=payload_dict.get('metadata', {})
            )
            
        except ImportError:
            # Fallback to base64 decoding
            try:
                decoded = base64.b64decode(token)
                payload_json = decoded[:-64]  # Remove signature
                signature = decoded[-64:]
                
                # Verify signature
                expected_signature = hmac.new(
                    self._secret_key,
                    payload_json,
                    hashlib.sha256
                ).digest()
                
                if not hmac.compare_digest(signature, expected_signature):
                    raise TokenValidationError("Invalid token signature")
                
                payload_dict = json.loads(payload_json)
                
                return TokenPayload(
                    user_id=payload_dict['user_id'],
                    username=payload_dict['username'],
                    role=payload_dict['role'],
                    issued_at=datetime.fromtimestamp(payload_dict['iat']),
                    expires_at=datetime.fromtimestamp(payload_dict['exp']),
                    token_type=payload_dict['token_type'],
                    jti=payload_dict['jti'],
                )
                
            except Exception as e:
                raise TokenValidationError(f"Invalid token: {str(e)}")
                
        except jwt.ExpiredSignatureError:
            raise TokenValidationError("Token has expired", "TOKEN_EXPIRED")
        except jwt.InvalidTokenError as e:
            raise TokenValidationError(f"Invalid token: {str(e)}")
    
    def register_user(
        self,
        username: str,
        email: Optional[str] = None,
        password: Optional[str] = None,
        role: Role = Role.USER,
        user_id: Optional[str] = None
    ) -> User:
        """
        Register a new user.

        Args:
            username: Username
            email: Email address
            password: Password (will be hashed)
            role: User role
            user_id: Optional user ID (auto-generated if not provided)

        Returns:
            Created user

        Raises:
            AuthenticationError: If user already exists
            ValueError: If password doesn't meet policy requirements
        """
        with self._lock:
            if username in self._users:
                self._audit_log('user.register_failed', username, {'reason': 'already_exists'})
                raise AuthenticationError(f"User '{username}' already exists", "USER_EXISTS")

            # Validate password if provided
            if password:
                is_valid, errors = self.validate_password(password)
                if not is_valid:
                    self._audit_log('user.register_failed', username, {'reason': 'weak_password', 'errors': errors})
                    raise ValueError(f"Password does not meet policy requirements: {', '.join(errors)}")

            user = User(
                id=user_id or secrets.token_hex(16),
                username=username,
                email=email,
                role=role
            )

            self._users[username] = user

            if password:
                self._password_hashes[username] = self._hash_password(password)
                # Store in password history
                self._password_history[username].append(self._password_hashes[username])
                if len(self._password_history[username]) > self.password_policy.history_count:
                    self._password_history[username].pop(0)

            self._audit_log('user.registered', username, {'role': role.value, 'email': email})
            logger.info(f"Registered user: {username} (role: {role.value})")
            return user
    
    def authenticate(
        self,
        username: str,
        password: str,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        totp_code: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Authenticate user with username and password.

        Args:
            username: Username
            password: Password
            ip_address: Client IP address for security tracking
            user_agent: Client user agent for security tracking
            totp_code: Optional TOTP code for 2FA verification

        Returns:
            Dictionary with access_token, refresh_token, and user info

        Raises:
            AuthenticationError: If authentication fails
            AccountLockoutError: If account is locked due to too many failed attempts
        """
        with self._lock:
            # Check if account is locked
            if self._is_account_locked(username):
                lockout_time = self._locked_accounts.get(username)
                self._audit_log('auth.lockout_detected', username, {'ip': ip_address})
                raise AuthenticationError(
                    f"Account is locked due to too many failed attempts. Try again after {lockout_time}",
                    "ACCOUNT_LOCKED"
                )
            
            # Find user
            user = self._users.get(username)
            if not user:
                self._record_failed_attempt(username, ip_address, user_agent, 'user_not_found')
                self._audit_log('auth.failed', username, {'reason': 'user_not_found', 'ip': ip_address})
                logger.warning(f"Authentication failed: user '{username}' not found")
                raise AuthenticationError("Invalid username or password", "INVALID_CREDENTIALS")

            # Check if user is active
            if not user.is_active:
                self._record_failed_attempt(username, ip_address, user_agent, 'user_inactive')
                self._audit_log('auth.failed', username, {'reason': 'user_inactive', 'ip': ip_address})
                logger.warning(f"Authentication failed: user '{username}' is inactive")
                raise AuthenticationError("User account is inactive", "USER_INACTIVE")

            # Verify password
            stored_hash = self._password_hashes.get(username)
            if not stored_hash:
                self._record_failed_attempt(username, ip_address, user_agent, 'no_password_set')
                self._audit_log('auth.failed', username, {'reason': 'no_password_set', 'ip': ip_address})
                logger.warning(f"Authentication failed: no password set for '{username}'")
                raise AuthenticationError("Invalid username or password", "INVALID_CREDENTIALS")

            if not self._verify_password(password, stored_hash):
                self._record_failed_attempt(username, ip_address, user_agent, 'incorrect_password')
                self._audit_log('auth.failed', username, {'reason': 'incorrect_password', 'ip': ip_address})
                logger.warning(f"Authentication failed: incorrect password for '{username}'")
                raise AuthenticationError("Invalid username or password", "INVALID_CREDENTIALS")

            # Check 2FA if enabled (REQUIRED when 2FA is configured for user)
            if self.enable_2fa:
                # Check if user has 2FA configured
                user_2fa_config = self._2fa_configs.get(username)
                if user_2fa_config and user_2fa_config.enabled:
                    # 2FA is required - must provide valid code
                    if not totp_code:
                        self._record_failed_attempt(username, ip_address, user_agent, '2fa_required')
                        self._audit_log('auth.failed', username, {'reason': '2fa_required', 'ip': ip_address})
                        raise AuthenticationError("2FA code required", "2FA_REQUIRED")
                    
                    if not self._verify_totp(username, totp_code):
                        self._record_failed_attempt(username, ip_address, user_agent, 'invalid_totp')
                        self._audit_log('auth.failed', username, {'reason': 'invalid_totp', 'ip': ip_address})
                        raise AuthenticationError("Invalid 2FA code", "INVALID_TOTP")
                else:
                    # 2FA enabled globally but user hasn't configured yet
                    # Allow login but could prompt user to set up 2FA
                    logger.info(f"User {username} logged in without 2FA (not configured for user)")
            
            # Clear failed attempts on successful login
            self._failed_attempts[username].clear()
            if username in self._locked_accounts:
                del self._locked_accounts[username]

            # Update last login
            user.last_login = datetime.now()
            
            # Check concurrent session limit
            active_sessions = sum(1 for s in self._sessions.values() 
                                 if s.user_id == user.id and s.is_valid)
            if active_sessions >= self.max_concurrent_sessions:
                # Invalidate oldest session
                oldest_session = min(
                    [s for s in self._sessions.values() if s.user_id == user.id and s.is_valid],
                    key=lambda s: s.last_activity
                )
                oldest_session.is_valid = False
                self._audit_log('session.invalidated', username, {'reason': 'max_sessions_reached'})

            # Record successful login
            self._record_login_attempt(username, ip_address, user_agent, True)
            self._audit_log('auth.success', username, {'ip': ip_address})

            # Generate tokens
            tokens = self._generate_tokens(user)

            logger.info(f"User '{username}' authenticated successfully")

            return {
                'access_token': tokens['access_token'],
                'refresh_token': tokens['refresh_token'],
                'user': {
                    'id': user.id,
                    'username': user.username,
                    'email': user.email,
                    'role': user.role.value
                }
            }
    
    def _generate_tokens(self, user: User) -> Dict[str, str]:
        """Generate access and refresh tokens for user."""
        now = datetime.now()
        
        # Generate access token
        access_payload = TokenPayload(
            user_id=user.id,
            username=user.username,
            role=user.role.value,
            issued_at=now,
            expires_at=now + self.access_token_expiry,
            token_type='access',
            jti=self._generate_token_id()
        )
        
        # Generate refresh token
        refresh_payload = TokenPayload(
            user_id=user.id,
            username=user.username,
            role=user.role.value,
            issued_at=now,
            expires_at=now + self.refresh_token_expiry,
            token_type='refresh',
            jti=self._generate_token_id()
        )
        
        return {
            'access_token': self._create_jwt(access_payload),
            'refresh_token': self._create_jwt(refresh_payload)
        }
    
    def validate_token(self, token: str) -> TokenPayload:
        """
        Validate JWT token and return payload.
        
        Args:
            token: JWT token
            
        Returns:
            Token payload
            
        Raises:
            TokenValidationError: If token is invalid
        """
        # Check blacklist
        payload = self._parse_jwt(token)
        
        if payload.jti in self._token_blacklist:
            raise TokenValidationError("Token has been revoked", "TOKEN_REVOKED")
        
        return payload
    
    def refresh_access_token(self, refresh_token: str) -> Dict[str, str]:
        """
        Refresh access token using refresh token.
        
        Args:
            refresh_token: Refresh token
            
        Returns:
            New tokens dictionary
            
        Raises:
            TokenValidationError: If refresh token is invalid
            AuthenticationError: If user not found
        """
        if not self.allow_refresh:
            raise AuthenticationError("Token refresh is disabled", "REFRESH_DISABLED")
        
        # Validate refresh token
        payload = self.validate_token(refresh_token)
        
        if payload.token_type != 'refresh':
            raise TokenValidationError("Invalid token type", "INVALID_TOKEN_TYPE")
        
        # Find user
        user = None
        for u in self._users.values():
            if u.id == payload.user_id:
                user = u
                break
        
        if not user:
            raise AuthenticationError("User not found", "USER_NOT_FOUND")
        
        # Generate new tokens
        return self._generate_tokens(user)
    
    def revoke_token(self, token: str) -> None:
        """
        Revoke a token (add to blacklist).
        
        Args:
            token: Token to revoke
        """
        try:
            payload = self._parse_jwt(token)
            with self._lock:
                self._token_blacklist.add(payload.jti)
            logger.info(f"Token {payload.jti} revoked")
        except TokenValidationError:
            pass  # Token already invalid
    
    def generate_api_key(self, username: str, scopes: Optional[List[str]] = None) -> str:
        """
        Generate API key for user.
        
        Args:
            username: Username
            scopes: Optional list of scopes
            
        Returns:
            API key string
            
        Raises:
            AuthenticationError: If user not found
        """
        with self._lock:
            user = self._users.get(username)
            if not user:
                raise AuthenticationError(f"User '{username}' not found", "USER_NOT_FOUND")
            
            # Generate API key
            key = f"{self.API_KEY_PREFIX}{secrets.token_urlsafe(32)}"
            
            # Store key mapping
            self._api_keys[key] = user.username
            
            logger.info(f"Generated API key for user '{username}'")
            return key
    
    def authenticate_api_key(self, api_key: str) -> Optional[User]:
        """
        Authenticate using API key.
        
        Args:
            api_key: API key
            
        Returns:
            User if authenticated, None otherwise
        """
        with self._lock:
            username = self._api_keys.get(api_key)
            if not username:
                return None
            
            return self._users.get(username)
    
    def create_session(
        self,
        user: User,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        duration_hours: int = 24
    ) -> Session:
        """
        Create user session.
        
        Args:
            user: User
            ip_address: Client IP address
            user_agent: Client user agent
            duration_hours: Session duration in hours
            
        Returns:
            Session object
        """
        with self._lock:
            session = Session(
                id=secrets.token_hex(32),
                user_id=user.id,
                username=user.username,
                role=user.role,
                created_at=datetime.now(),
                expires_at=datetime.now() + timedelta(hours=duration_hours),
                last_activity=datetime.now(),
                ip_address=ip_address,
                user_agent=user_agent
            )
            
            self._sessions[session.id] = session
            
            logger.info(f"Created session {session.id} for user '{user.username}'")
            return session
    
    def get_session(self, session_id: str) -> Optional[Session]:
        """
        Get session by ID.
        
        Args:
            session_id: Session ID
            
        Returns:
            Session if found and valid, None otherwise
        """
        with self._lock:
            session = self._sessions.get(session_id)
            
            if not session:
                return None
            
            # Check if session is expired
            if datetime.now() > session.expires_at:
                session.is_valid = False
                return None
            
            # Update last activity
            session.last_activity = datetime.now()
            
            return session if session.is_valid else None
    
    def invalidate_session(self, session_id: str) -> bool:
        """
        Invalidate a session.
        
        Args:
            session_id: Session ID
            
        Returns:
            True if invalidated, False if not found
        """
        with self._lock:
            if session_id in self._sessions:
                self._sessions[session_id].is_valid = False
                logger.info(f"Session {session_id} invalidated")
                return True
            return False
    
    def get_user(self, username: str) -> Optional[User]:
        """
        Get user by username.
        
        Args:
            username: Username
            
        Returns:
            User if found, None otherwise
        """
        return self._users.get(username)
    
    def list_users(self) -> List[Dict[str, Any]]:
        """
        List all users (metadata only).
        
        Returns:
            List of user metadata dictionaries
        """
        with self._lock:
            return [
                {
                    'id': user.id,
                    'username': user.username,
                    'email': user.email,
                    'role': user.role.value,
                    'created_at': user.created_at.isoformat(),
                    'last_login': user.last_login.isoformat() if user.last_login else None,
                    'is_active': user.is_active
                }
                for user in self._users.values()
            ]
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get authentication manager statistics.

        Returns:
            Dictionary with stats
        """
        with self._lock:
            return {
                'total_users': len(self._users),
                'active_sessions': sum(1 for s in self._sessions.values() if s.is_valid),
                'api_keys': len(self._api_keys),
                'revoked_tokens': len(self._token_blacklist),
                'access_token_expiry_minutes': self.access_token_expiry.total_seconds() / 60,
                'refresh_token_expiry_days': self.refresh_token_expiry.total_seconds() / 86400,
                'locked_accounts': len(self._locked_accounts),
                '2fa_enabled_users': sum(1 for config in self._2fa_configs.values() if config.enabled),
                'oauth_providers_configured': len(self._oauth_configs),
            }

    # ==================== ENHANCEMENT METHODS ====================

    def _record_failed_attempt(self, username: str, ip_address: Optional[str],
                               user_agent: Optional[str], reason: str) -> None:
        """Record a failed login attempt for brute force protection."""
        now = datetime.now()
        
        # Record failed attempt
        self._failed_attempts[username].append(now)
        
        # Clean old attempts outside window
        cutoff = now - timedelta(minutes=self.FAILED_ATTEMPT_WINDOW_MINUTES)
        self._failed_attempts[username] = [
            t for t in self._failed_attempts[username] if t > cutoff
        ]
        
        # Check if should lock account
        if len(self._failed_attempts[username]) >= self.MAX_FAILED_ATTEMPTS:
            self._locked_accounts[username] = now
            lockout_until = now + timedelta(minutes=self.LOCKOUT_DURATION_MINUTES)
            self._audit_log('auth.account_locked', username, {
                'ip': ip_address,
                'failed_attempts': len(self._failed_attempts[username]),
                'lockout_until': lockout_until.isoformat()
            })
            logger.warning(f"Account {username} locked until {lockout_until}")
        
        # Record in login history
        self._record_login_attempt(username, ip_address, user_agent, False, reason)

    def _record_login_attempt(self, username: str, ip_address: Optional[str],
                              user_agent: Optional[str], success: bool,
                              failure_reason: Optional[str] = None) -> None:
        """Record login attempt in history."""
        attempt = LoginAttempt(
            timestamp=datetime.now(),
            username=username,
            ip_address=ip_address,
            user_agent=user_agent,
            success=success,
            failure_reason=failure_reason
        )
        self._login_history[username].append(attempt)
        
        # Keep only last 100 attempts per user
        if len(self._login_history[username]) > 100:
            self._login_history[username] = self._login_history[username][-100:]

    def _is_account_locked(self, username: str) -> bool:
        """Check if account is locked."""
        if username not in self._locked_accounts:
            return False
        
        lockout_time = self._locked_accounts[username]
        if datetime.now() > lockout_time + timedelta(minutes=self.LOCKOUT_DURATION_MINUTES):
            # Lockout expired
            del self._locked_accounts[username]
            self._failed_attempts[username].clear()
            return False
        
        return True

    def validate_password(self, password: str) -> Tuple[bool, List[str]]:
        """
        Validate password against policy.

        Args:
            password: Password to validate

        Returns:
            Tuple of (is_valid, list of error messages)
        """
        errors = []
        policy = self.password_policy

        # Check minimum length
        if len(password) < policy.min_length:
            errors.append(f"Password must be at least {policy.min_length} characters")

        # Check character requirements
        if policy.require_uppercase and not re.search(r'[A-Z]', password):
            errors.append("Password must contain at least one uppercase letter")

        if policy.require_lowercase and not re.search(r'[a-z]', password):
            errors.append("Password must contain at least one lowercase letter")

        if policy.require_digit and not re.search(r'\d', password):
            errors.append("Password must contain at least one digit")

        if policy.require_special:
            special_count = len(re.findall(r'[!@#$%^&*(),.?":{}|<>]', password))
            if special_count < policy.min_special_chars:
                errors.append(f"Password must contain at least {policy.min_special_chars} special character(s)")

        # Check common passwords
        if policy.prevent_common_passwords and password.lower() in COMMON_WEAK_PASSWORDS:
            errors.append("Password is too common. Please choose a stronger password")

        return len(errors) == 0, errors

    def change_password(self, username: str, old_password: str, new_password: str) -> bool:
        """
        Change user password with validation.

        Args:
            username: Username
            old_password: Current password
            new_password: New password

        Returns:
            True if successful

        Raises:
            AuthenticationError: If old password is incorrect
            ValueError: If new password doesn't meet policy
        """
        with self._lock:
            # Verify old password
            if not self._verify_password(old_password, self._password_hashes.get(username, '')):
                raise AuthenticationError("Current password is incorrect", "INVALID_PASSWORD")

            # Validate new password
            is_valid, errors = self.validate_password(new_password)
            if not is_valid:
                raise ValueError(f"New password does not meet requirements: {', '.join(errors)}")

            # Check password history
            new_hash = self._hash_password(new_password)
            if new_hash in self._password_history.get(username, []):
                raise ValueError("Password has been used recently. Please choose a different password")

            # Update password
            self._password_hashes[username] = new_hash
            
            # Update history
            self._password_history[username].append(new_hash)
            if len(self._password_history[username]) > self.password_policy.history_count:
                self._password_history[username].pop(0)

            self._audit_log('password.changed', username)
            logger.info(f"Password changed for user {username}")
            return True

    def setup_2fa(self, username: str) -> Dict[str, Any]:
        """
        Setup two-factor authentication for user.

        Args:
            username: Username

        Returns:
            Dictionary with setup information including QR code URL
        """
        with self._lock:
            user = self._users.get(username)
            if not user:
                raise AuthenticationError(f"User {username} not found", "USER_NOT_FOUND")

            # Generate TOTP secret
            secret = base64.b32encode(os.urandom(20)).decode('utf-8')

            # Generate backup codes
            backup_codes = [secrets.token_hex(4).upper() for _ in range(10)]

            # Store config
            self._2fa_configs[username] = TwoFactorConfig(
                enabled=False,  # Not enabled until verified
                secret=secret,
                backup_codes=backup_codes,
                verified=False
            )

            # Generate QR code URL
            issuer = "NotebookML"
            qr_url = f"otpauth://totp/{issuer}:{username}?secret={secret}&issuer={issuer}"

            self._audit_log('2fa.setup_initiated', username)

            return {
                'secret': secret,
                'backup_codes': backup_codes,
                'qr_url': qr_url,
                'manual_entry_key': secret
            }

    def verify_2fa_setup(self, username: str, totp_code: str) -> bool:
        """
        Verify 2FA setup and enable it.

        Args:
            username: Username
            totp_code: TOTP code from authenticator app

        Returns:
            True if verified and enabled
        """
        with self._lock:
            config = self._2fa_configs.get(username)
            if not config or not config.secret:
                raise AuthenticationError("2FA not setup for this user", "2FA_NOT_CONFIGURED")

            if self._verify_totp(username, totp_code):
                config.enabled = True
                config.verified = True
                self._audit_log('2fa.enabled', username)
                logger.info(f"2FA enabled for user {username}")
                return True

            self._audit_log('2fa.verification_failed', username)
            return False

    def _verify_totp(self, username: str, totp_code: str) -> bool:
        """Verify TOTP code."""
        config = self._2fa_configs.get(username)
        if not config or not config.secret:
            return False

        try:
            # Import TOTP library if available
            import pyotp
            totp = pyotp.TOTP(config.secret)
            return totp.verify(totp_code, valid_window=1)
        except ImportError:
            logger.warning("pyotp not installed, 2FA verification unavailable")
            return False

    def disable_2fa(self, username: str, totp_code: Optional[str] = None,
                    backup_code: Optional[str] = None) -> bool:
        """
        Disable 2FA for user.

        Args:
            username: Username
            totp_code: TOTP code for verification
            backup_code: Backup code for verification (alternative to totp_code)

        Returns:
            True if disabled
        """
        with self._lock:
            config = self._2fa_configs.get(username)
            if not config or not config.enabled:
                raise AuthenticationError("2FA not enabled for this user", "2FA_NOT_ENABLED")

            # Verify with TOTP or backup code
            verified = False
            if totp_code and self._verify_totp(username, totp_code):
                verified = True
            elif backup_code and backup_code.upper() in config.backup_codes:
                config.backup_codes.remove(backup_code.upper())
                verified = True

            if not verified:
                self._audit_log('2fa.disable_failed', username, {'reason': 'verification_failed'})
                raise AuthenticationError("Invalid verification code", "INVALID_CODE")

            config.enabled = False
            config.verified = False
            self._audit_log('2fa.disabled', username)
            logger.info(f"2FA disabled for user {username}")
            return True

    def get_login_history(self, username: str, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Get login history for user.

        Args:
            username: Username
            limit: Maximum number of entries to return

        Returns:
            List of login attempt records
        """
        history = self._login_history.get(username, [])
        return [
            {
                'timestamp': attempt.timestamp.isoformat(),
                'ip_address': attempt.ip_address,
                'user_agent': attempt.user_agent,
                'success': attempt.success,
                'failure_reason': attempt.failure_reason
            }
            for attempt in history[-limit:]
        ]

    def configure_oauth(self, provider: str, client_id: str, client_secret: str,
                       redirect_uri: str, scopes: Optional[List[str]] = None) -> None:
        """
        Configure OAuth provider.

        Args:
            provider: Provider name ('google', 'github', 'azure')
            client_id: OAuth client ID
            client_secret: OAuth client secret
            redirect_uri: Redirect URI
            scopes: Optional list of scopes
        """
        provider_lower = provider.lower()
        
        # Get provider defaults
        provider_config = OAUTH_PROVIDERS.get(provider_lower, {})
        
        config = OAuthConfig(
            provider=provider_lower,
            client_id=client_id,
            client_secret=client_secret,
            redirect_uri=redirect_uri,
            scopes=scopes or provider_config.get('default_scopes', []),
            authorization_url=provider_config.get('authorization_url', ''),
            token_url=provider_config.get('token_url', ''),
            userinfo_url=provider_config.get('userinfo_url', '')
        )

        with self._lock:
            self._oauth_configs[provider_lower] = config
            self._audit_log('oauth.configured', None, {'provider': provider_lower})
            logger.info(f"OAuth configured for provider: {provider_lower}")

    def get_oauth_authorization_url(self, provider: str) -> Tuple[str, str]:
        """
        Get OAuth authorization URL.

        Args:
            provider: Provider name

        Returns:
            Tuple of (authorization_url, state)
        """
        with self._lock:
            config = self._oauth_configs.get(provider.lower())
            if not config:
                raise AuthenticationError(f"OAuth provider {provider} not configured", "OAUTH_NOT_CONFIGURED")

            # Generate state
            state = secrets.token_urlsafe(32)
            
            # Store state for verification
            self._oauth_states[state] = {
                'provider': provider.lower(),
                'created_at': datetime.now()
            }

            # Build authorization URL
            from urllib.parse import urlencode
            params = {
                'client_id': config.client_id,
                'redirect_uri': config.redirect_uri,
                'scope': ' '.join(config.scopes),
                'response_type': 'code',
                'state': state,
                'access_type': 'offline',
                'prompt': 'consent'
            }

            auth_url = f"{config.authorization_url}?{urlencode(params)}"
            return auth_url, state
