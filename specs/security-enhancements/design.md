# Design Document: Security Enhancements

## Overview

This design implements comprehensive security enhancements for the Notebook ML Orchestrator following defense-in-depth principles and OWASP guidelines. The security architecture consists of multiple layers: input validation at the entry points, authentication and authorization for access control, rate limiting for abuse prevention, and secure credential management for protecting sensitive data.

The design integrates security controls at every layer of the application stack:
- **Presentation Layer**: Input validation, XSS prevention, CSRF protection
- **Application Layer**: Authentication, authorization, session management
- **Business Logic Layer**: Rate limiting, secure error handling
- **Data Layer**: Encrypted credential storage, SQL injection prevention
- **Infrastructure Layer**: TLS encryption, secrets management integration

Key design principles:
- **Defense in Depth**: Multiple security layers so failure of one control doesn't compromise the system
- **Least Privilege**: Components and users have minimum necessary permissions
- **Fail Securely**: Security failures result in denial of access, not bypass
- **Secure by Default**: Security features enabled with strong defaults
- **Separation of Concerns**: Security logic isolated in dedicated components

## Architecture

### Component Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     Client Layer                             │
│  (CLI, GUI, API Clients)                                    │
└────────────────────┬────────────────────────────────────────┘
                     │ HTTPS/TLS
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                  API Gateway Layer                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ Rate Limiter │  │Input Validator│  │ TLS Handler  │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              Authentication & Authorization                  │
│  ┌──────────────────┐  ┌──────────────────┐               │
│  │ Authentication   │  │  Authorization    │               │
│  │    Manager       │  │     Manager       │               │
│  └──────────────────┘  └──────────────────┘               │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                 Application Layer                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │  Job Queue   │  │Backend Router│  │Workflow Engine│     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                   Data Layer                                 │
│  ┌──────────────────┐  ┌──────────────────┐               │
│  │  SQLite Database │  │  Credential Store │               │
│  │  (Parameterized) │  │   (Encrypted)     │               │
│  └──────────────────┘  └──────────────────┘               │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              External Services                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ Secrets Mgmt │  │Cloud Backends│  │  SIEM System │     │
│  │(Vault/AWS/Az)│  │(Modal/HF/etc)│  │              │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└─────────────────────────────────────────────────────────────┘
```

### Security Flow

1. **Request Reception**: Client sends request over TLS
2. **Rate Limiting**: Rate limiter checks request frequency
3. **Input Validation**: All inputs validated and sanitized
4. **Authentication**: User identity verified via token/API key
5. **Authorization**: User permissions checked for requested operation
6. **Business Logic**: Request processed by application components
7. **Data Access**: Database accessed via parameterized queries
8. **Credential Access**: Backend credentials retrieved from encrypted store
9. **Backend Communication**: Credentials transmitted over TLS to backends
10. **Response**: Sanitized response returned to client
11. **Logging**: Security events logged for monitoring

## Components and Interfaces

### 1. Input Validator

**Purpose**: Validate and sanitize all user inputs to prevent injection attacks and malformed data.

**Interface**:
```python
class InputValidator:
    def validate_job_input(self, job_data: dict) -> ValidationResult:
        """Validate job submission data against schema"""
        
    def validate_string(self, value: str, max_length: int, 
                       pattern: str = None) -> ValidationResult:
        """Validate string with length and pattern constraints"""
        
    def validate_path(self, path: str) -> ValidationResult:
        """Validate file path, prevent traversal attacks"""
        
    def validate_template_name(self, name: str) -> ValidationResult:
        """Validate template name (alphanumeric, hyphens, underscores only)"""
        
    def validate_json(self, json_str: str, schema: dict = None) -> ValidationResult:
        """Parse and validate JSON structure"""
        
    def sanitize_sql_input(self, value: str) -> str:
        """Sanitize input for SQL queries (use with parameterized queries)"""
        
    def sanitize_html(self, html: str) -> str:
        """Sanitize HTML content using allowlist-based approach"""
        
    def validate_file_upload(self, file_data: bytes, 
                            allowed_types: list) -> ValidationResult:
        """Validate file upload type and content"""

class ValidationResult:
    is_valid: bool
    sanitized_value: Any
    errors: list[str]
```

**Implementation Details**:
- Use Pydantic for schema-based validation
- Implement regex patterns for string validation
- Use `bleach` library for HTML sanitization
- Implement path validation using `pathlib` with checks for ".." and absolute paths
- Maximum string length: 10,000 characters (configurable)
- Reject null bytes (`\x00`) and control characters
- Template names: `^[a-zA-Z0-9_-]+$` pattern
- File type validation using magic bytes, not just extensions

### 2. Authentication Manager

**Purpose**: Verify user identity using multiple authentication methods.

**Interface**:
```python
class AuthenticationManager:
    def authenticate_api_key(self, api_key: str) -> AuthResult:
        """Authenticate using API key"""
        
    def authenticate_jwt(self, token: str) -> AuthResult:
        """Authenticate using JWT token"""
        
    def authenticate_oauth(self, oauth_token: str, 
                          provider: str) -> AuthResult:
        """Authenticate using OAuth token"""
        
    def create_session(self, user_id: str, 
                      metadata: dict) -> Session:
        """Create authenticated session"""
        
    def validate_session(self, session_id: str) -> SessionValidation:
        """Validate existing session"""
        
    def invalidate_session(self, session_id: str) -> bool:
        """Invalidate session (logout)"""
        
    def rotate_api_key(self, user_id: str, 
                      old_key: str) -> str:
        """Rotate API key for user"""
        
    def hash_password(self, password: str) -> str:
        """Hash password using Argon2"""
        
    def verify_password(self, password: str, 
                       password_hash: str) -> bool:
        """Verify password against hash"""

class AuthResult:
    success: bool
    user_id: str | None
    roles: list[str]
    error_message: str | None

class Session:
    session_id: str
    user_id: str
    created_at: datetime
    expires_at: datetime
    ip_address: str
    user_agent: str

class SessionValidation:
    is_valid: bool
    session: Session | None
    reason: str | None  # "expired", "invalid", "hijacked"
```

**Implementation Details**:
- Use `secrets` module for cryptographically secure random generation
- API keys: 32-byte random strings, base64-encoded
- JWT: Use PyJWT library with RS256 algorithm
- Password hashing: Argon2id with time cost=2, memory cost=65536, parallelism=4
- Session IDs: 32-byte random strings
- Session timeout: 24 hours (configurable)
- Inactive timeout: 2 hours (configurable)
- Account lockout: 3 failed attempts, 15-minute lockout
- Session binding: Store IP address and user agent, validate on each request
- OAuth: Support Google, GitHub providers using `authlib`

### 3. Authorization Manager

**Purpose**: Control user access to resources and operations based on roles and permissions.

**Interface**:
```python
class AuthorizationManager:
    def check_permission(self, user_id: str, 
                        resource: str, 
                        action: str) -> AuthzResult:
        """Check if user has permission for action on resource"""
        
    def get_user_roles(self, user_id: str) -> list[str]:
        """Get roles assigned to user"""
        
    def assign_role(self, user_id: str, role: str) -> bool:
        """Assign role to user (admin only)"""
        
    def check_resource_ownership(self, user_id: str, 
                                 resource_id: str) -> bool:
        """Check if user owns resource"""
        
    def get_permissions_for_role(self, role: str) -> list[Permission]:
        """Get all permissions for a role"""

class AuthzResult:
    allowed: bool
    reason: str | None

class Permission:
    resource: str  # "job", "template", "backend", "user"
    action: str    # "create", "read", "update", "delete", "execute"
    conditions: dict | None  # Additional conditions

class Role:
    name: str
    permissions: list[Permission]
```

**Role Definitions**:
- **Admin**: Full access to all resources and operations
  - Permissions: `*:*` (all actions on all resources)
- **User**: Can submit and manage own jobs
  - Permissions: `job:create`, `job:read:own`, `job:update:own`, `job:delete:own`, `template:read`, `backend:read`
- **Viewer**: Read-only access to own jobs
  - Permissions: `job:read:own`, `template:read`, `backend:read`

**Implementation Details**:
- Use role-based access control (RBAC) model
- Store role assignments in database
- Implement resource ownership checks for jobs
- Cache permissions in memory with TTL of 5 minutes
- Log all authorization failures with user, resource, action
- Support permission inheritance (admin inherits all permissions)

### 4. Rate Limiter

**Purpose**: Prevent abuse and DoS attacks by limiting request frequency.

**Interface**:
```python
class RateLimiter:
    def check_rate_limit(self, user_id: str, 
                        endpoint: str) -> RateLimitResult:
        """Check if request is within rate limits"""
        
    def record_request(self, user_id: str, endpoint: str) -> None:
        """Record request for rate limiting"""
        
    def get_limit_status(self, user_id: str, 
                        endpoint: str) -> LimitStatus:
        """Get current rate limit status"""
        
    def reset_limits(self, user_id: str) -> None:
        """Reset rate limits for user (admin only)"""

class RateLimitResult:
    allowed: bool
    retry_after: int | None  # Seconds until retry allowed
    remaining: int  # Remaining requests in window

class LimitStatus:
    limit: int
    remaining: int
    reset_at: datetime
```

**Rate Limit Configuration**:
- **Job Submission** (User): 100 requests/hour, 1000 requests/day
- **Job Submission** (Admin): 1000 requests/hour, 10000 requests/day
- **Job Status Query** (User): 1000 requests/hour
- **Authentication** (All): 10 requests/minute (per IP)
- **Global Limit**: 10,000 requests/hour across all users

**Implementation Details**:
- Use sliding window algorithm with Redis or in-memory store
- Store request timestamps in sorted set
- Window size: 1 hour for most endpoints
- Return HTTP 429 with `Retry-After` header
- Exempt health check endpoints (`/health`, `/metrics`)
- Log rate limit violations with user, endpoint, timestamp
- Implement distributed rate limiting for multi-instance deployments

### 5. Credential Store

**Purpose**: Securely store and manage API keys and sensitive credentials.

**Interface**:
```python
class CredentialStore:
    def store_credential(self, key: str, value: str, 
                        metadata: dict = None) -> bool:
        """Store encrypted credential"""
        
    def retrieve_credential(self, key: str) -> str | None:
        """Retrieve and decrypt credential"""
        
    def delete_credential(self, key: str) -> bool:
        """Delete credential"""
        
    def rotate_credential(self, key: str, new_value: str) -> bool:
        """Rotate credential value"""
        
    def list_credentials(self) -> list[CredentialMetadata]:
        """List credential metadata (not values)"""
        
    def audit_access(self, key: str, accessor: str) -> None:
        """Log credential access for auditing"""

class CredentialMetadata:
    key: str
    created_at: datetime
    updated_at: datetime
    accessed_at: datetime
    metadata: dict
```

**Implementation Details**:
- Use AES-256-GCM for encryption
- Derive encryption key from master key using PBKDF2 (100,000 iterations)
- Master key stored in environment variable `MASTER_ENCRYPTION_KEY`
- Generate unique nonce for each encryption operation
- Store encrypted credentials in SQLite with schema:
  ```sql
  CREATE TABLE credentials (
      key TEXT PRIMARY KEY,
      encrypted_value BLOB NOT NULL,
      nonce BLOB NOT NULL,
      created_at TIMESTAMP NOT NULL,
      updated_at TIMESTAMP NOT NULL,
      accessed_at TIMESTAMP,
      metadata TEXT
  );
  ```
- Clear decrypted credentials from memory immediately after use
- Implement credential caching with 5-minute TTL
- Log all credential access with timestamp and accessor component
- Support integration with external secrets managers (Vault, AWS Secrets Manager)

### 6. Secrets Manager Integration

**Purpose**: Integrate with external secrets management services for centralized credential management.

**Interface**:
```python
class SecretsManager:
    def get_secret(self, secret_name: str) -> str:
        """Retrieve secret from external service"""
        
    def put_secret(self, secret_name: str, secret_value: str) -> bool:
        """Store secret in external service"""
        
    def delete_secret(self, secret_name: str) -> bool:
        """Delete secret from external service"""
        
    def list_secrets(self) -> list[str]:
        """List available secret names"""
        
    def rotate_secret(self, secret_name: str) -> bool:
        """Trigger secret rotation"""

class VaultSecretsManager(SecretsManager):
    """HashiCorp Vault implementation"""
    
class AWSSecretsManager(SecretsManager):
    """AWS Secrets Manager implementation"""
    
class AzureKeyVaultManager(SecretsManager):
    """Azure Key Vault implementation"""
```

**Implementation Details**:
- Support multiple secrets backends via adapter pattern
- Cache secrets with configurable TTL (default 5 minutes)
- Implement automatic refresh on cache expiration
- Fail securely: reject operations if secrets service unavailable
- Validate connectivity during startup
- Support secret versioning where available
- Log all secrets operations for audit trail

### 7. Security Logger

**Purpose**: Log security events for monitoring and incident response.

**Interface**:
```python
class SecurityLogger:
    def log_authentication_attempt(self, user_id: str, 
                                   success: bool, 
                                   ip_address: str,
                                   method: str) -> None:
        """Log authentication attempt"""
        
    def log_authorization_failure(self, user_id: str, 
                                  resource: str, 
                                  action: str) -> None:
        """Log authorization failure"""
        
    def log_rate_limit_violation(self, user_id: str, 
                                 endpoint: str) -> None:
        """Log rate limit violation"""
        
    def log_input_validation_failure(self, input_type: str, 
                                     error: str) -> None:
        """Log input validation failure"""
        
    def log_credential_access(self, key: str, 
                             accessor: str) -> None:
        """Log credential access"""
        
    def log_security_alert(self, alert_type: str, 
                          details: dict) -> None:
        """Log security alert"""
        
    def get_security_events(self, start_time: datetime, 
                           end_time: datetime,
                           event_type: str = None) -> list[SecurityEvent]:
        """Query security events"""

class SecurityEvent:
    timestamp: datetime
    event_type: str
    user_id: str | None
    ip_address: str | None
    details: dict
    severity: str  # "info", "warning", "critical"
```

**Implementation Details**:
- Use structured logging (JSON format)
- Store security logs in separate SQLite table
- Implement log rotation (daily, keep 90 days)
- Never log sensitive data (passwords, API keys) in plain text
- Support integration with SIEM systems via syslog or HTTP
- Implement real-time alerting for critical events
- Log schema:
  ```sql
  CREATE TABLE security_logs (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      timestamp TIMESTAMP NOT NULL,
      event_type TEXT NOT NULL,
      user_id TEXT,
      ip_address TEXT,
      details TEXT,  -- JSON
      severity TEXT NOT NULL
  );
  CREATE INDEX idx_security_logs_timestamp ON security_logs(timestamp);
  CREATE INDEX idx_security_logs_event_type ON security_logs(event_type);
  ```

### 8. TLS Handler

**Purpose**: Manage TLS encryption for secure communication.

**Interface**:
```python
class TLSHandler:
    def configure_tls(self, cert_path: str, 
                     key_path: str,
                     min_version: str = "TLSv1.2") -> None:
        """Configure TLS settings"""
        
    def verify_certificate(self, hostname: str, 
                          cert: bytes) -> bool:
        """Verify TLS certificate"""
        
    def create_secure_context(self) -> ssl.SSLContext:
        """Create SSL context with secure settings"""
        
    def pin_certificate(self, hostname: str, 
                       cert_fingerprint: str) -> None:
        """Pin certificate for hostname"""
```

**Implementation Details**:
- Minimum TLS version: TLS 1.2
- Preferred TLS version: TLS 1.3
- Cipher suites: Use Mozilla Modern compatibility list
- Certificate verification: Always verify in production
- Support certificate pinning for critical backends
- Use Python `ssl` module with secure defaults
- Implement HSTS (HTTP Strict Transport Security) headers
- Set `Secure` flag on all cookies

## Data Models

### User Model
```python
class User:
    id: str  # UUID
    username: str
    email: str
    password_hash: str  # Argon2 hash
    roles: list[str]
    created_at: datetime
    updated_at: datetime
    last_login: datetime
    failed_login_attempts: int
    locked_until: datetime | None
    api_keys: list[APIKey]
```

### API Key Model
```python
class APIKey:
    id: str  # UUID
    user_id: str
    key_hash: str  # SHA-256 hash of key
    name: str  # User-friendly name
    created_at: datetime
    last_used: datetime
    expires_at: datetime | None
    scopes: list[str]  # Permissions for this key
```

### Session Model
```python
class Session:
    id: str  # Cryptographically random
    user_id: str
    created_at: datetime
    expires_at: datetime
    last_activity: datetime
    ip_address: str
    user_agent: str
    metadata: dict
```

### Rate Limit Record
```python
class RateLimitRecord:
    user_id: str
    endpoint: str
    window_start: datetime
    request_count: int
    last_request: datetime
```

### Credential Record
```python
class CredentialRecord:
    key: str
    encrypted_value: bytes
    nonce: bytes  # For AES-GCM
    created_at: datetime
    updated_at: datetime
    accessed_at: datetime
    metadata: dict
```

### Security Event
```python
class SecurityEvent:
    id: int
    timestamp: datetime
    event_type: str  # "auth_attempt", "authz_failure", "rate_limit", etc.
    user_id: str | None
    ip_address: str | None
    details: dict
    severity: str  # "info", "warning", "critical"
```

### Database Schema

```sql
-- Users table
CREATE TABLE users (
    id TEXT PRIMARY KEY,
    username TEXT UNIQUE NOT NULL,
    email TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    roles TEXT NOT NULL,  -- JSON array
    created_at TIMESTAMP NOT NULL,
    updated_at TIMESTAMP NOT NULL,
    last_login TIMESTAMP,
    failed_login_attempts INTEGER DEFAULT 0,
    locked_until TIMESTAMP
);

-- API Keys table
CREATE TABLE api_keys (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    key_hash TEXT UNIQUE NOT NULL,
    name TEXT NOT NULL,
    created_at TIMESTAMP NOT NULL,
    last_used TIMESTAMP,
    expires_at TIMESTAMP,
    scopes TEXT,  -- JSON array
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

-- Sessions table
CREATE TABLE sessions (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    created_at TIMESTAMP NOT NULL,
    expires_at TIMESTAMP NOT NULL,
    last_activity TIMESTAMP NOT NULL,
    ip_address TEXT NOT NULL,
    user_agent TEXT NOT NULL,
    metadata TEXT,  -- JSON
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

-- Rate limits table
CREATE TABLE rate_limits (
    user_id TEXT NOT NULL,
    endpoint TEXT NOT NULL,
    window_start TIMESTAMP NOT NULL,
    request_count INTEGER NOT NULL,
    last_request TIMESTAMP NOT NULL,
    PRIMARY KEY (user_id, endpoint, window_start)
);

-- Credentials table
CREATE TABLE credentials (
    key TEXT PRIMARY KEY,
    encrypted_value BLOB NOT NULL,
    nonce BLOB NOT NULL,
    created_at TIMESTAMP NOT NULL,
    updated_at TIMESTAMP NOT NULL,
    accessed_at TIMESTAMP,
    metadata TEXT  -- JSON
);

-- Security logs table
CREATE TABLE security_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TIMESTAMP NOT NULL,
    event_type TEXT NOT NULL,
    user_id TEXT,
    ip_address TEXT,
    details TEXT,  -- JSON
    severity TEXT NOT NULL
);

-- Indexes
CREATE INDEX idx_sessions_user_id ON sessions(user_id);
CREATE INDEX idx_sessions_expires_at ON sessions(expires_at);
CREATE INDEX idx_api_keys_user_id ON api_keys(user_id);
CREATE INDEX idx_rate_limits_user_endpoint ON rate_limits(user_id, endpoint);
CREATE INDEX idx_security_logs_timestamp ON security_logs(timestamp);
CREATE INDEX idx_security_logs_event_type ON security_logs(event_type);
CREATE INDEX idx_security_logs_user_id ON security_logs(user_id);
```

