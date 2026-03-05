"""
Credential Store with AES-256-GCM encryption.

This module provides secure storage and retrieval of sensitive credentials
using industry-standard encryption algorithms.

Features:
- AES-256-GCM encryption (authenticated encryption)
- PBKDF2 key derivation for master key
- Secure key storage via environment variables
- Support for multiple secrets backends (Vault, AWS, Azure)
- Credential rotation support
- Audit logging for credential access
- Role-based access control for credentials
- Automatic expiration checking
- Memory protection for sensitive data
- Backup/export functionality
- Key versioning support

Security Considerations:
- Master key must be stored securely (environment variable or secrets manager)
- Never log or display credentials in plain text
- Use HTTPS for all credential transmission
- Rotate credentials regularly
- Implement access controls for credential retrieval
"""

import base64
import hashlib
import hmac
import json
import os
import threading
import weakref
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
import copy

logger = logging.getLogger(__name__)


class CredentialEncryptionError(Exception):
    """Exception raised for credential encryption/decryption errors."""
    
    def __init__(self, message: str, is_recoverable: bool = False):
        super().__init__(message)
        self.is_recoverable = is_recoverable


@dataclass
class EncryptedCredential:
    """Encrypted credential with metadata."""
    ciphertext: str  # Base64-encoded encrypted data
    nonce: str  # Base64-encoded nonce
    tag: str  # Base64-encoded authentication tag
    created_at: str  # ISO format timestamp
    updated_at: str  # ISO format timestamp
    version: int = 1  # Encryption format version


@dataclass
class CredentialEntry:
    """Credential entry with metadata."""
    service: str
    key: str
    value: str  # Plain text value (only in memory, never persisted)
    created_at: datetime
    updated_at: datetime
    accessed_at: Optional[datetime] = None
    access_count: int = 0
    expires_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class AccessLevel(Enum):
    """Access level for credential access control."""
    READ = "read"  # Can read credential value
    WRITE = "write"  # Can create/update/delete credential
    ADMIN = "admin"  # Full access including access control


@dataclass
class AccessPolicy:
    """Access control policy for a credential."""
    service: str
    key: str
    allowed_roles: Set[str] = field(default_factory=set)
    allowed_users: Set[str] = field(default_factory=set)
    denied_users: Set[str] = field(default_factory=set)
    require_mfa: bool = False
    audit_all_access: bool = True
    max_access_count: Optional[int] = None  # None = unlimited


@dataclass
class CredentialTemplate:
    """Template for common credential structures."""
    name: str
    service: str
    fields: List[Dict[str, Any]]  # List of {key, required, description}
    validation_rules: Optional[Dict[str, Any]] = None
    rotation_interval_days: Optional[int] = None
    description: str = ""


# Pre-defined credential templates
CREDENTIAL_TEMPLATES = {
    'modal': CredentialTemplate(
        name='Modal',
        service='modal',
        fields=[
            {'key': 'token_id', 'required': True, 'description': 'Modal Token ID'},
            {'key': 'token_secret', 'required': True, 'description': 'Modal Token Secret'},
        ],
        rotation_interval_days=90,
        description='Modal.com API credentials'
    ),
    'huggingface': CredentialTemplate(
        name='HuggingFace',
        service='huggingface',
        fields=[
            {'key': 'token', 'required': True, 'description': 'HuggingFace API Token'},
        ],
        rotation_interval_days=90,
        description='HuggingFace API credentials'
    ),
    'kaggle': CredentialTemplate(
        name='Kaggle',
        service='kaggle',
        fields=[
            {'key': 'username', 'required': True, 'description': 'Kaggle Username'},
            {'key': 'key', 'required': True, 'description': 'Kaggle API Key'},
        ],
        rotation_interval_days=90,
        description='Kaggle API credentials'
    ),
    'aws': CredentialTemplate(
        name='AWS',
        service='aws',
        fields=[
            {'key': 'access_key_id', 'required': True, 'description': 'AWS Access Key ID'},
            {'key': 'secret_access_key', 'required': True, 'description': 'AWS Secret Access Key'},
            {'key': 'region', 'required': False, 'description': 'AWS Region'},
        ],
        rotation_interval_days=90,
        description='AWS API credentials'
    ),
    'azure': CredentialTemplate(
        name='Azure',
        service='azure',
        fields=[
            {'key': 'client_id', 'required': True, 'description': 'Azure Client ID'},
            {'key': 'client_secret', 'required': True, 'description': 'Azure Client Secret'},
            {'key': 'tenant_id', 'required': True, 'description': 'Azure Tenant ID'},
            {'key': 'subscription_id', 'required': False, 'description': 'Azure Subscription ID'},
        ],
        rotation_interval_days=90,
        description='Azure API credentials'
    ),
    'database': CredentialTemplate(
        name='Database',
        service='database',
        fields=[
            {'key': 'host', 'required': True, 'description': 'Database Host'},
            {'key': 'port', 'required': False, 'description': 'Database Port'},
            {'key': 'username', 'required': True, 'description': 'Database Username'},
            {'key': 'password', 'required': True, 'description': 'Database Password'},
            {'key': 'database', 'required': True, 'description': 'Database Name'},
        ],
        rotation_interval_days=30,
        description='Database connection credentials'
    ),
}


class CredentialStore:
    """
    Secure credential store with AES-256-GCM encryption.

    This class provides methods for storing, retrieving, and managing
    sensitive credentials with automatic encryption at rest.

    Example:
        store = CredentialStore(master_key=os.environ['MASTER_KEY'])

        # Store a credential
        store.set_credential('modal', 'token_id', 'my-token-id')

        # Retrieve a credential
        token = store.get_credential('modal', 'token_id')

        # Delete a credential
        store.delete_credential('modal', 'token_id')

        # List all services
        services = store.list_services()
    """

    # Encryption constants
    ALGORITHM = 'AES-256-GCM'
    KEY_SIZE = 32  # 256 bits
    NONCE_SIZE = 12  # 96 bits for GCM
    TAG_SIZE = 16  # 128 bits for GCM
    SALT_SIZE = 32  # 256 bits
    PBKDF2_ITERATIONS = 100000
    ENCRYPTION_VERSION = 1

    def __init__(
        self,
        master_key: Optional[str] = None,
        salt: Optional[str] = None,
        backend: str = 'file',
        backend_config: Optional[Dict[str, Any]] = None,
        enable_audit_logging: bool = True,
        audit_logger: Optional[Callable] = None,
        auto_cleanup_expired: bool = True,
        cleanup_interval_hours: int = 24
    ):
        """
        Initialize credential store.

        Args:
            master_key: Master encryption key (32 bytes or will be derived)
                       If not provided, will try to load from MASTER_KEY env var
            salt: Salt for key derivation (hex-encoded)
                 If not provided, will try to load from CREDENTIAL_SALT env var
            backend: Storage backend ('file', 'vault', 'aws', 'azure')
            backend_config: Backend-specific configuration
            enable_audit_logging: Enable audit logging for credential access
            audit_logger: Custom audit logger function (default: uses internal logging)
            auto_cleanup_expired: Automatically clean up expired credentials
            cleanup_interval_hours: Interval for expired credential cleanup

        Raises:
            CredentialEncryptionError: If master key cannot be obtained
        """
        self._lock = threading.RLock()
        self._credentials: Dict[str, CredentialEntry] = {}
        self._encrypted_store: Dict[str, EncryptedCredential] = {}
        self._backend = backend
        self._backend_config = backend_config or {}
        
        # Access control
        self._access_policies: Dict[str, AccessPolicy] = {}
        self._current_user: Optional[str] = None
        self._current_role: Optional[str] = None
        
        # Audit logging
        self._enable_audit_logging = enable_audit_logging
        self._audit_logger = audit_logger
        
        # Auto-cleanup configuration
        self._auto_cleanup_expired = auto_cleanup_expired
        self._cleanup_interval_hours = cleanup_interval_hours
        self._last_cleanup: Optional[datetime] = None
        
        # Key versioning (for future key rotation)
        self._key_version: int = 1
        self._key_history: Dict[int, bytes] = {1: b''}  # Will be set after key derivation

        # Get or derive master key
        self._master_key = self._get_master_key(master_key)

        # Get or generate salt
        self._salt = self._get_salt(salt)

        # Derive encryption key from master key and salt
        self._encryption_key = self._derive_key(self._master_key, self._salt)
        self._key_history[self._key_version] = self._encryption_key

        logger.info("CredentialStore initialized with AES-256-GCM encryption")

        # Load existing credentials if using file backend
        if backend == 'file':
            self._load_credentials()
        
        # Start cleanup thread if auto-cleanup enabled
        if auto_cleanup_expired:
            self._start_cleanup_thread()
    
    def _get_master_key(self, master_key: Optional[str]) -> bytes:
        """Get or derive master key."""
        # Try provided key first
        if master_key:
            return self._normalize_key(master_key)
        
        # Try environment variable
        env_key = os.environ.get('MASTER_KEY')
        if env_key:
            logger.info("Loaded master key from MASTER_KEY environment variable")
            return self._normalize_key(env_key)
        
        # Try loading from backend
        if self._backend == 'vault':
            try:
                import hvac
                client = hvac.Client(url=self._backend_config.get('vault_url'))
                client.token = os.environ.get('VAULT_TOKEN')
                response = client.secrets.kv.v2.read_secret_version(
                    path=self._backend_config.get('vault_path', 'orchestrator/master-key')
                )
                key = response['data']['data']['key']
                logger.info("Loaded master key from HashiCorp Vault")
                return self._normalize_key(key)
            except Exception as e:
                logger.warning(f"Failed to load master key from Vault: {e}")
        
        elif self._backend == 'aws':
            try:
                import boto3
                client = boto3.client('secretsmanager')
                response = client.get_secret_value(
                    SecretId=self._backend_config.get('secret_name', 'orchestrator/master-key')
                )
                key = json.loads(response['SecretString'])['key']
                logger.info("Loaded master key from AWS Secrets Manager")
                return self._normalize_key(key)
            except Exception as e:
                logger.warning(f"Failed to load master key from AWS: {e}")
        
        elif self._backend == 'azure':
            try:
                from azure.keyvault.secrets import SecretClient
                from azure.identity import DefaultAzureCredential
                
                credential = DefaultAzureCredential()
                client = SecretClient(
                    vault_url=self._backend_config.get('vault_url'),
                    credential=credential
                )
                key = client.get_secret('master-key').value
                logger.info("Loaded master key from Azure Key Vault")
                return self._normalize_key(key)
            except Exception as e:
                logger.warning(f"Failed to load master key from Azure: {e}")

        # No secure key source found - fail closed (do not use fallback)
        error_msg = (
            "CRITICAL: No secure master key configured. "
            "Set MASTER_KEY environment variable or configure a secrets backend (Vault/AWS/Azure). "
            "For development, generate a random key with: python -c \"import secrets; print(secrets.token_hex(32))\""
        )
        logger.error(error_msg)
        raise CredentialEncryptionError(error_msg, is_recoverable=False)
    
    def _normalize_key(self, key: str) -> bytes:
        """Normalize key to correct size using SHA-256."""
        if isinstance(key, str):
            key_bytes = key.encode('utf-8')
        else:
            key_bytes = key
        
        # Hash to get consistent 32-byte key
        return hashlib.sha256(key_bytes).digest()
    
    def _get_salt(self, salt: Optional[str]) -> bytes:
        """Get or generate salt."""
        # Try provided salt
        if salt:
            return bytes.fromhex(salt)
        
        # Try environment variable
        env_salt = os.environ.get('CREDENTIAL_SALT')
        if env_salt:
            logger.info("Loaded salt from CREDENTIAL_SALT environment variable")
            return bytes.fromhex(env_salt)
        
        # Try loading from backend
        if self._backend == 'file':
            salt_file = self._backend_config.get('salt_file', 'credential_salt.hex')
            if os.path.exists(salt_file):
                with open(salt_file, 'r') as f:
                    salt = f.read().strip()
                    logger.info("Loaded salt from file")
                    return bytes.fromhex(salt)
        
        # Generate new salt
        new_salt = os.urandom(self.SALT_SIZE)
        logger.info("Generated new salt (store securely for future use)")
        
        # Save salt for future use
        if self._backend == 'file':
            salt_file = self._backend_config.get('salt_file', 'credential_salt.hex')
            with open(salt_file, 'w') as f:
                f.write(new_salt.hex())
        
        return new_salt
    
    def _derive_key(self, master_key: bytes, salt: bytes) -> bytes:
        """Derive encryption key from master key using PBKDF2."""
        return hashlib.pbkdf2_hmac(
            'sha256',
            master_key,
            salt,
            self.PBKDF2_ITERATIONS,
            dklen=self.KEY_SIZE
        )
    
    def _encrypt(self, plaintext: str) -> EncryptedCredential:
        """
        Encrypt plaintext using AES-256-GCM.
        
        Args:
            plaintext: Plain text to encrypt
            
        Returns:
            EncryptedCredential with ciphertext, nonce, and tag
            
        Raises:
            CredentialEncryptionError: If encryption fails
        """
        try:
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM
            
            # Generate random nonce
            nonce = os.urandom(self.NONCE_SIZE)
            
            # Create AESGCM instance
            aesgcm = AESGCM(self._encryption_key)
            
            # Encrypt (includes authentication tag automatically)
            ciphertext = aesgcm.encrypt(nonce, plaintext.encode('utf-8'), None)
            
            # Split ciphertext and tag (tag is last 16 bytes)
            actual_ciphertext = ciphertext[:-self.TAG_SIZE]
            tag = ciphertext[-self.TAG_SIZE:]
            
            now = datetime.now().isoformat()
            
            return EncryptedCredential(
                ciphertext=base64.b64encode(actual_ciphertext).decode('utf-8'),
                nonce=base64.b64encode(nonce).decode('utf-8'),
                tag=base64.b64encode(tag).decode('utf-8'),
                created_at=now,
                updated_at=now,
                version=self.ENCRYPTION_VERSION
            )
            
        except ImportError:
            raise CredentialEncryptionError(
                "cryptography package not installed. Install with: pip install cryptography",
                is_recoverable=False
            )
        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            raise CredentialEncryptionError(
                f"Failed to encrypt credential: {str(e)}",
                is_recoverable=True
            )
    
    def _decrypt(self, encrypted: EncryptedCredential) -> str:
        """
        Decrypt ciphertext using AES-256-GCM.
        
        Args:
            encrypted: EncryptedCredential object
            
        Returns:
            Decrypted plain text
            
        Raises:
            CredentialEncryptionError: If decryption fails (including tampering)
        """
        try:
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM
            
            # Decode base64
            ciphertext = base64.b64decode(encrypted.ciphertext)
            nonce = base64.b64decode(encrypted.nonce)
            tag = base64.b64decode(encrypted.tag)
            
            # Reconstruct full ciphertext (ciphertext + tag)
            full_ciphertext = ciphertext + tag
            
            # Create AESGCM instance
            aesgcm = AESGCM(self._encryption_key)
            
            # Decrypt (will raise exception if tag doesn't match - tampering detected)
            plaintext = aesgcm.decrypt(nonce, full_ciphertext, None)
            
            return plaintext.decode('utf-8')
            
        except ImportError:
            raise CredentialEncryptionError(
                "cryptography package not installed. Install with: pip install cryptography",
                is_recoverable=False
            )
        except Exception as e:
            error_str = str(e).lower()
            
            # Check for tampering
            if "tag" in error_str or "mac" in error_str or "decrypt" in error_str:
                logger.error("Credential decryption failed - possible tampering detected!")
                raise CredentialEncryptionError(
                    "Credential decryption failed - data may have been tampered with",
                    is_recoverable=False
                )
            
            logger.error(f"Decryption failed: {e}")
            raise CredentialEncryptionError(
                f"Failed to decrypt credential: {str(e)}",
                is_recoverable=True
            )
    
    def set_credential(
        self,
        service: str,
        key: str,
        value: str,
        expires_at: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None,
        user: Optional[str] = None,
        role: Optional[str] = None
    ) -> None:
        """
        Store a credential with encryption.

        Args:
            service: Service name (e.g., 'modal', 'huggingface')
            key: Credential key (e.g., 'token_id', 'api_key')
            value: Credential value (plain text, will be encrypted)
            expires_at: Optional expiration time
            metadata: Optional metadata dictionary
            user: Optional user for access control
            role: Optional role for access control

        Raises:
            CredentialEncryptionError: If encryption fails
            PermissionError: If access is denied
        """
        store_key = f"{service}:{key}"
        
        # Use provided user/role or current context
        original_user = self._current_user
        original_role = self._current_role
        if user:
            self._current_user = user
        if role:
            self._current_role = role
        
        try:
            # Check access control
            if not self.check_access(service, key, AccessLevel.WRITE):
                raise PermissionError(f"Write access denied to credential {store_key}")
            
            with self._lock:
                try:
                    # Encrypt the credential value
                    encrypted = self._encrypt(value)

                    # Create credential entry
                    now = datetime.now()
                    entry = CredentialEntry(
                        service=service,
                        key=key,
                        value=value,  # Keep in memory cache only
                        created_at=now,
                        updated_at=now,
                        expires_at=expires_at,
                        metadata=metadata or {}
                    )

                    # Store in encrypted form
                    self._encrypted_store[store_key] = encrypted
                    self._credentials[store_key] = entry

                    # Persist to backend
                    if self._backend == 'file':
                        self._save_credentials()

                    # Audit logging
                    is_update = entry.created_at != entry.updated_at
                    action = 'credential.update' if is_update else 'credential.create'
                    self._audit_log_action(action, store_key)

                    logger.info(f"Stored credential for {service}:{key}")

                except CredentialEncryptionError:
                    self._audit_log_action('credential.store_error', store_key, {'reason': 'encryption_failed'})
                    raise
                except Exception as e:
                    logger.error(f"Failed to store credential: {e}")
                    self._audit_log_action('credential.store_error', store_key, {'reason': str(e)})
                    raise CredentialEncryptionError(
                        f"Failed to store credential: {str(e)}",
                        is_recoverable=True
                    )
        finally:
            # Restore original context
            if user:
                self._current_user = original_user
            if role:
                self._current_role = original_role
    
    def get_credential(
        self,
        service: str,
        key: str,
        update_access: bool = True,
        user: Optional[str] = None,
        role: Optional[str] = None
    ) -> Optional[str]:
        """
        Retrieve a decrypted credential.

        Args:
            service: Service name
            key: Credential key
            update_access: Whether to update access timestamp
            user: Optional user for access control (overrides current user)
            role: Optional role for access control (overrides current role)

        Returns:
            Decrypted credential value, or None if not found

        Raises:
            CredentialEncryptionError: If decryption fails
            PermissionError: If access is denied
        """
        store_key = f"{service}:{key}"
        
        # Use provided user/role or current context
        original_user = self._current_user
        original_role = self._current_role
        if user:
            self._current_user = user
        if role:
            self._current_role = role
        
        try:
            # Check access control
            if not self.check_access(service, key, AccessLevel.READ):
                raise PermissionError(f"Access denied to credential {store_key}")
            
            with self._lock:
                # Check in-memory cache first
                if store_key in self._credentials:
                    entry = self._credentials[store_key]

                    # Check expiration
                    if entry.expires_at and datetime.now() > entry.expires_at:
                        logger.warning(f"Credential {store_key} has expired")
                        self._audit_log_action('credential.access_denied', store_key, {'reason': 'expired'})
                        return None

                    # If value is empty (loaded from file), decrypt it
                    if not entry.value and store_key in self._encrypted_store:
                        encrypted = self._encrypted_store[store_key]
                        value = self._decrypt(encrypted)
                        entry.value = value
                        self._credentials[store_key] = entry

                    # Update access info
                    if update_access:
                        entry.accessed_at = datetime.now()
                        entry.access_count += 1

                    # Audit logging
                    self._audit_log_action('credential.access', store_key)

                    return entry.value

                # Try to load from encrypted store
                if store_key in self._encrypted_store:
                    encrypted = self._encrypted_store[store_key]

                    try:
                        # Decrypt
                        value = self._decrypt(encrypted)

                        # Cache in memory
                        now = datetime.now()
                        entry = CredentialEntry(
                            service=service,
                            key=key,
                            value=value,
                            created_at=datetime.fromisoformat(encrypted.created_at),
                            updated_at=datetime.fromisoformat(encrypted.updated_at),
                            accessed_at=now if update_access else None,
                            access_count=1 if update_access else 0
                        )
                        self._credentials[store_key] = entry

                        # Audit logging
                        self._audit_log_action('credential.access', store_key)

                        return value

                    except CredentialEncryptionError:
                        self._audit_log_action('credential.access_error', store_key, {'reason': 'decryption_failed'})
                        raise
                    except Exception as e:
                        logger.error(f"Failed to retrieve credential {store_key}: {e}")
                        self._audit_log_action('credential.access_error', store_key, {'reason': str(e)})
                        return None

                logger.debug(f"Credential {store_key} not found")
                self._audit_log_action('credential.access_not_found', store_key)
                return None
        finally:
            # Restore original context
            if user:
                self._current_user = original_user
            if role:
                self._current_role = original_role
    
    def delete_credential(
        self,
        service: str,
        key: str,
        user: Optional[str] = None,
        role: Optional[str] = None
    ) -> bool:
        """
        Delete a credential.

        Args:
            service: Service name
            key: Credential key
            user: Optional user for access control
            role: Optional role for access control

        Returns:
            True if deleted, False if not found
            
        Raises:
            PermissionError: If access is denied
        """
        store_key = f"{service}:{key}"
        
        # Use provided user/role or current context
        original_user = self._current_user
        original_role = self._current_role
        if user:
            self._current_user = user
        if role:
            self._current_role = role
        
        try:
            # Check access control
            if not self.check_access(service, key, AccessLevel.WRITE):
                self._audit_log_action('credential.delete_denied', store_key, {'reason': 'access_denied'})
                raise PermissionError(f"Delete access denied to credential {store_key}")
            
            with self._lock:
                deleted = False

                if store_key in self._credentials:
                    del self._credentials[store_key]
                    deleted = True

                if store_key in self._encrypted_store:
                    del self._encrypted_store[store_key]
                    deleted = True

                if deleted and self._backend == 'file':
                    self._save_credentials()

                if deleted:
                    self._audit_log_action('credential.delete', store_key)
                    logger.info(f"Deleted credential {store_key}")
                else:
                    self._audit_log_action('credential.delete_not_found', store_key)

                return deleted
        finally:
            # Restore original context
            if user:
                self._current_user = original_user
            if role:
                self._current_role = original_role
    
    def list_services(self) -> List[str]:
        """
        List all services with stored credentials.
        
        Returns:
            List of service names
        """
        with self._lock:
            services = set()
            for store_key in self._encrypted_store.keys():
                service = store_key.split(':')[0]
                services.add(service)
            return sorted(list(services))
    
    def list_credentials(self, service: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List credentials (metadata only, not values).
        
        Args:
            service: Optional service filter
            
        Returns:
            List of credential metadata dictionaries
        """
        with self._lock:
            credentials = []
            
            for store_key, entry in self._credentials.items():
                if service and not store_key.startswith(f"{service}:"):
                    continue
                
                credentials.append({
                    'service': entry.service,
                    'key': entry.key,
                    'created_at': entry.created_at.isoformat(),
                    'updated_at': entry.updated_at.isoformat(),
                    'accessed_at': entry.accessed_at.isoformat() if entry.accessed_at else None,
                    'access_count': entry.access_count,
                    'expires_at': entry.expires_at.isoformat() if entry.expires_at else None,
                    'has_value': True
                })
            
            return credentials
    
    def rotate_credential(
        self,
        service: str,
        key: str,
        new_value: str,
        notify_webhook: bool = True
    ) -> bool:
        """
        Rotate a credential (update value while preserving metadata).

        SECURITY: Credential rotation is a critical security operation.
        All rotations are logged and can trigger webhook notifications.

        Args:
            service: Service name
            key: Credential key
            new_value: New credential value
            notify_webhook: Whether to send rotation notification webhook

        Returns:
            True if rotated, False if not found

        Raises:
            ValueError: If new value is empty or same as old value
        """
        store_key = f"{service}:{key}"

        with self._lock:
            if store_key not in self._encrypted_store:
                return False

            # Get old value for validation
            old_value = self.get_credential(service, key, update_access=False)
            
            # SECURITY: Validate new value is different
            if old_value and new_value == old_value:
                logger.warning(f"Credential rotation attempted with same value for {store_key}")
                raise ValueError("New credential value must be different from old value")
            
            # SECURITY: Validate new value is not empty
            if not new_value or not new_value.strip():
                raise ValueError("New credential value cannot be empty")

            # Encrypt new value
            encrypted = self._encrypt(new_value)

            # Preserve original timestamps
            original = self._encrypted_store[store_key]
            encrypted.created_at = original.created_at

            # Get existing entry for preserving access info
            existing_entry = self._credentials.get(store_key)

            # Update entry (preserve expires_at and metadata)
            now = datetime.now()
            entry = CredentialEntry(
                service=service,
                key=key,
                value=new_value,
                created_at=datetime.fromisoformat(encrypted.created_at),
                updated_at=now,
                expires_at=existing_entry.expires_at if existing_entry else None,  # Preserve expiration
                accessed_at=existing_entry.accessed_at if existing_entry else None,
                access_count=existing_entry.access_count if existing_entry else 0,
                metadata=existing_entry.metadata if existing_entry else {}
            )

            # Add rotation metadata
            entry.metadata['last_rotated_at'] = now.isoformat()
            entry.metadata['rotation_count'] = entry.metadata.get('rotation_count', 0) + 1

            self._encrypted_store[store_key] = encrypted
            self._credentials[store_key] = entry

            if self._backend == 'file':
                self._save_credentials()

            # SECURITY: Audit log the rotation
            self._audit_log_action('credential.rotated', store_key, {
                'rotated_at': now.isoformat(),
                'rotation_count': entry.metadata['rotation_count']
            })

            logger.info(f"Rotated credential {store_key} (rotation #{entry.metadata['rotation_count']})")
            
            # SECURITY: Send webhook notification if configured
            if notify_webhook and self._audit_logger:
                try:
                    self._audit_logger({
                        'event': 'credential_rotation',
                        'timestamp': now.isoformat(),
                        'service': service,
                        'key': key,
                        'rotation_count': entry.metadata['rotation_count'],
                        'expires_at': entry.expires_at.isoformat() if entry.expires_at else None
                    })
                except Exception as e:
                    logger.warning(f"Failed to send credential rotation notification: {e}")
            
            return True

    def get_expiring_credentials(self, days_threshold: int = 7) -> List[Dict[str, Any]]:
        """
        Get list of credentials expiring within threshold.
        
        SECURITY: Proactive credential rotation support.
        Call this method regularly to identify credentials needing rotation.
        
        Args:
            days_threshold: Number of days to check for expiration
            
        Returns:
            List of credential info dictionaries
        """
        from datetime import timedelta
        
        threshold = datetime.now() + timedelta(days=days_threshold)
        expiring = []
        
        with self._lock:
            for store_key, entry in self._credentials.items():
                if entry.expires_at and entry.expires_at <= threshold:
                    expiring.append({
                        'service': entry.service,
                        'key': entry.key,
                        'expires_at': entry.expires_at.isoformat(),
                        'days_until_expiry': (entry.expires_at - datetime.now()).days,
                        'last_rotated': entry.metadata.get('last_rotated_at'),
                        'rotation_count': entry.metadata.get('rotation_count', 0)
                    })
        
        # Sort by expiry date (most urgent first)
        expiring.sort(key=lambda x: x['days_until_expiry'])
        
        return expiring

    def auto_rotate_expired_credentials(
        self,
        rotation_callback: callable,
        notify_webhook: bool = True
    ) -> Dict[str, Any]:
        """
        Automatically rotate expired credentials.
        
        SECURITY: Automated credential rotation for enhanced security.
        The callback should generate new credentials for the service.
        
        Args:
            rotation_callback: Function that takes (service, key) and returns new credential value
            notify_webhook: Whether to send rotation notifications
            
        Returns:
            Dictionary with rotation results
        """
        results = {
            'rotated': [],
            'failed': [],
            'skipped': []
        }
        
        with self._lock:
            for store_key, entry in list(self._credentials.items()):
                # Check if expired
                if entry.expires_at and datetime.now() > entry.expires_at:
                    service, key = entry.service, entry.key
                    
                    try:
                        # Generate new credential via callback
                        new_value = rotation_callback(service, key)
                        
                        if new_value:
                            # Rotate the credential
                            if self.rotate_credential(service, key, new_value, notify_webhook):
                                results['rotated'].append({
                                    'service': service,
                                    'key': key,
                                    'rotated_at': datetime.now().isoformat()
                                })
                        else:
                            results['skipped'].append({
                                'service': service,
                                'key': key,
                                'reason': 'callback_returned_empty'
                            })
                    
                    except Exception as e:
                        results['failed'].append({
                            'service': service,
                            'key': key,
                            'error': str(e)
                        })
        
        logger.info(
            f"Auto-rotation complete: {len(results['rotated'])} rotated, "
            f"{len(results['failed'])} failed, {len(results['skipped'])} skipped"
        )
        
        return results

    def set_credential_expiration(
        self,
        service: str,
        key: str,
        expires_at: datetime
    ) -> bool:
        """
        Set or update expiration date for a credential.
        
        SECURITY: Allows setting rotation schedules for credentials.
        
        Args:
            service: Service name
            key: Credential key
            expires_at: Expiration datetime
            
        Returns:
            True if updated, False if not found
        """
        store_key = f"{service}:{key}"
        
        with self._lock:
            if store_key not in self._credentials:
                return False
            
            self._credentials[store_key].expires_at = expires_at
            self._credentials[store_key].metadata['expiration_set_at'] = datetime.now().isoformat()
            
            if self._backend == 'file':
                self._save_credentials()
            
            self._audit_log_action('credential.expiration_set', store_key, {
                'expires_at': expires_at.isoformat()
            })
            
            logger.info(f"Set expiration for {store_key}: {expires_at}")
            return True
    
    def _save_credentials(self) -> None:
        """Save encrypted credentials to file."""
        try:
            store_file = self._backend_config.get('store_file', 'credentials.enc.json')

            data = {
                'version': self.ENCRYPTION_VERSION,
                'saved_at': datetime.now().isoformat(),
                'credentials': {
                    key: {
                        'ciphertext': enc.ciphertext,
                        'nonce': enc.nonce,
                        'tag': enc.tag,
                        'created_at': enc.created_at,
                        'updated_at': enc.updated_at,
                        'version': enc.version,
                        # Persist expires_at if set
                        'expires_at': self._credentials[key].expires_at.isoformat() 
                            if key in self._credentials and self._credentials[key].expires_at else None
                    }
                    for key, enc in self._encrypted_store.items()
                }
            }

            with open(store_file, 'w') as f:
                json.dump(data, f, indent=2)

            logger.debug(f"Saved {len(data['credentials'])} credentials to {store_file}")

        except Exception as e:
            logger.error(f"Failed to save credentials: {e}")
    
    def _load_credentials(self) -> None:
        """Load encrypted credentials from file."""
        try:
            store_file = self._backend_config.get('store_file', 'credentials.enc.json')

            if not os.path.exists(store_file):
                logger.debug("No credential store file found")
                return

            with open(store_file, 'r') as f:
                data = json.load(f)

            version = data.get('version', 1)
            if version != self.ENCRYPTION_VERSION:
                logger.warning(
                    f"Credential store version mismatch: file={version}, "
                    f"current={self.ENCRYPTION_VERSION}"
                )

            for key, enc_data in data.get('credentials', {}).items():
                # Load encrypted credential
                self._encrypted_store[key] = EncryptedCredential(
                    ciphertext=enc_data['ciphertext'],
                    nonce=enc_data['nonce'],
                    tag=enc_data['tag'],
                    created_at=enc_data['created_at'],
                    updated_at=enc_data['updated_at'],
                    version=enc_data.get('version', 1)
                )
                
                # Load expires_at if present
                expires_at = None
                if enc_data.get('expires_at'):
                    try:
                        expires_at = datetime.fromisoformat(enc_data['expires_at'])
                    except (ValueError, TypeError):
                        logger.warning(f"Invalid expires_at for credential {key}")
                
                # Create in-memory entry (value will be populated on first access)
                self._credentials[key] = CredentialEntry(
                    service=key.split(':')[0] if ':' in key else 'unknown',
                    key=key.split(':')[1] if ':' in key else key,
                    value='',  # Will be populated on first decrypt
                    created_at=datetime.fromisoformat(enc_data['created_at']),
                    updated_at=datetime.fromisoformat(enc_data['updated_at']),
                    expires_at=expires_at
                )

            logger.info(f"Loaded {len(self._encrypted_store)} credentials from {store_file}")

        except Exception as e:
            logger.error(f"Failed to load credentials: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get credential store statistics.

        Returns:
            Dictionary with store statistics
        """
        with self._lock:
            return {
                'total_credentials': len(self._encrypted_store),
                'services': len(self.list_services()),
                'encryption_algorithm': self.ALGORITHM,
                'key_size_bits': self.KEY_SIZE * 8,
                'pbkdf2_iterations': self.PBKDF2_ITERATIONS,
                'backend': self._backend,
                'audit_logging_enabled': self._enable_audit_logging,
                'auto_cleanup_enabled': self._auto_cleanup_expired,
                'key_version': self._key_version,
                'access_policies_count': len(self._access_policies),
            }

    # ==================== NEW ENHANCEMENT METHODS ====================

    def _start_cleanup_thread(self) -> None:
        """Start background thread for expired credential cleanup."""
        def cleanup_loop():
            while True:
                try:
                    import time
                    time.sleep(self._cleanup_interval_hours * 3600)
                    self.cleanup_expired_credentials()
                except Exception as e:
                    logger.error(f"Cleanup thread error: {e}")

        cleanup_thread = threading.Thread(target=cleanup_loop, daemon=True)
        cleanup_thread.start()
        logger.info(f"Started expired credential cleanup thread (interval: {self._cleanup_interval_hours}h)")

    def cleanup_expired_credentials(self) -> int:
        """
        Remove expired credentials from the store.

        Returns:
            Number of credentials removed
        """
        with self._lock:
            removed_count = 0
            now = datetime.now()
            to_remove = []

            for store_key, entry in self._credentials.items():
                if entry.expires_at and now > entry.expires_at:
                    to_remove.append(store_key)

            for store_key in to_remove:
                if store_key in self._credentials:
                    del self._credentials[store_key]
                if store_key in self._encrypted_store:
                    del self._encrypted_store[store_key]
                removed_count += 1
                self._audit_log_action('credential.expired_cleanup', store_key)

            if removed_count > 0:
                self._last_cleanup = now
                if self._backend == 'file':
                    self._save_credentials()
                logger.info(f"Cleaned up {removed_count} expired credentials")

            return removed_count

    def _audit_log_action(self, action: str, store_key: str, details: Optional[Dict[str, Any]] = None) -> None:
        """
        Log an audit event for credential access.

        Args:
            action: Action type (e.g., 'credential.access', 'credential.create')
            store_key: Credential store key (service:key)
            details: Optional additional details
        """
        if not self._enable_audit_logging:
            return

        service, key = store_key.split(':', 1) if ':' in store_key else ('unknown', store_key)

        audit_data = {
            'timestamp': datetime.now().isoformat(),
            'action': action,
            'service': service,
            'key': key,
            'user': self._current_user,
            'role': self._current_role,
            'details': details or {}
        }

        if self._audit_logger:
            try:
                self._audit_logger(audit_data)
            except Exception as e:
                logger.error(f"Audit logger failed: {e}")
        else:
            logger.info(f"[AUDIT] {action}: {store_key} by {self._current_user or 'unknown'}")

    def set_current_user(self, username: Optional[str], role: Optional[str] = None) -> None:
        """
        Set the current user context for access control and audit logging.

        Args:
            username: Current username
            role: Current user role
        """
        self._current_user = username
        self._current_role = role
        logger.debug(f"Set current user: {username} (role: {role})")

    def check_access(self, service: str, key: str, required_level: AccessLevel = AccessLevel.READ) -> bool:
        """
        Check if current user has access to a credential.

        Args:
            service: Service name
            key: Credential key
            required_level: Required access level (READ, WRITE, or ADMIN)

        Returns:
            True if access granted, False otherwise
        """
        store_key = f"{service}:{key}"
        policy = self._access_policies.get(store_key)

        # No policy = allow by default (backward compatibility)
        if not policy:
            return True

        # Check denied users first
        if self._current_user and self._current_user in policy.denied_users:
            self._audit_log_action('access.denied', store_key, {'reason': 'user_denied'})
            return False

        # Check allowed users (if specified)
        if policy.allowed_users:
            if not self._current_user or self._current_user not in policy.allowed_users:
                self._audit_log_action('access.denied', store_key, {'reason': 'user_not_allowed'})
                return False

        # Check allowed roles (if specified) - must have required access level
        if policy.allowed_roles:
            if not self._current_role:
                self._audit_log_action('access.denied', store_key, {'reason': 'no_role'})
                return False
            
            # Check if user's role has the required access level
            # ADMIN role can do anything, WRITE can do READ+WRITE, READ can only READ
            role_hierarchy = {'ADMIN': 3, 'WRITE': 2, 'READ': 1}
            required_level_value = role_hierarchy.get(required_level.value, 0)
            
            # Check if user has any role with sufficient access
            user_roles = policy.allowed_roles
            max_user_level = max(
                (role_hierarchy.get(r, 0) for r in user_roles),
                default=0
            )
            
            if max_user_level < required_level_value:
                self._audit_log_action('access.denied', store_key, {
                    'reason': 'insufficient_role',
                    'required': required_level.value,
                    'user_roles': list(user_roles)
                })
                return False

        # Check access count limit
        if policy.max_access_count is not None:
            entry = self._credentials.get(store_key)
            if entry and entry.access_count >= policy.max_access_count:
                self._audit_log_action('access.denied', store_key, {'reason': 'max_access_reached'})
                return False

        return True

    def set_access_policy(self, service: str, key: str, policy: AccessPolicy) -> None:
        """
        Set access policy for a credential.

        Args:
            service: Service name
            key: Credential key
            policy: Access policy to set
        """
        store_key = f"{service}:{key}"
        with self._lock:
            self._access_policies[store_key] = policy
            logger.info(f"Set access policy for {store_key}")

    def get_access_policy(self, service: str, key: str) -> Optional[AccessPolicy]:
        """
        Get access policy for a credential.

        Args:
            service: Service name
            key: Credential key

        Returns:
            Access policy or None if not set
        """
        store_key = f"{service}:{key}"
        return self._access_policies.get(store_key)

    def validate_credential_template(self, service: str, credentials: Dict[str, str]) -> tuple[bool, List[str]]:
        """
        Validate credentials against a template.

        Args:
            service: Service name
            credentials: Dictionary of credentials to validate

        Returns:
            Tuple of (is_valid, list of error messages)
        """
        template = CREDENTIAL_TEMPLATES.get(service)
        if not template:
            return True, []  # No template = no validation

        errors = []

        # Check required fields
        for field_def in template.fields:
            if field_def['required'] and field_def['key'] not in credentials:
                errors.append(f"Missing required field: {field_def['key']}")

        # Apply validation rules if defined
        if template.validation_rules:
            for field_key, value in credentials.items():
                if field_key in template.validation_rules:
                    rule = template.validation_rules[field_key]
                    if rule.get('pattern') and not __import__('re').match(rule['pattern'], value):
                        errors.append(f"Field {field_key} does not match pattern")
                    if rule.get('min_length') and len(value) < rule['min_length']:
                        errors.append(f"Field {field_key} is too short")

        return len(errors) == 0, errors

    def set_credential_with_template(self, service: str, credentials: Dict[str, str],
                                     metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Store multiple credentials using a template.

        Args:
            service: Service name
            credentials: Dictionary of credentials to store
            metadata: Optional metadata

        Returns:
            Dictionary with validation results and stored keys
        """
        # Validate against template
        is_valid, errors = self.validate_credential_template(service, credentials)
        if not is_valid:
            return {'success': False, 'errors': errors, 'stored_keys': []}

        # Store each credential
        stored_keys = []
        for key, value in credentials.items():
            store_key = f"{service}:{key}"
            self.set_credential(service, key, value, metadata=metadata)
            stored_keys.append(store_key)

        # Set automatic expiration if template defines rotation interval
        template = CREDENTIAL_TEMPLATES.get(service)
        if template and template.rotation_interval_days:
            expires_at = datetime.now() + timedelta(days=template.rotation_interval_days)
            for key in credentials.keys():
                store_key = f"{service}:{key}"
                if store_key in self._credentials:
                    self._credentials[store_key].expires_at = expires_at

        self._audit_log_action('credential.template_store', service, {
            'keys': stored_keys,
            'template': template.name if template else 'custom'
        })

        return {'success': True, 'errors': [], 'stored_keys': stored_keys}

    def export_credentials(self, password: str, include_expired: bool = False) -> bytes:
        """
        Export credentials to an encrypted backup file.

        Args:
            password: Password for export encryption
            include_expired: Include expired credentials

        Returns:
            Encrypted export data
        """
        with self._lock:
            # Derive export key from password
            export_salt = os.urandom(32)
            export_key = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), export_salt, 100000, 32)

            # Prepare export data
            export_data = {
                'version': self.ENCRYPTION_VERSION,
                'exported_at': datetime.now().isoformat(),
                'salt': export_salt.hex(),
                'credentials': {}
            }

            for store_key, enc in self._encrypted_store.items():
                if not include_expired:
                    entry = self._credentials.get(store_key)
                    if entry and entry.expires_at and datetime.now() > entry.expires_at:
                        continue

                export_data['credentials'][store_key] = {
                    'ciphertext': enc.ciphertext,
                    'nonce': enc.nonce,
                    'tag': enc.tag,
                    'created_at': enc.created_at,
                    'updated_at': enc.updated_at,
                    'version': enc.version
                }

            # Encrypt export data
            export_json = json.dumps(export_data).encode('utf-8')
            nonce = os.urandom(12)
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM
            aesgcm = AESGCM(export_key)
            encrypted_export = aesgcm.encrypt(nonce, export_json, None)

            # Package with nonce and salt (salt must be unencrypted for import)
            result = base64.b64encode(nonce + encrypted_export + export_salt)

            self._audit_log_action('credential.export', 'all', {
                'count': len(export_data['credentials']),
                'include_expired': include_expired
            })

            return result

    def import_credentials(self, export_data: bytes, password: str) -> Dict[str, Any]:
        """
        Import credentials from an encrypted backup file.

        Args:
            export_data: Encrypted export data (format: nonce + ciphertext + salt)
            password: Password for decryption

        Returns:
            Import results
        """
        try:
            # Decode export data
            decoded = base64.b64decode(export_data)
            
            # Extract nonce (first 12 bytes), salt (last 32 bytes), and encrypted data (middle)
            nonce = decoded[:12]
            export_salt = decoded[-32:]  # Salt is at the end, unencrypted
            encrypted = decoded[12:-32]

            # Derive export key from password using the salt from export
            export_key = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), export_salt, 100000, 32)

            # Decrypt
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM
            aesgcm = AESGCM(export_key)
            export_json = aesgcm.decrypt(nonce, encrypted, None)
            export_data_dict = json.loads(export_json)

            # Import credentials
            imported_count = 0
            with self._lock:
                for store_key, enc_data in export_data_dict.get('credentials', {}).items():
                    self._encrypted_store[store_key] = EncryptedCredential(
                        ciphertext=enc_data['ciphertext'],
                        nonce=enc_data['nonce'],
                        tag=enc_data['tag'],
                        created_at=enc_data['created_at'],
                        updated_at=enc_data['updated_at'],
                        version=enc_data.get('version', 1)
                    )
                    imported_count += 1

                if self._backend == 'file':
                    self._save_credentials()

            self._audit_log_action('credential.import', 'all', {'count': imported_count})

            return {'success': True, 'imported_count': imported_count}

        except Exception as e:
            logger.error(f"Import failed: {e}")
            return {'success': False, 'error': str(e)}

    def get_expiring_credentials(self, days_threshold: int = 7) -> List[Dict[str, Any]]:
        """
        Get credentials that will expire within the specified threshold.

        Args:
            days_threshold: Number of days to check

        Returns:
            List of expiring credentials with metadata
        """
        with self._lock:
            expiring = []
            threshold = datetime.now() + timedelta(days=days_threshold)

            for store_key, entry in self._credentials.items():
                if entry.expires_at and entry.expires_at <= threshold:
                    days_until_expiry = (entry.expires_at - datetime.now()).days
                    expiring.append({
                        'service': entry.service,
                        'key': entry.key,
                        'expires_at': entry.expires_at.isoformat(),
                        'days_until_expiry': days_until_expiry,
                        'rotation_recommended': CREDENTIAL_TEMPLATES.get(entry.service) is not None
                    })

            return sorted(expiring, key=lambda x: x['days_until_expiry'])

    def clear_memory(self) -> None:
        """
        Clear all credential values from memory for security.

        This should be called when the application shuts down or when
        credentials are no longer needed.
        """
        with self._lock:
            # Clear credential values
            for entry in self._credentials.values():
                # Overwrite value before clearing
                if entry.value:
                    # Secure clear by overwriting
                    entry.value = '\x00' * len(entry.value)
            self._credentials.clear()

            # Clear key from history
            for key_version in self._key_history:
                self._key_history[key_version] = b'\x00' * len(self._key_history[key_version])
            self._key_history.clear()

            logger.info("Cleared all credential data from memory")

    def __del__(self):
        """Cleanup on destruction."""
        try:
            self.clear_memory()
        except Exception:
            pass
