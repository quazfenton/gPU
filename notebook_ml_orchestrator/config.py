"""
Configuration management for the Notebook ML Orchestrator.

This module provides centralized configuration management with environment
variable support and validation.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import dataclass, field


@dataclass
class DatabaseConfig:
    """Database configuration settings."""
    path: str = "orchestrator.db"
    timeout: float = 30.0
    max_connections: int = 10
    enable_wal: bool = True
    
    @classmethod
    def from_env(cls) -> 'DatabaseConfig':
        """Create database config from environment variables."""
        return cls(
            path=os.getenv('ORCHESTRATOR_DB_PATH', cls.path),
            timeout=float(os.getenv('ORCHESTRATOR_DB_TIMEOUT', cls.timeout)),
            max_connections=int(os.getenv('ORCHESTRATOR_DB_MAX_CONNECTIONS', cls.max_connections)),
            enable_wal=os.getenv('ORCHESTRATOR_DB_ENABLE_WAL', 'true').lower() == 'true'
        )


@dataclass
class LoggingConfig:
    """Logging configuration settings."""
    level: str = "INFO"
    file_path: Optional[str] = "logs/orchestrator.log"
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5
    enable_console: bool = True
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    @classmethod
    def from_env(cls) -> 'LoggingConfig':
        """Create logging config from environment variables."""
        return cls(
            level=os.getenv('ORCHESTRATOR_LOG_LEVEL', cls.level),
            file_path=os.getenv('ORCHESTRATOR_LOG_FILE'),
            max_file_size=int(os.getenv('ORCHESTRATOR_LOG_MAX_SIZE', cls.max_file_size)),
            backup_count=int(os.getenv('ORCHESTRATOR_LOG_BACKUP_COUNT', cls.backup_count)),
            enable_console=os.getenv('ORCHESTRATOR_LOG_CONSOLE', 'true').lower() == 'true'
        )


@dataclass
class JobQueueConfig:
    """Job queue configuration settings."""
    max_retries: int = 3
    base_retry_delay: float = 1.0
    max_retry_delay: float = 300.0
    exponential_base: float = 2.0
    cleanup_interval_hours: int = 24
    cleanup_age_days: int = 30
    
    @classmethod
    def from_env(cls) -> 'JobQueueConfig':
        """Create job queue config from environment variables."""
        return cls(
            max_retries=int(os.getenv('ORCHESTRATOR_MAX_RETRIES', cls.max_retries)),
            base_retry_delay=float(os.getenv('ORCHESTRATOR_BASE_RETRY_DELAY', cls.base_retry_delay)),
            max_retry_delay=float(os.getenv('ORCHESTRATOR_MAX_RETRY_DELAY', cls.max_retry_delay)),
            exponential_base=float(os.getenv('ORCHESTRATOR_EXPONENTIAL_BASE', cls.exponential_base)),
            cleanup_interval_hours=int(os.getenv('ORCHESTRATOR_CLEANUP_INTERVAL', cls.cleanup_interval_hours)),
            cleanup_age_days=int(os.getenv('ORCHESTRATOR_CLEANUP_AGE_DAYS', cls.cleanup_age_days))
        )


@dataclass
class BatchProcessingConfig:
    """Batch processing configuration settings."""
    max_parallel_items: int = 4
    max_batch_size: int = 1000
    progress_update_interval: float = 1.0
    
    @classmethod
    def from_env(cls) -> 'BatchProcessingConfig':
        """Create batch processing config from environment variables."""
        return cls(
            max_parallel_items=int(os.getenv('ORCHESTRATOR_MAX_PARALLEL_ITEMS', cls.max_parallel_items)),
            max_batch_size=int(os.getenv('ORCHESTRATOR_MAX_BATCH_SIZE', cls.max_batch_size)),
            progress_update_interval=float(os.getenv('ORCHESTRATOR_PROGRESS_INTERVAL', cls.progress_update_interval))
        )


@dataclass
class SecurityConfig:
    """Security configuration settings."""
    enable_encryption: bool = False  # Disabled by default for development
    secret_key: Optional[str] = None
    token_expiry_hours: int = 24
    max_file_size_mb: int = 100
    allowed_file_types: list = field(default_factory=lambda: ['.jpg', '.png', '.wav', '.mp3', '.txt', '.pdf'])
    
    @classmethod
    def from_env(cls) -> 'SecurityConfig':
        """Create security config from environment variables."""
        allowed_types = os.getenv('ORCHESTRATOR_ALLOWED_FILE_TYPES', '')
        if allowed_types:
            allowed_types = [t.strip() for t in allowed_types.split(',')]
        else:
            allowed_types = ['.jpg', '.png', '.wav', '.mp3', '.txt', '.pdf']
            
        return cls(
            enable_encryption=os.getenv('ORCHESTRATOR_ENABLE_ENCRYPTION', 'false').lower() == 'true',
            secret_key=os.getenv('ORCHESTRATOR_SECRET_KEY'),
            token_expiry_hours=int(os.getenv('ORCHESTRATOR_TOKEN_EXPIRY', str(cls.__dataclass_fields__['token_expiry_hours'].default))),
            max_file_size_mb=int(os.getenv('ORCHESTRATOR_MAX_FILE_SIZE', str(cls.__dataclass_fields__['max_file_size_mb'].default))),
            allowed_file_types=allowed_types
        )


@dataclass
class OrchestratorConfig:
    """Main orchestrator configuration."""
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    job_queue: JobQueueConfig = field(default_factory=JobQueueConfig)
    batch_processing: BatchProcessingConfig = field(default_factory=BatchProcessingConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    
    # Environment settings
    environment: str = "development"
    debug: bool = False
    
    @classmethod
    def from_env(cls) -> 'OrchestratorConfig':
        """Create configuration from environment variables."""
        return cls(
            database=DatabaseConfig.from_env(),
            logging=LoggingConfig.from_env(),
            job_queue=JobQueueConfig.from_env(),
            batch_processing=BatchProcessingConfig.from_env(),
            security=SecurityConfig.from_env(),
            environment=os.getenv('ORCHESTRATOR_ENVIRONMENT', 'development'),
            debug=os.getenv('ORCHESTRATOR_DEBUG', 'false').lower() == 'true'
        )
    
    def validate(self) -> bool:
        """Validate configuration settings."""
        errors = []
        
        # Validate database path
        if not self.database.path:
            errors.append("Database path cannot be empty")
        
        # Validate logging level
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if self.logging.level.upper() not in valid_levels:
            errors.append(f"Invalid logging level: {self.logging.level}")
        
        # Validate retry settings
        if self.job_queue.max_retries < 0:
            errors.append("Max retries cannot be negative")
        
        if self.job_queue.base_retry_delay <= 0:
            errors.append("Base retry delay must be positive")
        
        # Validate batch processing settings
        if self.batch_processing.max_parallel_items <= 0:
            errors.append("Max parallel items must be positive")
        
        if self.batch_processing.max_batch_size <= 0:
            errors.append("Max batch size must be positive")
        
        # Validate security settings
        if self.security.enable_encryption and not self.security.secret_key:
            errors.append("Secret key required when encryption is enabled")
        
        if errors:
            raise ValueError(f"Configuration validation failed: {'; '.join(errors)}")
        
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'database': {
                'path': self.database.path,
                'timeout': self.database.timeout,
                'max_connections': self.database.max_connections,
                'enable_wal': self.database.enable_wal
            },
            'logging': {
                'level': self.logging.level,
                'file_path': self.logging.file_path,
                'max_file_size': self.logging.max_file_size,
                'backup_count': self.logging.backup_count,
                'enable_console': self.logging.enable_console
            },
            'job_queue': {
                'max_retries': self.job_queue.max_retries,
                'base_retry_delay': self.job_queue.base_retry_delay,
                'max_retry_delay': self.job_queue.max_retry_delay,
                'exponential_base': self.job_queue.exponential_base
            },
            'batch_processing': {
                'max_parallel_items': self.batch_processing.max_parallel_items,
                'max_batch_size': self.batch_processing.max_batch_size,
                'progress_update_interval': self.batch_processing.progress_update_interval
            },
            'security': {
                'enable_encryption': self.security.enable_encryption,
                'token_expiry_hours': self.security.token_expiry_hours,
                'max_file_size_mb': self.security.max_file_size_mb,
                'allowed_file_types': self.security.allowed_file_types
            },
            'environment': self.environment,
            'debug': self.debug
        }


# Global configuration instance
_config: Optional[OrchestratorConfig] = None


def get_config() -> OrchestratorConfig:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = OrchestratorConfig.from_env()
        _config.validate()
    return _config


def set_config(config: OrchestratorConfig):
    """Set the global configuration instance."""
    global _config
    config.validate()
    _config = config


def reset_config():
    """Reset the global configuration instance."""
    global _config
    _config = None