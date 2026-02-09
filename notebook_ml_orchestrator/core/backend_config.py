"""
Backend configuration management for the Notebook ML Orchestrator.

This module handles loading, validating, and managing backend configurations
from environment variables and configuration files.
"""

import os
import yaml
import threading
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from pathlib import Path
import re

from .models import BackendType
from .exceptions import ConfigurationError
from .logging_config import LoggerMixin


# Required credentials for each backend type
REQUIRED_CREDENTIALS = {
    BackendType.MODAL: ["token_id", "token_secret"],
    BackendType.HUGGINGFACE: ["token"],
    BackendType.KAGGLE: ["username", "key"],
    BackendType.COLAB: ["client_id", "client_secret", "refresh_token"],
}


@dataclass
class BackendConfig:
    """Configuration for a specific backend."""
    backend_type: BackendType
    enabled: bool
    credentials: Dict[str, str] = field(default_factory=dict)
    options: Dict[str, Any] = field(default_factory=dict)
    
    def validate(self) -> bool:
        """
        Validate that required credentials are present.
        
        Returns:
            True if valid, False otherwise
        """
        if not self.enabled:
            return True
        
        required_creds = REQUIRED_CREDENTIALS.get(self.backend_type, [])
        return all(key in self.credentials and self.credentials[key] for key in required_creds)
    
    def get_missing_credentials(self) -> list:
        """
        Get list of missing required credentials.
        
        Returns:
            List of missing credential keys
        """
        required_creds = REQUIRED_CREDENTIALS.get(self.backend_type, [])
        return [key for key in required_creds if key not in self.credentials or not self.credentials[key]]


@dataclass
class RoutingConfig:
    """Configuration for routing behavior."""
    strategy: str = "cost-optimized"  # or "round-robin", "least-loaded"
    prefer_free_tier: bool = True
    health_check_interval: int = 300  # seconds
    max_retries: int = 3
    retry_backoff_base: float = 2.0  # exponential backoff multiplier


class BackendConfigManager(LoggerMixin):
    """Manages backend configurations with hot-reload support."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the configuration manager.
        
        Args:
            config_path: Path to configuration file (optional)
        """
        self.config_path = config_path or self._get_default_config_path()
        self.backend_configs: Dict[BackendType, BackendConfig] = {}
        self.routing_config = RoutingConfig()
        self._lock = threading.RLock()
        self._config_data = {}
        
        self.logger.info(f"Backend configuration manager initialized with path: {self.config_path}")
    
    def _get_default_config_path(self) -> str:
        """Get default configuration file path."""
        # Check for .kiro/config.yaml in current directory
        kiro_config = Path(".kiro/config.yaml")
        if kiro_config.exists():
            return str(kiro_config)
        
        # Check for config.yaml in current directory
        local_config = Path("config.yaml")
        if local_config.exists():
            return str(local_config)
        
        # Return default path (may not exist)
        return ".kiro/config.yaml"
    
    def load_config(self) -> None:
        """
        Load configuration from environment variables and files.
        Environment variables take precedence over file configuration.
        """
        with self._lock:
            self._config_data = {}
            
            # Load from file if it exists
            if os.path.exists(self.config_path):
                try:
                    with open(self.config_path, 'r') as f:
                        self._config_data = yaml.safe_load(f) or {}
                    self.logger.info(f"Loaded configuration from {self.config_path}")
                except Exception as e:
                    self.logger.warning(f"Failed to load config file {self.config_path}: {e}")
                    self._config_data = {}
            else:
                self.logger.info(f"Config file {self.config_path} not found, using environment variables only")
            
            # Parse backend configurations
            self._parse_backend_configs()
            
            # Parse routing configuration
            self._parse_routing_config()
            
            self.logger.info(f"Configuration loaded: {len(self.backend_configs)} backends configured")
    
    def _substitute_env_vars(self, value: str) -> str:
        """
        Substitute environment variables in configuration values.
        Supports ${VAR_NAME} syntax.
        
        Args:
            value: String value that may contain env var references
            
        Returns:
            String with env vars substituted
        """
        if not isinstance(value, str):
            return value
        
        # Find all ${VAR_NAME} patterns
        pattern = r'\$\{([^}]+)\}'
        matches = re.findall(pattern, value)
        
        result = value
        for var_name in matches:
            env_value = os.getenv(var_name, "")
            result = result.replace(f"${{{var_name}}}", env_value)
        
        return result
    
    def _parse_backend_configs(self) -> None:
        """Parse backend configurations from loaded data."""
        backends_data = self._config_data.get('backends', {})
        
        # Modal configuration
        self._parse_modal_config(backends_data.get('modal', {}))
        
        # HuggingFace configuration
        self._parse_huggingface_config(backends_data.get('huggingface', {}))
        
        # Kaggle configuration
        self._parse_kaggle_config(backends_data.get('kaggle', {}))
        
        # Colab configuration
        self._parse_colab_config(backends_data.get('colab', {}))
    
    def _parse_modal_config(self, modal_data: Dict[str, Any]) -> None:
        """Parse Modal backend configuration."""
        enabled = modal_data.get('enabled', False)
        
        credentials = {
            'token_id': self._substitute_env_vars(modal_data.get('token_id', '')),
            'token_secret': self._substitute_env_vars(modal_data.get('token_secret', ''))
        }
        
        # Also check direct environment variables
        if not credentials['token_id']:
            credentials['token_id'] = os.getenv('MODAL_TOKEN_ID', '')
        if not credentials['token_secret']:
            credentials['token_secret'] = os.getenv('MODAL_TOKEN_SECRET', '')
        
        options = {
            'default_gpu': modal_data.get('default_gpu', 'A10G'),
            'timeout': modal_data.get('timeout', 300),
        }
        
        config = BackendConfig(
            backend_type=BackendType.MODAL,
            enabled=enabled,
            credentials=credentials,
            options=options
        )
        
        self.backend_configs[BackendType.MODAL] = config
    
    def _parse_huggingface_config(self, hf_data: Dict[str, Any]) -> None:
        """Parse HuggingFace backend configuration."""
        enabled = hf_data.get('enabled', False)
        
        credentials = {
            'token': self._substitute_env_vars(hf_data.get('token', ''))
        }
        
        # Also check direct environment variables
        if not credentials['token']:
            credentials['token'] = os.getenv('HF_TOKEN', '') or os.getenv('HUGGINGFACE_TOKEN', '')
        
        options = {
            'default_space_hardware': hf_data.get('default_space_hardware', 'cpu-basic'),
        }
        
        config = BackendConfig(
            backend_type=BackendType.HUGGINGFACE,
            enabled=enabled,
            credentials=credentials,
            options=options
        )
        
        self.backend_configs[BackendType.HUGGINGFACE] = config
    
    def _parse_kaggle_config(self, kaggle_data: Dict[str, Any]) -> None:
        """Parse Kaggle backend configuration."""
        enabled = kaggle_data.get('enabled', False)
        
        credentials = {
            'username': self._substitute_env_vars(kaggle_data.get('username', '')),
            'key': self._substitute_env_vars(kaggle_data.get('key', ''))
        }
        
        # Also check direct environment variables
        if not credentials['username']:
            credentials['username'] = os.getenv('KAGGLE_USERNAME', '')
        if not credentials['key']:
            credentials['key'] = os.getenv('KAGGLE_KEY', '')
        
        options = {
            'max_concurrent_kernels': kaggle_data.get('max_concurrent_kernels', 1),
        }
        
        config = BackendConfig(
            backend_type=BackendType.KAGGLE,
            enabled=enabled,
            credentials=credentials,
            options=options
        )
        
        self.backend_configs[BackendType.KAGGLE] = config
    
    def _parse_colab_config(self, colab_data: Dict[str, Any]) -> None:
        """Parse Colab backend configuration."""
        enabled = colab_data.get('enabled', False)
        
        credentials = {
            'client_id': self._substitute_env_vars(colab_data.get('client_id', '')),
            'client_secret': self._substitute_env_vars(colab_data.get('client_secret', '')),
            'refresh_token': self._substitute_env_vars(colab_data.get('refresh_token', ''))
        }
        
        # Also check direct environment variables
        if not credentials['client_id']:
            credentials['client_id'] = os.getenv('GOOGLE_CLIENT_ID', '')
        if not credentials['client_secret']:
            credentials['client_secret'] = os.getenv('GOOGLE_CLIENT_SECRET', '')
        if not credentials['refresh_token']:
            credentials['refresh_token'] = os.getenv('GOOGLE_REFRESH_TOKEN', '')
        
        options = {}
        
        config = BackendConfig(
            backend_type=BackendType.COLAB,
            enabled=enabled,
            credentials=credentials,
            options=options
        )
        
        self.backend_configs[BackendType.COLAB] = config
    
    def _parse_routing_config(self) -> None:
        """Parse routing configuration."""
        routing_data = self._config_data.get('routing', {})
        
        self.routing_config = RoutingConfig(
            strategy=routing_data.get('strategy', 'cost-optimized'),
            prefer_free_tier=routing_data.get('prefer_free_tier', True),
            health_check_interval=routing_data.get('health_check_interval', 300),
            max_retries=routing_data.get('max_retries', 3),
            retry_backoff_base=routing_data.get('retry_backoff_base', 2.0)
        )
    
    def validate_backend_config(self, backend_type: BackendType) -> bool:
        """
        Validate configuration for a specific backend.
        
        Args:
            backend_type: Type of backend to validate
            
        Returns:
            True if valid, False otherwise
        """
        with self._lock:
            config = self.backend_configs.get(backend_type)
            if not config:
                return False
            
            return config.validate()
    
    def get_backend_config(self, backend_type: BackendType) -> Optional[BackendConfig]:
        """
        Get configuration for a specific backend.
        
        Args:
            backend_type: Type of backend
            
        Returns:
            BackendConfig or None if not found
        """
        with self._lock:
            return self.backend_configs.get(backend_type)
    
    def get_enabled_backends(self) -> list:
        """
        Get list of enabled backend types.
        
        Returns:
            List of enabled BackendType values
        """
        with self._lock:
            return [
                backend_type
                for backend_type, config in self.backend_configs.items()
                if config.enabled and config.validate()
            ]
    
    def reload_config(self) -> None:
        """
        Hot-reload configuration without restart.
        Re-loads from file and environment variables.
        """
        self.logger.info("Reloading configuration...")
        self.load_config()
        self.logger.info("Configuration reloaded successfully")
    
    def get_routing_config(self) -> RoutingConfig:
        """
        Get routing configuration.
        
        Returns:
            RoutingConfig instance
        """
        with self._lock:
            return self.routing_config
    
    def log_configuration_status(self) -> None:
        """Log current configuration status (without exposing credentials)."""
        with self._lock:
            self.logger.info("=== Backend Configuration Status ===")
            
            for backend_type, config in self.backend_configs.items():
                status = "ENABLED" if config.enabled else "DISABLED"
                valid = "VALID" if config.validate() else "INVALID"
                
                self.logger.info(f"{backend_type.value}: {status}, {valid}")
                
                if config.enabled and not config.validate():
                    missing = config.get_missing_credentials()
                    self.logger.warning(f"  Missing credentials: {missing}")
            
            self.logger.info(f"Routing strategy: {self.routing_config.strategy}")
            self.logger.info(f"Prefer free tier: {self.routing_config.prefer_free_tier}")
            self.logger.info("=" * 40)
