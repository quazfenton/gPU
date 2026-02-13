"""
Unit tests for backend configuration management.
"""

import os
import tempfile
import pytest
from pathlib import Path

from notebook_ml_orchestrator.core.backend_config import (
    BackendConfig,
    BackendConfigManager,
    RoutingConfig,
    REQUIRED_CREDENTIALS
)
from notebook_ml_orchestrator.core.models import BackendType


class TestBackendConfig:
    """Test BackendConfig dataclass."""
    
    def test_validate_with_all_credentials(self):
        """Test validation succeeds when all credentials present."""
        config = BackendConfig(
            backend_type=BackendType.MODAL,
            enabled=True,
            credentials={'token_id': 'test_id', 'token_secret': 'test_secret'}
        )
        assert config.validate() is True
    
    def test_validate_with_missing_credentials(self):
        """Test validation fails when credentials missing."""
        config = BackendConfig(
            backend_type=BackendType.MODAL,
            enabled=True,
            credentials={'token_id': 'test_id'}  # Missing token_secret
        )
        assert config.validate() is False
    
    def test_validate_disabled_backend(self):
        """Test validation passes for disabled backends."""
        config = BackendConfig(
            backend_type=BackendType.MODAL,
            enabled=False,
            credentials={}  # No credentials
        )
        assert config.validate() is True
    
    def test_get_missing_credentials(self):
        """Test getting list of missing credentials."""
        config = BackendConfig(
            backend_type=BackendType.MODAL,
            enabled=True,
            credentials={'token_id': 'test_id'}
        )
        missing = config.get_missing_credentials()
        assert 'token_secret' in missing
        assert 'token_id' not in missing


class TestBackendConfigManager:
    """Test BackendConfigManager."""
    
    def test_init_with_default_path(self):
        """Test initialization with default config path."""
        manager = BackendConfigManager()
        assert manager.config_path is not None
    
    def test_init_with_custom_path(self):
        """Test initialization with custom config path."""
        custom_path = "/custom/path/config.yaml"
        manager = BackendConfigManager(config_path=custom_path)
        assert manager.config_path == custom_path
    
    def test_load_config_from_file(self):
        """Test loading configuration from YAML file."""
        # Create temporary config file
        config_content = """
backends:
  modal:
    enabled: true
    token_id: test_modal_id
    token_secret: test_modal_secret
    default_gpu: "T4"
  
  huggingface:
    enabled: false
    token: test_hf_token

routing:
  strategy: "round-robin"
  prefer_free_tier: false
  max_retries: 5
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(config_content)
            temp_path = f.name
        
        try:
            manager = BackendConfigManager(config_path=temp_path)
            manager.load_config()
            
            # Check Modal config
            modal_config = manager.get_backend_config(BackendType.MODAL)
            assert modal_config is not None
            assert modal_config.enabled is True
            assert modal_config.credentials['token_id'] == 'test_modal_id'
            assert modal_config.options['default_gpu'] == 'T4'
            
            # Check HuggingFace config
            hf_config = manager.get_backend_config(BackendType.HUGGINGFACE)
            assert hf_config is not None
            assert hf_config.enabled is False
            
            # Check routing config
            routing = manager.get_routing_config()
            assert routing.strategy == "round-robin"
            assert routing.prefer_free_tier is False
            assert routing.max_retries == 5
        finally:
            os.unlink(temp_path)
    
    def test_env_var_substitution(self):
        """Test environment variable substitution in config."""
        # Set environment variables
        os.environ['TEST_MODAL_ID'] = 'env_modal_id'
        os.environ['TEST_MODAL_SECRET'] = 'env_modal_secret'
        
        config_content = """
backends:
  modal:
    enabled: true
    token_id: ${TEST_MODAL_ID}
    token_secret: ${TEST_MODAL_SECRET}
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(config_content)
            temp_path = f.name
        
        try:
            manager = BackendConfigManager(config_path=temp_path)
            manager.load_config()
            
            modal_config = manager.get_backend_config(BackendType.MODAL)
            assert modal_config.credentials['token_id'] == 'env_modal_id'
            assert modal_config.credentials['token_secret'] == 'env_modal_secret'
        finally:
            os.unlink(temp_path)
            del os.environ['TEST_MODAL_ID']
            del os.environ['TEST_MODAL_SECRET']
    
    def test_direct_env_vars(self):
        """Test loading credentials directly from environment variables."""
        # Set environment variables
        os.environ['MODAL_TOKEN_ID'] = 'direct_modal_id'
        os.environ['MODAL_TOKEN_SECRET'] = 'direct_modal_secret'
        
        config_content = """
backends:
  modal:
    enabled: true
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(config_content)
            temp_path = f.name
        
        try:
            manager = BackendConfigManager(config_path=temp_path)
            manager.load_config()
            
            modal_config = manager.get_backend_config(BackendType.MODAL)
            assert modal_config.credentials['token_id'] == 'direct_modal_id'
            assert modal_config.credentials['token_secret'] == 'direct_modal_secret'
        finally:
            os.unlink(temp_path)
            del os.environ['MODAL_TOKEN_ID']
            del os.environ['MODAL_TOKEN_SECRET']
    
    def test_validate_backend_config(self):
        """Test backend configuration validation."""
        config_content = """
backends:
  modal:
    enabled: true
    token_id: test_id
    token_secret: test_secret
  
  huggingface:
    enabled: true
    # Missing token
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(config_content)
            temp_path = f.name
        
        try:
            manager = BackendConfigManager(config_path=temp_path)
            manager.load_config()
            
            # Modal should be valid
            assert manager.validate_backend_config(BackendType.MODAL) is True
            
            # HuggingFace should be invalid (missing token)
            assert manager.validate_backend_config(BackendType.HUGGINGFACE) is False
        finally:
            os.unlink(temp_path)
    
    def test_get_enabled_backends(self):
        """Test getting list of enabled and valid backends."""
        config_content = """
backends:
  modal:
    enabled: true
    token_id: test_id
    token_secret: test_secret
  
  huggingface:
    enabled: true
    token: test_token
  
  kaggle:
    enabled: false
    username: test_user
    key: test_key
  
  colab:
    enabled: true
    # Missing credentials
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(config_content)
            temp_path = f.name
        
        try:
            manager = BackendConfigManager(config_path=temp_path)
            manager.load_config()
            
            enabled = manager.get_enabled_backends()
            
            # Should include Modal and HuggingFace (enabled and valid)
            assert BackendType.MODAL in enabled
            assert BackendType.HUGGINGFACE in enabled
            
            # Should not include Kaggle (disabled) or Colab (invalid)
            assert BackendType.KAGGLE not in enabled
            assert BackendType.COLAB not in enabled
        finally:
            os.unlink(temp_path)
    
    def test_reload_config(self):
        """Test hot-reloading configuration."""
        config_content_v1 = """
backends:
  modal:
    enabled: true
    token_id: v1_id
    token_secret: v1_secret

routing:
  strategy: "round-robin"
"""
        config_content_v2 = """
backends:
  modal:
    enabled: true
    token_id: v2_id
    token_secret: v2_secret

routing:
  strategy: "cost-optimized"
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(config_content_v1)
            temp_path = f.name
        
        try:
            manager = BackendConfigManager(config_path=temp_path)
            manager.load_config()
            
            # Check initial config
            modal_config = manager.get_backend_config(BackendType.MODAL)
            assert modal_config.credentials['token_id'] == 'v1_id'
            assert manager.get_routing_config().strategy == "round-robin"
            
            # Update config file
            with open(temp_path, 'w') as f:
                f.write(config_content_v2)
            
            # Reload
            manager.reload_config()
            
            # Check updated config
            modal_config = manager.get_backend_config(BackendType.MODAL)
            assert modal_config.credentials['token_id'] == 'v2_id'
            assert manager.get_routing_config().strategy == "cost-optimized"
        finally:
            os.unlink(temp_path)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
