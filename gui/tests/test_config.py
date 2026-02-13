"""Tests for GUIConfig."""

import os
import tempfile
from pathlib import Path

import pytest

from gui.config import GUIConfig


class TestGUIConfig:
    """Test suite for GUIConfig dataclass."""
    
    def test_default_values(self):
        """Test that default values are set correctly."""
        config = GUIConfig()
        
        assert config.host == "0.0.0.0"
        assert config.port == 7860
        assert config.websocket_port == 7861
        assert config.enable_auth is False
        assert config.auth_provider is None
        assert config.enable_websocket is True
        assert config.theme == "default"
        assert config.page_size == 50
        assert config.auto_refresh_interval == 5
        assert config.session_timeout == 3600
    
    def test_from_env_with_defaults(self):
        """Test loading from environment with no env vars set uses defaults."""
        # Clear any GUI-related env vars
        env_vars = [k for k in os.environ.keys() if k.startswith("GUI_")]
        original_values = {}
        for var in env_vars:
            original_values[var] = os.environ.pop(var)
        
        try:
            config = GUIConfig.from_env()
            
            assert config.host == "0.0.0.0"
            assert config.port == 7860
            assert config.websocket_port == 7861
            assert config.enable_auth is False
            assert config.auth_provider is None
            assert config.enable_websocket is True
            assert config.theme == "default"
            assert config.page_size == 50
            assert config.auto_refresh_interval == 5
            assert config.session_timeout == 3600
        finally:
            # Restore original env vars
            for var, value in original_values.items():
                os.environ[var] = value
    
    def test_from_env_with_custom_values(self):
        """Test loading from environment with custom values."""
        # Set custom env vars
        os.environ["GUI_HOST"] = "127.0.0.1"
        os.environ["GUI_PORT"] = "8080"
        os.environ["GUI_WEBSOCKET_PORT"] = "8081"
        os.environ["GUI_ENABLE_AUTH"] = "true"
        os.environ["GUI_AUTH_PROVIDER"] = "oauth"
        os.environ["GUI_ENABLE_WEBSOCKET"] = "false"
        os.environ["GUI_THEME"] = "dark"
        os.environ["GUI_PAGE_SIZE"] = "100"
        os.environ["GUI_AUTO_REFRESH_INTERVAL"] = "10"
        os.environ["GUI_SESSION_TIMEOUT"] = "7200"
        
        try:
            config = GUIConfig.from_env()
            
            assert config.host == "127.0.0.1"
            assert config.port == 8080
            assert config.websocket_port == 8081
            assert config.enable_auth is True
            assert config.auth_provider == "oauth"
            assert config.enable_websocket is False
            assert config.theme == "dark"
            assert config.page_size == 100
            assert config.auto_refresh_interval == 10
            assert config.session_timeout == 7200
        finally:
            # Clean up env vars
            for var in ["GUI_HOST", "GUI_PORT", "GUI_WEBSOCKET_PORT", 
                       "GUI_ENABLE_AUTH", "GUI_AUTH_PROVIDER", "GUI_ENABLE_WEBSOCKET",
                       "GUI_THEME", "GUI_PAGE_SIZE", "GUI_AUTO_REFRESH_INTERVAL",
                       "GUI_SESSION_TIMEOUT"]:
                os.environ.pop(var, None)
    
    def test_from_file_success(self):
        """Test loading configuration from a file."""
        # Create a temporary config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
            f.write("GUI_HOST=192.168.1.1\n")
            f.write("GUI_PORT=9000\n")
            f.write("GUI_WEBSOCKET_PORT=9001\n")
            f.write("GUI_ENABLE_AUTH=true\n")
            f.write("GUI_AUTH_PROVIDER=ldap\n")
            f.write("GUI_THEME=light\n")
            f.write("GUI_PAGE_SIZE=25\n")
            temp_path = f.name
        
        try:
            config = GUIConfig.from_file(temp_path)
            
            assert config.host == "192.168.1.1"
            assert config.port == 9000
            assert config.websocket_port == 9001
            assert config.enable_auth is True
            assert config.auth_provider == "ldap"
            assert config.theme == "light"
            assert config.page_size == 25
            # Values not in file should use defaults
            assert config.enable_websocket is True
            assert config.auto_refresh_interval == 5
            assert config.session_timeout == 3600
        finally:
            # Clean up
            Path(temp_path).unlink()
            # Clean up env vars that were loaded
            for var in ["GUI_HOST", "GUI_PORT", "GUI_WEBSOCKET_PORT", 
                       "GUI_ENABLE_AUTH", "GUI_AUTH_PROVIDER", "GUI_THEME",
                       "GUI_PAGE_SIZE"]:
                os.environ.pop(var, None)
    
    def test_from_file_not_found(self):
        """Test that from_file raises FileNotFoundError for non-existent file."""
        with pytest.raises(FileNotFoundError, match="Configuration file not found"):
            GUIConfig.from_file("/nonexistent/path/to/config.env")
    
    def test_boolean_parsing_case_insensitive(self):
        """Test that boolean values are parsed case-insensitively."""
        test_cases = [
            ("true", True),
            ("True", True),
            ("TRUE", True),
            ("false", False),
            ("False", False),
            ("FALSE", False),
            ("yes", False),  # Only "true" should be True
            ("1", False),    # Only "true" should be True
        ]
        
        for value, expected in test_cases:
            os.environ["GUI_ENABLE_AUTH"] = value
            try:
                config = GUIConfig.from_env()
                assert config.enable_auth == expected, f"Failed for value: {value}"
            finally:
                os.environ.pop("GUI_ENABLE_AUTH", None)
    
    def test_direct_instantiation(self):
        """Test creating config by directly passing values."""
        config = GUIConfig(
            host="10.0.0.1",
            port=5000,
            websocket_port=5001,
            enable_auth=True,
            auth_provider="custom",
            enable_websocket=False,
            theme="custom-theme",
            page_size=75,
            auto_refresh_interval=15,
            session_timeout=1800
        )
        
        assert config.host == "10.0.0.1"
        assert config.port == 5000
        assert config.websocket_port == 5001
        assert config.enable_auth is True
        assert config.auth_provider == "custom"
        assert config.enable_websocket is False
        assert config.theme == "custom-theme"
        assert config.page_size == 75
        assert config.auto_refresh_interval == 15
        assert config.session_timeout == 1800
