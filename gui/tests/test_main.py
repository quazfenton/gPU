"""
Tests for the main entry point module.

This module tests the startup logging functionality and main entry point.
"""

import pytest
import logging
from unittest.mock import Mock, patch, call
from io import StringIO

from gui.config import GUIConfig
from gui.main import log_startup_info


class TestStartupLogging:
    """Test suite for startup logging functionality."""
    
    @pytest.fixture
    def config(self):
        """Create a test configuration."""
        return GUIConfig(
            host="127.0.0.1",
            port=7860,
            websocket_port=7861,
            enable_websocket=True,
            enable_auth=False,
            theme="default",
            page_size=50,
            auto_refresh_interval=5,
            session_timeout=3600
        )
    
    def test_log_startup_info_logs_version(self, config, caplog):
        """Test that startup logging includes version information.
        
        Requirements: 11.7 - Log version information
        """
        with caplog.at_level(logging.INFO):
            log_startup_info(config, "test.db", share=False)
        
        # Check that version information is logged
        assert "Notebook ML Orchestrator - GUI Interface" in caplog.text
        assert "GUI Version:" in caplog.text
    
    def test_log_startup_info_logs_configuration(self, config, caplog):
        """Test that startup logging includes configuration values.
        
        Requirements: 11.7 - Log configuration values
        """
        with caplog.at_level(logging.INFO):
            log_startup_info(config, "test.db", share=False)
        
        # Check that configuration values are logged
        assert "Configuration:" in caplog.text
        assert "Host: 127.0.0.1" in caplog.text
        assert "Port: 7860" in caplog.text
        assert "Database: test.db" in caplog.text
        assert "Theme: default" in caplog.text
        assert "Page Size: 50" in caplog.text
        assert "Auto Refresh Interval: 5s" in caplog.text
        assert "Session Timeout: 3600s" in caplog.text
    
    def test_log_startup_info_logs_available_features(self, config, caplog):
        """Test that startup logging includes available features.
        
        Requirements: 11.7 - Log available features
        """
        with caplog.at_level(logging.INFO):
            log_startup_info(config, "test.db", share=False)
        
        # Check that features are logged
        assert "Available Features:" in caplog.text
        assert "Job Submission" in caplog.text
        assert "Job Monitoring" in caplog.text
        assert "Workflow Builder" in caplog.text
        assert "Template Management" in caplog.text
        assert "Backend Status Monitoring" in caplog.text
    
    def test_log_startup_info_websocket_enabled(self, config, caplog):
        """Test that WebSocket feature is logged when enabled."""
        config.enable_websocket = True
        
        with caplog.at_level(logging.INFO):
            log_startup_info(config, "test.db", share=False)
        
        assert "Real-time Updates (WebSocket on port 7861)" in caplog.text
    
    def test_log_startup_info_websocket_disabled(self, config, caplog):
        """Test that WebSocket feature is logged when disabled."""
        config.enable_websocket = False
        
        with caplog.at_level(logging.INFO):
            log_startup_info(config, "test.db", share=False)
        
        assert "Real-time Updates (WebSocket disabled)" in caplog.text
    
    def test_log_startup_info_auth_enabled(self, config, caplog):
        """Test that authentication feature is logged when enabled."""
        config.enable_auth = True
        config.auth_provider = "oauth"
        
        with caplog.at_level(logging.INFO):
            log_startup_info(config, "test.db", share=False)
        
        assert "Authentication (Provider: oauth)" in caplog.text
    
    def test_log_startup_info_auth_disabled(self, config, caplog):
        """Test that authentication feature is logged when disabled."""
        config.enable_auth = False
        
        with caplog.at_level(logging.INFO):
            log_startup_info(config, "test.db", share=False)
        
        assert "Authentication (Disabled)" in caplog.text
    
    def test_log_startup_info_share_enabled(self, config, caplog):
        """Test that public sharing feature is logged when enabled."""
        with caplog.at_level(logging.INFO):
            log_startup_info(config, "test.db", share=True)
        
        assert "Public Sharing (Gradio share link)" in caplog.text
    
    def test_log_startup_info_share_disabled(self, config, caplog):
        """Test that public sharing feature is logged when disabled."""
        with caplog.at_level(logging.INFO):
            log_startup_info(config, "test.db", share=False)
        
        assert "Public Sharing (Disabled)" in caplog.text
    
    def test_log_startup_info_auth_default_provider(self, config, caplog):
        """Test that default auth provider is used when none specified."""
        config.enable_auth = True
        config.auth_provider = None
        
        with caplog.at_level(logging.INFO):
            log_startup_info(config, "test.db", share=False)
        
        assert "Authentication (Provider: default)" in caplog.text
