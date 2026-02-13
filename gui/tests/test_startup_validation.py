"""Tests for startup validation module.

This module tests the startup validation functionality including:
- Dependency validation
- Database connectivity validation
- Configuration validation
"""

import pytest
import tempfile
import sqlite3
from pathlib import Path
from unittest.mock import patch, MagicMock

from gui.validation import (
    validate_dependencies,
    validate_database_connectivity,
    validate_configuration,
    validate_startup,
    ValidationError
)
from gui.config import GUIConfig


class TestValidateDependencies:
    """Test dependency validation."""
    
    def test_validate_dependencies_success(self):
        """Test successful dependency validation."""
        success, errors = validate_dependencies()
        
        # Should succeed since all dependencies are installed in test environment
        assert success is True
        assert len(errors) == 0
    
    @patch('gui.validation.importlib.import_module')
    def test_validate_dependencies_missing_gradio(self, mock_import):
        """Test dependency validation with missing Gradio."""
        def import_side_effect(name):
            if name == 'gradio':
                raise ImportError("No module named 'gradio'")
            return MagicMock()
        
        mock_import.side_effect = import_side_effect
        
        success, errors = validate_dependencies()
        
        assert success is False
        assert len(errors) > 0
        assert any('gradio' in error.lower() for error in errors)
    
    @patch('gui.validation.importlib.import_module')
    def test_validate_dependencies_multiple_missing(self, mock_import):
        """Test dependency validation with multiple missing packages."""
        def import_side_effect(name):
            if name in ['gradio', 'fastapi']:
                raise ImportError(f"No module named '{name}'")
            return MagicMock()
        
        mock_import.side_effect = import_side_effect
        
        success, errors = validate_dependencies()
        
        assert success is False
        assert len(errors) >= 2
        assert any('gradio' in error.lower() for error in errors)
        assert any('fastapi' in error.lower() for error in errors)


class TestValidateDatabaseConnectivity:
    """Test database connectivity validation."""
    
    def test_validate_database_new_file(self):
        """Test validation with new database file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test.db")
            
            success, error = validate_database_connectivity(db_path)
            
            assert success is True
            assert error is None
            # Database file should be created
            assert Path(db_path).exists()
    
    def test_validate_database_existing_file(self):
        """Test validation with existing database file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test.db")
            
            # Create database with jobs table
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE jobs (
                    id TEXT PRIMARY KEY,
                    status TEXT
                )
            """)
            conn.commit()
            conn.close()
            
            success, error = validate_database_connectivity(db_path)
            
            assert success is True
            assert error is None
    
    def test_validate_database_invalid_directory(self):
        """Test validation with non-existent parent directory."""
        db_path = "/nonexistent/directory/test.db"
        
        success, error = validate_database_connectivity(db_path)
        
        assert success is False
        assert error is not None
        assert "does not exist" in error.lower()
    
    def test_validate_database_corrupted_file(self):
        """Test validation with corrupted database file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test.db")
            
            # Create a corrupted database file
            with open(db_path, 'w') as f:
                f.write("This is not a valid SQLite database")
            
            success, error = validate_database_connectivity(db_path)
            
            assert success is False
            assert error is not None


class TestValidateConfiguration:
    """Test configuration validation."""
    
    def test_validate_configuration_valid(self):
        """Test validation with valid configuration."""
        config = GUIConfig(
            host="0.0.0.0",
            port=7860,
            websocket_port=7861,
            enable_websocket=True,
            theme="default",
            page_size=50,
            auto_refresh_interval=5,
            session_timeout=3600
        )
        
        success, errors = validate_configuration(config)
        
        assert success is True
        assert len(errors) == 0
    
    def test_validate_configuration_invalid_port(self):
        """Test validation with invalid port."""
        config = GUIConfig(port=70000)  # Port out of range
        
        success, errors = validate_configuration(config)
        
        assert success is False
        assert len(errors) > 0
        assert any('port' in error.lower() for error in errors)
    
    def test_validate_configuration_invalid_websocket_port(self):
        """Test validation with invalid WebSocket port."""
        config = GUIConfig(
            port=7860,
            websocket_port=70000,  # Port out of range
            enable_websocket=True
        )
        
        success, errors = validate_configuration(config)
        
        assert success is False
        assert len(errors) > 0
        assert any('websocket' in error.lower() for error in errors)
    
    def test_validate_configuration_port_conflict(self):
        """Test validation with port conflict."""
        config = GUIConfig(
            port=7860,
            websocket_port=7860,  # Same as GUI port
            enable_websocket=True
        )
        
        success, errors = validate_configuration(config)
        
        assert success is False
        assert len(errors) > 0
        # Check for "conflict" or "conflicts" in error message
        assert any('conflict' in error.lower() or 'same' in error.lower() for error in errors)
    
    def test_validate_configuration_empty_host(self):
        """Test validation with empty host."""
        config = GUIConfig(host="")
        
        success, errors = validate_configuration(config)
        
        assert success is False
        assert len(errors) > 0
        assert any('host' in error.lower() for error in errors)
    
    def test_validate_configuration_negative_page_size(self):
        """Test validation with negative page size."""
        config = GUIConfig(page_size=-10)
        
        success, errors = validate_configuration(config)
        
        assert success is False
        assert len(errors) > 0
        assert any('page size' in error.lower() for error in errors)
    
    def test_validate_configuration_negative_refresh_interval(self):
        """Test validation with negative refresh interval."""
        config = GUIConfig(auto_refresh_interval=-5)
        
        success, errors = validate_configuration(config)
        
        assert success is False
        assert len(errors) > 0
        assert any('refresh' in error.lower() for error in errors)
    
    def test_validate_configuration_negative_session_timeout(self):
        """Test validation with negative session timeout."""
        config = GUIConfig(session_timeout=-100)
        
        success, errors = validate_configuration(config)
        
        assert success is False
        assert len(errors) > 0
        assert any('timeout' in error.lower() for error in errors)
    
    def test_validate_configuration_unknown_theme(self):
        """Test validation with unknown theme (should warn but not fail)."""
        config = GUIConfig(theme="unknown_theme")
        
        success, errors = validate_configuration(config)
        
        # Should succeed (theme validation is a warning, not an error)
        assert success is True
        assert len(errors) == 0


class TestValidateStartup:
    """Test complete startup validation."""
    
    def test_validate_startup_success(self):
        """Test successful startup validation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test.db")
            config = GUIConfig()
            
            success, errors = validate_startup(config, db_path)
            
            assert success is True
            assert len(errors) == 0
    
    def test_validate_startup_multiple_failures(self):
        """Test startup validation with multiple failures."""
        # Invalid config and invalid database path
        config = GUIConfig(port=70000)  # Invalid port
        db_path = "/nonexistent/directory/test.db"
        
        with patch('gui.validation.validate_dependencies') as mock_deps:
            # Mock dependency failure
            mock_deps.return_value = (False, ["Missing gradio"])
            
            success, errors = validate_startup(config, db_path)
            
            assert success is False
            assert len(errors) >= 2  # At least dependency and config errors
    
    def test_validate_startup_dependency_failure_only(self):
        """Test startup validation with only dependency failure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test.db")
            config = GUIConfig()
            
            with patch('gui.validation.validate_dependencies') as mock_deps:
                mock_deps.return_value = (False, ["Missing gradio"])
                
                success, errors = validate_startup(config, db_path)
                
                assert success is False
                assert len(errors) >= 1
                assert any('gradio' in error.lower() for error in errors)
