"""Startup validation utilities for GUI application.

This module provides validation functions to ensure the GUI can start successfully
by checking dependencies, database connectivity, and configuration.

Requirements:
    - 11.4: Validate required dependencies are installed
    - 11.5: Verify connectivity to Job_Queue database
"""

import sys
import importlib
import sqlite3
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import logging

from gui.config import GUIConfig


logger = logging.getLogger('gui.validation')


class ValidationError(Exception):
    """Exception raised when validation fails."""
    pass


def validate_dependencies() -> Tuple[bool, List[str]]:
    """Validate that all required dependencies are installed.
    
    Checks for the presence of required packages:
    - gradio: Web interface framework
    - fastapi: WebSocket server
    - websockets: WebSocket protocol support
    - hypothesis: Property-based testing (optional for runtime)
    - dotenv: Configuration file loading
    
    Returns:
        Tuple of (success: bool, errors: List[str])
        
    Requirements:
        - 11.4: Validate required dependencies are installed
    """
    logger.info("Validating dependencies...")
    
    required_packages = [
        ('gradio', 'Gradio web interface framework'),
        ('fastapi', 'FastAPI for WebSocket server'),
        ('websockets', 'WebSocket protocol support'),
        ('dotenv', 'Configuration file loading (python-dotenv)'),
    ]
    
    optional_packages = [
        ('hypothesis', 'Property-based testing framework'),
    ]
    
    errors = []
    
    # Check required packages
    for package_name, description in required_packages:
        try:
            importlib.import_module(package_name)
            logger.debug(f"  ✓ {package_name} - {description}")
        except ImportError:
            error_msg = f"Missing required dependency: {package_name} ({description})"
            logger.error(f"  ✗ {error_msg}")
            errors.append(error_msg)
    
    # Check optional packages (log warning but don't fail)
    for package_name, description in optional_packages:
        try:
            importlib.import_module(package_name)
            logger.debug(f"  ✓ {package_name} - {description} (optional)")
        except ImportError:
            logger.warning(f"  ! {package_name} not installed ({description}) - optional")
    
    if errors:
        logger.error(f"Dependency validation failed with {len(errors)} error(s)")
        return False, errors
    
    logger.info("All required dependencies are installed")
    return True, []


def validate_database_connectivity(db_path: str) -> Tuple[bool, Optional[str]]:
    """Verify connectivity to the Job_Queue database.
    
    Attempts to:
    1. Check if database file exists (or can be created)
    2. Open a connection to the database
    3. Execute a simple query to verify it's a valid SQLite database
    
    Args:
        db_path: Path to the SQLite database file
        
    Returns:
        Tuple of (success: bool, error_message: Optional[str])
        
    Requirements:
        - 11.5: Verify connectivity to Job_Queue database
    """
    logger.info(f"Validating database connectivity: {db_path}")
    
    try:
        # Check if database file exists
        db_file = Path(db_path)
        db_exists = db_file.exists()
        
        if db_exists:
            logger.debug(f"  Database file exists: {db_path}")
        else:
            logger.debug(f"  Database file will be created: {db_path}")
            # Check if parent directory exists and is writable
            parent_dir = db_file.parent
            if not parent_dir.exists():
                error_msg = f"Database parent directory does not exist: {parent_dir}"
                logger.error(f"  ✗ {error_msg}")
                return False, error_msg
            
            if not os.access(parent_dir, os.W_OK):
                error_msg = f"Database parent directory is not writable: {parent_dir}"
                logger.error(f"  ✗ {error_msg}")
                return False, error_msg
        
        # Try to connect to the database
        logger.debug("  Attempting database connection...")
        conn = None
        try:
            conn = sqlite3.connect(db_path)
            
            # Execute a simple query to verify it's a valid database
            cursor = conn.cursor()
            cursor.execute("SELECT sqlite_version()")
            version = cursor.fetchone()[0]
            logger.debug(f"  ✓ Connected to SQLite database (version: {version})")
            
            # Check if the database has the expected tables (if it exists)
            if db_exists:
                cursor.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='jobs'"
                )
                result = cursor.fetchone()
                if result:
                    logger.debug("  ✓ Found 'jobs' table in database")
                else:
                    logger.warning("  ! 'jobs' table not found - database may be uninitialized")
        finally:
            # Always close connection
            if conn:
                conn.close()
        
        logger.info("Database connectivity validated successfully")
        return True, None
        
    except sqlite3.Error as e:
        error_msg = f"Database error: {e}"
        logger.error(f"  ✗ {error_msg}")
        return False, error_msg
    except Exception as e:
        error_msg = f"Unexpected error validating database: {e}"
        logger.error(f"  ✗ {error_msg}")
        return False, error_msg


def validate_configuration(config: GUIConfig) -> Tuple[bool, List[str]]:
    """Validate GUI configuration values.
    
    Checks that configuration values are valid:
    - Port numbers are in valid range (1-65535)
    - Host address is not empty
    - Theme is a valid option
    - Numeric values are positive
    
    Args:
        config: GUI configuration to validate
        
    Returns:
        Tuple of (success: bool, errors: List[str])
        
    Requirements:
        - 11.4: Validate configuration
    """
    logger.info("Validating configuration...")
    
    errors = []
    
    # Validate host
    if not config.host or not config.host.strip():
        errors.append("Host address cannot be empty")
        logger.error("  ✗ Host address is empty")
    else:
        logger.debug(f"  ✓ Host: {config.host}")
    
    # Validate port
    if not (1 <= config.port <= 65535):
        errors.append(f"Port must be between 1 and 65535 (got: {config.port})")
        logger.error(f"  ✗ Invalid port: {config.port}")
    else:
        logger.debug(f"  ✓ Port: {config.port}")
    
    # Validate WebSocket port
    if config.enable_websocket:
        if not (1 <= config.websocket_port <= 65535):
            errors.append(
                f"WebSocket port must be between 1 and 65535 (got: {config.websocket_port})"
            )
            logger.error(f"  ✗ Invalid WebSocket port: {config.websocket_port}")
        elif config.websocket_port == config.port:
            errors.append(
                f"WebSocket port cannot be the same as GUI port ({config.port})"
            )
            logger.error(f"  ✗ WebSocket port conflicts with GUI port")
        else:
            logger.debug(f"  ✓ WebSocket port: {config.websocket_port}")
    
    # Validate theme
    valid_themes = ['default', 'soft', 'monochrome']
    if config.theme not in valid_themes:
        logger.warning(
            f"  ! Unknown theme '{config.theme}' - will use 'default'. "
            f"Valid themes: {', '.join(valid_themes)}"
        )
        # This is a warning, not an error - we'll fall back to default
    else:
        logger.debug(f"  ✓ Theme: {config.theme}")
    
    # Validate page size
    if config.page_size <= 0:
        errors.append(f"Page size must be positive (got: {config.page_size})")
        logger.error(f"  ✗ Invalid page size: {config.page_size}")
    else:
        logger.debug(f"  ✓ Page size: {config.page_size}")
    
    # Validate auto refresh interval
    if config.auto_refresh_interval <= 0:
        errors.append(
            f"Auto refresh interval must be positive (got: {config.auto_refresh_interval})"
        )
        logger.error(f"  ✗ Invalid auto refresh interval: {config.auto_refresh_interval}")
    else:
        logger.debug(f"  ✓ Auto refresh interval: {config.auto_refresh_interval}s")
    
    # Validate session timeout
    if config.session_timeout <= 0:
        errors.append(f"Session timeout must be positive (got: {config.session_timeout})")
        logger.error(f"  ✗ Invalid session timeout: {config.session_timeout}")
    else:
        logger.debug(f"  ✓ Session timeout: {config.session_timeout}s")
    
    # Validate authentication configuration
    if config.enable_auth:
        logger.debug(f"  ✓ Authentication enabled")
        if config.auth_provider:
            logger.debug(f"  ✓ Auth provider: {config.auth_provider}")
        else:
            logger.debug("  ✓ Auth provider: default")
    else:
        logger.debug("  ✓ Authentication disabled")
    
    if errors:
        logger.error(f"Configuration validation failed with {len(errors)} error(s)")
        return False, errors
    
    logger.info("Configuration validated successfully")
    return True, []


def validate_startup(config: GUIConfig, db_path: str) -> Tuple[bool, List[str]]:
    """Perform all startup validations.
    
    This is a convenience function that runs all validation checks:
    1. Dependency validation
    2. Database connectivity validation
    3. Configuration validation
    
    Args:
        config: GUI configuration
        db_path: Path to database file
        
    Returns:
        Tuple of (success: bool, errors: List[str])
        
    Requirements:
        - 11.4: Validate required dependencies are installed
        - 11.5: Verify connectivity to Job_Queue database
    """
    logger.info("=" * 60)
    logger.info("Starting startup validation")
    logger.info("=" * 60)
    
    all_errors = []
    
    # 1. Validate dependencies
    deps_ok, deps_errors = validate_dependencies()
    if not deps_ok:
        all_errors.extend(deps_errors)
    
    # 2. Validate database connectivity
    db_ok, db_error = validate_database_connectivity(db_path)
    if not db_ok:
        all_errors.append(db_error)
    
    # 3. Validate configuration
    config_ok, config_errors = validate_configuration(config)
    if not config_ok:
        all_errors.extend(config_errors)
    
    # Summary
    if all_errors:
        logger.error("=" * 60)
        logger.error(f"Startup validation FAILED with {len(all_errors)} error(s)")
        logger.error("=" * 60)
        for i, error in enumerate(all_errors, 1):
            logger.error(f"  {i}. {error}")
        logger.error("=" * 60)
        return False, all_errors
    else:
        logger.info("=" * 60)
        logger.info("Startup validation PASSED - all checks successful")
        logger.info("=" * 60)
        return True, []


# Import os for file access checks
import os



def validate_inputs(template, inputs: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate inputs against template schema.
    
    Args:
        template: Template instance with input schema
        inputs: Dictionary of input values to validate
        
    Returns:
        Tuple of (is_valid, list of error messages)
    """
    errors = []
    
    # Get input schema from template
    input_schema = template.get_input_schema()
    
    # Check required fields
    for field_name, field_def in input_schema.items():
        if field_def.get('required', False) and field_name not in inputs:
            errors.append(f"Required field '{field_name}' is missing")
    
    # Check field types and constraints
    for field_name, value in inputs.items():
        if field_name not in input_schema:
            errors.append(f"Unknown field '{field_name}'")
            continue
        
        field_def = input_schema[field_name]
        field_type = field_def.get('type', 'string')
        
        # Type validation
        if field_type == 'number' and not isinstance(value, (int, float)):
            errors.append(f"Field '{field_name}' must be a number")
        elif field_type == 'string' and not isinstance(value, str):
            errors.append(f"Field '{field_name}' must be a string")
        elif field_type == 'boolean' and not isinstance(value, bool):
            errors.append(f"Field '{field_name}' must be a boolean")
        
        # Range validation for numbers
        if field_type == 'number' and isinstance(value, (int, float)):
            if 'min' in field_def and value < field_def['min']:
                errors.append(f"Field '{field_name}' must be >= {field_def['min']}")
            if 'max' in field_def and value > field_def['max']:
                errors.append(f"Field '{field_name}' must be <= {field_def['max']}")
    
    return len(errors) == 0, errors


def format_validation_errors(errors: List[str]) -> str:
    """
    Format validation errors into a user-friendly message.
    
    Args:
        errors: List of error messages
        
    Returns:
        Formatted error message string
    """
    if not errors:
        return ""
    
    if len(errors) == 1:
        return f"Validation error: {errors[0]}"
    
    error_list = "\n".join(f"  - {error}" for error in errors)
    return f"Validation errors:\n{error_list}"
