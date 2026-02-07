"""
Logging configuration for the Notebook ML Orchestrator.

This module provides centralized logging configuration with proper formatting,
log levels, and file rotation for the entire orchestration system.
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    max_file_size: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
    enable_console: bool = True
) -> logging.Logger:
    """
    Set up logging configuration for the orchestrator.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (optional)
        max_file_size: Maximum size of log file before rotation
        backup_count: Number of backup files to keep
        enable_console: Whether to enable console logging

    Returns:
        Configured logger instance
    """
    # Create logs directory if it doesn't exist
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

    # Configure root logger
    root_logger = logging.getLogger()
    try:
        level = getattr(logging, log_level.upper())
    except AttributeError:
        level = logging.INFO
        root_logger.warning(f"Invalid log level '{log_level}', defaulting to INFO")
    root_logger.setLevel(level)

    # Clear existing handlers
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_level.upper()))
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    # File handler with rotation
    if log_file:
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_file_size,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Create orchestrator-specific logger
    orchestrator_logger = logging.getLogger('notebook_ml_orchestrator')
    
    return orchestrator_logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a specific module.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Logger instance
    """
    return logging.getLogger(f'notebook_ml_orchestrator.{name}')


class LoggerMixin:
    """Mixin class to add logging capabilities to any class."""
    
    @property
    def logger(self) -> logging.Logger:
        """Get logger for this class."""
        return get_logger(self.__class__.__module__ + '.' + self.__class__.__name__)


# Default logging configuration
DEFAULT_LOG_CONFIG = {
    'log_level': 'INFO',
    'log_file': 'logs/orchestrator.log',
    'max_file_size': 10 * 1024 * 1024,  # 10MB
    'backup_count': 5,
    'enable_console': True
}


def configure_default_logging():
    """Configure logging with default settings."""
    return setup_logging(**DEFAULT_LOG_CONFIG)