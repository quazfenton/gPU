#!/usr/bin/env python3
"""Main entry point for the Notebook ML Orchestrator GUI.

This script provides a command-line interface for launching the GUI application
with configurable settings.

Usage:
    python -m gui.main [OPTIONS]

Examples:
    # Launch with default settings
    python -m gui.main

    # Launch with custom host and port
    python -m gui.main --host 127.0.0.1 --port 8080

    # Launch with configuration file
    python -m gui.main --config /path/to/config.env

    # Launch with authentication enabled
    python -m gui.main --enable-auth

    # Launch with WebSocket disabled
    python -m gui.main --no-websocket

Requirements:
    - 11.1: Read configuration from environment variables or configuration files
    - 11.2: Support configurable host and port settings
"""

import argparse
import sys
import os
import logging
from pathlib import Path

from notebook_ml_orchestrator.core.job_queue import JobQueueManager
from notebook_ml_orchestrator.core.backend_router import MultiBackendRouter
from notebook_ml_orchestrator.core.workflow_engine import WorkflowEngine
from notebook_ml_orchestrator.core.template_registry import TemplateRegistry
from notebook_ml_orchestrator.core.logging_config import setup_logging

from gui.app import GradioApp
from gui.config import GUIConfig
from gui.validation import validate_startup, ValidationError
import gui


def parse_arguments():
    """Parse command-line arguments.
    
    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Notebook ML Orchestrator GUI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                    # Launch with default settings
  %(prog)s --host 127.0.0.1 --port 8080      # Custom host and port
  %(prog)s --config /path/to/config.env      # Use configuration file
  %(prog)s --enable-auth                     # Enable authentication
  %(prog)s --no-websocket                    # Disable WebSocket
  %(prog)s --share                           # Create public Gradio link
        """
    )
    
    # Configuration file
    parser.add_argument(
        '--config',
        type=str,
        help='Path to configuration file (.env format)'
    )
    
    # Host and port overrides
    parser.add_argument(
        '--host',
        type=str,
        help='Host address to bind (overrides config)'
    )
    
    parser.add_argument(
        '--port',
        type=int,
        help='Port number to bind (overrides config)'
    )
    
    # WebSocket configuration
    parser.add_argument(
        '--websocket-port',
        type=int,
        help='WebSocket port number (overrides config)'
    )
    
    parser.add_argument(
        '--no-websocket',
        action='store_true',
        help='Disable WebSocket for real-time updates'
    )
    
    # Authentication configuration
    parser.add_argument(
        '--enable-auth',
        action='store_true',
        help='Enable authentication'
    )
    
    parser.add_argument(
        '--auth-provider',
        type=str,
        help='Authentication provider name'
    )
    
    # Theme configuration
    parser.add_argument(
        '--theme',
        type=str,
        choices=['default', 'soft', 'monochrome'],
        help='Gradio theme to use'
    )
    
    # Database configuration
    parser.add_argument(
        '--db-path',
        type=str,
        default='orchestrator.db',
        help='Path to SQLite database (default: orchestrator.db)'
    )
    
    # Gradio-specific options
    parser.add_argument(
        '--share',
        action='store_true',
        help='Create a public Gradio link'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode with verbose logging'
    )
    
    return parser.parse_args()


def load_config(args):
    """Load configuration from file and command-line arguments.
    
    Command-line arguments take precedence over configuration file values.
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        GUIConfig instance
        
    Requirements:
        - 11.1: Read configuration from environment variables or configuration files
        - 11.2: Support configurable host and port settings
    """
    # Load from configuration file if provided
    if args.config:
        config_path = Path(args.config)
        if not config_path.exists():
            print(f"Error: Configuration file not found: {args.config}", file=sys.stderr)
            sys.exit(1)
        
        try:
            config = GUIConfig.from_file(args.config)
            print(f"Loaded configuration from {args.config}")
        except Exception as e:
            print(f"Error loading configuration file: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        # Load from environment variables
        config = GUIConfig.from_env()
    
    # Apply command-line overrides
    if args.host:
        config.host = args.host
    
    if args.port:
        config.port = args.port
    
    if args.websocket_port:
        config.websocket_port = args.websocket_port
    
    if args.no_websocket:
        config.enable_websocket = False
    
    if args.enable_auth:
        config.enable_auth = True
    
    if args.auth_provider:
        config.auth_provider = args.auth_provider
    
    if args.theme:
        config.theme = args.theme
    
    return config


def initialize_orchestrator_components(db_path: str):
    """Initialize the core orchestrator components.
    
    Args:
        db_path: Path to SQLite database
        
    Returns:
        Tuple of (job_queue, backend_router, workflow_engine, template_registry)
        
    Requirements:
        - 10.1: GUI uses same Job_Queue as CLI
        - 10.2: GUI retrieves data from same database as CLI
        - 10.3: GUI uses same Workflow_Engine as CLI
        - 10.4: GUI uses same Template_Registry as CLI
        - 10.5: GUI uses same Backend_Router as CLI
    """
    print("Initializing orchestrator components...")
    
    # Initialize job queue with database
    job_queue = JobQueueManager(db_path=db_path)
    print(f"  ✓ Job queue initialized (database: {db_path})")
    
    # Initialize backend router
    backend_router = MultiBackendRouter()
    print("  ✓ Backend router initialized")
    
    # Initialize workflow engine
    workflow_engine = WorkflowEngine()
    print("  ✓ Workflow engine initialized")
    
    # Initialize template registry and discover templates
    template_registry = TemplateRegistry()
    template_count = template_registry.discover_templates()
    print(f"  ✓ Template registry initialized ({template_count} templates discovered)")
    
    # Note: Backends need to be registered separately based on your deployment configuration
    # For example, you would register Modal, Kaggle, HuggingFace, or local backends here
    # Example: backend_router.register_backend(modal_backend)
    
    if len(backend_router.backends) == 0:
        print("  ⚠ Warning: No backends registered. Jobs cannot be executed until backends are configured.")
        print("    To register backends, see the documentation on backend configuration.")
    
    return job_queue, backend_router, workflow_engine, template_registry


def log_startup_info(config: GUIConfig, db_path: str, share: bool = False):
    """Log startup information including version, configuration, and features.
    
    This function logs:
    - Version information
    - Configuration values
    - Available features
    
    Args:
        config: GUI configuration
        db_path: Path to database
        share: Whether public sharing is enabled
        
    Requirements:
        - 11.7: Log startup information including version, configuration, and available features
    """
    logger = logging.getLogger('gui.main')
    
    # Log version information
    logger.info("=" * 60)
    logger.info("Notebook ML Orchestrator - GUI Interface")
    logger.info(f"GUI Version: {gui.__version__}")
    logger.info("=" * 60)
    
    # Log configuration values
    logger.info("Configuration:")
    logger.info(f"  Host: {config.host}")
    logger.info(f"  Port: {config.port}")
    logger.info(f"  Database: {db_path}")
    logger.info(f"  Theme: {config.theme}")
    logger.info(f"  Page Size: {config.page_size}")
    logger.info(f"  Auto Refresh Interval: {config.auto_refresh_interval}s")
    logger.info(f"  Session Timeout: {config.session_timeout}s")
    
    # Log available features
    logger.info("Available Features:")
    
    # WebSocket feature
    if config.enable_websocket:
        logger.info(f"  ✓ Real-time Updates (WebSocket on port {config.websocket_port})")
    else:
        logger.info("  ✗ Real-time Updates (WebSocket disabled)")
    
    # Authentication feature
    if config.enable_auth:
        auth_provider = config.auth_provider or 'default'
        logger.info(f"  ✓ Authentication (Provider: {auth_provider})")
    else:
        logger.info("  ✗ Authentication (Disabled)")
    
    # Core features (always available)
    logger.info("  ✓ Job Submission")
    logger.info("  ✓ Job Monitoring")
    logger.info("  ✓ Workflow Builder")
    logger.info("  ✓ Template Management")
    logger.info("  ✓ Backend Status Monitoring")
    
    # Public sharing feature
    if share:
        logger.info("  ✓ Public Sharing (Gradio share link)")
    else:
        logger.info("  ✗ Public Sharing (Disabled)")
    
    logger.info("=" * 60)


def main():
    """Main entry point for the GUI application.
    
    This function:
    1. Parses command-line arguments
    2. Loads configuration
    3. Initializes orchestrator components
    4. Creates and launches the Gradio app
    """
    # Parse arguments
    args = parse_arguments()
    
    # Setup logging
    log_level = 'DEBUG' if args.debug else 'INFO'
    setup_logging(log_level=log_level)
    
    # Get logger for main module
    logger = logging.getLogger('gui.main')
    
    # Load configuration
    config = load_config(args)
    
    # Perform startup validation (Requirements: 11.4, 11.5)
    logger.info("Performing startup validation...")
    validation_ok, validation_errors = validate_startup(config, args.db_path)
    
    if not validation_ok:
        logger.error("Startup validation failed!")
        print("\n" + "=" * 60, file=sys.stderr)
        print("ERROR: Startup validation failed", file=sys.stderr)
        print("=" * 60, file=sys.stderr)
        for i, error in enumerate(validation_errors, 1):
            print(f"  {i}. {error}", file=sys.stderr)
        print("=" * 60, file=sys.stderr)
        print("\nPlease fix the above errors and try again.", file=sys.stderr)
        sys.exit(1)
    
    logger.info("Startup validation passed")
    
    # Log startup information (Requirements: 11.7)
    log_startup_info(config, args.db_path, args.share)
    
    # Print startup banner to console (for user visibility)
    print("=" * 60)
    print("Notebook ML Orchestrator - GUI Interface")
    print("=" * 60)
    print(f"Host: {config.host}")
    print(f"Port: {config.port}")
    print(f"WebSocket: {'Enabled' if config.enable_websocket else 'Disabled'}")
    if config.enable_websocket:
        print(f"WebSocket Port: {config.websocket_port}")
    print(f"Authentication: {'Enabled' if config.enable_auth else 'Disabled'}")
    if config.enable_auth:
        print(f"Auth Provider: {config.auth_provider or 'default'}")
    print(f"Theme: {config.theme}")
    print(f"Database: {args.db_path}")
    print("=" * 60)
    print()
    
    try:
        # Initialize orchestrator components
        logger.info("Initializing orchestrator components...")
        job_queue, backend_router, workflow_engine, template_registry = \
            initialize_orchestrator_components(args.db_path)
        logger.info("Orchestrator components initialized successfully")
        
        # Create Gradio app
        logger.info("Creating Gradio application...")
        app = GradioApp(
            job_queue=job_queue,
            backend_router=backend_router,
            workflow_engine=workflow_engine,
            template_registry=template_registry,
            config=config
        )
        logger.info("Gradio application created successfully")
        print("  ✓ Gradio application created")
        print()
        
        # Launch the app
        logger.info("Launching GUI...")
        print("Launching GUI...")
        print(f"Access the GUI at: http://{config.host}:{config.port}")
        if args.share:
            logger.info("Creating public Gradio link...")
            print("Creating public Gradio link...")
        print()
        
        # Prepare launch kwargs
        launch_kwargs = {}
        if args.share:
            launch_kwargs['share'] = True
        
        # Launch
        logger.info("GUI startup complete")
        app.launch(**launch_kwargs)
        
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, shutting down gracefully...")
        print("\n\nShutting down gracefully...")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error during startup: {e}", exc_info=args.debug)
        print(f"\n\nError: {e}", file=sys.stderr)
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
