"""Configuration management for GUI application."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class GUIConfig:
    """Configuration for GUI application.
    
    Configuration can be loaded from environment variables or provided directly.
    Environment variables take precedence over default values.
    
    Attributes:
        host: Host address to bind the GUI server (default: "0.0.0.0")
        port: Port for the GUI server (default: 7860)
        websocket_port: Port for WebSocket connections (default: 7861)
        enable_auth: Whether to enable authentication (default: False)
        auth_provider: Authentication provider name (default: None)
        enable_websocket: Whether to enable WebSocket for real-time updates (default: True)
        theme: Gradio theme name (default: "default")
        page_size: Number of items per page in lists (default: 50)
        auto_refresh_interval: Auto-refresh interval in seconds (default: 5)
        session_timeout: Session timeout in seconds (default: 3600)
    """
    
    host: str = field(default="0.0.0.0")
    port: int = field(default=7860)
    websocket_port: int = field(default=7861)
    enable_auth: bool = field(default=False)
    auth_provider: Optional[str] = field(default=None)
    enable_websocket: bool = field(default=True)
    theme: str = field(default="default")
    page_size: int = field(default=50)
    auto_refresh_interval: int = field(default=5)  # seconds
    session_timeout: int = field(default=3600)  # seconds
    
    @classmethod
    def from_env(cls) -> "GUIConfig":
        """Load configuration from environment variables.
        
        Environment variables:
            GUI_HOST: Host address (default: "0.0.0.0")
            GUI_PORT: Port number (default: 7860)
            GUI_WEBSOCKET_PORT: WebSocket port (default: 7861)
            GUI_ENABLE_AUTH: Enable authentication ("true"/"false", default: "false")
            GUI_AUTH_PROVIDER: Authentication provider name
            GUI_ENABLE_WEBSOCKET: Enable WebSocket ("true"/"false", default: "true")
            GUI_THEME: Gradio theme name (default: "default")
            GUI_PAGE_SIZE: Items per page (default: 50)
            GUI_AUTO_REFRESH_INTERVAL: Auto-refresh interval in seconds (default: 5)
            GUI_SESSION_TIMEOUT: Session timeout in seconds (default: 3600)
        
        Returns:
            GUIConfig instance with values from environment variables
        """
        return cls(
            host=os.getenv("GUI_HOST", "0.0.0.0"),
            port=int(os.getenv("GUI_PORT", "7860")),
            websocket_port=int(os.getenv("GUI_WEBSOCKET_PORT", "7861")),
            enable_auth=os.getenv("GUI_ENABLE_AUTH", "false").lower() == "true",
            auth_provider=os.getenv("GUI_AUTH_PROVIDER"),
            enable_websocket=os.getenv("GUI_ENABLE_WEBSOCKET", "true").lower() == "true",
            theme=os.getenv("GUI_THEME", "default"),
            page_size=int(os.getenv("GUI_PAGE_SIZE", "50")),
            auto_refresh_interval=int(os.getenv("GUI_AUTO_REFRESH_INTERVAL", "5")),
            session_timeout=int(os.getenv("GUI_SESSION_TIMEOUT", "3600"))
        )
    
    @classmethod
    def from_file(cls, config_path: str) -> "GUIConfig":
        """Load configuration from a configuration file.
        
        Supports .env file format with key=value pairs.
        
        Args:
            config_path: Path to configuration file (supports .env format)
            
        Returns:
            GUIConfig instance with values from configuration file
            
        Raises:
            FileNotFoundError: If the configuration file does not exist
        """
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        # Load environment variables from file
        from dotenv import load_dotenv
        load_dotenv(config_path)
        
        # Use from_env to read the loaded variables
        return cls.from_env()
