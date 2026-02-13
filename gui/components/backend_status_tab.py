"""
Backend Status Tab component for GUI interface.

This module provides the UI component for monitoring backend health and performance.
"""

from typing import Any, Dict, List, Optional, Tuple
import gradio as gr
import pandas as pd

from gui.services.backend_monitor_service import BackendMonitorService
from notebook_ml_orchestrator.core.logging_config import LoggerMixin


class BackendStatusTab(LoggerMixin):
    """UI component for monitoring backend status."""
    
    def __init__(self, backend_monitor: