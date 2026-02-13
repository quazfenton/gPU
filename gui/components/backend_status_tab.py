"""
Backend Status Tab component for GUI interface.

This module provides the UI component for monitoring backend health and performance.
"""

from typing import Any, Dict, List, Tuple
import gradio as gr
import pandas as pd
from datetime import datetime

from gui.services.backend_monitor_service import BackendMonitorService
from gui.error_handling import (
    format_backend_error,
    format_generic_error,
    create_success_message,
    create_warning_message,
    sanitize_error_message
)
from notebook_ml_orchestrator.core.logging_config import LoggerMixin


class BackendStatusTab(LoggerMixin):
    """UI component for monitoring backend status."""
    
    def __init__(self, backend_monitor: BackendMonitorService):
        """
        Initialize backend status tab.
        
        Args:
            backend_monitor: Backend monitor service instance
        """
        self.backend_monitor = backend_monitor
        self.logger.info("BackendStatusTab initialized")
    
    def render(self) -> gr.Blocks:
        """
        Render the backend status tab within a Gradio Blocks context.
        
        Returns:
            Gradio Blocks component containing the backend status interface
            
        Requirements:
            - 5.1: Display all registered backends with health status indicators
            - 5.2: Update status indicators in real-time using WebSocket connections
            - 5.3: Display health metrics
            - 5.7: Provide manual health check triggers
        """
        with gr.Blocks() as tab:
            gr.Markdown("## Backend Status Panel")
            gr.Markdown("Monitor backend health and performance metrics.")
            
            # WebSocket client code for real-time updates
            gr.HTML(self._get_websocket_client_code())
            
            with gr.Row():
                # Backend status table
                with gr.Column(scale=2):
                    gr.Markdown("### Backends")
                    
                    backend_status_table = gr.Dataframe(
                        headers=["Backend", "Status", "Uptime %", "Avg Response Time", "Jobs Executed"],
                        datatype=["str", "str", "number", "number", "number"],
                        interactive=False,
                        wrap=True,
                        value=self._get_initial_backends()
                    )
                    
                    # Refresh controls
                    with gr.Row():
                        refresh_button = gr.Button("Refresh", variant="secondary")
                        auto_refresh = gr.Checkbox(
                            label="Auto-refresh",
                            value=False
                        )
                
                # Backend selection and controls
                with gr.Column(scale=1):
                    gr.Markdown("### Controls")
                    
                    selected_backend_name = gr.Textbox(
                        label="Selected Backend",
                        value="",
                        interactive=True,
                        placeholder="Enter backend name or select from table"
                    )
                    
                    load_details_button = gr.Button("Load Details", variant="primary")
                    
                    health_check_button = gr.Button(
                        "Manual Health Check",
                        variant="secondary"
                    )
                    
                    health_check_result = gr.Textbox(
                        label="Health Check Result",
                        value="",
                        interactive=False,
                        visible=False
                    )
            
            # Backend details section
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Backend Details")
                    
                    backend_details_panel = gr.JSON(
                        label="Backend Information",
                        value={}
                    )
            
            # Health metrics section
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Health Metrics")
                    
                    health_metrics_display = gr.Markdown(
                        value="Select a backend to view health metrics."
                    )
            
            # Event handlers
            refresh_button.click(
                fn=self.on_refresh_backends,
                inputs=[],
                outputs=[backend_status_table]
            )
            
            load_details_button.click(
                fn=self.on_load_backend_details,
                inputs=[selected_backend_name],
                outputs=[backend_details_panel, health_metrics_display]
            )
            
            health_check_button.click(
                fn=self.on_trigger_health_check,
                inputs=[selected_backend_name],
                outputs=[health_check_result, backend_status_table]
            )
            
            # Table row selection (if supported by Gradio)
            backend_status_table.select(
                fn=self.on_backend_selected_from_table,
                inputs=[backend_status_table],
                outputs=[selected_backend_name]
            )
        
        return tab

    
    def _get_initial_backends(self) -> pd.DataFrame:
        """
        Get initial backend status list for display.
        
        Returns:
            DataFrame with all backend statuses
        """
        try:
            backends = self.backend_monitor.get_backends_status()
            return self._backends_to_dataframe(backends)
        except Exception as e:
            self.logger.error(f"Failed to load initial backends: {e}")
            return self._get_empty_dataframe()
    
    def _get_empty_dataframe(self) -> pd.DataFrame:
        """
        Get empty dataframe with correct columns.
        
        Returns:
            Empty pandas DataFrame with backend columns
        """
        return pd.DataFrame(columns=["Backend", "Status", "Uptime %", "Avg Response Time", "Jobs Executed"])
    
    def _backends_to_dataframe(self, backends: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Convert backend list to DataFrame format.
        
        Args:
            backends: List of backend status dictionaries
            
        Returns:
            DataFrame with backend data
        """
        if not backends:
            return self._get_empty_dataframe()
        
        rows = []
        for backend in backends:
            rows.append({
                "Backend": backend['name'],
                "Status": backend['status'],
                "Uptime %": round(backend['uptime_percentage'], 2),
                "Avg Response Time": round(backend['avg_response_time'], 2),
                "Jobs Executed": backend['jobs_executed']
            })
        
        return pd.DataFrame(rows)
    
    def on_refresh_backends(self) -> pd.DataFrame:
        """
        Refresh backend status list.
        
        Returns:
            Updated DataFrame with backend statuses
        """
        try:
            backends = self.backend_monitor.get_backends_status()
            
            self.logger.debug(f"Refreshed status for {len(backends)} backends")
            
            return self._backends_to_dataframe(backends)
            
        except Exception as e:
            self.logger.error(f"Failed to refresh backends: {e}")
            return self._get_empty_dataframe()
    
    def on_load_backend_details(
        self,
        backend_name: str
    ) -> Tuple[Dict[str, Any], str]:
        """
        Load backend details and health metrics.
        
        Args:
            backend_name: Backend name to load
            
        Returns:
            Tuple of (backend_details, health_metrics_markdown)
        """
        if not backend_name:
            return (
                {},
                "Select a backend to view health metrics."
            )
        
        try:
            # Get backend details
            backend_details = self.backend_monitor.get_backend_details(backend_name)
            
            # Format health metrics as markdown
            health_metrics_md = self._format_health_metrics(backend_details)
            
            return (
                backend_details,
                health_metrics_md
            )
            
        except ValueError as e:
            # Backend not found
            self.logger.warning(f"Backend not found: {backend_name}")
            return (
                {},
                f"Backend '{backend_name}' not found."
            )
        except Exception as e:
            self.logger.error(f"Failed to load backend details for {backend_name}: {e}")
            return (
                {},
                f"Error loading backend: {str(e)}"
            )
    
    def on_trigger_health_check(
        self,
        backend_name: str
    ) -> Tuple[str, pd.DataFrame]:
        """
        Trigger manual health check for a backend.
        
        Args:
            backend_name: Backend name to check
            
        Returns:
            Tuple of (health_check_result_message, updated_backend_table)
        """
        if not backend_name:
            return (
                "⚠️ Please select a backend first.",
                self._get_empty_dataframe()
            )
        
        try:
            # Trigger health check
            result = self.backend_monitor.trigger_health_check(backend_name)
            
            # Format result message
            status_icon = "✅" if result['status'] == 'healthy' else "❌"
            message = (
                f"{status_icon} **Health check completed for '{result['backend_name']}'**\n\n"
                f"**Status:** {result['status']}\n\n"
                f"**Timestamp:** {result['timestamp']}\n\n"
                f"**Message:** {result['message']}"
            )
            
            self.logger.info(f"Manual health check triggered for {backend_name}: {result['status']}")
            
            # Refresh backend table
            backends = self.backend_monitor.get_backends_status()
            updated_table = self._backends_to_dataframe(backends)
            
            return (message, updated_table)
            
        except ValueError as e:
            # Backend not found
            self.logger.warning(f"Backend not found: {backend_name}")
            return (
                f"❌ Backend '{backend_name}' not found.",
                self._get_empty_dataframe()
            )
        except Exception as e:
            self.logger.error(f"Failed to trigger health check for {backend_name}: {e}")
            return (
                f"❌ **Error:** Failed to trigger health check\n\n**Details:** {str(e)}",
                self._get_empty_dataframe()
            )
    
    def on_backend_selected_from_table(
        self,
        table_data: pd.DataFrame
    ) -> str:
        """
        Handle backend selection from table.
        
        Args:
            table_data: Selected row data from table
            
        Returns:
            Backend name from selected row
        """
        try:
            if table_data is not None and len(table_data) > 0:
                # First column is backend name
                return str(table_data.iloc[0, 0])
        except Exception as e:
            self.logger.error(f"Failed to extract backend name from table selection: {e}")
        
        return ""
    
    def _format_health_metrics(self, backend_details: Dict[str, Any]) -> str:
        """
        Format health metrics as markdown.
        
        Args:
            backend_details: Backend details dictionary
            
        Returns:
            Markdown formatted health metrics
        """
        health_metrics = backend_details.get('health_metrics', {})
        
        lines = [
            "### Health Status",
            f"**Current Status:** {backend_details.get('status', 'unknown')}",
            "",
            "### Uptime Metrics",
            f"**Uptime Percentage:** {health_metrics.get('uptime_percentage', 0.0):.2f}%",
            f"**Total Checks:** {health_metrics.get('total_checks', 0)}",
            f"**Healthy Checks:** {health_metrics.get('healthy_checks', 0)}",
            f"**Failure Rate:** {health_metrics.get('failure_rate', 0.0):.2f}%",
            "",
            "### Recent Activity",
            f"**Last Check:** {health_metrics.get('last_check', 'N/A')}",
            f"**Consecutive Failures:** {health_metrics.get('consecutive_failures', 0)}",
            f"**Consecutive Job Failures:** {health_metrics.get('consecutive_job_failures', 0)}",
        ]
        
        # Add error information if available
        last_error = health_metrics.get('last_error')
        last_error_timestamp = health_metrics.get('last_error_timestamp')
        if last_error:
            lines.extend([
                "",
                "### Last Error",
                f"**Error:** {last_error}",
                f"**Timestamp:** {last_error_timestamp or 'N/A'}"
            ])
        
        # Add capabilities
        capabilities = backend_details.get('capabilities', {})
        lines.extend([
            "",
            "### Capabilities",
            f"**GPU Support:** {'Yes' if capabilities.get('supports_gpu', False) else 'No'}",
            f"**Batch Support:** {'Yes' if capabilities.get('supports_batch', False) else 'No'}",
            f"**Max Concurrent Jobs:** {capabilities.get('max_concurrent_jobs', 'N/A')}",
            f"**Max Job Duration:** {capabilities.get('max_job_duration_minutes', 'N/A')} minutes",
            f"**Cost per Hour:** ${capabilities.get('cost_per_hour', 0.0):.2f}",
        ])
        
        # Add cost metrics
        cost_metrics = backend_details.get('cost_metrics', {})
        lines.extend([
            "",
            "### Cost Tracking",
            f"**Total Cost:** ${cost_metrics.get('total_cost', 0.0):.2f}"
        ])
        
        # Add configuration status
        lines.extend([
            "",
            "### Configuration",
            f"**Status:** {backend_details.get('configuration_status', 'unknown')}"
        ])
        
        return "\n".join(lines)
    
    def _get_websocket_client_code(self) -> str:
        """
        Generate JavaScript code for WebSocket client connection.
        
        This method creates the JavaScript code that:
        - Connects to the WebSocket server
        - Handles backend status update events
        - Updates the backend status table when events are received
        
        Returns:
            HTML string containing JavaScript code for WebSocket client
            
        Requirements validated: 5.2
        """
        return """
        <script>
        (function() {
            // WebSocket connection for real-time backend status updates
            let ws = null;
            let reconnectAttempts = 0;
            const maxReconnectAttempts = 10;
            const reconnectDelay = 3000; // 3 seconds
            
            function connectWebSocket() {
                // Determine WebSocket URL based on current location
                const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                const host = window.location.hostname;
                const port = 7861; // WebSocket server port from config
                const wsUrl = `${protocol}//${host}:${port}/ws`;
                
                console.log('Connecting to WebSocket:', wsUrl);
                
                try {
                    ws = new WebSocket(wsUrl);
                    
                    ws.onopen = function(event) {
                        console.log('WebSocket connected for backend status updates');
                        reconnectAttempts = 0;
                    };
                    
                    ws.onmessage = function(event) {
                        try {
                            const message = JSON.parse(event.data);
                            console.log('WebSocket message received:', message);
                            
                            // Handle backend status change events
                            if (message.event_type === 'backend.status_changed') {
                                handleBackendStatusChanged(message.data);
                            }
                        } catch (error) {
                            console.error('Error parsing WebSocket message:', error);
                        }
                    };
                    
                    ws.onerror = function(error) {
                        console.error('WebSocket error:', error);
                    };
                    
                    ws.onclose = function(event) {
                        console.log('WebSocket disconnected');
                        ws = null;
                        
                        // Attempt to reconnect with exponential backoff
                        if (reconnectAttempts < maxReconnectAttempts) {
                            reconnectAttempts++;
                            const delay = reconnectDelay * Math.pow(1.5, reconnectAttempts - 1);
                            console.log(`Reconnecting in ${delay}ms (attempt ${reconnectAttempts}/${maxReconnectAttempts})`);
                            setTimeout(connectWebSocket, delay);
                        } else {
                            console.error('Max reconnection attempts reached');
                        }
                    };
                } catch (error) {
                    console.error('Error creating WebSocket connection:', error);
                }
            }
            
            function handleBackendStatusChanged(data) {
                console.log('Backend status changed:', data);
                
                // Trigger refresh of backend status table
                // Find the refresh button and click it programmatically
                const refreshButtons = document.querySelectorAll('button');
                for (let button of refreshButtons) {
                    if (button.textContent.trim() === 'Refresh') {
                        button.click();
                        break;
                    }
                }
                
                // If the updated backend is currently being viewed, refresh details
                const selectedBackendInput = document.querySelector('input[placeholder="Enter backend name or select from table"]');
                if (selectedBackendInput && selectedBackendInput.value === data.backend_name) {
                    const loadDetailsButtons = document.querySelectorAll('button');
                    for (let button of loadDetailsButtons) {
                        if (button.textContent.trim() === 'Load Details') {
                            button.click();
                            break;
                        }
                    }
                }
                
                // Show notification for backend status changes
                const statusMessage = `Backend ${data.backend_name} status changed to ${data.status}`;
                showNotification(statusMessage, getNotificationType(data.status));
            }
            
            function getNotificationType(status) {
                // Map backend status to notification type
                if (status === 'healthy') {
                    return 'success';
                } else if (status === 'unhealthy') {
                    return 'error';
                } else if (status === 'degraded') {
                    return 'warning';
                } else {
                    return 'info';
                }
            }
            
            function showNotification(message, type) {
                // Create a simple notification element
                const notification = document.createElement('div');
                notification.textContent = message;
                notification.style.position = 'fixed';
                notification.style.top = '20px';
                notification.style.right = '20px';
                notification.style.padding = '15px 20px';
                notification.style.borderRadius = '5px';
                notification.style.zIndex = '10000';
                notification.style.boxShadow = '0 2px 10px rgba(0,0,0,0.2)';
                notification.style.maxWidth = '400px';
                notification.style.fontSize = '14px';
                
                if (type === 'success') {
                    notification.style.backgroundColor = '#4CAF50';
                    notification.style.color = 'white';
                } else if (type === 'error') {
                    notification.style.backgroundColor = '#f44336';
                    notification.style.color = 'white';
                } else if (type === 'warning') {
                    notification.style.backgroundColor = '#ff9800';
                    notification.style.color = 'white';
                } else {
                    notification.style.backgroundColor = '#2196F3';
                    notification.style.color = 'white';
                }
                
                document.body.appendChild(notification);
                
                // Remove notification after 5 seconds
                setTimeout(function() {
                    notification.style.transition = 'opacity 0.5s';
                    notification.style.opacity = '0';
                    setTimeout(function() {
                        document.body.removeChild(notification);
                    }, 500);
                }, 5000);
            }
            
            // Connect when the page loads
            if (document.readyState === 'loading') {
                document.addEventListener('DOMContentLoaded', connectWebSocket);
            } else {
                connectWebSocket();
            }
            
            // Cleanup on page unload
            window.addEventListener('beforeunload', function() {
                if (ws) {
                    ws.close();
                }
            });
        })();
        </script>
        """
