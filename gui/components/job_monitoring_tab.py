"""
Job Monitoring Tab component for GUI interface.

This module provides the UI component for monitoring job status and viewing results
with real-time updates.
"""

from typing import Any, Dict, List, Optional, Tuple
import gradio as gr
import pandas as pd
from datetime import datetime

from gui.services.job_service import JobService
from gui.error_handling import (
    format_generic_error,
    format_validation_error,
    create_success_message,
    create_warning_message,
    sanitize_error_message
)
from notebook_ml_orchestrator.core.logging_config import LoggerMixin


class JobMonitoringTab(LoggerMixin):
    """UI component for monitoring job status and viewing results."""
    
    def __init__(self, job_service: JobService):
        """
        Initialize job monitoring tab.
        
        Args:
            job_service: Job service instance for job monitoring
        """
        self.job_service = job_service
        self.logger.info("JobMonitoringTab initialized")
    
    def render(self) -> gr.Blocks:
        """
        Render the job monitoring tab within a Gradio Blocks context.
        
        Returns:
            Gradio Blocks component containing the job monitoring interface
        """
        with gr.Blocks() as tab:
            gr.Markdown("## Job Monitoring Dashboard")
            gr.Markdown("Monitor job status and view results in real-time.")
            
            # WebSocket client code for real-time updates
            gr.HTML(self._get_websocket_client_code())
            
            with gr.Row():
                # Filter controls
                with gr.Column(scale=1):
                    gr.Markdown("### Filters")
                    
                    status_filter = gr.Dropdown(
                        label="Status",
                        choices=["all", "queued", "running", "completed", "failed", "cancelled"],
                        value="all",
                        interactive=True
                    )
                    
                    template_filter = gr.Dropdown(
                        label="Template",
                        choices=["all"] + self._get_template_choices(),
                        value="all",
                        interactive=True
                    )
                    
                    backend_filter = gr.Dropdown(
                        label="Backend",
                        choices=["all", "colab", "kaggle", "huggingface", "modal"],
                        value="all",
                        interactive=True
                    )
                    
                    # Page size configuration
                    page_size_dropdown = gr.Dropdown(
                        label="Jobs per page",
                        choices=["10", "25", "50", "100", "200"],
                        value="50",
                        interactive=True
                    )
                    
                    # Refresh controls
                    with gr.Row():
                        refresh_button = gr.Button("Refresh", variant="secondary")
                        auto_refresh = gr.Checkbox(
                            label="Auto-refresh",
                            value=False
                        )
                
                # Job list table
                with gr.Column(scale=2):
                    gr.Markdown("### Jobs")
                    
                    # Pagination info and controls
                    with gr.Row():
                        page_info = gr.Textbox(
                            label="Page Info",
                            value="No jobs loaded",
                            interactive=False,
                            scale=2
                        )
                        
                        prev_page_button = gr.Button(
                            "◀ Previous",
                            variant="secondary",
                            scale=1,
                            visible=False
                        )
                        
                        next_page_button = gr.Button(
                            "Next ▶",
                            variant="secondary",
                            scale=1,
                            visible=False
                        )
                    
                    job_list_table = gr.Dataframe(
                        headers=["Job ID", "Template", "Status", "Backend", "Submitted", "Duration"],
                        datatype=["str", "str", "str", "str", "str", "str"],
                        interactive=False,
                        wrap=True,
                        value=self._get_empty_dataframe()
                    )
                    
                    # Hidden state for pagination
                    current_page = gr.State(value=1)
            
            # Job details section
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Job Details")
                    
                    selected_job_id = gr.Textbox(
                        label="Selected Job ID",
                        value="",
                        interactive=True,
                        placeholder="Enter job ID or select from table"
                    )
                    
                    load_details_button = gr.Button("Load Details", variant="primary")
                    
                    # Job details panel
                    job_details_panel = gr.JSON(
                        label="Job Information",
                        value={}
                    )
            
            # Job logs section
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Job Logs")
                    
                    # Pagination controls
                    with gr.Row():
                        log_page_info = gr.Textbox(
                            label="Log Page Info",
                            value="No logs loaded",
                            interactive=False,
                            scale=2
                        )
                        
                        prev_log_page_button = gr.Button(
                            "◀ Previous",
                            variant="secondary",
                            scale=1,
                            visible=False
                        )
                        
                        next_log_page_button = gr.Button(
                            "Next ▶",
                            variant="secondary",
                            scale=1,
                            visible=False
                        )
                    
                    job_logs_display = gr.Textbox(
                        label="Execution Logs",
                        value="",
                        lines=10,
                        max_lines=20,
                        interactive=False
                    )
                    
                    # Hidden state for pagination
                    log_start_line = gr.State(value=0)
                    log_max_lines = gr.State(value=1000)
                    current_job_id = gr.State(value="")
            
            # Job results section
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Job Results")
                    
                    job_results_display = gr.JSON(
                        label="Job Results",
                        value={}
                    )
                    
                    # Download buttons for results
                    with gr.Row():
                        download_results_button = gr.Button(
                            "Download Results (JSON)",
                            variant="secondary",
                            visible=False
                        )
                        download_outputs_button = gr.Button(
                            "Download Outputs",
                            variant="secondary",
                            visible=False
                        )
            
            # Event handlers
            refresh_button.click(
                fn=self.on_refresh_jobs,
                inputs=[status_filter, template_filter, backend_filter, current_page, page_size_dropdown],
                outputs=[job_list_table, page_info, prev_page_button, next_page_button, current_page]
            )
            
            # Pagination event handlers
            prev_page_button.click(
                fn=self.on_prev_page,
                inputs=[status_filter, template_filter, backend_filter, current_page, page_size_dropdown],
                outputs=[job_list_table, page_info, prev_page_button, next_page_button, current_page]
            )
            
            next_page_button.click(
                fn=self.on_next_page,
                inputs=[status_filter, template_filter, backend_filter, current_page, page_size_dropdown],
                outputs=[job_list_table, page_info, prev_page_button, next_page_button, current_page]
            )
            
            # Page size change handler
            page_size_dropdown.change(
                fn=self.on_page_size_changed,
                inputs=[status_filter, template_filter, backend_filter, page_size_dropdown],
                outputs=[job_list_table, page_info, prev_page_button, next_page_button, current_page]
            )
            
            load_details_button.click(
                fn=self.on_load_job_details,
                inputs=[selected_job_id, log_start_line, log_max_lines],
                outputs=[
                    job_details_panel,
                    job_logs_display,
                    log_page_info,
                    prev_log_page_button,
                    next_log_page_button,
                    log_start_line,
                    current_job_id,
                    job_results_display,
                    download_results_button,
                    download_outputs_button
                ]
            )
            
            # Log pagination event handlers
            prev_log_page_button.click(
                fn=self.on_prev_log_page,
                inputs=[current_job_id, log_start_line, log_max_lines],
                outputs=[
                    job_logs_display,
                    log_page_info,
                    prev_log_page_button,
                    next_log_page_button,
                    log_start_line
                ]
            )
            
            next_log_page_button.click(
                fn=self.on_next_log_page,
                inputs=[current_job_id, log_start_line, log_max_lines],
                outputs=[
                    job_logs_display,
                    log_page_info,
                    prev_log_page_button,
                    next_log_page_button,
                    log_start_line
                ]
            )
            
            # Table row selection (if supported by Gradio)
            job_list_table.select(
                fn=self.on_job_selected_from_table,
                inputs=[job_list_table],
                outputs=[selected_job_id]
            )
        
        return tab
    
    def _get_template_choices(self) -> List[str]:
        """
        Get list of unique template names from jobs.
        
        Returns:
            List of template names
        """
        try:
            result = self.job_service.get_jobs()
            jobs = result['jobs'] if isinstance(result, dict) else result
            templates = list(set(job['template'] for job in jobs if job.get('template')))
            return sorted(templates)
        except Exception as e:
            self.logger.error(f"Failed to retrieve template choices: {e}")
            return []
    
    def _get_empty_dataframe(self) -> pd.DataFrame:
        """
        Get empty dataframe with correct columns.
        
        Returns:
            Empty pandas DataFrame with job columns
        """
        return pd.DataFrame(columns=["Job ID", "Template", "Status", "Backend", "Submitted", "Duration"])
    
    def on_refresh_jobs(
        self,
        status_filter: str,
        template_filter: str,
        backend_filter: str,
        page: int,
        page_size_str: str
    ) -> Tuple[pd.DataFrame, str, gr.Button, gr.Button, int]:
        """
        Refresh job list with filters and pagination applied.
        
        Args:
            status_filter: Status filter value ("all" or specific status)
            template_filter: Template filter value ("all" or specific template)
            backend_filter: Backend filter value ("all" or specific backend)
            page: Current page number
            page_size_str: Page size as string
            
        Returns:
            Tuple of (job_dataframe, page_info, prev_button, next_button, current_page)
        """
        try:
            # Parse page size
            page_size = int(page_size_str)
            
            # Build filter dictionary
            filters = {
                'page': page,
                'page_size': page_size
            }
            
            if status_filter != "all":
                filters['status'] = status_filter
            
            if template_filter != "all":
                filters['template'] = template_filter
            
            if backend_filter != "all":
                filters['backend'] = backend_filter
            
            # Get paginated jobs
            result = self.job_service.get_jobs(filters)
            jobs = result['jobs']
            
            # Convert to DataFrame format
            rows = []
            for job in jobs:
                rows.append({
                    "Job ID": job['job_id'],
                    "Template": job['template'],
                    "Status": job['status'],
                    "Backend": job['backend'] or "N/A",
                    "Submitted": self._format_datetime(job['created_at']),
                    "Duration": self._format_duration(job['duration'])
                })
            
            # Create DataFrame
            if not rows:
                df = self._get_empty_dataframe()
            else:
                df = pd.DataFrame(rows)
            
            # Format page info
            page_info = f"Page {result['page']} of {result['total_pages']} ({result['total_count']} total jobs)"
            
            # Show/hide pagination buttons
            show_prev = result['has_prev']
            show_next = result['has_next']
            
            return (
                df,
                page_info,
                gr.Button(visible=show_prev),
                gr.Button(visible=show_next),
                page
            )
            
        except Exception as e:
            self.logger.error(f"Failed to refresh jobs: {e}")
            return (
                self._get_empty_dataframe(),
                f"Error: {str(e)}",
                gr.Button(visible=False),
                gr.Button(visible=False),
                1
            )
    
    def on_prev_page(
        self,
        status_filter: str,
        template_filter: str,
        backend_filter: str,
        current_page: int,
        page_size_str: str
    ) -> Tuple[pd.DataFrame, str, gr.Button, gr.Button, int]:
        """
        Load previous page of jobs.
        
        Args:
            status_filter: Status filter value
            template_filter: Template filter value
            backend_filter: Backend filter value
            current_page: Current page number
            page_size_str: Page size as string
            
        Returns:
            Tuple of (job_dataframe, page_info, prev_button, next_button, new_page)
        """
        new_page = max(1, current_page - 1)
        return self.on_refresh_jobs(status_filter, template_filter, backend_filter, new_page, page_size_str)
    
    def on_next_page(
        self,
        status_filter: str,
        template_filter: str,
        backend_filter: str,
        current_page: int,
        page_size_str: str
    ) -> Tuple[pd.DataFrame, str, gr.Button, gr.Button, int]:
        """
        Load next page of jobs.
        
        Args:
            status_filter: Status filter value
            template_filter: Template filter value
            backend_filter: Backend filter value
            current_page: Current page number
            page_size_str: Page size as string
            
        Returns:
            Tuple of (job_dataframe, page_info, prev_button, next_button, new_page)
        """
        new_page = current_page + 1
        return self.on_refresh_jobs(status_filter, template_filter, backend_filter, new_page, page_size_str)
    
    def on_page_size_changed(
        self,
        status_filter: str,
        template_filter: str,
        backend_filter: str,
        page_size_str: str
    ) -> Tuple[pd.DataFrame, str, gr.Button, gr.Button, int]:
        """
        Handle page size change by resetting to page 1.
        
        Args:
            status_filter: Status filter value
            template_filter: Template filter value
            backend_filter: Backend filter value
            page_size_str: New page size as string
            
        Returns:
            Tuple of (job_dataframe, page_info, prev_button, next_button, new_page)
        """
        return self.on_refresh_jobs(status_filter, template_filter, backend_filter, 1, page_size_str)
    
    def on_load_job_details(
        self,
        job_id: str,
        start_line: int,
        max_lines: int
    ) -> Tuple[Dict[str, Any], str, str, gr.Button, gr.Button, int, str, Dict[str, Any], gr.Button, gr.Button]:
        """
        Load job details, logs, and results with pagination support.
        
        Args:
            job_id: Job ID to load
            start_line: Starting line for log pagination
            max_lines: Maximum lines per page
            
        Returns:
            Tuple of (job_details, job_logs, log_page_info, prev_button, next_button, 
                     start_line, current_job_id, job_results, download_results_button, 
                     download_outputs_button)
        """
        if not job_id:
            return (
                {},
                "⚠️ Please enter a job ID",
                "No logs loaded",
                gr.Button(visible=False),
                gr.Button(visible=False),
                0,
                "",
                {},
                gr.Button(visible=False),
                gr.Button(visible=False)
            )
        
        try:
            # Get job status/details
            job_details = self.job_service.get_job_status(job_id)
            
            # Get job logs with pagination
            try:
                log_data = self.job_service.get_job_logs(job_id, start_line, max_lines)
                job_logs = log_data['logs']
                
                # Format page info
                page_info = f"Lines {log_data['start_line'] + 1}-{log_data['end_line']} of {log_data['total_lines']}"
                
                # Show/hide pagination buttons
                show_prev = log_data['start_line'] > 0
                show_next = log_data['has_more']
                
            except Exception as e:
                job_logs = f"❌ Failed to load logs: {str(e)}"
                page_info = "Error loading logs"
                show_prev = False
                show_next = False
            
            # Get job results (if completed)
            job_results = {}
            show_download = False
            if job_details['status'] in ['completed', 'failed']:
                try:
                    job_results = self.job_service.get_job_results(job_id)
                    show_download = job_results.get('success', False)
                except Exception as e:
                    job_results = {'error': f"Failed to load results: {str(e)}"}
            
            return (
                job_details,
                job_logs,
                page_info,
                gr.Button(visible=show_prev),
                gr.Button(visible=show_next),
                start_line,
                job_id,
                job_results,
                gr.Button(visible=show_download),
                gr.Button(visible=show_download)
            )
            
        except Exception as e:
            self.logger.error(f"Failed to load job details for {job_id}: {e}")
            return (
                {},
                f"❌ Error loading job: {str(e)}",
                "Error loading logs",
                gr.Button(visible=False),
                gr.Button(visible=False),
                0,
                "",
                {},
                gr.Button(visible=False),
                gr.Button(visible=False)
            )
    
    def on_job_selected_from_table(
        self,
        table_data: pd.DataFrame
    ) -> str:
        """
        Handle job selection from table.
        
        Args:
            table_data: Selected row data from table
            
        Returns:
            Job ID from selected row
        """
        # Note: Gradio table selection behavior may vary
        # This is a placeholder implementation
        try:
            if table_data is not None and len(table_data) > 0:
                # Assuming first column is Job ID
                return str(table_data.iloc[0, 0])
        except Exception as e:
            self.logger.error(f"Failed to extract job ID from table selection: {e}")
        
        return ""
    
    def on_prev_log_page(
        self,
        job_id: str,
        current_start_line: int,
        max_lines: int
    ) -> Tuple[str, str, gr.Button, gr.Button, int]:
        """
        Load previous page of logs.
        
        Args:
            job_id: Job ID to load logs for
            current_start_line: Current starting line
            max_lines: Maximum lines per page
            
        Returns:
            Tuple of (job_logs, log_page_info, prev_button, next_button, new_start_line)
        """
        if not job_id:
            return (
                "No job selected",
                "No logs loaded",
                gr.Button(visible=False),
                gr.Button(visible=False),
                0
            )
        
        try:
            # Calculate new start line
            new_start_line = max(0, current_start_line - max_lines)
            
            # Get logs for new page
            log_data = self.job_service.get_job_logs(job_id, new_start_line, max_lines)
            
            # Format page info
            page_info = f"Lines {log_data['start_line'] + 1}-{log_data['end_line']} of {log_data['total_lines']}"
            
            # Show/hide pagination buttons
            show_prev = log_data['start_line'] > 0
            show_next = log_data['has_more']
            
            return (
                log_data['logs'],
                page_info,
                gr.Button(visible=show_prev),
                gr.Button(visible=show_next),
                new_start_line
            )
            
        except Exception as e:
            self.logger.error(f"Failed to load previous log page for {job_id}: {e}")
            return (
                f"Error loading logs: {str(e)}",
                "Error loading logs",
                gr.Button(visible=False),
                gr.Button(visible=False),
                current_start_line
            )
    
    def on_next_log_page(
        self,
        job_id: str,
        current_start_line: int,
        max_lines: int
    ) -> Tuple[str, str, gr.Button, gr.Button, int]:
        """
        Load next page of logs.
        
        Args:
            job_id: Job ID to load logs for
            current_start_line: Current starting line
            max_lines: Maximum lines per page
            
        Returns:
            Tuple of (job_logs, log_page_info, prev_button, next_button, new_start_line)
        """
        if not job_id:
            return (
                "No job selected",
                "No logs loaded",
                gr.Button(visible=False),
                gr.Button(visible=False),
                0
            )
        
        try:
            # Calculate new start line
            new_start_line = current_start_line + max_lines
            
            # Get logs for new page
            log_data = self.job_service.get_job_logs(job_id, new_start_line, max_lines)
            
            # Format page info
            page_info = f"Lines {log_data['start_line'] + 1}-{log_data['end_line']} of {log_data['total_lines']}"
            
            # Show/hide pagination buttons
            show_prev = log_data['start_line'] > 0
            show_next = log_data['has_more']
            
            return (
                log_data['logs'],
                page_info,
                gr.Button(visible=show_prev),
                gr.Button(visible=show_next),
                new_start_line
            )
            
        except Exception as e:
            self.logger.error(f"Failed to load next log page for {job_id}: {e}")
            return (
                f"Error loading logs: {str(e)}",
                "Error loading logs",
                gr.Button(visible=False),
                gr.Button(visible=False),
                current_start_line
            )
    
    def _format_datetime(self, dt_str: Optional[str]) -> str:
        """
        Format datetime string for display.
        
        Args:
            dt_str: ISO format datetime string
            
        Returns:
            Formatted datetime string
        """
        if not dt_str:
            return "N/A"
        
        try:
            dt = datetime.fromisoformat(dt_str)
            return dt.strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            return dt_str
    
    def _format_duration(self, duration: Optional[float]) -> str:
        """
        Format duration for display.
        
        Args:
            duration: Duration in seconds
            
        Returns:
            Formatted duration string
        """
        if duration is None:
            return "N/A"
        
        if duration < 60:
            return f"{duration:.1f}s"
        elif duration < 3600:
            minutes = duration / 60
            return f"{minutes:.1f}m"
        else:
            hours = duration / 3600
            return f"{hours:.1f}h"
    
    def _get_websocket_client_code(self) -> str:
        """
        Generate JavaScript code for WebSocket client connection.
        
        This method creates the JavaScript code that:
        - Connects to the WebSocket server
        - Handles job status update events
        - Updates the job list table when events are received
        - Updates the job details panel when events are received
        
        Returns:
            HTML string containing JavaScript code for WebSocket client
            
        Requirements validated: 2.2
        """
        return """
        <script>
        (function() {
            // WebSocket connection for real-time job updates
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
                        console.log('WebSocket connected');
                        reconnectAttempts = 0;
                    };
                    
                    ws.onmessage = function(event) {
                        try {
                            const message = JSON.parse(event.data);
                            console.log('WebSocket message received:', message);
                            
                            // Handle different event types
                            if (message.event_type === 'job.status_changed') {
                                handleJobStatusChanged(message.data);
                            } else if (message.event_type === 'job.completed') {
                                handleJobCompleted(message.data);
                            } else if (message.event_type === 'job.failed') {
                                handleJobFailed(message.data);
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
            
            function handleJobStatusChanged(data) {
                console.log('Job status changed:', data);
                
                // Trigger refresh of job list table
                // Find the refresh button and click it programmatically
                const refreshButton = document.querySelector('button[value="Refresh"]');
                if (refreshButton) {
                    refreshButton.click();
                }
                
                // If the updated job is currently being viewed, refresh details
                const selectedJobIdInput = document.querySelector('input[placeholder="Enter job ID or select from table"]');
                if (selectedJobIdInput && selectedJobIdInput.value === data.job_id) {
                    const loadDetailsButton = document.querySelector('button[value="Load Details"]');
                    if (loadDetailsButton) {
                        loadDetailsButton.click();
                    }
                }
            }
            
            function handleJobCompleted(data) {
                console.log('Job completed:', data);
                
                // Show notification
                showNotification(`Job ${data.job_id} completed successfully`, 'success');
                
                // Trigger refresh
                handleJobStatusChanged(data);
            }
            
            function handleJobFailed(data) {
                console.log('Job failed:', data);
                
                // Show notification
                showNotification(`Job ${data.job_id} failed: ${data.error || 'Unknown error'}`, 'error');
                
                // Trigger refresh
                handleJobStatusChanged(data);
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
