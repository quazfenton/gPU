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
                    
                    job_list_table = gr.Dataframe(
                        headers=["Job ID", "Template", "Status", "Backend", "Submitted", "Duration"],
                        datatype=["str", "str", "str", "str", "str", "str"],
                        interactive=False,
                        wrap=True,
                        value=self._get_empty_dataframe()
                    )
            
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
                    
                    job_logs_display = gr.Textbox(
                        label="Execution Logs",
                        value="",
                        lines=10,
                        max_lines=20,
                        interactive=False
                    )
            
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
                            v