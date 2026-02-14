"""
Job Monitoring Tab component - V2 with real implementation.

This module provides the UI component for monitoring job status and viewing results.
"""

from typing import Any, Dict, List, Optional, Tuple
import gradio as gr
import pandas as pd
from datetime import datetime

from gui.services.job_service import JobService
from notebook_ml_orchestrator.core.logging_config import LoggerMixin


class JobMonitoringTabV2(LoggerMixin):
    """UI component for monitoring job status and viewing results."""
    
    def __init__(self, job_service: JobService):
        """Initialize job monitoring tab."""
        self.job_service = job_service
        self.logger.info("JobMonitoringTabV2 initialized")
    
    def render(self) -> gr.Blocks:
        """Render the job monitoring tab."""
        with gr.Blocks() as tab:
            gr.Markdown("## Job Monitoring Dashboard")
            gr.Markdown("Monitor job status and view results in real-time.")
            
            with gr.Row():
                # Filters
                status_filter = gr.Dropdown(
                    label="Status Filter",
                    choices=["all", "queued", "running", "completed", "failed", "cancelled"],
                    value="all",
                    interactive=True
                )
                
                refresh_button = gr.Button("🔄 Refresh", variant="secondary")
            
            # Job list table
            job_list_df = gr.Dataframe(
                label="Jobs",
                headers=["Job ID", "Template", "Status", "Backend", "Submitted", "Duration"],
                datatype=["str", "str", "str", "str", "str", "str"],
                interactive=False,
                wrap=True
            )
            
            # Job details section
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Job Details")
                    selected_job_id = gr.Textbox(
                        label="Job ID",
                        placeholder="Enter job ID to view details",
                        interactive=True
                    )
                    load_details_btn = gr.Button("Load Details", variant="secondary")
                    
                    job_details_text = gr.JSON(
                        label="Job Information",
                        value={}
                    )
                    
                    # Output files section
                    gr.Markdown("### Output Files")
                    output_files_list = gr.Textbox(
                        label="Available Output Files",
                        value="No output files available",
                        lines=5,
                        interactive=False
                    )
                    download_output_btn = gr.Button(
                        "📥 Download Output",
                        variant="primary",
                        visible=False
                    )
                    output_file_download = gr.File(
                        label="Download",
                        visible=False
                    )
                
                with gr.Column():
                    gr.Markdown("### Job Logs")
                    job_logs_text = gr.Textbox(
                        label="Job execution logs",
                        value="",
                        lines=15,
                        interactive=False
                    )
            
            # Event handlers
            def load_jobs(status_filter_val):
                return self.get_jobs_list(status_filter_val)
            
            # Load jobs on tab open
            tab.load(
                fn=lambda: self.get_jobs_list("all"),
                outputs=[job_list_df]
            )
            
            # Refresh button
            refresh_button.click(
                fn=load_jobs,
                inputs=[status_filter],
                outputs=[job_list_df]
            )
            
            # Status filter change
            status_filter.change(
                fn=load_jobs,
                inputs=[status_filter],
                outputs=[job_list_df]
            )
            
            # Load job details button
            load_details_btn.click(
                fn=self.on_load_job_details,
                inputs=[selected_job_id],
                outputs=[
                    job_details_text,
                    job_logs_text,
                    output_files_list,
                    download_output_btn,
                    output_file_download
                ]
            )
            
            # Download output button
            download_output_btn.click(
                fn=self.on_download_output,
                inputs=[selected_job_id],
                outputs=[output_file_download]
            )
        
        return tab
    
    def get_jobs_list(self, status_filter: str = "all") -> pd.DataFrame:
        """Get list of jobs as DataFrame."""
        try:
            # Get jobs from service
            filters = {}
            if status_filter and status_filter != "all":
                filters['status'] = status_filter
            
            result = self.job_service.get_jobs(filters)
            
            # Extract jobs list from pagination result
            jobs = result.get('jobs', []) if isinstance(result, dict) else []
            
            if not jobs:
                # Return empty DataFrame with correct structure
                return pd.DataFrame(columns=[
                    "Job ID", "Template", "Status", "Backend", "Submitted", "Duration"
                ])
            
            # Convert to DataFrame format
            rows = []
            for job in jobs:
                # Calculate duration
                duration = "N/A"
                if job.get('completed_at') and job.get('created_at'):
                    try:
                        completed = datetime.fromisoformat(job['completed_at'])
                        created = datetime.fromisoformat(job['created_at'])
                        duration_sec = (completed - created).total_seconds()
                        duration = f"{duration_sec:.1f}s"
                    except:
                        pass
                elif job.get('started_at') and job.get('created_at'):
                    try:
                        started = datetime.fromisoformat(job['started_at'])
                        created = datetime.fromisoformat(job['created_at'])
                        duration_sec = (datetime.now() - started).total_seconds()
                        duration = f"{duration_sec:.1f}s (running)"
                    except:
                        pass
                
                # Format submitted time
                submitted = "N/A"
                if job.get('created_at'):
                    try:
                        created = datetime.fromisoformat(job['created_at'])
                        submitted = created.strftime("%Y-%m-%d %H:%M:%S")
                    except:
                        submitted = str(job['created_at'])
                
                rows.append([
                    job.get('job_id', 'N/A'),
                    job.get('template', 'N/A'),
                    job.get('status', 'N/A'),
                    job.get('backend', 'auto'),
                    submitted,
                    duration
                ])
            
            return pd.DataFrame(rows, columns=[
                "Job ID", "Template", "Status", "Backend", "Submitted", "Duration"
            ])
            
        except Exception as e:
            self.logger.error(f"Failed to load jobs: {e}")
            return pd.DataFrame(columns=[
                "Job ID", "Template", "Status", "Backend", "Submitted", "Duration"
            ])
    
    def on_load_job_details(
        self,
        job_id: str
    ) -> Tuple[Dict, str, str, gr.Button, gr.File]:
        """Load job details, logs, and output files."""
        if not job_id or not job_id.strip():
            return (
                {},
                "Enter a job ID and click 'Load Details'",
                "No output files available",
                gr.Button(visible=False),
                gr.File(visible=False)
            )
        
        try:
            # Get job status
            job_status = self.job_service.get_job_status(job_id)
            
            if not job_status:
                return (
                    {"error": f"Job {job_id} not found"},
                    "Job not found",
                    "No output files available",
                    gr.Button(visible=False),
                    gr.File(visible=False)
                )
            
            # Get job logs
            logs_result = self.job_service.get_job_logs(job_id)
            logs_text = logs_result.get('logs', 'No logs available') if isinstance(logs_result, dict) else str(logs_result)
            
            # Get job results to check for output files
            results = self.job_service.get_job_results(job_id)
            
            output_files_text = "No output files available"
            show_download_btn = False
            
            if results and isinstance(results, dict):
                # Check for file outputs
                file_outputs = []
                for key, value in results.items():
                    if isinstance(value, str) and (
                        value.startswith('/') or 
                        value.startswith('http://') or 
                        value.startswith('https://') or
                        '.' in value.split('/')[-1]  # Has file extension
                    ):
                        file_outputs.append(f"{key}: {value}")
                
                if file_outputs:
                    output_files_text = "Output files:\n" + "\n".join(f"- {f}" for f in file_outputs)
                    show_download_btn = True
            
            return (
                job_status,
                logs_text,
                output_files_text,
                gr.Button(visible=show_download_btn),
                gr.File(visible=False)
            )
            
        except Exception as e:
            self.logger.error(f"Failed to load job details: {e}")
            return (
                {"error": str(e)},
                f"Error loading logs: {str(e)}",
                "Error loading output files",
                gr.Button(visible=False),
                gr.File(visible=False)
            )
    
    def on_download_output(self, job_id: str) -> gr.File:
        """Download job output files."""
        if not job_id or not job_id.strip():
            return gr.File(visible=False)
        
        try:
            # Get job results
            results = self.job_service.get_job_results(job_id)
            
            if not results or not isinstance(results, dict):
                self.logger.warning(f"No results found for job {job_id}")
                return gr.File(visible=False)
            
            # Find first file output
            for key, value in results.items():
                if isinstance(value, str) and (
                    value.startswith('/') or 
                    value.startswith('http://') or 
                    value.startswith('https://')
                ):
                    # Return the file path for download
                    import os
                    if os.path.exists(value):
                        return gr.File(value=value, visible=True)
                    else:
                        self.logger.warning(f"Output file not found: {value}")
            
            return gr.File(visible=False)
            
        except Exception as e:
            self.logger.error(f"Failed to download output: {e}")
            return gr.File(visible=False)

