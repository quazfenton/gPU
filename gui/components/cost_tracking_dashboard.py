"""
Cost Tracking Dashboard component for GUI interface.

This module provides a dashboard for tracking ML job costs across all backends
with real-time cost estimation, budget alerts, and cost breakdown by backend/template.
"""

from typing import Any, Dict, List, Optional
import gradio as gr
from datetime import datetime, timedelta

from gui.services.job_service import JobService
from gui.services.backend_monitor_service import BackendMonitorService
from notebook_ml_orchestrator.core.logging_config import LoggerMixin


class CostTrackingDashboard(LoggerMixin):
    """UI component for tracking ML job costs."""

    def __init__(self, job_service: JobService, backend_monitor: BackendMonitorService):
        """Initialize cost tracking dashboard."""
        self.job_service = job_service
        self.backend_monitor = backend_monitor
        self.logger.info("CostTrackingDashboard initialized")

    def render(self) -> gr.Blocks:
        """Render the cost tracking dashboard."""
        with gr.Blocks() as tab:
            gr.Markdown("## Cost Tracking Dashboard")
            gr.Markdown("Monitor and analyze ML job costs across all backends")
            
            # Summary cards
            with gr.Row():
                total_cost = gr.Number(
                    label="Total Cost (USD)",
                    value=0.0,
                    interactive=False,
                    precision=2
                )
                jobs_count = gr.Number(
                    label="Total Jobs",
                    value=0,
                    interactive=False,
                    precision=0
                )
                avg_cost_per_job = gr.Number(
                    label="Avg Cost per Job (USD)",
                    value=0.0,
                    interactive=False,
                    precision=2
                )
                estimated_monthly = gr.Number(
                    label="Estimated Monthly (USD)",
                    value=0.0,
                    interactive=False,
                    precision=2
                )
            
            gr.Markdown("---")
            
            with gr.Row():
                # Cost by backend
                with gr.Column(scale=1):
                    gr.Markdown("### Cost by Backend")
                    backend_cost_chart = gr.Plot(
                        label="Backend Costs",
                        show_label=False
                    )
                    
                    backend_cost_table = gr.Dataframe(
                        label="Backend Cost Breakdown",
                        headers=["Backend", "Jobs", "Total Cost", "Avg Cost", "% of Total"],
                        datatype=["str", "number", "number", "number", "str"],
                        interactive=False,
                        wrap=True
                    )
                
                # Cost by template
                with gr.Column(scale=1):
                    gr.Markdown("### Cost by Template")
                    template_cost_chart = gr.Plot(
                        label="Template Costs",
                        show_label=False
                    )
                    
                    template_cost_table = gr.Dataframe(
                        label="Template Cost Breakdown",
                        headers=["Template", "Jobs", "Total Cost", "Avg Cost"],
                        datatype=["str", "number", "number", "number"],
                        interactive=False,
                        wrap=True
                    )
            
            gr.Markdown("---")
            
            # Budget settings
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Budget Alerts")
                    
                    budget_input = gr.Number(
                        label="Monthly Budget (USD)",
                        value=100.0,
                        interactive=True,
                        precision=2
                    )
                    
                    alert_threshold = gr.Slider(
                        minimum=50,
                        maximum=100,
                        value=80,
                        label="Alert Threshold (%)",
                        interactive=True
                    )
                    
                    save_budget_btn = gr.Button("Save Budget Settings", variant="primary")
                    
                    budget_status = gr.Textbox(
                        label="Budget Status",
                        value="No budget set",
                        interactive=False
                    )
                
                # Recent expensive jobs
                with gr.Column(scale=1):
                    gr.Markdown("### Most Expensive Jobs (Last 7 Days)")
                    
                    expensive_jobs = gr.Dataframe(
                        label="Expensive Jobs",
                        headers=["Job ID", "Template", "Backend", "Cost", "Duration"],
                        datatype=["str", "str", "str", "number", "str"],
                        interactive=False,
                        wrap=True,
                        max_rows=10
                    )
            
            # Time range selector
            with gr.Row():
                time_range = gr.Radio(
                    choices=["Last 24 Hours", "Last 7 Days", "Last 30 Days", "All Time"],
                    value="Last 7 Days",
                    label="Time Range",
                    interactive=True
                )
                
                refresh_btn = gr.Button("🔄 Refresh Costs", variant="secondary")
            
            # Event handlers
            tab.load(
                fn=self._load_cost_data,
                inputs=[time_range],
                outputs=[
                    total_cost, jobs_count, avg_cost_per_job, estimated_monthly,
                    backend_cost_chart, backend_cost_table,
                    template_cost_chart, template_cost_table,
                    expensive_jobs, budget_status
                ]
            )
            
            refresh_btn.click(
                fn=self._load_cost_data,
                inputs=[time_range],
                outputs=[
                    total_cost, jobs_count, avg_cost_per_job, estimated_monthly,
                    backend_cost_chart, backend_cost_table,
                    template_cost_chart, template_cost_table,
                    expensive_jobs, budget_status
                ]
            )
            
            time_range.change(
                fn=self._load_cost_data,
                inputs=[time_range],
                outputs=[
                    total_cost, jobs_count, avg_cost_per_job, estimated_monthly,
                    backend_cost_chart, backend_cost_table,
                    template_cost_chart, template_cost_table,
                    expensive_jobs, budget_status
                ]
            )
            
            save_budget_btn.click(
                fn=self._save_budget_settings,
                inputs=[budget_input, alert_threshold],
                outputs=[budget_status]
            )
        
        return tab

    def _load_cost_data(self, time_range: str) -> tuple:
        """Load cost data for display."""
        try:
            import matplotlib.pyplot as plt
            
            # Get job data
            jobs_data = self._get_jobs_with_costs(time_range)
            
            # Calculate totals
            total_cost = sum(job['cost'] for job in jobs_data)
            jobs_count = len(jobs_data)
            avg_cost = total_cost / jobs_count if jobs_count > 0 else 0.0
            
            # Estimate monthly cost
            if time_range == "Last 24 Hours":
                estimated_monthly = total_cost * 30
            elif time_range == "Last 7 Days":
                estimated_monthly = total_cost * 4.33
            elif time_range == "Last 30 Days":
                estimated_monthly = total_cost
            else:  # All time
                estimated_monthly = total_cost  # Best guess
            
            # Cost by backend
            backend_costs = self._calculate_backend_costs(jobs_data)
            backend_chart = self._create_backend_chart(backend_costs)
            backend_table = self._create_backend_table(backend_costs, total_cost)
            
            # Cost by template
            template_costs = self._calculate_template_costs(jobs_data)
            template_chart = self._create_template_chart(template_costs)
            template_table = self._create_template_table(template_costs)
            
            # Most expensive jobs
            expensive = sorted(jobs_data, key=lambda x: x['cost'], reverse=True)[:10]
            expensive_table = [
                [job['id'], job['template'], job['backend'], f"${job['cost']:.4f}", job['duration']]
                for job in expensive
            ]
            
            # Budget status
            budget_status = "Budget tracking not yet implemented"
            
            return (
                total_cost, jobs_count, avg_cost, estimated_monthly,
                backend_chart, backend_table,
                template_chart, template_table,
                expensive_table, budget_status
            )
            
        except Exception as e:
            self.logger.error(f"Failed to load cost data: {e}")
            # Return empty data on error
            return (0.0, 0, 0.0, 0.0, None, [], None, [], [], "Error loading cost data")

    def _get_jobs_with_costs(self, time_range: str) -> List[Dict[str, Any]]:
        """Get jobs with cost calculations."""
        # Get jobs from job service
        filters = {}
        jobs_result = self.job_service.get_jobs(filters)
        jobs = jobs_result.get('jobs', []) if isinstance(jobs_result, dict) else []
        
        # Filter by time range
        now = datetime.now()
        if time_range == "Last 24 Hours":
            cutoff = now - timedelta(hours=24)
        elif time_range == "Last 7 Days":
            cutoff = now - timedelta(days=7)
        elif time_range == "Last 30 Days":
            cutoff = now - timedelta(days=30)
        else:  # All time
            cutoff = datetime(2000, 1, 1)
        
        # Calculate cost for each job
        jobs_with_costs = []
        for job in jobs:
            try:
                created = datetime.fromisoformat(job['created_at']) if job.get('created_at') else now
                if created < cutoff:
                    continue
                
                # Calculate cost based on backend and duration
                cost = self._calculate_job_cost(job)
                
                jobs_with_costs.append({
                    'id': job['job_id'],
                    'template': job['template'],
                    'backend': job.get('backend', 'unknown'),
                    'cost': cost,
                    'duration': job.get('duration', 'N/A'),
                    'created_at': created
                })
            except Exception as e:
                self.logger.debug(f"Error processing job {job.get('job_id', 'unknown')}: {e}")
                continue
        
        return jobs_with_costs

    def _calculate_job_cost(self, job: Dict[str, Any]) -> float:
        """Calculate cost for a single job."""
        backend = job.get('backend', 'unknown')
        duration_str = job.get('duration', 'N/A')
        
        # Parse duration
        try:
            if duration_str == 'N/A' or not duration_str:
                duration_hours = 0.01  # Default minimum
            else:
                # Parse "X.Xs" or "X.Xm" format
                if duration_str.endswith('s'):
                    duration_seconds = float(duration_str[:-1].split()[0])
                elif duration_str.endswith('m'):
                    duration_seconds = float(duration_str[:-1].split()[0]) * 60
                else:
                    duration_seconds = 60  # Default 1 minute
                
                duration_hours = duration_seconds / 3600
                if duration_hours < 0.01:
                    duration_hours = 0.01
        except:
            duration_hours = 0.01
        
        # Get backend cost rate
        cost_rates = {
            'modal': 1.10,  # Average $1.10/hr (A10G)
            'huggingface': 0.0,  # Free tier
            'kaggle': 0.0,  # Free tier
            'colab': 0.0,  # Free tier
            'auto': 0.55,  # Average estimate
            'unknown': 0.55
        }
        
        rate = cost_rates.get(backend.lower() if backend else 'unknown', 0.55)
        
        return rate * duration_hours

    def _calculate_backend_costs(self, jobs: List[Dict]) -> Dict[str, Dict]:
        """Calculate costs grouped by backend."""
        backend_data = {}
        
        for job in jobs:
            backend = job['backend']
            if backend not in backend_data:
                backend_data[backend] = {'jobs': 0, 'total_cost': 0.0}
            
            backend_data[backend]['jobs'] += 1
            backend_data[backend]['total_cost'] += job['cost']
        
        # Calculate averages
        for backend in backend_data:
            jobs_count = backend_data[backend]['jobs']
            backend_data[backend]['avg_cost'] = backend_data[backend]['total_cost'] / jobs_count if jobs_count > 0 else 0.0
        
        return backend_data

    def _calculate_template_costs(self, jobs: List[Dict]) -> Dict[str, Dict]:
        """Calculate costs grouped by template."""
        template_data = {}
        
        for job in jobs:
            template = job['template']
            if template not in template_data:
                template_data[template] = {'jobs': 0, 'total_cost': 0.0}
            
            template_data[template]['jobs'] += 1
            template_data[template]['total_cost'] += job['cost']
        
        # Calculate averages
        for template in template_data:
            jobs_count = template_data[template]['jobs']
            template_data[template]['avg_cost'] = template_data[template]['total_cost'] / jobs_count if jobs_count > 0 else 0.0
        
        return template_data

    def _create_backend_chart(self, backend_costs: Dict) -> Any:
        """Create matplotlib chart for backend costs."""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            
            backends = list(backend_costs.keys())
            costs = [backend_costs[b]['total_cost'] for b in backends]
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(backends, costs, color=['#2196F3', '#4CAF50', '#FF9800', '#9C27B0', '#F44336'])
            ax.set_xlabel('Backend')
            ax.set_ylabel('Cost (USD)')
            ax.set_title('Cost by Backend')
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            return fig
            
        except ImportError:
            return None

    def _create_template_chart(self, template_costs: Dict) -> Any:
        """Create matplotlib chart for template costs."""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            
            # Get top 10 templates
            sorted_templates = sorted(template_costs.items(), key=lambda x: x[1]['total_cost'], reverse=True)[:10]
            
            templates = [t[0] for t in sorted_templates]
            costs = [t[1]['total_cost'] for t in sorted_templates]
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.barh(templates, costs, color='#2196F3')
            ax.set_xlabel('Cost (USD)')
            ax.set_ylabel('Template')
            ax.set_title('Cost by Template (Top 10)')
            plt.tight_layout()
            
            return fig
            
        except ImportError:
            return None

    def _create_backend_table(self, backend_costs: Dict, total_cost: float) -> List[List]:
        """Create backend cost breakdown table."""
        table = []
        
        for backend, data in backend_costs.items():
            pct = (data['total_cost'] / total_cost * 100) if total_cost > 0 else 0
            table.append([
                backend,
                data['jobs'],
                f"${data['total_cost']:.4f}",
                f"${data['avg_cost']:.4f}",
                f"{pct:.1f}%"
            ])
        
        return sorted(table, key=lambda x: float(x[2].replace('$', '')), reverse=True)

    def _create_template_table(self, template_costs: Dict) -> List[List]:
        """Create template cost breakdown table."""
        table = []
        
        for template, data in template_costs.items():
            table.append([
                template,
                data['jobs'],
                f"${data['total_cost']:.4f}",
                f"${data['avg_cost']:.4f}"
            ])
        
        return sorted(table, key=lambda x: float(x[2].replace('$', '')), reverse=True)[:20]

    def _save_budget_settings(self, budget: float, threshold: float) -> str:
        """Save budget settings."""
        # In a real implementation, this would save to database
        return f"Budget set to ${budget:.2f}/month with {threshold}% alert threshold"
