"""Main Gradio application for the GUI interface."""

from typing import Optional
import gradio as gr

from notebook_ml_orchestrator.core.job_queue import JobQueue
from notebook_ml_orchestrator.core.backend_router import BackendRouter
from notebook_ml_orchestrator.core.workflow_engine import WorkflowEngine
from notebook_ml_orchestrator.core.template_registry import TemplateRegistry
from gui.config import GUIConfig


class GradioApp:
    """Main Gradio application orchestrating all UI components."""
    
    def __init__(
        self,
        job_queue: JobQueue,
        backend_router: BackendRouter,
        workflow_engine: WorkflowEngine,
        template_registry: TemplateRegistry,
        config: Optional[GUIConfig] = None
    ):
        """Initialize the Gradio app with orchestrator components.
        
        Args:
            job_queue: Job queue for job persistence
            backend_router: Backend router for job routing
            workflow_engine: Workflow engine for DAG execution
            template_registry: Template registry for template discovery
            config: GUI configuration (uses defaults if not provided)
        """
        self.job_queue = job_queue
        self.backend_router = backend_router
        self.workflow_engine = workflow_engine
        self.template_registry = template_registry
        self.config = config or GUIConfig()
        
    def build_interface(self) -> gr.Blocks:
        """Build the complete Gradio interface with all tabs.
        
        Returns:
            Gradio Blocks interface
        """
        with gr.Blocks(title="Notebook ML Orchestrator", theme=self.config.theme) as interface:
            gr.Markdown("# Notebook ML Orchestrator")
            gr.Markdown("Web-based interface for ML job orchestration")
            
            with gr.Tabs():
                with gr.Tab("Job Submission"):
                    gr.Markdown("Submit ML jobs using templates")
                    # Placeholder for job submission components
                    
                with gr.Tab("Job Monitoring"):
                    gr.Markdown("Monitor job status and view results")
                    # Placeholder for job monitoring components
                    
                with gr.Tab("Workflow Builder"):
                    gr.Markdown("Build and execute multi-step workflows")
                    # Placeholder for workflow builder components
                    
                with gr.Tab("Template Management"):
                    gr.Markdown("Browse and explore available templates")
                    # Placeholder for template management components
                    
                with gr.Tab("Backend Status"):
                    gr.Markdown("Monitor backend health and performance")
                    # Placeholder for backend status components
        
        return interface
        
    def launch(self, host: Optional[str] = None, port: Optional[int] = None):
        """Launch the Gradio application.
        
        Args:
            host: Host address (uses config default if not provided)
            port: Port number (uses config default if not provided)
        """
        interface = self.build_interface()
        interface.launch(
            server_name=host or self.config.host,
            server_port=port or self.config.port
        )
