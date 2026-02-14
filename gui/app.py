"""Main Gradio application for the GUI interface."""

from typing import Optional
import gradio as gr
import threading
import time

from notebook_ml_orchestrator.core.interfaces import (
    JobQueueInterface,
    BackendRouterInterface,
    WorkflowEngineInterface
)
from notebook_ml_orchestrator.core.template_registry import TemplateRegistry
from notebook_ml_orchestrator.core.logging_config import LoggerMixin
from notebook_ml_orchestrator.core.models import JobStatus, HealthStatus, WorkflowStatus

from gui.config import GUIConfig
from gui.services.job_service import JobService
from gui.services.template_service import TemplateService
from gui.services.workflow_service import WorkflowService
from gui.services.backend_monitor_service import BackendMonitorService
from gui.components.job_submission_tab_v2 import JobSubmissionTabV2
from gui.components.job_monitoring_tab_v2 import JobMonitoringTabV2
from gui.components.workflow_builder_tab import WorkflowBuilderTab
from gui.components.template_management_tab_v2 import TemplateManagementTabV2
from gui.components.backend_status_tab import BackendStatusTab
from gui.components.backend_registration_tab import BackendRegistrationTab
from gui.components.file_manager_tab import FileManagerTab
from gui.components.file_upload_handler import FileUploadHandler
from gui.events import EventEmitter
from gui.websocket_server import WebSocketServer
from gui.auth import AuthenticationMiddleware, SimpleAuthProvider, SessionManager, Role
from gui.health import HealthChecker, create_health_check_handler
from gui.rate_limiter import RateLimiter, RateLimitConfig, RateLimitError
import gui


class GradioApp(LoggerMixin):
    """Main Gradio application orchestrating all UI components.
    
    This class initializes all service layer components and UI tabs,
    builds the complete Gradio interface, and provides a launch method
    to start the web server.
    
    Requirements:
        - 7.1: Implemented using Gradio framework
        - 7.2: Uses Gradio Blocks for custom layout composition
        - 7.3: Organizes components into tabs
        - 7.6: Uses Gradio themes for consistent visual styling
    """
    
    def __init__(
        self,
        job_queue: JobQueueInterface,
        backend_router: BackendRouterInterface,
        workflow_engine: WorkflowEngineInterface,
        template_registry: TemplateRegistry,
        config: Optional[GUIConfig] = None
    ):
        """Initialize the Gradio app with orchestrator components.
        
        This method initializes all service layer components and UI tabs
        that will be used in the interface. It also sets up event emitters,
        WebSocket server, authentication, and observers for real-time updates.
        
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
        
        # Initialize event emitter for real-time updates
        self.logger.info("Initializing event emitter")
        self.event_emitter = EventEmitter()
        
        # Initialize WebSocket server if enabled
        self.websocket_server = None
        if self.config.enable_websocket:
            self.logger.info("Initializing WebSocket server")
            self.websocket_server = WebSocketServer(self.event_emitter)
            self.websocket_server.setup_listeners()
        
        # Initialize authentication if enabled
        self.auth_middleware = None
        if self.config.enable_auth:
            self.logger.info(f"Initializing authentication with provider: {self.config.auth_provider}")
            # For now, use SimpleAuthProvider with default users
            # In production, this would be configured based on auth_provider
            auth_provider = SimpleAuthProvider({
                'admin': ('admin', Role.ADMIN),
                'user': ('user', Role.USER),
                'viewer': ('viewer', Role.VIEWER)
            })
            session_manager = SessionManager(timeout_seconds=self.config.session_timeout)
            self.auth_middleware = AuthenticationMiddleware(
                provider=auth_provider,
                session_manager=session_manager,
                enabled=True
            )
        else:
            self.logger.info("Authentication disabled")
            # Create a disabled auth middleware for consistency
            auth_provider = SimpleAuthProvider()
            session_manager = SessionManager()
            self.auth_middleware = AuthenticationMiddleware(
                provider=auth_provider,
                session_manager=session_manager,
                enabled=False
            )
        
        # Initialize rate limiter if enabled
        self.rate_limiter = None
        if self.config.enable_rate_limiting:
            self.logger.info(
                f"Initializing rate limiter: {self.config.rate_limit_per_minute} req/min, "
                f"{self.config.rate_limit_per_hour} req/hour"
            )
            rate_limit_config = RateLimitConfig(
                requests_per_minute=self.config.rate_limit_per_minute,
                requests_per_hour=self.config.rate_limit_per_hour,
                enabled=True
            )
            self.rate_limiter = RateLimiter(rate_limit_config)
        else:
            self.logger.info("Rate limiting disabled")
            # Create a disabled rate limiter for consistency
            rate_limit_config = RateLimitConfig(enabled=False)
            self.rate_limiter = RateLimiter(rate_limit_config)
        
        # Initialize service layer components
        self.logger.info("Initializing service layer components")
        self.job_service = JobService(job_queue, backend_router)
        self.template_service = TemplateService(template_registry)
        self.workflow_service = WorkflowService(workflow_engine)
        self.backend_monitor = BackendMonitorService(backend_router)
        
        # Initialize health checker
        self.logger.info("Initializing health checker")
        self.health_checker = HealthChecker(
            job_queue=job_queue,
            backend_router=backend_router,
            workflow_engine=workflow_engine,
            template_registry=template_registry,
            version=gui.__version__
        )
        
        # Initialize file upload handler
        self.file_upload_handler = FileUploadHandler(upload_dir=self.config.upload_dir)
        
        # Initialize UI components
        self.logger.info("Initializing UI components")
        self.job_submission_tab = JobSubmissionTabV2(
            self.job_service, 
            self.template_service
        )
        self.job_monitoring_tab = JobMonitoringTabV2(self.job_service)
        self.workflow_builder_tab = WorkflowBuilderTab(self.workflow_service, self.template_service)
        self.template_management_tab = TemplateManagementTabV2(self.template_service)
        self.backend_status_tab = BackendStatusTab(self.backend_monitor)
        self.backend_registration_tab = BackendRegistrationTab(self.backend_router, self.backend_monitor)
        self.file_manager_tab = FileManagerTab(self.file_upload_handler)
        
        # Setup observers for orchestrator components
        self.logger.info("Setting up observers for real-time updates")
        self._setup_observers()
        
        # Start observer thread
        self._observer_running = False
        self._observer_thread = None
        self._start_observer_thread()
        
        self.logger.info("GradioApp initialized successfully")
        
    def build_interface(self) -> gr.Blocks:
        """Build the complete Gradio interface with all tabs.
        
        This method creates the main Gradio Blocks interface with all UI tabs
        organized in a tabbed layout. Each tab is rendered by its respective
        component class.
        
        Returns:
            Gradio Blocks interface with all tabs
            
        Requirements:
            - 7.2: Uses Gradio Blocks for custom layout composition
            - 7.3: Organizes components into tabs for job submission, monitoring,
                   workflows, templates, and backend status
            - 7.6: Uses Gradio themes for consistent visual styling
        """
        self.logger.info("Building Gradio interface")
        
        # Apply theme configuration
        theme = self.config.theme
        if theme == "default":
            theme = gr.themes.Default()
        elif theme == "soft":
            theme = gr.themes.Soft()
        elif theme == "monochrome":
            theme = gr.themes.Monochrome()
        else:
            # Use default theme if unknown theme specified
            theme = gr.themes.Default()
        
        with gr.Blocks(
            title="Notebook ML Orchestrator",
            theme=theme,
            css=self._get_custom_css()
        ) as interface:
            # Header
            gr.Markdown("# Notebook ML Orchestrator")
            gr.Markdown("Web-based interface for ML job orchestration across multiple cloud backends")
            
            # Main tabbed interface
            with gr.Tabs():
                # Job Submission Tab
                with gr.Tab("Job Submission"):
                    self.job_submission_tab.render()
                    
                # Job Monitoring Tab
                with gr.Tab("Job Monitoring"):
                    self.job_monitoring_tab.render()
                    
                # Workflow Builder Tab
                with gr.Tab("Workflow Builder"):
                    self.workflow_builder_tab.render()
                    
                # Template Management Tab
                with gr.Tab("Template Management"):
                    self.template_management_tab.render()
                    
                # Backend Status Tab
                with gr.Tab("Backend Status"):
                    self.backend_status_tab.render()
                    
                # Backend Registration Tab
                with gr.Tab("Backend Registration"):
                    self.backend_registration_tab.render()
                    
                # File Manager Tab
                with gr.Tab("File Manager"):
                    self.file_manager_tab.render()
            
            # Footer
            gr.Markdown("---")
            gr.Markdown(
                "Notebook ML Orchestrator - Unified interface for ML job orchestration | "
                f"Version: {self._get_version()}"
            )
            
            # Add health check API endpoint
            # Note: Gradio allows adding custom API routes via the API
            # We'll expose the health check as a hidden component that can be queried
            health_check_output = gr.JSON(visible=False, elem_id="health_check")
            health_check_btn = gr.Button("Check Health", visible=False, elem_id="health_check_btn")
            
            # Wire up health check
            health_check_handler = create_health_check_handler(self.health_checker)
            health_check_btn.click(
                fn=health_check_handler,
                outputs=health_check_output
            )
        
        self.logger.info("Gradio interface built successfully")
        return interface
    
    def _get_custom_css(self) -> str:
        """Get custom CSS for the interface.
        
        Returns:
            CSS string for custom styling
        """
        return """
        /* Custom CSS for Notebook ML Orchestrator */
        .gradio-container {
            max-width: 1400px !important;
        }
        
        /* Improve table readability */
        .dataframe {
            font-size: 14px;
        }
        
        /* Status indicators */
        .status-healthy {
            color: #4CAF50;
            font-weight: bold;
        }
        
        .status-unhealthy {
            color: #f44336;
            font-weight: bold;
        }
        
        .status-degraded {
            color: #ff9800;
            font-weight: bold;
        }
        """
    
    def _get_version(self) -> str:
        """Get application version.
        
        Returns:
            Version string
        """
        return gui.__version__
    
    def get_health_status(self) -> dict:
        """Get current health status of the application.
        
        This method provides programmatic access to the health check
        without going through the Gradio interface.
        
        Returns:
            Health check response as dictionary
            
        Requirements:
            - 11.6: Provide health check endpoint for monitoring
        """
        response = self.health_checker.check_health()
        return response.to_dict()
    
    def check_rate_limit(self, client_id: str) -> None:
        """Check rate limit for a client.
        
        This method checks if the client has exceeded rate limits and raises
        an exception if so. It should be called before processing requests.
        
        Args:
            client_id: Unique identifier for the client (username or IP)
            
        Raises:
            RateLimitError: If rate limit is exceeded
            
        Requirements:
            - 12.5: Implement rate limiting to prevent abuse
        """
        if self.rate_limiter:
            self.rate_limiter.check_rate_limit(client_id)
    
    def get_client_id(self, request=None) -> str:
        """Get client identifier for rate limiting.
        
        This method extracts a unique identifier for the client from the request.
        If authentication is enabled, it uses the username. Otherwise, it uses
        the IP address or a default identifier.
        
        Args:
            request: Optional request object (Gradio request)
            
        Returns:
            Client identifier string
        """
        # If authentication is enabled, use the username
        if self.config.enable_auth and self.auth_middleware:
            # Try to get current session
            # Note: In a real implementation, this would extract from the request
            # For now, we'll use a default identifier
            return "authenticated_user"
        
        # If request is available, try to extract IP address
        if request and hasattr(request, 'client'):
            return str(request.client.host)
        
        # Default identifier
        return "default_client"
    
    def _setup_observers(self) -> None:
        """Setup observers for job queue, backend router, and workflow engine.
        
        This method sets up polling-based observers that monitor state changes
        in the orchestrator components and emit events for real-time updates.
        
        Requirements:
            - 10.1: GUI uses same Job_Queue as CLI
            - 10.2: GUI retrieves data from same database as CLI
            - 10.3: GUI uses same Workflow_Engine as CLI
            - 10.4: GUI uses same Template_Registry as CLI
            - 10.5: GUI uses same Backend_Router as CLI
        """
        # Store last known states for change detection
        self._last_job_states = {}
        self._last_backend_states = {}
        self._last_workflow_states = {}
        
        self.logger.info("Observers configured for job queue, backend router, and workflow engine")
    
    def _start_observer_thread(self) -> None:
        """Start the observer thread for monitoring state changes."""
        if self._observer_thread and self._observer_thread.is_alive():
            return
        
        self._observer_running = True
        self._observer_thread = threading.Thread(target=self._observe_state_changes, daemon=True)
        self._observer_thread.start()
        self.logger.info("Observer thread started")
    
    def _stop_observer_thread(self) -> None:
        """Stop the observer thread."""
        self._observer_running = False
        if self._observer_thread:
            self._observer_thread.join(timeout=5.0)
        self.logger.info("Observer thread stopped")
    
    def _observe_state_changes(self) -> None:
        """Observer thread that polls for state changes and emits events.
        
        This method runs in a background thread and periodically checks for:
        - Job status changes
        - Backend health status changes
        - Workflow execution progress
        
        When changes are detected, it emits events through the EventEmitter
        which are then broadcast to WebSocket clients for real-time updates.
        """
        while self._observer_running:
            try:
                # Check for job status changes
                self._check_job_status_changes()
                
                # Check for backend health changes
                self._check_backend_status_changes()
                
                # Check for workflow execution changes
                self._check_workflow_status_changes()
                
                # Sleep before next check (use configured interval)
                time.sleep(self.config.auto_refresh_interval)
                
            except Exception as e:
                self.logger.error(f"Error in observer thread: {e}")
                time.sleep(10)  # Wait longer on error
    
    def _check_job_status_changes(self) -> None:
        """Check for job status changes and emit events."""
        try:
            # Get recent jobs (last 100)
            # Note: This is a simplified implementation. In production, you'd want
            # to track specific jobs or use database triggers for efficiency.
            stats = self.job_queue.get_queue_statistics()
            
            # For now, we'll just emit a general update event
            # A more sophisticated implementation would track individual job changes
            self.event_emitter.emit('job.status_changed', {
                'timestamp': time.time(),
                'statistics': stats
            })
            
        except Exception as e:
            self.logger.error(f"Error checking job status changes: {e}")
    
    def _check_backend_status_changes(self) -> None:
        """Check for backend health status changes and emit events."""
        try:
            # Get current backend status
            current_status = self.backend_router.get_backend_status()
            
            # Check for changes
            for backend_id, health_status in current_status.items():
                last_status = self._last_backend_states.get(backend_id)
                
                if last_status != health_status:
                    # Status changed, emit event
                    self.event_emitter.emit('backend.status_changed', {
                        'backend_id': backend_id,
                        'status': health_status.value,
                        'timestamp': time.time()
                    })
                    
                    self._last_backend_states[backend_id] = health_status
            
        except Exception as e:
            self.logger.error(f"Error checking backend status changes: {e}")
    
    def _check_workflow_status_changes(self) -> None:
        """Check for workflow execution changes and emit events."""
        try:
            # Get recent workflow executions
            executions = self.workflow_engine.list_executions()
            
            # Check for changes in execution status
            for execution in executions:
                last_status = self._last_workflow_states.get(execution.id)
                
                if last_status != execution.status:
                    # Status changed, emit event
                    self.event_emitter.emit('workflow.step_completed', {
                        'workflow_id': execution.workflow_id,
                        'execution_id': execution.id,
                        'status': execution.status.value,
                        'timestamp': time.time()
                    })
                    
                    self._last_workflow_states[execution.id] = execution.status
            
        except Exception as e:
            self.logger.error(f"Error checking workflow status changes: {e}")
        
    def launch(self, host: Optional[str] = None, port: Optional[int] = None, **kwargs):
        """Launch the Gradio application.
        
        This method builds the interface and starts the Gradio web server.
        Additional keyword arguments are passed through to Gradio's launch method.
        
        Args:
            host: Host address (uses config default if not provided)
            port: Port number (uses config default if not provided)
            **kwargs: Additional arguments to pass to Gradio's launch method
                     (e.g., share=True, auth=None, etc.)
                     
        Requirements:
            - 7.1: Launches Gradio web interface
            - 11.2: Supports configurable host and port settings
        """
        self.logger.info(
            f"Launching Gradio application on {host or self.config.host}:{port or self.config.port}"
        )
        
        interface = self.build_interface()
        
        # Setup authentication for Gradio if enabled
        auth_fn = None
        if self.config.enable_auth and 'auth' not in kwargs:
            def gradio_auth(username: str, password: str) -> bool:
                """Gradio authentication callback."""
                session = self.auth_middleware.authenticate(username, password)
                return session is not None
            
            auth_fn = gradio_auth
            kwargs['auth'] = auth_fn
            self.logger.info("Gradio authentication enabled")
        
        # Launch with configuration
        interface.launch(
            server_name=host or self.config.host,
            server_port=port or self.config.port,
            **kwargs
        )
        
        self.logger.info("Gradio application launched successfully")
    
    def shutdown(self) -> None:
        """Shutdown the application and cleanup resources.
        
        This method stops the observer thread and performs any necessary cleanup.
        """
        self.logger.info("Shutting down GradioApp")
        self._stop_observer_thread()
        
        # Shutdown rate limiter
        if self.rate_limiter:
            self.rate_limiter.shutdown()
        
        self.logger.info("GradioApp shutdown complete")
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        try:
            self.shutdown()
        except Exception:
            pass  # Ignore errors during cleanup
