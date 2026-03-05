"""
Main Orchestrator class that initializes and coordinates all components.

This module provides the central Orchestrator class that manages the lifecycle
of all orchestrator components including job queue, backend router, workflow engine,
batch processor, and template registry.

SECURITY FIXED: Comprehensive input validation added to prevent:
- Injection attacks via user_id
- DoS via large inputs
- Invalid routing strategy values
- Deeply nested data structures
"""

import logging
import re
import sys
from pathlib import Path
from typing import Optional, Dict, Any

from .core.job_queue import JobQueueManager
from .core.backend_router import MultiBackendRouter
from .core.workflow_engine import WorkflowEngine
from .core.batch_processor import BatchProcessor
from .core.template_registry import TemplateRegistry
from .core.logging_config import configure_default_logging
from .core.interfaces import Job, Backend
from .core.models import ResourceEstimate
from .core.exceptions import TemplateNotFoundError, JobValidationError
from .config import get_config, OrchestratorConfig


logger = logging.getLogger(__name__)

# SECURITY: Input validation constants
MAX_USER_ID_LENGTH = 64
MAX_INPUT_SIZE_BYTES = 10 * 1024 * 1024  # 10MB
MAX_INPUT_DEPTH = 10  # Maximum nesting depth for input dictionaries
VALID_ROUTING_STRATEGIES = {"cost-optimized", "round-robin", "least-loaded", "performance"}

# SECURITY: User ID pattern - alphanumeric, underscore, hyphen only
USER_ID_PATTERN = re.compile(r'^[a-zA-Z0-9_-]{1,64}$')


class Orchestrator:
    """
    Main orchestrator that coordinates all system components.
    
    The Orchestrator initializes and manages:
    - Template Registry: Discovers and manages ML templates
    - Job Queue: Manages job submission and execution
    - Backend Router: Routes jobs to appropriate backends
    - Workflow Engine: Executes multi-step workflows
    - Batch Processor: Handles batch job processing
    """
    
    def __init__(
        self,
        config: Optional[OrchestratorConfig] = None,
        templates_dir: str = "templates",
        db_path: Optional[str] = None
    ):
        """
        Initialize the orchestrator with all components.
        
        Args:
            config: Orchestrator configuration (uses default if None)
            templates_dir: Directory containing template files
            db_path: Path to SQLite database (uses config default if None)
        """
        self.config = config or get_config()
        self.templates_dir = templates_dir
        self.db_path = db_path or self.config.database.path
        
        # Initialize logging
        configure_default_logging()
        logger.info("Initializing Notebook ML Orchestrator")
        
        # Initialize components
        self._initialize_components()
        
        logger.info("Orchestrator initialization complete")
    
    def _initialize_components(self):
        """Initialize all orchestrator components."""
        # Initialize Template Registry first
        logger.info(f"Initializing Template Registry from {self.templates_dir}")
        self.template_registry = TemplateRegistry(templates_dir=self.templates_dir)
        
        # Discover and register templates
        template_count = self.template_registry.discover_templates()
        logger.info(f"Template discovery complete: {template_count} templates registered")
        
        # Log template statistics
        stats = self.template_registry.get_registry_stats()
        logger.info(f"Templates by category: {stats['templates_by_category']}")
        if stats['failed_templates']:
            logger.warning(f"Failed to load {stats['failed_templates']} templates: {stats['failed_template_list']}")
        
        # Initialize Job Queue
        logger.info(f"Initializing Job Queue with database: {self.db_path}")
        self.job_queue = JobQueueManager(db_path=self.db_path)
        
        # Initialize Backend Router
        logger.info("Initializing Backend Router")
        self.backend_router = MultiBackendRouter()
        
        # Initialize Workflow Engine
        logger.info("Initializing Workflow Engine")
        self.workflow_engine = WorkflowEngine()
        
        # Initialize Batch Processor
        logger.info("Initializing Batch Processor")
        self.batch_processor = BatchProcessor()
    
    def get_template_registry(self) -> TemplateRegistry:
        """
        Get the template registry instance.
        
        Returns:
            TemplateRegistry instance
        """
        return self.template_registry
    
    def get_job_queue(self) -> JobQueueManager:
        """
        Get the job queue manager instance.
        
        Returns:
            JobQueueManager instance
        """
        return self.job_queue
    
    def get_backend_router(self) -> MultiBackendRouter:
        """
        Get the backend router instance.
        
        Returns:
            MultiBackendRouter instance
        """
        return self.backend_router
    
    def get_workflow_engine(self) -> WorkflowEngine:
        """
        Get the workflow engine instance.
        
        Returns:
            WorkflowEngine instance
        """
        return self.workflow_engine
    
    def get_batch_processor(self) -> BatchProcessor:
        """
        Get the batch processor instance.

        Returns:
            BatchProcessor instance
        """
        return self.batch_processor

    def _validate_user_id(self, user_id: str) -> None:
        """
        Validate user_id format to prevent injection attacks.
        
        SECURITY: Validates user_id against strict pattern.
        
        Args:
            user_id: User ID to validate
            
        Raises:
            JobValidationError: If user_id is invalid
        """
        if not user_id:
            raise JobValidationError("User ID is required")
        
        if len(user_id) > MAX_USER_ID_LENGTH:
            raise JobValidationError(
                f"User ID exceeds maximum length of {MAX_USER_ID_LENGTH} characters"
            )
        
        if not USER_ID_PATTERN.match(user_id):
            raise JobValidationError(
                "User ID can only contain alphanumeric characters, underscores, and hyphens"
            )

    def _validate_inputs_size(self, inputs: Dict[str, Any]) -> None:
        """
        Validate input size to prevent DoS attacks.
        
        SECURITY: Checks serialized input size and nesting depth.
        
        Args:
            inputs: Input dictionary to validate
            
        Raises:
            JobValidationError: If inputs are too large or deeply nested
        """
        import json
        
        # Check serialized size
        try:
            serialized_size = sys.getsizeof(json.dumps(inputs))
            if serialized_size > MAX_INPUT_SIZE_BYTES:
                raise JobValidationError(
                    f"Input size ({serialized_size} bytes) exceeds maximum allowed "
                    f"({MAX_INPUT_SIZE_BYTES} bytes)"
                )
        except (TypeError, ValueError) as e:
            raise JobValidationError(f"Inputs are not JSON serializable: {str(e)}")
        
        # Check nesting depth
        def get_depth(obj, current_depth=0):
            if not isinstance(obj, dict):
                return current_depth
            if not obj:
                return current_depth
            return max(get_depth(v, current_depth + 1) for v in obj.values())
        
        depth = get_depth(inputs)
        if depth > MAX_INPUT_DEPTH:
            raise JobValidationError(
                f"Input nesting depth ({depth}) exceeds maximum allowed ({MAX_INPUT_DEPTH})"
            )

    def _validate_routing_strategy(self, routing_strategy: str) -> None:
        """
        Validate routing strategy is a known value.
        
        SECURITY: Prevents injection of arbitrary strategy values.
        
        Args:
            routing_strategy: Strategy name to validate
            
        Raises:
            JobValidationError: If strategy is invalid
        """
        if not routing_strategy:
            raise JobValidationError("Routing strategy is required")
        
        if routing_strategy not in VALID_ROUTING_STRATEGIES:
            raise JobValidationError(
                f"Invalid routing strategy '{routing_strategy}'. "
                f"Valid strategies: {', '.join(sorted(VALID_ROUTING_STRATEGIES))}"
            )

    def submit_job(
        self,
        template_name: str,
        inputs: Dict[str, Any],
        user_id: str = "default",
        routing_strategy: str = "cost-optimized"
    ) -> str:
        """
        Submit a job with template validation and automatic backend routing.

        This method integrates templates with the job submission flow by:
        1. Retrieving the template from the registry
        2. SECURITY: Validating user_id format
        3. SECURITY: Validating input size and structure
        4. SECURITY: Validating routing strategy
        5. Validating inputs against the template schema
        6. Estimating resource requirements from the template
        7. Creating a job and submitting it to the queue
        8. Routing the job to an appropriate backend

        Args:
            template_name: Name of the template to execute
            inputs: Input parameters for the template
            user_id: User ID submitting the job (default: "default")
            routing_strategy: Backend routing strategy (default: "cost-optimized")

        Returns:
            Job ID

        Raises:
            TemplateNotFoundError: If template is not found in registry
            JobValidationError: If input validation fails
            BackendNotAvailableError: If no suitable backend is available
        """
        # SECURITY VALIDATION 1: Validate user_id format
        self._validate_user_id(user_id)
        
        # SECURITY VALIDATION 2: Validate input size and structure
        self._validate_inputs_size(inputs)
        
        # SECURITY VALIDATION 3: Validate routing strategy
        self._validate_routing_strategy(routing_strategy)
        
        # Get template from registry
        template = self.template_registry.get_template(template_name)
        if not template:
            raise TemplateNotFoundError(
                f"Template '{template_name}' not found in registry. "
                f"Available templates: {list(self.template_registry.list_templates())}"
            )

        # Validate inputs against template schema
        try:
            template.validate_inputs(**inputs)
        except ValueError as e:
            raise JobValidationError(f"Input validation failed: {str(e)}") from e

        # Estimate resource requirements from template
        resource_estimate = self.backend_router.template_to_resource_estimate(template)

        logger.info(
            f"Submitting job for template '{template_name}' with resource requirements: "
            f"GPU={resource_estimate.requires_gpu}, "
            f"memory={resource_estimate.memory_gb}GB, "
            f"duration={resource_estimate.estimated_duration_minutes}min"
        )

        # Create job
        job = Job(
            template_name=template_name,
            inputs=inputs,
            user_id=user_id,
            metadata={
                'resource_estimate': {
                    'requires_gpu': resource_estimate.requires_gpu,
                    'memory_gb': resource_estimate.memory_gb,
                    'gpu_memory_gb': resource_estimate.gpu_memory_gb,
                    'estimated_duration_minutes': resource_estimate.estimated_duration_minutes,
                    'cpu_cores': resource_estimate.cpu_cores
                }
            }
        )
        
        # Submit to job queue
        job_id = self.job_queue.submit_job(job)
        
        # Route job to backend
        try:
            backend = self.backend_router.route_job(
                job,
                routing_strategy=routing_strategy,
                resource_estimate=resource_estimate
            )
            
            logger.info(
                f"Job {job_id} routed to backend '{backend.id}' "
                f"(type: {backend.type.value})"
            )
            
            # Update job with selected backend
            job.backend_id = backend.id
            
            # Persist the backend assignment to the database
            self.job_queue.update_job_backend(job_id, backend.id)
            
            logger.info(
                f"Job {job_id} assigned to backend '{backend.id}' (persisted)"
            )
            
        except Exception as e:
            logger.error(f"Failed to route job {job_id}: {str(e)}")
            raise
        
        return job_id
    
    def execute_job(
        self,
        template_name: str,
        inputs: Dict[str, Any],
        user_id: str = "default",
        routing_strategy: str = "cost-optimized"
    ) -> Dict[str, Any]:
        """
        Submit and execute a job synchronously with template validation.
        
        This is a convenience method that submits a job and waits for completion.
        For asynchronous execution, use submit_job() instead.
        
        Args:
            template_name: Name of the template to execute
            inputs: Input parameters for the template
            user_id: User ID submitting the job (default: "default")
            routing_strategy: Backend routing strategy (default: "cost-optimized")
            
        Returns:
            Job result dictionary with outputs
            
        Raises:
            TemplateNotFoundError: If template is not found in registry
            JobValidationError: If input validation fails
            BackendNotAvailableError: If no suitable backend is available
            JobExecutionError: If job execution fails
        """
        # Submit the job
        job_id = self.submit_job(template_name, inputs, user_id, routing_strategy)
        
        # Get the job
        job = self.job_queue.get_job(job_id)
        if not job:
            raise JobValidationError(f"Job {job_id} not found after submission")
        
        # Get the backend
        backend = self.backend_router.get_backend(job.backend_id)
        if not backend:
            raise JobValidationError(f"Backend {job.backend_id} not found")
        
        # Get the template
        template = self.template_registry.get_template(template_name)
        
        # Execute the job on the backend
        result = backend.execute_job(job, template)
        
        # Update job status
        self.job_queue.update_job_status(job_id, job.status, result)
        
        logger.info(f"Job {job_id} completed successfully")
        
        return result.outputs if result else {}
    
    def get_template(self, template_name: str):
        """
        Get a template by name.
        
        Args:
            template_name: Name of the template
            
        Returns:
            Template instance or None if not found
        """
        return self.template_registry.get_template(template_name)
    
    def list_templates(self, category: Optional[str] = None):
        """
        List available templates, optionally filtered by category.
        
        Args:
            category: Optional category filter
            
        Returns:
            List of template instances
        """
        return self.template_registry.list_templates(category)
    
    def get_template_metadata(self, template_name: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata for a specific template.
        
        Args:
            template_name: Name of the template
            
        Returns:
            Template metadata dictionary or None if not found
        """
        return self.template_registry.get_template_metadata(template_name)
    
    def shutdown(self):
        """Shutdown the orchestrator and cleanup resources."""
        logger.info("Shutting down orchestrator")
        
        # Stop job queue retry processor
        if hasattr(self, 'job_queue'):
            self.job_queue.stop_retry_processor()
        
        logger.info("Orchestrator shutdown complete")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.shutdown()
        return False
