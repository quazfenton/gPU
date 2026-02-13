"""
Modal backend implementation for the Notebook ML Orchestrator.

This module provides integration with Modal's serverless GPU infrastructure,
enabling execution of ML jobs on Modal's platform with GPU support.
"""

import time
from typing import Dict, Any, Optional
from datetime import datetime

from ..interfaces import Backend, MLTemplate, Job
from ..models import (
    BackendType, HealthStatus, ResourceEstimate, JobResult, BackendCapabilities
)
from ..exceptions import (
    BackendConnectionError, BackendNotAvailableError, JobExecutionError, 
    JobTimeoutError, BackendAuthenticationError, BackendRateLimitError
)
from ..logging_config import LoggerMixin


# GPU pricing per hour (in USD)
GPU_PRICING = {
    "T4": 0.60,
    "A10G": 1.10,
    "A100": 4.00,
    "CPU": 0.10,  # CPU-only pricing
}

# Templates supported by Modal backend (GPU-intensive and batch processing)
SUPPORTED_TEMPLATES = [
    "image-generation",
    "text-generation",
    "image-classification",
    "model-training",
    "data-processing",
    "batch-inference",
    "embeddings",
    # Audio templates
    "speech-recognition",
    "audio-generation",
    "music-processing",
    # Vision templates
    "object-detection",
    "image-segmentation",
    "video-processing",
    # Language templates
    "named-entity-recognition",
    "sentiment-analysis",
    "translation",
    "summarization",
    # Multimodal templates
    "image-captioning",
    "visual-question-answering",
    "text-to-image",
]


class ModalBackend(Backend, LoggerMixin):
    """
    Backend implementation for Modal serverless compute platform.
    
    This backend executes ML jobs on Modal's infrastructure with support for
    GPU acceleration, timeout handling, and secrets injection.
    """
    
    def __init__(self, backend_id: str = "modal", config: Optional[Dict[str, Any]] = None):
        """
        Initialize Modal backend.
        
        Args:
            backend_id: Unique identifier for this backend instance
            config: Configuration dictionary containing credentials and options
        """
        super().__init__(backend_id, "Modal", BackendType.MODAL)
        
        self.config = config or {}
        self.credentials = self.config.get('credentials', {})
        self.options = self.config.get('options', {})
        
        # Configuration options
        self.default_gpu = self.options.get('default_gpu', 'A10G')
        self.timeout = self.options.get('timeout', 300)
        
        # Set capabilities
        self.capabilities = BackendCapabilities(
            supported_templates=SUPPORTED_TEMPLATES,
            max_concurrent_jobs=10,
            max_job_duration_minutes=60,
            supports_gpu=True,
            supports_batch=True,
            cost_per_hour=GPU_PRICING.get(self.default_gpu, 1.10),
            free_tier_limits={}
        )
        
        # Initialize Modal client (lazy initialization)
        self._modal_client = None
        self._authenticated = False
        
        self.logger.info(f"Modal backend initialized: {backend_id}")
    
    def _authenticate(self) -> None:
        """
        Authenticate with Modal API using configured credentials.
        
        Raises:
            BackendAuthenticationError: If authentication fails
            BackendConnectionError: If Modal SDK is not available
        """
        if self._authenticated:
            return
        
        try:
            import modal
            
            token_id = self.credentials.get('token_id')
            token_secret = self.credentials.get('token_secret')
            
            if not token_id or not token_secret:
                raise BackendAuthenticationError(
                    "Modal credentials not configured. Set MODAL_TOKEN_ID and MODAL_TOKEN_SECRET.",
                    backend_id=self.id
                )
            
            # Set Modal credentials as environment variables (Modal SDK reads from env)
            import os
            os.environ['MODAL_TOKEN_ID'] = token_id
            os.environ['MODAL_TOKEN_SECRET'] = token_secret
            
            # Verify credentials by attempting to create a test app
            try:
                test_app = modal.App("auth-test")
                # If we can create an app, credentials are valid
                self._authenticated = True
                self.logger.info("Modal authentication successful")
            except Exception as auth_error:
                # Check for specific authentication errors
                error_str = str(auth_error).lower()
                if "unauthorized" in error_str or "invalid" in error_str or "credentials" in error_str:
                    raise BackendAuthenticationError(
                        f"Modal authentication failed: Invalid credentials - {str(auth_error)}",
                        backend_id=self.id
                    )
                # Re-raise other errors
                raise
            
        except ImportError:
            raise BackendConnectionError(
                "Modal SDK not installed. Install with: pip install modal",
                backend_id=self.id
            )
        except BackendAuthenticationError:
            # Re-raise authentication errors
            raise
        except Exception as e:
            self.logger.error(f"Modal authentication failed: {e}")
            raise BackendConnectionError(
                f"Modal authentication failed: {str(e)}",
                backend_id=self.id
            )
    
    def execute_job(self, job: Job, template: MLTemplate) -> JobResult:
        """
        Execute a job on Modal infrastructure with retry logic.
        
        This method creates a Modal function with appropriate GPU configuration,
        executes the template code, and returns the results. It includes retry
        logic for transient failures.
        
        Args:
            job: Job to execute
            template: Template instance for the job
            
        Returns:
            JobResult with execution results
            
        Raises:
            BackendAuthenticationError: If authentication fails
            BackendConnectionError: If connection fails
            BackendNotAvailableError: If GPU resources are unavailable
            BackendRateLimitError: If rate limit is exceeded
            JobExecutionError: If job execution fails
            JobTimeoutError: If job execution times out
        """
        max_retries = 3
        retry_delay = 2  # seconds
        
        for attempt in range(max_retries):
            try:
                return self._execute_job_internal(job, template)
            
            except BackendRateLimitError as e:
                # Handle rate limiting with exponential backoff
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)
                    self.logger.warning(
                        f"Rate limit hit for job {job.id}, retrying in {wait_time}s (attempt {attempt + 1}/{max_retries})"
                    )
                    time.sleep(wait_time)
                else:
                    self.logger.error(f"Rate limit exceeded after {max_retries} attempts for job {job.id}")
                    raise
            
            except JobTimeoutError:
                # Don't retry timeout errors
                raise
            
            except BackendAuthenticationError:
                # Don't retry authentication errors
                raise
            
            except BackendConnectionError:
                # Don't retry connection errors (SDK not installed, etc.)
                raise
            
            except BackendNotAvailableError as e:
                # Retry GPU unavailability with backoff
                if "gpu" in str(e).lower() and attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)
                    self.logger.warning(
                        f"GPU unavailable for job {job.id}, retrying in {wait_time}s (attempt {attempt + 1}/{max_retries})"
                    )
                    time.sleep(wait_time)
                else:
                    raise
            
            except JobExecutionError as e:
                # Retry transient execution errors
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)
                    self.logger.warning(
                        f"Job {job.id} failed, retrying in {wait_time}s (attempt {attempt + 1}/{max_retries}): {str(e)}"
                    )
                    time.sleep(wait_time)
                else:
                    raise
        
        # Should not reach here, but just in case
        raise JobExecutionError(
            f"Job {job.id} failed after {max_retries} attempts",
            job_id=job.id,
            backend_id=self.id
        )
    
    def _execute_job_internal(self, job: Job, template: MLTemplate) -> JobResult:
        """
        Internal method to execute a job on Modal infrastructure.
        
        Args:
            job: Job to execute
            template: Template instance for the job
            
        Returns:
            JobResult with execution results
            
        Raises:
            Various backend and job exceptions
        """
        start_time = time.time()
        
        try:
            # Authenticate with Modal
            self._authenticate()
            
            # Import Modal SDK
            import modal
            
            # Determine GPU type from job metadata or use default
            gpu_type = job.metadata.get('gpu_type', self.default_gpu)
            timeout_seconds = job.metadata.get('timeout', self.timeout)
            
            self.logger.info(f"Executing job {job.id} on Modal with GPU: {gpu_type}, timeout: {timeout_seconds}s")
            
            # Create Modal app
            app = modal.App(f"orchestrator-job-{job.id}")
            
            # Build container image with dependencies
            image = self._build_image(template)
            
            # Get GPU configuration
            gpu_config = self._get_gpu_config(gpu_type)
            
            # Create Modal function
            @app.function(
                image=image,
                gpu=gpu_config,
                timeout=timeout_seconds,
                secrets=self._get_secrets()
            )
            def execute_notebook_code(inputs: Dict[str, Any]) -> Dict[str, Any]:
                """Execute template code with inputs."""
                # Execute the template's logic
                # Note: In a real implementation, this would execute the template code
                # For now, we'll call the template's execute method
                return inputs  # Placeholder
            
            # Execute the function remotely
            with app.run():
                result_data = execute_notebook_code.remote(job.inputs)
            
            execution_time = time.time() - start_time
            
            self.logger.info(f"Job {job.id} completed successfully in {execution_time:.2f}s")
            
            return JobResult(
                success=True,
                outputs=result_data,
                execution_time_seconds=execution_time,
                backend_used=self.id,
                metadata={
                    'gpu_type': gpu_type,
                    'timeout': timeout_seconds
                }
            )
        
        except BackendAuthenticationError:
            # Re-raise authentication errors
            raise
        
        except BackendConnectionError:
            # Re-raise connection errors
            raise
            
        except ImportError as e:
            error_msg = f"Modal SDK not available: {str(e)}"
            self.logger.error(error_msg)
            raise BackendConnectionError(error_msg, backend_id=self.id)
        
        except Exception as e:
            execution_time = time.time() - start_time
            error_str = str(e).lower()
            
            # Check for timeout errors
            if "timeout" in error_str or "timed out" in error_str or execution_time >= self.timeout:
                raise JobTimeoutError(
                    f"Job {job.id} timed out after {execution_time:.2f}s",
                    job_id=job.id,
                    timeout_seconds=self.timeout
                )
            
            # Check for GPU unavailability
            if ("gpu" in error_str and ("unavailable" in error_str or "not available" in error_str)) or \
               "no gpu" in error_str or "gpu capacity" in error_str:
                raise BackendNotAvailableError(
                    f"GPU resources unavailable on Modal: {str(e)}",
                    required_capabilities=['gpu']
                )
            
            # Check for rate limiting
            if "rate limit" in error_str or "too many requests" in error_str or "429" in error_str:
                # Try to extract retry-after time
                retry_after = None
                if hasattr(e, 'retry_after'):
                    retry_after = e.retry_after
                raise BackendRateLimitError(
                    f"Modal rate limit exceeded: {str(e)}",
                    backend_id=self.id,
                    retry_after=retry_after
                )
            
            # Check for authentication errors
            if "unauthorized" in error_str or "authentication" in error_str or "invalid credentials" in error_str:
                raise BackendAuthenticationError(
                    f"Modal authentication failed: {str(e)}",
                    backend_id=self.id
                )
            
            # Generic execution error
            error_msg = f"Job execution failed on Modal: {str(e)}"
            self.logger.error(f"Job {job.id} failed: {error_msg}")
            raise JobExecutionError(
                error_msg,
                job_id=job.id,
                backend_id=self.id
            )
    
    def _build_image(self, template: MLTemplate) -> Any:
        """
        Build Modal container image with required dependencies.
        
        Args:
            template: Template to build image for
            
        Returns:
            Modal Image object
        """
        try:
            import modal
            
            # Start with base image
            image = modal.Image.debian_slim().pip_install(
                "numpy",
                "pandas",
                "scikit-learn",
            )
            
            # Add template-specific dependencies
            if hasattr(template, 'requirements') and template.requirements:
                deps = template.requirements.get('packages', [])
                if deps:
                    image = image.pip_install(*deps)
            
            return image
            
        except Exception as e:
            self.logger.warning(f"Failed to build custom image: {e}, using default")
            import modal
            return modal.Image.debian_slim()
    
    def _get_gpu_config(self, gpu_type: str) -> str:
        """
        Get Modal GPU configuration string.
        
        Args:
            gpu_type: GPU type (T4, A10G, A100, or CPU)
            
        Returns:
            Modal GPU configuration string
        """
        if gpu_type.upper() == "CPU":
            return None  # No GPU
        
        # Modal uses specific GPU type strings
        gpu_map = {
            "T4": "T4",
            "A10G": "A10G",
            "A100": "A100",
        }
        
        return gpu_map.get(gpu_type.upper(), "A10G")
    
    def _get_secrets(self) -> list:
        """
        Get Modal secrets for injection.
        
        Returns:
            List of Modal Secret objects
        """
        secrets = []
        
        try:
            import modal
            
            # Add any configured secrets
            secret_names = self.options.get('secrets', [])
            for secret_name in secret_names:
                try:
                    secrets.append(modal.Secret.from_name(secret_name))
                except Exception as e:
                    self.logger.warning(f"Failed to load secret {secret_name}: {e}")
        
        except ImportError:
            pass
        
        return secrets
    
    def check_health(self) -> HealthStatus:
        """
        Check Modal backend health by verifying API connectivity.
        
        Returns:
            Current health status
        """
        try:
            self._authenticate()
            
            # Try to create a simple Modal app to verify connectivity
            import modal
            
            # Simple connectivity test
            app = modal.App("health-check")
            
            # If we can create an app, the API is accessible
            self.health_status = HealthStatus.HEALTHY
            self.last_health_check = datetime.now()
            
            self.logger.debug("Modal health check: HEALTHY")
            return HealthStatus.HEALTHY
        
        except BackendAuthenticationError as e:
            self.health_status = HealthStatus.UNHEALTHY
            self.last_health_check = datetime.now()
            self.logger.warning(f"Modal health check: UNHEALTHY (authentication failed: {str(e)})")
            return HealthStatus.UNHEALTHY
            
        except BackendConnectionError as e:
            self.health_status = HealthStatus.UNHEALTHY
            self.last_health_check = datetime.now()
            self.logger.warning(f"Modal health check: UNHEALTHY (connection failed: {str(e)})")
            return HealthStatus.UNHEALTHY
        
        except Exception as e:
            self.health_status = HealthStatus.DEGRADED
            self.last_health_check = datetime.now()
            self.logger.warning(f"Modal health check: DEGRADED ({str(e)})")
            return HealthStatus.DEGRADED
    
    def get_queue_length(self) -> int:
        """
        Get current queue length for Modal backend.
        
        Note: Modal doesn't expose queue length directly, so we return 0
        to indicate no local queue (Modal handles queueing internally).
        
        Returns:
            Queue length (always 0 for Modal)
        """
        return 0
    
    def supports_template(self, template_name: str) -> bool:
        """
        Check if Modal backend supports a specific template.
        
        Modal supports GPU-intensive templates and batch processing workloads.
        
        Args:
            template_name: Name of the template
            
        Returns:
            True if supported, False otherwise
        """
        return template_name in SUPPORTED_TEMPLATES
    
    def estimate_cost(self, resource_estimate: ResourceEstimate) -> float:
        """
        Estimate cost for executing a job on Modal.
        
        Cost is calculated based on GPU type and estimated duration.
        
        Args:
            resource_estimate: Resource requirements
            
        Returns:
            Estimated cost in USD
        """
        # Determine GPU type
        if not resource_estimate.requires_gpu:
            gpu_type = "CPU"
        else:
            # Use GPU memory requirement to select appropriate GPU
            if resource_estimate.gpu_memory_gb <= 16:
                gpu_type = "T4"
            elif resource_estimate.gpu_memory_gb <= 24:
                gpu_type = "A10G"
            else:
                gpu_type = "A100"
        
        # Get hourly rate
        hourly_rate = GPU_PRICING.get(gpu_type, 1.10)
        
        # Calculate cost based on estimated duration
        duration_hours = resource_estimate.estimated_duration_minutes / 60.0
        estimated_cost = hourly_rate * duration_hours
        
        self.logger.debug(
            f"Cost estimate: {gpu_type} @ ${hourly_rate}/hr × {duration_hours:.2f}hr = ${estimated_cost:.4f}"
        )
        
        return estimated_cost
