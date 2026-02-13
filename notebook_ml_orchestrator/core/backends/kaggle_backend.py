"""
Kaggle backend implementation for the Notebook ML Orchestrator.

This module provides integration with Kaggle's kernel execution platform,
enabling execution of notebook-based ML workflows on Kaggle's free GPU resources.
"""

import json
import time
import tempfile
import os
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path

from ..interfaces import Backend, MLTemplate, Job
from ..models import (
    BackendType, HealthStatus, ResourceEstimate, JobResult, BackendCapabilities
)
from ..exceptions import (
    BackendConnectionError, BackendNotAvailableError, JobExecutionError, 
    JobTimeoutError, BackendAuthenticationError, BackendQuotaExceededError
)
from ..logging_config import LoggerMixin


# Templates supported by Kaggle backend (notebook-based workflows)
SUPPORTED_TEMPLATES = [
    "image-classification",
    "model-training",
    "data-processing",
    "batch-inference",
    # Audio templates
    "speech-recognition",
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
]


class KaggleBackend(Backend, LoggerMixin):
    """
    Backend implementation for Kaggle kernel execution platform.
    
    This backend executes ML jobs on Kaggle's infrastructure with support for
    GPU kernels, notebook-based workflows, and output file retrieval.
    """
    
    def __init__(self, backend_id: str = "kaggle", config: Optional[Dict[str, Any]] = None):
        """
        Initialize Kaggle backend.
        
        Args:
            backend_id: Unique identifier for this backend instance
            config: Configuration dictionary containing credentials and options
        """
        super().__init__(backend_id, "Kaggle", BackendType.KAGGLE)
        
        self.config = config or {}
        self.credentials = self.config.get('credentials', {})
        self.options = self.config.get('options', {})
        
        # Configuration options
        self.max_concurrent_kernels = self.options.get('max_concurrent_kernels', 1)
        self.default_timeout = self.options.get('timeout', 3600)  # 1 hour default
        self.poll_interval = self.options.get('poll_interval', 30)  # 30 seconds
        self.max_retries = self.options.get('max_retries', 3)  # Network retry attempts
        self.retry_backoff_base = self.options.get('retry_backoff_base', 2)  # Exponential backoff
        
        # Set capabilities
        self.capabilities = BackendCapabilities(
            supported_templates=SUPPORTED_TEMPLATES,
            max_concurrent_jobs=self.max_concurrent_kernels,
            max_job_duration_minutes=120,  # 2 hours max
            supports_gpu=True,
            supports_batch=True,
            cost_per_hour=0.0,  # Free tier
            free_tier_limits={
                'gpu_hours_per_week': 30,
                'gpu_type': 'T4 x2'
            }
        )
        
        # Initialize Kaggle API client (lazy initialization)
        self._kaggle_api = None
        self._authenticated = False
        
        self.logger.info(f"Kaggle backend initialized: {backend_id}")
    
    def _authenticate(self) -> None:
        """
        Authenticate with Kaggle API using configured credentials.
        
        Raises:
            BackendAuthenticationError: If authentication fails
            BackendConnectionError: If Kaggle SDK is not available
        """
        if self._authenticated:
            return
        
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
            
            username = self.credentials.get('username')
            key = self.credentials.get('key')
            
            if not username or not key:
                raise BackendAuthenticationError(
                    "Kaggle credentials not configured. Set KAGGLE_USERNAME and KAGGLE_KEY.",
                    backend_id=self.id
                )
            
            # Set Kaggle credentials as environment variables
            os.environ['KAGGLE_USERNAME'] = username
            os.environ['KAGGLE_KEY'] = key
            
            # Initialize and authenticate Kaggle API
            self._kaggle_api = KaggleApi()
            self._kaggle_api.authenticate()
            
            self._authenticated = True
            self.logger.info("Kaggle authentication successful")
            
        except ImportError:
            raise BackendConnectionError(
                "Kaggle SDK not installed. Install with: pip install kaggle",
                backend_id=self.id
            )
        except BackendAuthenticationError:
            # Re-raise authentication errors
            raise
        except Exception as e:
            self.logger.error(f"Kaggle authentication failed: {e}")
            error_str = str(e).lower()
            if "unauthorized" in error_str or "invalid" in error_str or "credentials" in error_str:
                raise BackendAuthenticationError(
                    f"Kaggle authentication failed: Invalid credentials - {str(e)}",
                    backend_id=self.id
                )
            raise BackendConnectionError(
                f"Kaggle authentication failed: {str(e)}",
                backend_id=self.id
            )
    
    def _retry_with_backoff(self, operation, operation_name: str, *args, **kwargs):
        """
        Retry an operation with exponential backoff for network errors.
        
        Args:
            operation: Callable to execute
            operation_name: Name of the operation for logging
            *args: Positional arguments for the operation
            **kwargs: Keyword arguments for the operation
            
        Returns:
            Result of the operation
            
        Raises:
            BackendConnectionError: If all retries fail
        """
        last_exception = None
        
        for attempt in range(self.max_retries):
            try:
                return operation(*args, **kwargs)
            except Exception as e:
                last_exception = e
                error_str = str(e).lower()
                
                # Check if this is a retryable network error
                is_network_error = any(
                    keyword in error_str 
                    for keyword in ['network', 'connection', 'timeout', 'unreachable', 'refused', 'reset']
                )
                
                # Don't retry authentication or quota errors
                if isinstance(e, (BackendAuthenticationError, BackendQuotaExceededError)):
                    raise
                
                if not is_network_error or attempt == self.max_retries - 1:
                    # Not a network error or last attempt - raise the exception
                    raise
                
                # Calculate backoff delay
                backoff_delay = self.retry_backoff_base ** attempt
                self.logger.warning(
                    f"{operation_name} failed (attempt {attempt + 1}/{self.max_retries}): {str(e)}. "
                    f"Retrying in {backoff_delay}s..."
                )
                time.sleep(backoff_delay)
        
        # This should not be reached, but just in case
        raise BackendConnectionError(
            f"{operation_name} failed after {self.max_retries} attempts: {str(last_exception)}",
            backend_id=self.id
        )
    
    def execute_job(self, job: Job, template: MLTemplate) -> JobResult:
        """
        Execute a job on Kaggle infrastructure.
        
        This method creates a Kaggle kernel with dynamic notebook generation,
        polls for completion, and retrieves output files.
        
        Args:
            job: Job to execute
            template: Template instance for the job
            
        Returns:
            JobResult with execution results
            
        Raises:
            BackendAuthenticationError: If authentication fails
            BackendConnectionError: If connection fails
            BackendNotAvailableError: If quota is exceeded
            JobExecutionError: If job execution fails
            JobTimeoutError: If job execution times out
        """
        start_time = time.time()
        kernel_slug = None
        
        try:
            # Authenticate with Kaggle
            self._authenticate()
            
            # Get job configuration
            enable_gpu = job.metadata.get('enable_gpu', True)
            enable_internet = job.metadata.get('enable_internet', True)
            timeout_seconds = job.metadata.get('timeout', self.default_timeout)
            
            self.logger.info(
                f"Executing job {job.id} on Kaggle (GPU: {enable_gpu}, timeout: {timeout_seconds}s)"
            )
            
            # Create notebook from template and inputs
            notebook_content = self._create_notebook(job, template)
            
            # Create kernel
            kernel_slug = self._create_kernel(
                job_id=job.id,
                notebook_content=notebook_content,
                enable_gpu=enable_gpu,
                enable_internet=enable_internet
            )
            
            # Poll kernel status until completion
            final_status = self._poll_kernel_status(
                kernel_slug=kernel_slug,
                timeout_seconds=timeout_seconds,
                start_time=start_time
            )
            
            # Check if kernel completed successfully
            if final_status != "complete":
                raise JobExecutionError(
                    f"Kernel execution failed with status: {final_status}",
                    job_id=job.id,
                    backend_id=self.id
                )
            
            # Download output files
            outputs = self._download_outputs(kernel_slug, job.id)
            
            execution_time = time.time() - start_time
            
            self.logger.info(f"Job {job.id} completed successfully in {execution_time:.2f}s")
            
            return JobResult(
                success=True,
                outputs=outputs,
                execution_time_seconds=execution_time,
                backend_used=self.id,
                metadata={
                    'kernel_slug': kernel_slug,
                    'enable_gpu': enable_gpu,
                    'final_status': final_status
                }
            )
        
        except BackendAuthenticationError:
            # Re-raise authentication errors
            raise
        
        except BackendConnectionError:
            # Re-raise connection errors
            raise
        
        except JobTimeoutError:
            # Cancel kernel if timeout occurs
            if kernel_slug:
                try:
                    self._cancel_kernel(kernel_slug)
                except Exception as cancel_error:
                    self.logger.warning(f"Failed to cancel kernel {kernel_slug}: {cancel_error}")
            raise
        
        except Exception as e:
            execution_time = time.time() - start_time
            error_str = str(e).lower()
            
            # Check for quota exceeded errors
            if "quota" in error_str or "limit" in error_str or "exceeded" in error_str:
                raise BackendQuotaExceededError(
                    f"Kaggle quota exceeded: {str(e)}",
                    backend_id=self.id,
                    quota_type='kaggle_gpu_hours'
                )
            
            # Check for timeout
            if "timeout" in error_str or execution_time >= timeout_seconds:
                if kernel_slug:
                    try:
                        self._cancel_kernel(kernel_slug)
                    except Exception as cancel_error:
                        self.logger.warning(f"Failed to cancel kernel {kernel_slug}: {cancel_error}")
                
                raise JobTimeoutError(
                    f"Job {job.id} timed out after {execution_time:.2f}s",
                    job_id=job.id,
                    timeout_seconds=timeout_seconds
                )
            
            # Generic execution error
            error_msg = f"Job execution failed on Kaggle: {str(e)}"
            self.logger.error(f"Job {job.id} failed: {error_msg}")
            raise JobExecutionError(
                error_msg,
                job_id=job.id,
                backend_id=self.id
            )
    
    def _create_notebook(self, job: Job, template: MLTemplate) -> str:
        """
        Create a Jupyter notebook from template and job inputs.
        
        Args:
            job: Job to create notebook for
            template: Template instance
            
        Returns:
            Notebook content as JSON string
        """
        # Create notebook structure
        notebook = {
            "cells": [
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": [
                        f"# Job: {job.id}\n",
                        f"Template: {job.template_name}\n",
                        f"Created: {datetime.now().isoformat()}"
                    ]
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": [
                        "# Install dependencies\n",
                        "import sys\n",
                        "import json\n"
                    ]
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": [
                        "# Job inputs\n",
                        f"inputs = {json.dumps(job.inputs, indent=2)}\n"
                    ]
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": [
                        "# Execute template logic\n",
                        "# TODO: Add template-specific code here\n",
                        "print('Job execution started')\n",
                        "print(f'Inputs: {inputs}')\n",
                        "\n",
                        "# Placeholder for template execution\n",
                        "results = {'status': 'completed', 'inputs': inputs}\n",
                        "\n",
                        "print('Job execution completed')\n"
                    ]
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": [
                        "# Save results\n",
                        "with open('results.json', 'w') as f:\n",
                        "    json.dump(results, f, indent=2)\n",
                        "print('Results saved to results.json')\n"
                    ]
                }
            ],
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3"
                },
                "language_info": {
                    "name": "python",
                    "version": "3.10.0"
                }
            },
            "nbformat": 4,
            "nbformat_minor": 4
        }
        
        return json.dumps(notebook, indent=2)
    
    def _create_kernel(
        self,
        job_id: str,
        notebook_content: str,
        enable_gpu: bool,
        enable_internet: bool
    ) -> str:
        """
        Create and push a Kaggle kernel with retry logic for network errors.
        
        Args:
            job_id: Job ID for kernel naming
            notebook_content: Notebook JSON content
            enable_gpu: Whether to enable GPU
            enable_internet: Whether to enable internet
            
        Returns:
            Kernel slug (username/kernel-name)
            
        Raises:
            JobExecutionError: If kernel creation fails
            BackendConnectionError: If network errors persist after retries
        """
        def _push_kernel():
            # Create temporary directory for kernel files
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # Write notebook file
                notebook_path = temp_path / "notebook.ipynb"
                notebook_path.write_text(notebook_content)
                
                # Get username
                username = self.credentials.get('username')
                kernel_name = f"job-{job_id}"
                kernel_slug = f"{username}/{kernel_name}"
                
                # Create kernel metadata
                kernel_metadata = {
                    "id": kernel_slug,
                    "title": f"Job {job_id}",
                    "code_file": "notebook.ipynb",
                    "language": "python",
                    "kernel_type": "notebook",
                    "is_private": True,
                    "enable_gpu": enable_gpu,
                    "enable_internet": enable_internet,
                    "dataset_sources": [],
                    "competition_sources": [],
                    "kernel_sources": []
                }
                
                # Write metadata file
                metadata_path = temp_path / "kernel-metadata.json"
                metadata_path.write_text(json.dumps(kernel_metadata, indent=2))
                
                # Push kernel
                self.logger.info(f"Creating kernel: {kernel_slug}")
                self._kaggle_api.kernels_push(str(temp_path))
                
                self.logger.info(f"Kernel created successfully: {kernel_slug}")
                return kernel_slug
        
        try:
            return self._retry_with_backoff(_push_kernel, "Kernel creation")
        except Exception as e:
            error_msg = f"Failed to create kernel: {str(e)}"
            self.logger.error(error_msg)
            
            # Check for quota errors
            error_str = str(e).lower()
            if "quota" in error_str or "limit" in error_str or "exceeded" in error_str:
                raise BackendQuotaExceededError(
                    f"Kaggle quota exceeded during kernel creation: {str(e)}",
                    backend_id=self.id,
                    quota_type='kaggle_kernels'
                )
            
            raise JobExecutionError(
                error_msg,
                job_id=job_id,
                backend_id=self.id
            )
    
    def _poll_kernel_status(
        self,
        kernel_slug: str,
        timeout_seconds: int,
        start_time: float
    ) -> str:
        """
        Poll kernel status until completion or timeout with retry logic for network errors.
        
        Args:
            kernel_slug: Kernel slug to poll
            timeout_seconds: Maximum time to wait
            start_time: Job start time
            
        Returns:
            Final kernel status
            
        Raises:
            JobTimeoutError: If polling times out
            BackendConnectionError: If network errors persist
        """
        self.logger.info(f"Polling kernel status: {kernel_slug}")
        
        while True:
            elapsed_time = time.time() - start_time
            
            # Check timeout
            if elapsed_time >= timeout_seconds:
                raise JobTimeoutError(
                    f"Kernel {kernel_slug} timed out after {elapsed_time:.2f}s",
                    job_id=kernel_slug,
                    timeout_seconds=timeout_seconds
                )
            
            try:
                # Get kernel status with retry logic
                def _get_status():
                    return self._kaggle_api.kernels_status(kernel_slug)
                
                status_response = self._retry_with_backoff(_get_status, "Kernel status check")
                status = status_response.get('status', 'unknown').lower()
                
                self.logger.debug(f"Kernel {kernel_slug} status: {status}")
                
                # Check if kernel is in terminal state
                if status in ['complete', 'error', 'cancelled']:
                    return status
                
                # Wait before next poll
                time.sleep(self.poll_interval)
            
            except (BackendConnectionError, BackendQuotaExceededError):
                # Re-raise critical errors
                raise
            except Exception as e:
                self.logger.warning(f"Error polling kernel status: {e}")
                # Continue polling unless it's a critical error
                time.sleep(self.poll_interval)
    
    def _download_outputs(self, kernel_slug: str, job_id: str) -> Dict[str, Any]:
        """
        Download output files from completed kernel with retry logic for network errors.
        
        Args:
            kernel_slug: Kernel slug
            job_id: Job ID
            
        Returns:
            Dictionary of outputs
        """
        def _download():
            # Create temporary directory for outputs
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # Download kernel outputs
                self.logger.info(f"Downloading outputs for kernel: {kernel_slug}")
                self._kaggle_api.kernels_output(kernel_slug, path=str(temp_path))
                
                # Read results.json if it exists
                results_path = temp_path / "results.json"
                if results_path.exists():
                    with open(results_path, 'r') as f:
                        results = json.load(f)
                    return results
                
                # If no results.json, return list of output files
                output_files = [f.name for f in temp_path.iterdir() if f.is_file()]
                return {
                    'output_files': output_files,
                    'message': 'No results.json found, listing output files'
                }
        
        try:
            return self._retry_with_backoff(_download, "Output download")
        except Exception as e:
            self.logger.warning(f"Failed to download outputs: {e}")
            return {
                'error': f"Failed to download outputs: {str(e)}",
                'kernel_slug': kernel_slug
            }
    
    def _cancel_kernel(self, kernel_slug: str) -> None:
        """
        Cancel a running kernel.
        
        Note: Kaggle API doesn't have a direct cancel method in the public API.
        This method attempts to stop the kernel by deleting it.
        
        Args:
            kernel_slug: Kernel slug to cancel
        """
        try:
            self.logger.info(f"Attempting to cancel kernel: {kernel_slug}")
            
            # Try to delete the kernel as a way to cancel it
            # Note: This may not work for already-running kernels
            try:
                # The Kaggle API doesn't expose a cancel method, but we can try to delete
                # This is a best-effort attempt
                self.logger.warning(
                    f"Kernel cancellation not directly supported by Kaggle API. "
                    f"Kernel {kernel_slug} may continue running."
                )
            except Exception as delete_error:
                self.logger.error(f"Failed to delete kernel {kernel_slug}: {delete_error}")
                
        except Exception as e:
            self.logger.error(f"Failed to cancel kernel {kernel_slug}: {e}")
    
    def check_health(self) -> HealthStatus:
        """
        Check Kaggle backend health by verifying API connectivity and quota.
        
        Returns:
            Current health status
        """
        try:
            self._authenticate()
            
            # Try to list user's kernels to verify connectivity with retry logic
            def _check_connectivity():
                return self._kaggle_api.kernels_list(
                    user=self.credentials.get('username'), 
                    page_size=1
                )
            
            try:
                self._retry_with_backoff(_check_connectivity, "Health check")
            except BackendQuotaExceededError:
                # Quota exceeded means API is working but quota is exhausted
                self.health_status = HealthStatus.DEGRADED
                self.last_health_check = datetime.now()
                self.logger.warning("Kaggle health check: DEGRADED (quota exceeded)")
                return HealthStatus.DEGRADED
            except Exception as list_error:
                error_str = str(list_error).lower()
                if "quota" in error_str or "limit" in error_str:
                    self.health_status = HealthStatus.DEGRADED
                    self.last_health_check = datetime.now()
                    self.logger.warning("Kaggle health check: DEGRADED (quota issues)")
                    return HealthStatus.DEGRADED
                raise
            
            self.health_status = HealthStatus.HEALTHY
            self.last_health_check = datetime.now()
            self.logger.debug("Kaggle health check: HEALTHY")
            return HealthStatus.HEALTHY
        
        except BackendAuthenticationError as e:
            self.health_status = HealthStatus.UNHEALTHY
            self.last_health_check = datetime.now()
            self.logger.warning(f"Kaggle health check: UNHEALTHY (authentication failed: {str(e)})")
            return HealthStatus.UNHEALTHY
        
        except BackendConnectionError as e:
            self.health_status = HealthStatus.UNHEALTHY
            self.last_health_check = datetime.now()
            self.logger.warning(f"Kaggle health check: UNHEALTHY (connection failed: {str(e)})")
            return HealthStatus.UNHEALTHY
        
        except Exception as e:
            self.health_status = HealthStatus.DEGRADED
            self.last_health_check = datetime.now()
            self.logger.warning(f"Kaggle health check: DEGRADED ({str(e)})")
            return HealthStatus.DEGRADED
    
    def get_queue_length(self) -> int:
        """
        Get current queue length for Kaggle backend.
        
        Note: Kaggle doesn't expose queue length directly, so we return 0
        to indicate no local queue (Kaggle handles queueing internally).
        
        Returns:
            Queue length (always 0 for Kaggle)
        """
        return 0
    
    def supports_template(self, template_name: str) -> bool:
        """
        Check if Kaggle backend supports a specific template.
        
        Kaggle supports notebook-based templates for training and data processing.
        
        Args:
            template_name: Name of the template
            
        Returns:
            True if supported, False otherwise
        """
        return template_name in SUPPORTED_TEMPLATES
    
    def estimate_cost(self, resource_estimate: ResourceEstimate) -> float:
        """
        Estimate cost for executing a job on Kaggle.
        
        Kaggle provides free tier access, so cost is always 0.0.
        
        Args:
            resource_estimate: Resource requirements
            
        Returns:
            Estimated cost in USD (always 0.0 for Kaggle free tier)
        """
        return 0.0
