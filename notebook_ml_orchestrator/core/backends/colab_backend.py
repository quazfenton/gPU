"""
Google Colab backend implementation for the Notebook ML Orchestrator.

This module provides integration with Google Colab's notebook execution platform,
enabling execution of ML experiments on Colab's free GPU resources with Google Drive integration.
"""

import json
import time
import tempfile
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path

from ..interfaces import Backend, MLTemplate, Job
from ..models import (
    BackendType, HealthStatus, ResourceEstimate, JobResult, BackendCapabilities
)
from ..exceptions import (
    BackendConnectionError, BackendNotAvailableError, JobExecutionError, 
    JobTimeoutError, BackendAuthenticationError
)
from ..logging_config import LoggerMixin


# Templates supported by Colab backend (interactive notebook templates)
SUPPORTED_TEMPLATES = [
    "model-training",
    "data-processing",
    "image-classification",
    "text-generation",
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


class ColabBackend(Backend, LoggerMixin):
    """
    Backend implementation for Google Colab notebook execution platform.
    
    This backend executes ML jobs on Google Colab's infrastructure with support for
    GPU runtimes, OAuth 2.0 authentication, and Google Drive integration for result storage.
    """
    
    def __init__(self, backend_id: str = "colab", config: Optional[Dict[str, Any]] = None):
        """
        Initialize Colab backend.
        
        Args:
            backend_id: Unique identifier for this backend instance
            config: Configuration dictionary containing credentials and options
        """
        super().__init__(backend_id, "Google Colab", BackendType.COLAB)
        
        self.config = config or {}
        self.credentials = self.config.get('credentials', {})
        self.options = self.config.get('options', {})
        
        # Configuration options
        self.default_timeout = self.options.get('timeout', 3600)  # 1 hour default
        self.enable_gpu = self.options.get('enable_gpu', True)
        self.drive_folder = self.options.get('drive_folder', 'orchestrator_jobs')
        
        # Set capabilities
        self.capabilities = BackendCapabilities(
            supported_templates=SUPPORTED_TEMPLATES,
            max_concurrent_jobs=1,  # Colab typically runs one notebook at a time
            max_job_duration_minutes=720,  # 12 hours max for Colab Pro
            supports_gpu=True,
            supports_batch=False,
            cost_per_hour=0.0,  # Free tier
            free_tier_limits={
                'gpu_type': 'T4',
                'session_duration_hours': 12,
                'note': 'Free tier with usage limits'
            }
        )
        
        # Initialize Google API clients (lazy initialization)
        self._drive_service = None
        self._credentials = None
        self._authenticated = False
        
        self.logger.info(f"Colab backend initialized: {backend_id}")
    
    def _handle_token_expiration(self, error: Exception) -> bool:
        """
        Handle OAuth token expiration by attempting to refresh.
        
        Args:
            error: The exception that occurred
            
        Returns:
            True if token was refreshed successfully, False otherwise
        """
        error_str = str(error).lower()
        
        # Check if error is related to token expiration
        if any(keyword in error_str for keyword in ['expired', 'unauthorized', 'invalid_grant', 'token']):
            self.logger.warning("OAuth token may have expired, attempting refresh...")
            
            try:
                # Force re-authentication with token refresh
                self._authenticated = False
                self._authenticate(force_refresh=True)
                return True
            except Exception as refresh_error:
                self.logger.error(f"Token refresh failed: {refresh_error}")
                return False
        
        return False
    
    def _execute_with_retry(self, operation, max_retries: int = 2):
        """
        Execute an operation with automatic retry on token expiration.
        
        Args:
            operation: Callable to execute
            max_retries: Maximum number of retries
            
        Returns:
            Result of the operation
            
        Raises:
            Exception: If operation fails after all retries
        """
        last_error = None
        
        for attempt in range(max_retries + 1):
            try:
                return operation()
            except Exception as e:
                last_error = e
                
                # Try to handle token expiration
                if attempt < max_retries and self._handle_token_expiration(e):
                    self.logger.info(f"Retrying operation after token refresh (attempt {attempt + 1}/{max_retries})")
                    continue
                
                # If not a token issue or out of retries, raise
                if attempt >= max_retries:
                    break
        
        # All retries exhausted
        raise last_error
    
    def _authenticate(self, force_refresh: bool = False) -> None:
        """
        Authenticate with Google OAuth 2.0 using configured credentials.
        
        Args:
            force_refresh: Force token refresh even if already authenticated
        
        Raises:
            BackendAuthenticationError: If authentication fails
            BackendConnectionError: If Google API client libraries are not available
        """
        if self._authenticated and not force_refresh:
            return
        
        try:
            from google.oauth2.credentials import Credentials
            from googleapiclient.discovery import build
            from google.auth.transport.requests import Request
            
            client_id = self.credentials.get('client_id')
            client_secret = self.credentials.get('client_secret')
            refresh_token = self.credentials.get('refresh_token')
            
            if not client_id or not client_secret or not refresh_token:
                raise BackendAuthenticationError(
                    "Google OAuth credentials not configured. Set GOOGLE_CLIENT_ID, "
                    "GOOGLE_CLIENT_SECRET, and GOOGLE_REFRESH_TOKEN.",
                    backend_id=self.id
                )
            
            # Create credentials object
            self._credentials = Credentials(
                token=None,  # Will be refreshed
                refresh_token=refresh_token,
                token_uri="https://oauth2.googleapis.com/token",
                client_id=client_id,
                client_secret=client_secret,
                scopes=[
                    'https://www.googleapis.com/auth/drive',
                    'https://www.googleapis.com/auth/drive.file'
                ]
            )
            
            # Refresh the access token
            try:
                self._credentials.refresh(Request())
                self.logger.info("OAuth token refreshed successfully")
            except Exception as refresh_error:
                error_str = str(refresh_error).lower()
                if "invalid" in error_str or "unauthorized" in error_str:
                    raise BackendAuthenticationError(
                        f"Google OAuth token refresh failed: Invalid credentials - {str(refresh_error)}",
                        backend_id=self.id
                    )
                raise
            
            # Build Drive service
            self._drive_service = build('drive', 'v3', credentials=self._credentials)
            
            # Verify credentials by attempting to list files (limit 1)
            try:
                self._drive_service.files().list(pageSize=1).execute()
                self._authenticated = True
                self.logger.info("Google OAuth authentication successful")
            except Exception as verify_error:
                error_str = str(verify_error).lower()
                if "unauthorized" in error_str or "invalid" in error_str:
                    raise BackendAuthenticationError(
                        f"Google Drive API access failed: {str(verify_error)}",
                        backend_id=self.id
                    )
                raise
            
        except ImportError as e:
            raise BackendConnectionError(
                f"Google API client libraries not installed. Install with: "
                f"pip install google-auth google-auth-oauthlib google-auth-httplib2 google-api-python-client - {str(e)}",
                backend_id=self.id
            )
        except BackendAuthenticationError:
            # Re-raise authentication errors
            raise
        except Exception as e:
            self.logger.error(f"Google OAuth authentication failed: {e}")
            raise BackendConnectionError(
                f"Google OAuth authentication failed: {str(e)}",
                backend_id=self.id
            )
    
    def execute_job(self, job: Job, template: MLTemplate) -> JobResult:
        """
        Execute a job on Google Colab infrastructure.
        
        This method creates a Colab notebook in Google Drive, executes it
        (note: actual execution requires manual intervention or automation tools),
        and retrieves results from Google Drive.
        
        Args:
            job: Job to execute
            template: Template instance for the job
            
        Returns:
            JobResult with execution results
            
        Raises:
            BackendAuthenticationError: If authentication fails
            BackendConnectionError: If connection fails
            BackendNotAvailableError: If GPU is unavailable or Drive quota exceeded
            JobExecutionError: If job execution fails
            JobTimeoutError: If job execution times out
        """
        start_time = time.time()
        notebook_id = None
        
        try:
            # Authenticate with Google
            self._authenticate()
            
            # Get job configuration
            enable_gpu = job.metadata.get('enable_gpu', self.enable_gpu)
            timeout_seconds = job.metadata.get('timeout', self.default_timeout)
            
            self.logger.info(
                f"Executing job {job.id} on Colab (GPU: {enable_gpu}, timeout: {timeout_seconds}s)"
            )
            
            # Check GPU availability if required
            if enable_gpu:
                self._check_gpu_availability()
            
            # Create notebook from template and inputs
            notebook_content = self._create_notebook(job, template, enable_gpu)
            
            # Upload notebook to Google Drive with retry on token expiration
            notebook_id = self._execute_with_retry(
                lambda: self._upload_notebook_to_drive(
                    job_id=job.id,
                    notebook_content=notebook_content
                )
            )
            
            self.logger.info(f"Notebook uploaded to Drive: {notebook_id}")
            
            # Note: Colab doesn't have an official API for programmatic execution
            # This is a limitation - in a real implementation, you would need to:
            # 1. Use selenium/puppeteer to automate browser interaction
            # 2. Use unofficial Colab API wrappers
            # 3. Manually execute the notebook and save results
            
            # For this implementation, we'll simulate the execution flow
            # and provide a placeholder for where execution would happen
            
            self.logger.warning(
                "Colab programmatic execution not fully implemented. "
                "Notebook created in Drive but requires manual execution or automation tool."
            )
            
            # Simulate execution (in real implementation, this would poll for completion)
            execution_result = self._simulate_execution(job, notebook_id, timeout_seconds, start_time)
            
            execution_time = time.time() - start_time
            
            self.logger.info(f"Job {job.id} completed in {execution_time:.2f}s")
            
            return JobResult(
                success=True,
                outputs=execution_result,
                execution_time_seconds=execution_time,
                backend_used=self.id,
                metadata={
                    'notebook_id': notebook_id,
                    'enable_gpu': enable_gpu,
                    'drive_folder': self.drive_folder,
                    'note': 'Notebook created in Drive - manual execution may be required'
                }
            )
        
        except BackendAuthenticationError:
            # Re-raise authentication errors
            raise
        
        except BackendConnectionError:
            # Re-raise connection errors
            raise
        
        except BackendNotAvailableError:
            # Re-raise availability errors
            raise
        
        except JobTimeoutError:
            # Timeout occurred
            raise
        
        except Exception as e:
            execution_time = time.time() - start_time
            error_str = str(e).lower()
            
            # Check for Drive quota exceeded
            if "quota" in error_str or "storage" in error_str or "limit" in error_str:
                raise BackendNotAvailableError(
                    f"Google Drive quota exceeded: {str(e)}",
                    required_capabilities=['drive_storage']
                )
            
            # Check for GPU unavailability
            if "gpu" in error_str and ("unavailable" in error_str or "not available" in error_str):
                raise BackendNotAvailableError(
                    f"GPU unavailable on Colab: {str(e)}",
                    required_capabilities=['gpu']
                )
            
            # Check for runtime disconnection
            if "disconnect" in error_str or "connection" in error_str or "runtime" in error_str:
                raise BackendConnectionError(
                    f"Colab runtime disconnected: {str(e)}",
                    backend_id=self.id
                )
            
            # Check for timeout
            if "timeout" in error_str or execution_time >= timeout_seconds:
                raise JobTimeoutError(
                    f"Job {job.id} timed out after {execution_time:.2f}s",
                    job_id=job.id,
                    timeout_seconds=timeout_seconds
                )
            
            # Check for authentication errors
            if "unauthorized" in error_str or "authentication" in error_str or "invalid credentials" in error_str:
                raise BackendAuthenticationError(
                    f"Google authentication failed: {str(e)}",
                    backend_id=self.id
                )
            
            # Generic execution error
            error_msg = f"Job execution failed on Colab: {str(e)}"
            self.logger.error(f"Job {job.id} failed: {error_msg}")
            raise JobExecutionError(
                error_msg,
                job_id=job.id,
                backend_id=self.id
            )
    
    def _check_gpu_availability(self) -> None:
        """
        Check if GPU is available on Colab.
        
        Note: This is a placeholder as we can't actually check GPU availability
        without executing code on Colab. In a real implementation, this would
        query Colab's API or check recent execution history.
        
        Raises:
            BackendNotAvailableError: If GPU is known to be unavailable
        """
        # Placeholder implementation
        # In production, you would check:
        # 1. Recent execution history for GPU availability
        # 2. Colab API status (if available)
        # 3. User's quota/limits
        
        self.logger.debug("GPU availability check passed (placeholder)")
        pass
    
    def _create_notebook(self, job: Job, template: MLTemplate, enable_gpu: bool) -> str:
        """
        Create a Jupyter notebook from template and job inputs.
        
        Args:
            job: Job to create notebook for
            template: Template instance
            enable_gpu: Whether to enable GPU
            
        Returns:
            Notebook content as JSON string
        """
        # Create notebook structure
        cells = [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    f"# Job: {job.id}\n",
                    f"Template: {job.template_name}\n",
                    f"Created: {datetime.now().isoformat()}\n",
                    f"GPU Enabled: {enable_gpu}"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Mount Google Drive\n",
                    "from google.colab import drive\n",
                    "drive.mount('/content/drive')\n"
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
                    "import json\n",
                    "import os\n"
                ]
            }
        ]
        
        # Add GPU check if enabled
        if enable_gpu:
            cells.append({
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Check GPU availability\n",
                    "import torch\n",
                    "if torch.cuda.is_available():\n",
                    "    print(f'GPU available: {torch.cuda.get_device_name(0)}')\n",
                    "    print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB')\n",
                    "else:\n",
                    "    print('WARNING: GPU not available')\n"
                ]
            })
        
        # Add job inputs
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Job inputs\n",
                f"inputs = {json.dumps(job.inputs, indent=2)}\n"
            ]
        })
        
        # Add template execution code
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Execute template logic\n",
                "print('Job execution started')\n",
                "print(f'Inputs: {inputs}')\n",
                "\n",
                "# TODO: Add template-specific code here\n",
                "# Placeholder for template execution\n",
                "results = {'status': 'completed', 'inputs': inputs}\n",
                "\n",
                "print('Job execution completed')\n"
            ]
        })
        
        # Add results saving
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Save results to Google Drive\n",
                f"output_dir = '/content/drive/MyDrive/{self.drive_folder}'\n",
                "os.makedirs(output_dir, exist_ok=True)\n",
                f"output_file = os.path.join(output_dir, 'results_{job.id}.json')\n",
                "\n",
                "with open(output_file, 'w') as f:\n",
                "    json.dump(results, f, indent=2)\n",
                "\n",
                "print(f'Results saved to {output_file}')\n"
            ]
        })
        
        notebook = {
            "cells": cells,
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3"
                },
                "language_info": {
                    "name": "python",
                    "version": "3.10.0"
                },
                "accelerator": "GPU" if enable_gpu else "None",
                "colab": {
                    "name": f"job_{job.id}.ipynb",
                    "provenance": []
                }
            },
            "nbformat": 4,
            "nbformat_minor": 0
        }
        
        return json.dumps(notebook, indent=2)
    
    def _upload_notebook_to_drive(self, job_id: str, notebook_content: str) -> str:
        """
        Upload notebook to Google Drive with retry on token expiration.
        
        Args:
            job_id: Job ID for notebook naming
            notebook_content: Notebook JSON content
            
        Returns:
            File ID of uploaded notebook
            
        Raises:
            JobExecutionError: If upload fails
            BackendAuthenticationError: If authentication fails
        """
        try:
            from googleapiclient.http import MediaInMemoryUpload
            
            # Create folder if it doesn't exist (with retry)
            folder_id = self._execute_with_retry(
                lambda: self._get_or_create_folder(self.drive_folder)
            )
            
            # Prepare file metadata
            file_metadata = {
                'name': f'job_{job_id}.ipynb',
                'mimeType': 'application/x-ipynb+json',
                'parents': [folder_id]
            }
            
            # Upload file
            media = MediaInMemoryUpload(
                notebook_content.encode('utf-8'),
                mimetype='application/x-ipynb+json',
                resumable=True
            )
            
            file = self._drive_service.files().create(
                body=file_metadata,
                media_body=media,
                fields='id'
            ).execute()
            
            file_id = file.get('id')
            self.logger.info(f"Notebook uploaded to Drive with ID: {file_id}")
            
            return file_id
        
        except BackendAuthenticationError:
            # Re-raise authentication errors
            raise
        
        except Exception as e:
            error_msg = f"Failed to upload notebook to Drive: {str(e)}"
            self.logger.error(error_msg)
            raise JobExecutionError(
                error_msg,
                job_id=job_id,
                backend_id=self.id
            )
    
    def _get_or_create_folder(self, folder_name: str) -> str:
        """
        Get or create a folder in Google Drive.
        
        Args:
            folder_name: Name of the folder
            
        Returns:
            Folder ID
        """
        try:
            # Search for existing folder
            query = f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
            results = self._drive_service.files().list(
                q=query,
                spaces='drive',
                fields='files(id, name)'
            ).execute()
            
            files = results.get('files', [])
            
            if files:
                # Folder exists
                folder_id = files[0]['id']
                self.logger.debug(f"Found existing folder: {folder_name} (ID: {folder_id})")
                return folder_id
            
            # Create new folder
            file_metadata = {
                'name': folder_name,
                'mimeType': 'application/vnd.google-apps.folder'
            }
            
            folder = self._drive_service.files().create(
                body=file_metadata,
                fields='id'
            ).execute()
            
            folder_id = folder.get('id')
            self.logger.info(f"Created folder: {folder_name} (ID: {folder_id})")
            
            return folder_id
        
        except Exception as e:
            self.logger.error(f"Failed to get/create folder: {e}")
            # Return root folder as fallback
            return 'root'
    
    def _simulate_execution(
        self,
        job: Job,
        notebook_id: str,
        timeout_seconds: int,
        start_time: float
    ) -> Dict[str, Any]:
        """
        Simulate notebook execution (placeholder for actual execution).
        
        In a real implementation, this would:
        1. Use selenium/puppeteer to open Colab and execute cells
        2. Poll for completion
        3. Retrieve results from Drive
        
        Args:
            job: Job being executed
            notebook_id: Notebook file ID in Drive
            timeout_seconds: Maximum time to wait
            start_time: Job start time
            
        Returns:
            Execution results
        """
        # This is a placeholder implementation
        # In production, you would need to implement actual execution logic
        
        self.logger.warning(
            "Using simulated execution - real Colab execution requires automation tools"
        )
        
        # Return placeholder results
        return {
            'status': 'completed',
            'notebook_id': notebook_id,
            'inputs': job.inputs,
            'note': 'Notebook created in Drive - requires manual execution or automation',
            'drive_path': f'{self.drive_folder}/job_{job.id}.ipynb'
        }
    
    def check_health(self) -> HealthStatus:
        """
        Check Colab backend health by verifying OAuth token validity and Drive access.
        
        Returns:
            Current health status
        """
        try:
            self._authenticate()
            
            # Try to list files to verify Drive access
            try:
                self._drive_service.files().list(pageSize=1).execute()
            except Exception as drive_error:
                error_str = str(drive_error).lower()
                if "quota" in error_str or "limit" in error_str:
                    self.health_status = HealthStatus.DEGRADED
                    self.last_health_check = datetime.now()
                    self.logger.warning("Colab health check: DEGRADED (Drive quota issues)")
                    return HealthStatus.DEGRADED
                raise
            
            self.health_status = HealthStatus.HEALTHY
            self.last_health_check = datetime.now()
            self.logger.debug("Colab health check: HEALTHY")
            return HealthStatus.HEALTHY
        
        except BackendAuthenticationError as e:
            self.health_status = HealthStatus.UNHEALTHY
            self.last_health_check = datetime.now()
            self.logger.warning(f"Colab health check: UNHEALTHY (authentication failed: {str(e)})")
            return HealthStatus.UNHEALTHY
        
        except BackendConnectionError as e:
            self.health_status = HealthStatus.UNHEALTHY
            self.last_health_check = datetime.now()
            self.logger.warning(f"Colab health check: UNHEALTHY (connection failed: {str(e)})")
            return HealthStatus.UNHEALTHY
        
        except Exception as e:
            self.health_status = HealthStatus.DEGRADED
            self.last_health_check = datetime.now()
            self.logger.warning(f"Colab health check: DEGRADED ({str(e)})")
            return HealthStatus.DEGRADED
    
    def get_queue_length(self) -> int:
        """
        Get current queue length for Colab backend.
        
        Note: Colab doesn't have a queue concept for free tier,
        so we return 0 to indicate no local queue.
        
        Returns:
            Queue length (always 0 for Colab)
        """
        return 0
    
    def supports_template(self, template_name: str) -> bool:
        """
        Check if Colab backend supports a specific template.
        
        Colab supports interactive notebook templates for experimentation and training.
        
        Args:
            template_name: Name of the template
            
        Returns:
            True if supported, False otherwise
        """
        return template_name in SUPPORTED_TEMPLATES
    
    def estimate_cost(self, resource_estimate: ResourceEstimate) -> float:
        """
        Estimate cost for executing a job on Colab.
        
        Colab provides free tier access, so cost is always 0.0.
        
        Args:
            resource_estimate: Resource requirements
            
        Returns:
            Estimated cost in USD (always 0.0 for Colab free tier)
        """
        return 0.0
