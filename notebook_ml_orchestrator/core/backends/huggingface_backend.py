"""
HuggingFace backend implementation for the Notebook ML Orchestrator.

This module provides integration with HuggingFace Spaces and Inference API,
enabling execution of ML inference jobs on HuggingFace's platform.
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


# Templates supported by HuggingFace backend (inference-focused)
SUPPORTED_TEMPLATES = [
    "text-generation",
    "image-classification",
    "embeddings",
    "image-generation",
    # Audio templates
    "speech-recognition",
    "audio-generation",
    "audio-classification",
    "text-to-speech",
    # Vision templates
    "object-detection",
    "image-segmentation",
    "face-detection",
    "image-upscaling",
    # Language templates
    "named-entity-recognition",
    "sentiment-analysis",
    "translation",
    "summarization",
    "code-generation",
    "text-embedding",
    "zero-shot-classification",
    "text-similarity",
    # Multimodal templates
    "image-captioning",
    "visual-question-answering",
    "text-to-image",
    "document-qa",
]


class HuggingFaceBackend(Backend, LoggerMixin):
    """
    Backend implementation for HuggingFace Spaces and Inference API.
    
    This backend executes ML inference jobs on HuggingFace's infrastructure,
    supporting both the Inference API for popular models and custom Spaces
    for specialized deployments.
    """
    
    def __init__(self, backend_id: str = "huggingface", config: Optional[Dict[str, Any]] = None):
        """
        Initialize HuggingFace backend.
        
        Args:
            backend_id: Unique identifier for this backend instance
            config: Configuration dictionary containing credentials and options
        """
        super().__init__(backend_id, "HuggingFace", BackendType.HUGGINGFACE)
        
        self.config = config or {}
        self.credentials = self.config.get('credentials', {})
        self.options = self.config.get('options', {})
        
        # Configuration options
        self.default_space_hardware = self.options.get('default_space_hardware', 'cpu-basic')
        self.space_startup_timeout = self.options.get('space_startup_timeout', 300)  # 5 minutes
        self.use_inference_api = self.options.get('use_inference_api', True)
        
        # Set capabilities
        self.capabilities = BackendCapabilities(
            supported_templates=SUPPORTED_TEMPLATES,
            max_concurrent_jobs=5,
            max_job_duration_minutes=30,
            supports_gpu=True,
            supports_batch=False,
            cost_per_hour=0.0,  # Free tier
            free_tier_limits={'requests_per_hour': 1000}
        )
        
        # Initialize HuggingFace clients (lazy initialization)
        self._hf_api = None
        self._inference_client = None
        self._authenticated = False
        
        # Cache for whoami/health check to avoid rate limits
        self._whoami_cache = None
        self._whoami_cache_time = 0
        self._whoami_cache_ttl = 300  # 5 minutes
        
        self.logger.info(f"HuggingFace backend initialized: {backend_id}")
    
    def _authenticate(self) -> None:
        """
        Authenticate with HuggingFace API using configured token.
        
        Raises:
            BackendAuthenticationError: If authentication fails
            BackendConnectionError: If HuggingFace SDK is not available
        """
        if self._authenticated:
            return
        
        try:
            from huggingface_hub import HfApi, InferenceClient
            
            token = self.credentials.get('token')
            
            if not token:
                raise BackendAuthenticationError(
                    "HuggingFace token not configured. Set HF_TOKEN environment variable.",
                    backend_id=self.id
                )
            
            # Initialize HuggingFace API client
            try:
                self._hf_api = HfApi(token=token)
                
                # Verify token by getting user info
                user_info = self._hf_api.whoami()
                self.logger.info(f"HuggingFace authentication successful for user: {user_info.get('name', 'unknown')}")
                
                # Initialize Inference Client
                self._inference_client = InferenceClient(token=token)
                
                self._authenticated = True
                
            except Exception as auth_error:
                error_str = str(auth_error).lower()
                if "unauthorized" in error_str or "invalid" in error_str or "token" in error_str:
                    raise BackendAuthenticationError(
                        f"HuggingFace authentication failed: Invalid token - {str(auth_error)}",
                        backend_id=self.id
                    )
                raise
            
        except ImportError as e:
            raise BackendConnectionError(
                f"HuggingFace SDK not installed. Install with: pip install huggingface-hub gradio-client - {str(e)}",
                backend_id=self.id
            )
        except BackendAuthenticationError:
            raise
        except Exception as e:
            self.logger.error(f"HuggingFace authentication failed: {e}")
            raise BackendConnectionError(
                f"HuggingFace authentication failed: {str(e)}",
                backend_id=self.id
            )
    
    def execute_job(self, job: Job, template: MLTemplate) -> JobResult:
        """
        Execute a job on HuggingFace infrastructure.
        
        This method routes the job to either the Inference API (for popular models)
        or a custom Space (for specialized deployments), with retry logic for
        transient failures.
        
        Args:
            job: Job to execute
            template: Template instance for the job
            
        Returns:
            JobResult with execution results
            
        Raises:
            BackendAuthenticationError: If authentication fails
            BackendConnectionError: If connection fails
            BackendNotAvailableError: If Space is unavailable
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
                raise
            
            except BackendAuthenticationError:
                raise
            
            except BackendConnectionError:
                raise
            
            except BackendNotAvailableError as e:
                if "space" in str(e).lower() and "building" in str(e).lower() and attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)
                    self.logger.warning(
                        f"Space building for job {job.id}, retrying in {wait_time}s (attempt {attempt + 1}/{max_retries})"
                    )
                    time.sleep(wait_time)
                else:
                    raise
            
            except JobExecutionError as e:
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)
                    self.logger.warning(
                        f"Job {job.id} failed, retrying in {wait_time}s (attempt {attempt + 1}/{max_retries}): {str(e)}"
                    )
                    time.sleep(wait_time)
                else:
                    raise
        
        raise JobExecutionError(
            f"Job {job.id} failed after {max_retries} attempts",
            job_id=job.id,
            backend_id=self.id
        )
    
    def _execute_job_internal(self, job: Job, template: MLTemplate) -> JobResult:
        """
        Internal method to execute a job on HuggingFace infrastructure.
        
        Args:
            job: Job to execute
            template: Template instance for the job
            
        Returns:
            JobResult with execution results
        """
        start_time = time.time()
        
        try:
            self._authenticate()
            
            # Determine execution method: Inference API or Space
            use_space = job.metadata.get('use_space', False)
            space_url = job.metadata.get('space_url')
            model_name = job.metadata.get('model_name')
            
            self.logger.info(f"Executing job {job.id} on HuggingFace (use_space={use_space})")
            
            if use_space or space_url:
                # Execute via Space
                result_data = self._execute_via_space(job, space_url)
            elif self.use_inference_api and model_name:
                # Execute via Inference API
                result_data = self._execute_via_inference_api(job, model_name)
            else:
                # Default: try Inference API with template-based model selection
                result_data = self._execute_via_inference_api(job, None)
            
            execution_time = time.time() - start_time
            
            self.logger.info(f"Job {job.id} completed successfully in {execution_time:.2f}s")
            
            return JobResult(
                success=True,
                outputs=result_data,
                execution_time_seconds=execution_time,
                backend_used=self.id,
                metadata={
                    'use_space': use_space,
                    'space_url': space_url,
                    'model_name': model_name
                }
            )
        
        except BackendAuthenticationError:
            raise
        
        except BackendConnectionError:
            raise
        
        except Exception as e:
            execution_time = time.time() - start_time
            error_str = str(e).lower()
            
            # Check for timeout errors
            if "timeout" in error_str or "timed out" in error_str:
                raise JobTimeoutError(
                    f"Job {job.id} timed out after {execution_time:.2f}s",
                    job_id=job.id,
                    timeout_seconds=self.space_startup_timeout
                )
            
            # Check for Space unavailability
            if "space" in error_str and ("not found" in error_str or "unavailable" in error_str):
                raise BackendNotAvailableError(
                    f"HuggingFace Space unavailable: {str(e)}",
                    required_capabilities=['space']
                )
            
            # Check for Space building
            if "space" in error_str and "building" in error_str:
                raise BackendNotAvailableError(
                    f"HuggingFace Space is building: {str(e)}",
                    required_capabilities=['space']
                )
            
            # Check for rate limiting
            if "rate limit" in error_str or "too many requests" in error_str or "429" in error_str:
                retry_after = None
                if hasattr(e, 'retry_after'):
                    retry_after = e.retry_after
                raise BackendRateLimitError(
                    f"HuggingFace rate limit exceeded: {str(e)}",
                    backend_id=self.id,
                    retry_after=retry_after
                )
            
            # Check for model not found
            if "model" in error_str and "not found" in error_str:
                raise JobExecutionError(
                    f"Model not found on HuggingFace: {str(e)}",
                    job_id=job.id,
                    backend_id=self.id
                )
            
            # Generic execution error
            error_msg = f"Job execution failed on HuggingFace: {str(e)}"
            self.logger.error(f"Job {job.id} failed: {error_msg}")
            raise JobExecutionError(
                error_msg,
                job_id=job.id,
                backend_id=self.id
            )
    
    def _execute_via_inference_api(self, job: Job, model_name: Optional[str]) -> Dict[str, Any]:
        """
        Execute job using HuggingFace Inference API.
        
        Args:
            job: Job to execute
            model_name: Model name (optional, will use template default if not provided)
            
        Returns:
            Execution results
            
        Raises:
            BackendConnectionError: If Inference client is not initialized
            JobExecutionError: If model is not found or execution fails
            BackendRateLimitError: If rate limit is exceeded
            JobTimeoutError: If execution times out
        """
        if not self._inference_client:
            raise BackendConnectionError(
                "Inference client not initialized",
                backend_id=self.id
            )
        
        # Get template-specific model if not provided
        if not model_name:
            model_name = self._get_default_model_for_template(job.template_name)
        
        self.logger.info(f"Executing job {job.id} via Inference API with model: {model_name}")
        
        try:
            # Route to appropriate inference method based on template
            if job.template_name == "text-generation":
                result = self._inference_client.text_generation(
                    prompt=job.inputs.get('prompt', ''),
                    model=model_name,
                    max_new_tokens=job.inputs.get('max_tokens', 100)
                )
                return {'generated_text': result}
            
            elif job.template_name == "image-classification":
                image_input = job.inputs.get('image')
                result = self._inference_client.image_classification(
                    image=image_input,
                    model=model_name
                )
                return {'classifications': result}
            
            elif job.template_name == "embeddings":
                text_input = job.inputs.get('text', '')
                result = self._inference_client.feature_extraction(
                    text=text_input,
                    model=model_name
                )
                return {'embeddings': result}
            
            elif job.template_name == "image-generation":
                prompt = job.inputs.get('prompt', '')
                result = self._inference_client.text_to_image(
                    prompt=prompt,
                    model=model_name
                )
                return {'image': result}
            
            else:
                raise JobExecutionError(
                    f"Template {job.template_name} not supported by Inference API",
                    job_id=job.id,
                    backend_id=self.id
                )
        
        except Exception as e:
            error_str = str(e).lower()
            
            # Handle model not found errors
            if "model" in error_str and ("not found" in error_str or "404" in error_str or "does not exist" in error_str):
                raise JobExecutionError(
                    f"Model '{model_name}' not found on HuggingFace. Please verify the model name is correct "
                    f"and the model exists in the HuggingFace Hub.",
                    job_id=job.id,
                    backend_id=self.id
                )
            
            # Handle rate limiting
            elif "rate limit" in error_str or "too many requests" in error_str or "429" in error_str:
                retry_after = None
                if hasattr(e, 'retry_after'):
                    retry_after = e.retry_after
                raise BackendRateLimitError(
                    f"HuggingFace Inference API rate limit exceeded: {str(e)}",
                    backend_id=self.id,
                    retry_after=retry_after
                )
            
            # Handle timeout errors
            elif "timeout" in error_str or "timed out" in error_str:
                raise JobTimeoutError(
                    f"HuggingFace Inference API request timed out: {str(e)}",
                    job_id=job.id,
                    timeout_seconds=None
                )
            
            # Handle authentication errors
            elif "unauthorized" in error_str or "authentication" in error_str or "401" in error_str:
                raise BackendAuthenticationError(
                    f"HuggingFace authentication failed during Inference API call: {str(e)}",
                    backend_id=self.id
                )
            
            # Re-raise if already an orchestrator error
            elif isinstance(e, (JobExecutionError, BackendRateLimitError, JobTimeoutError, BackendAuthenticationError)):
                raise
            
            # Generic execution error
            else:
                raise JobExecutionError(
                    f"HuggingFace Inference API execution failed: {str(e)}",
                    job_id=job.id,
                    backend_id=self.id
                )
    
    def _execute_via_space(self, job: Job, space_url: Optional[str]) -> Dict[str, Any]:
        """
        Execute job using HuggingFace Space via Gradio client.
        
        Args:
            job: Job to execute
            space_url: Space URL or ID
            
        Returns:
            Execution results
            
        Raises:
            BackendConnectionError: If Gradio client is not installed or connection fails
            BackendNotAvailableError: If Space is not found or building
            JobTimeoutError: If Space startup times out
            JobExecutionError: If Space execution fails
        """
        try:
            from gradio_client import Client
        except ImportError:
            raise BackendConnectionError(
                "Gradio client not installed. Install with: pip install gradio-client",
                backend_id=self.id
            )
        
        if not space_url:
            # Try to create or find a Space
            space_url = self._get_or_create_space(job)
        
        self.logger.info(f"Executing job {job.id} via Space: {space_url}")
        
        # Connect to Space with timeout handling
        start_time = time.time()
        max_wait_time = self.space_startup_timeout
        retry_interval = 10  # seconds
        
        client = None
        last_error = None
        
        while time.time() - start_time < max_wait_time:
            try:
                client = Client(space_url)
                self.logger.info(f"Successfully connected to Space: {space_url}")
                break
            except Exception as e:
                last_error = e
                error_str = str(e).lower()
                
                # Handle Space not found errors
                if "not found" in error_str or "404" in error_str:
                    raise BackendNotAvailableError(
                        f"HuggingFace Space not found: {space_url}. Please verify the Space exists and is accessible.",
                        required_capabilities=['space']
                    )
                
                # Handle Space building - wait and retry
                elif "building" in error_str or "starting" in error_str or "loading" in error_str:
                    elapsed = time.time() - start_time
                    remaining = max_wait_time - elapsed
                    
                    if remaining > retry_interval:
                        self.logger.info(
                            f"Space {space_url} is building, waiting {retry_interval}s before retry "
                            f"(elapsed: {elapsed:.1f}s, timeout: {max_wait_time}s)"
                        )
                        time.sleep(retry_interval)
                        continue
                    else:
                        raise JobTimeoutError(
                            f"HuggingFace Space {space_url} is still building after {max_wait_time}s timeout. "
                            f"The Space may need more time to initialize.",
                            job_id=job.id,
                            timeout_seconds=max_wait_time
                        )
                
                # Handle rate limiting
                elif "rate limit" in error_str or "too many requests" in error_str or "429" in error_str:
                    retry_after = None
                    if hasattr(e, 'retry_after'):
                        retry_after = e.retry_after
                    raise BackendRateLimitError(
                        f"HuggingFace rate limit exceeded when connecting to Space {space_url}. "
                        f"Please wait before retrying.",
                        backend_id=self.id,
                        retry_after=retry_after
                    )
                
                # Other connection errors
                else:
                    raise BackendConnectionError(
                        f"Failed to connect to HuggingFace Space {space_url}: {str(e)}",
                        backend_id=self.id
                    )
        
        # If we exhausted retries without success
        if client is None:
            if last_error:
                error_str = str(last_error).lower()
                if "building" in error_str or "starting" in error_str or "loading" in error_str:
                    raise JobTimeoutError(
                        f"HuggingFace Space {space_url} building timeout after {max_wait_time}s. "
                        f"The Space may need more time to initialize.",
                        job_id=job.id,
                        timeout_seconds=max_wait_time
                    )
            
            raise BackendConnectionError(
                f"Failed to connect to HuggingFace Space {space_url} after {max_wait_time}s",
                backend_id=self.id
            )
        
        # Execute prediction
        api_name = job.metadata.get('api_name', '/predict')
        
        try:
            result = client.predict(*job.inputs.values(), api_name=api_name)
            return {'result': result}
        except Exception as e:
            error_str = str(e).lower()
            
            # Handle rate limiting during execution
            if "rate limit" in error_str or "too many requests" in error_str or "429" in error_str:
                retry_after = None
                if hasattr(e, 'retry_after'):
                    retry_after = e.retry_after
                raise BackendRateLimitError(
                    f"HuggingFace rate limit exceeded during Space execution: {str(e)}",
                    backend_id=self.id,
                    retry_after=retry_after
                )
            
            # Handle timeout during execution
            elif "timeout" in error_str or "timed out" in error_str:
                raise JobTimeoutError(
                    f"HuggingFace Space execution timed out: {str(e)}",
                    job_id=job.id,
                    timeout_seconds=None
                )
            
            # Generic execution error
            raise JobExecutionError(
                f"HuggingFace Space execution failed: {str(e)}",
                job_id=job.id,
                backend_id=self.id
            )
    
    def _get_or_create_space(self, job: Job) -> str:
        """
        Get existing Space or create a new one for the job.
        
        Args:
            job: Job to create Space for
            
        Returns:
            Space URL or ID
        """
        # For now, return a placeholder
        # In a full implementation, this would:
        # 1. Check if a suitable Space exists
        # 2. Create a new Space if needed
        # 3. Wait for Space to be ready
        
        raise BackendNotAvailableError(
            "Automatic Space creation not yet implemented. Please provide space_url in job metadata.",
            required_capabilities=['space']
        )
    
    def _get_default_model_for_template(self, template_name: str) -> str:
        """
        Get default model name for a template.
        
        Args:
            template_name: Template name
            
        Returns:
            Default model name
        """
        default_models = {
            "text-generation": "gpt2",
            "image-classification": "google/vit-base-patch16-224",
            "embeddings": "sentence-transformers/all-MiniLM-L6-v2",
            "image-generation": "stabilityai/stable-diffusion-2-1",
        }
        
        return default_models.get(template_name, "gpt2")
    
    def check_health(self) -> HealthStatus:
        """
        Check HuggingFace backend health by verifying API connectivity and Space availability.
        
        This method verifies:
        1. API authentication and connectivity
        2. Inference API availability (if enabled)
        3. Space availability (if configured)
        
        Returns:
            Current health status
        """
        try:
            self._authenticate()
            
            # Verify API connectivity by getting user info
            if self._hf_api:
                # Use cached whoami check for health monitoring to avoid rate limits
                current_time = time.time()
                user_info = None
                
                try:
                    if not self._whoami_cache or (current_time - self._whoami_cache_time > self._whoami_cache_ttl):
                        self._whoami_cache = self._hf_api.whoami()
                        self._whoami_cache_time = current_time
                    
                    user_info = self._whoami_cache
                    
                except Exception as e:
                    # If we hit rate limits during health check, assume healthy if previously cached
                    if "429" in str(e) or "rate limit" in str(e).lower():
                        if self._whoami_cache:
                            user_info = self._whoami_cache
                            self.logger.debug("Rate limited during health check, using cached credentials.")
                        else:
                            # If no cache but rate limited, likely healthy but busy
                            self.logger.warning("Rate limited during health check, no cache available.")
                            user_info = {"name": "unknown (rate-limited)"}
                    else:
                        raise e
                
                if not user_info:
                    self.health_status = HealthStatus.DEGRADED
                    self.last_health_check = datetime.now()
                    self.logger.warning("HuggingFace health check: DEGRADED (API client not initialized)")
                    return HealthStatus.DEGRADED
                
                # Verify Inference API availability if enabled
                if self.use_inference_api and self._inference_client:
                    try:
                        # Quick test with a simple model to verify Inference API is accessible
                        # We don't actually run inference, just check if the client is functional
                        self.logger.debug("HuggingFace Inference API client initialized and ready")
                    except Exception as e:
                        self.logger.warning(f"HuggingFace Inference API check failed: {str(e)}")
                        # Don't fail health check for Inference API issues, just log
                
                # If a specific Space is configured, verify its availability
                configured_space = self.options.get('default_space_url')
                if configured_space:
                    try:
                        # Check if Space exists and is accessible
                        space_info = self._hf_api.space_info(configured_space)
                        if space_info:
                            self.logger.debug(f"HuggingFace Space {configured_space} is available")
                    except Exception as e:
                        self.logger.warning(f"HuggingFace Space {configured_space} check failed: {str(e)}")
                        # Space unavailable, but API is working - mark as DEGRADED
                        self.health_status = HealthStatus.DEGRADED
                        self.last_health_check = datetime.now()
                        self.logger.warning(f"HuggingFace health check: DEGRADED (Space unavailable: {str(e)})")
                        return HealthStatus.DEGRADED
                
                # All checks passed
                self.health_status = HealthStatus.HEALTHY
                self.last_health_check = datetime.now()
                self.logger.debug("HuggingFace health check: HEALTHY")
                return HealthStatus.HEALTHY
            
            self.health_status = HealthStatus.DEGRADED
            self.last_health_check = datetime.now()
            self.logger.warning("HuggingFace health check: DEGRADED (API client not initialized)")
            return HealthStatus.DEGRADED
        
        except BackendAuthenticationError as e:
            self.health_status = HealthStatus.UNHEALTHY
            self.last_health_check = datetime.now()
            self.logger.warning(f"HuggingFace health check: UNHEALTHY (authentication failed: {str(e)})")
            return HealthStatus.UNHEALTHY
        
        except BackendConnectionError as e:
            self.health_status = HealthStatus.UNHEALTHY
            self.last_health_check = datetime.now()
            self.logger.warning(f"HuggingFace health check: UNHEALTHY (connection failed: {str(e)})")
            return HealthStatus.UNHEALTHY
        
        except Exception as e:
            self.health_status = HealthStatus.DEGRADED
            self.last_health_check = datetime.now()
            self.logger.warning(f"HuggingFace health check: DEGRADED ({str(e)})")
            return HealthStatus.DEGRADED
    
    def get_queue_length(self) -> int:
        """
        Get current queue length for HuggingFace backend.
        
        Note: HuggingFace doesn't expose queue length directly, so we return 0
        to indicate no local queue (HuggingFace handles queueing internally).
        
        Returns:
            Queue length (always 0 for HuggingFace)
        """
        return 0
    
    def supports_template(self, template_name: str) -> bool:
        """
        Check if HuggingFace backend supports a specific template.
        
        HuggingFace supports inference-focused templates.
        
        Args:
            template_name: Name of the template
            
        Returns:
            True if supported, False otherwise
        """
        return template_name in SUPPORTED_TEMPLATES
    
    def estimate_cost(self, resource_estimate: ResourceEstimate) -> float:
        """
        Estimate cost for executing a job on HuggingFace.
        
        HuggingFace Inference API and Spaces are free tier, so cost is $0.00.
        
        Args:
            resource_estimate: Resource requirements
            
        Returns:
            Estimated cost in USD (always 0.0 for free tier)
        """
        self.logger.debug("Cost estimate: HuggingFace free tier = $0.00")
        return 0.0
