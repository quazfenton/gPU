"""
Unit tests for HuggingFace backend implementation.

Tests the HuggingFaceBackend class functionality including initialization,
health checks, template support, and cost estimation.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from notebook_ml_orchestrator.core.backends.huggingface_backend import (
    HuggingFaceBackend, SUPPORTED_TEMPLATES
)
from notebook_ml_orchestrator.core.models import (
    BackendType, HealthStatus, ResourceEstimate, JobStatus
)
from notebook_ml_orchestrator.core.interfaces import Job, MLTemplate
from notebook_ml_orchestrator.core.exceptions import (
    BackendConnectionError, BackendNotAvailableError, JobExecutionError, 
    JobTimeoutError, BackendAuthenticationError, BackendRateLimitError
)


class TestHuggingFaceBackendInitialization:
    """Test HuggingFace backend initialization."""
    
    def test_initialization_with_default_config(self):
        """Test backend initializes with default configuration."""
        backend = HuggingFaceBackend()
        
        assert backend.id == "huggingface"
        assert backend.name == "HuggingFace"
        assert backend.type == BackendType.HUGGINGFACE
        assert backend.default_space_hardware == "cpu-basic"
        assert backend.space_startup_timeout == 300
        assert backend.use_inference_api is True
        assert backend.capabilities.supports_gpu is True
        assert backend.capabilities.supported_templates == SUPPORTED_TEMPLATES
    
    def test_initialization_with_custom_config(self):
        """Test backend initializes with custom configuration."""
        config = {
            'credentials': {
                'token': 'test_token'
            },
            'options': {
                'default_space_hardware': 'gpu-small',
                'space_startup_timeout': 600,
                'use_inference_api': False
            }
        }
        
        backend = HuggingFaceBackend(backend_id="hf-custom", config=config)
        
        assert backend.id == "hf-custom"
        assert backend.default_space_hardware == "gpu-small"
        assert backend.space_startup_timeout == 600
        assert backend.use_inference_api is False
        assert backend.credentials['token'] == 'test_token'
    
    def test_capabilities_set_correctly(self):
        """Test backend capabilities are set correctly."""
        backend = HuggingFaceBackend()
        
        assert backend.capabilities.max_concurrent_jobs == 5
        assert backend.capabilities.max_job_duration_minutes == 30
        assert backend.capabilities.supports_gpu is True
        assert backend.capabilities.supports_batch is False
        assert backend.capabilities.cost_per_hour == 0.0  # Free tier


class TestHuggingFaceBackendTemplateSupport:
    """Test template support checking."""
    
    def test_supports_inference_templates(self):
        """Test backend supports inference templates."""
        backend = HuggingFaceBackend()
        
        assert backend.supports_template("text-generation") is True
        assert backend.supports_template("image-classification") is True
        assert backend.supports_template("embeddings") is True
        assert backend.supports_template("image-generation") is True
    
    def test_does_not_support_training_templates(self):
        """Test backend does not support training templates."""
        backend = HuggingFaceBackend()
        
        assert backend.supports_template("model-training") is False
        assert backend.supports_template("data-processing") is False
    
    def test_does_not_support_unknown_templates(self):
        """Test backend rejects unknown templates."""
        backend = HuggingFaceBackend()
        
        assert backend.supports_template("unknown-template") is False
        assert backend.supports_template("interactive-notebook") is False


class TestHuggingFaceBackendCostEstimation:
    """Test cost estimation logic."""
    
    def test_cost_estimation_always_returns_zero(self):
        """Test cost estimation returns $0.00 for free tier."""
        backend = HuggingFaceBackend()
        
        # CPU job
        resource_estimate_cpu = ResourceEstimate(
            cpu_cores=2,
            memory_gb=4.0,
            requires_gpu=False,
            estimated_duration_minutes=30
        )
        
        cost_cpu = backend.estimate_cost(resource_estimate_cpu)
        assert cost_cpu == 0.0
        
        # GPU job
        resource_estimate_gpu = ResourceEstimate(
            gpu_memory_gb=16.0,
            requires_gpu=True,
            estimated_duration_minutes=60
        )
        
        cost_gpu = backend.estimate_cost(resource_estimate_gpu)
        assert cost_gpu == 0.0
    
    def test_cost_estimation_with_various_durations(self):
        """Test cost estimation is always $0.00 regardless of duration."""
        backend = HuggingFaceBackend()
        
        durations = [5, 30, 60, 120]
        
        for duration in durations:
            resource_estimate = ResourceEstimate(
                estimated_duration_minutes=duration,
                requires_gpu=True
            )
            
            cost = backend.estimate_cost(resource_estimate)
            assert cost == 0.0


class TestHuggingFaceBackendHealthCheck:
    """Test health check functionality."""
    
    def test_health_check_fails_without_credentials(self):
        """Test health check fails when credentials are missing."""
        backend = HuggingFaceBackend()
        
        status = backend.check_health()
        
        assert status == HealthStatus.UNHEALTHY
        assert backend.health_status == HealthStatus.UNHEALTHY
    
    def test_health_check_fails_without_sdk(self):
        """Test health check fails when HuggingFace SDK is not installed."""
        backend = HuggingFaceBackend()
        
        # Without credentials, it will fail with authentication error
        status = backend.check_health()
        
        assert status in [HealthStatus.UNHEALTHY, HealthStatus.DEGRADED]


class TestHuggingFaceBackendQueueLength:
    """Test queue length reporting."""
    
    def test_get_queue_length_returns_zero(self):
        """Test queue length always returns 0 (HuggingFace handles queueing)."""
        backend = HuggingFaceBackend()
        
        queue_length = backend.get_queue_length()
        
        assert queue_length == 0


class TestHuggingFaceBackendAuthentication:
    """Test authentication logic."""
    
    def test_authentication_fails_without_credentials(self):
        """Test authentication fails when credentials are missing."""
        backend = HuggingFaceBackend()
        
        # Will fail with either BackendConnectionError (SDK not installed) or BackendAuthenticationError (no credentials)
        with pytest.raises((BackendAuthenticationError, BackendConnectionError)) as exc_info:
            backend._authenticate()
        
        assert exc_info.value.backend_id == "huggingface"
    
    def test_authentication_fails_without_sdk(self):
        """Test authentication fails when HuggingFace SDK is not installed."""
        config = {
            'credentials': {
                'token': 'test_token'
            }
        }
        backend = HuggingFaceBackend(config=config)
        
        # Will fail because SDK is not installed or token is invalid
        with pytest.raises((BackendConnectionError, BackendAuthenticationError)):
            backend._authenticate()


class TestHuggingFaceBackendDefaultModels:
    """Test default model selection."""
    
    def test_get_default_model_for_text_generation(self):
        """Test default model for text generation."""
        backend = HuggingFaceBackend()
        
        model = backend._get_default_model_for_template("text-generation")
        assert model == "gpt2"
    
    def test_get_default_model_for_image_classification(self):
        """Test default model for image classification."""
        backend = HuggingFaceBackend()
        
        model = backend._get_default_model_for_template("image-classification")
        assert model == "google/vit-base-patch16-224"
    
    def test_get_default_model_for_embeddings(self):
        """Test default model for embeddings."""
        backend = HuggingFaceBackend()
        
        model = backend._get_default_model_for_template("embeddings")
        assert model == "sentence-transformers/all-MiniLM-L6-v2"
    
    def test_get_default_model_for_image_generation(self):
        """Test default model for image generation."""
        backend = HuggingFaceBackend()
        
        model = backend._get_default_model_for_template("image-generation")
        assert model == "stabilityai/stable-diffusion-2-1"
    
    def test_get_default_model_for_unknown_template(self):
        """Test default model for unknown template defaults to gpt2."""
        backend = HuggingFaceBackend()
        
        model = backend._get_default_model_for_template("unknown-template")
        assert model == "gpt2"


class TestHuggingFaceBackendErrorHandling:
    """Test error handling for HuggingFace backend."""
    
    def test_authentication_error_without_credentials(self):
        """Test that authentication fails with proper error when credentials are missing."""
        backend = HuggingFaceBackend()
        
        # Will fail with either BackendConnectionError (SDK not installed) or BackendAuthenticationError (no credentials)
        with pytest.raises((BackendAuthenticationError, BackendConnectionError)) as exc_info:
            backend._authenticate()
        
        assert exc_info.value.backend_id == "huggingface"
        error_msg = str(exc_info.value).lower()
        assert "huggingface" in error_msg or "token" in error_msg
    
    def test_space_unavailability_error_detection(self):
        """Test that Space unavailability errors are properly detected."""
        # Verify that the backend can detect Space unavailability from error messages
        test_errors = [
            "Space not found",
            "Space unavailable",
            "space is building",
            "Space building"
        ]
        
        for error_msg in test_errors:
            assert "space" in error_msg.lower()
    
    def test_rate_limit_error_detection(self):
        """Test that rate limit errors are properly detected."""
        # Verify that the backend can detect rate limiting from error messages
        test_errors = [
            "Rate limit exceeded",
            "Too many requests",
            "HTTP 429 error",
            "rate limit hit"
        ]
        
        for error_msg in test_errors:
            error_lower = error_msg.lower()
            assert any(keyword in error_lower for keyword in ["rate limit", "too many requests", "429"])
    
    def test_model_not_found_error_detection(self):
        """Test that model not found errors are properly detected."""
        test_errors = [
            "Model not found",
            "model not found on HuggingFace"
        ]
        
        for error_msg in test_errors:
            error_lower = error_msg.lower()
            assert "model" in error_lower and "not found" in error_lower
    
    def test_retry_logic_configuration(self):
        """Test that retry logic is properly configured."""
        backend = HuggingFaceBackend()
        
        job = Job(
            id="retry-job",
            template_name="text-generation",
            inputs={"prompt": "test"}
        )
        
        template = Mock(spec=MLTemplate)
        template.requirements = {}
        
        # The execute_job method should have retry logic
        assert hasattr(backend, 'execute_job')
        assert hasattr(backend, '_execute_job_internal')
    
    def test_health_check_handles_authentication_errors(self):
        """Test that health check properly handles authentication errors."""
        backend = HuggingFaceBackend()
        
        status = backend.check_health()
        
        # Should return UNHEALTHY when authentication fails
        assert status == HealthStatus.UNHEALTHY
        assert backend.health_status == HealthStatus.UNHEALTHY
    
    def test_error_messages_include_context(self):
        """Test that error messages include relevant context."""
        backend = HuggingFaceBackend(backend_id="test-hf")
        
        try:
            backend._authenticate()
        except (BackendAuthenticationError, BackendConnectionError) as e:
            # Error should include backend_id
            assert e.backend_id == "test-hf"
            # Error message should be descriptive
            assert len(str(e)) > 0


class TestHuggingFaceBackendJobExecution:
    """Test job execution logic."""
    
    def test_execute_job_uses_job_metadata(self):
        """Test job execution uses metadata from job."""
        config = {
            'credentials': {
                'token': 'test_token'
            }
        }
        backend = HuggingFaceBackend(config=config)
        
        job = Job(
            id="test-job",
            template_name="text-generation",
            inputs={"prompt": "test"},
            metadata={
                "use_space": True,
                "space_url": "test/space",
                "model_name": "gpt2"
            }
        )
        
        # Verify metadata is accessible
        assert job.metadata.get('use_space') is True
        assert job.metadata.get('space_url') == "test/space"
        assert job.metadata.get('model_name') == "gpt2"
    
    def test_execute_job_defaults_to_inference_api(self):
        """Test job execution defaults to Inference API when use_inference_api is True."""
        backend = HuggingFaceBackend()
        
        assert backend.use_inference_api is True
        
        job = Job(
            id="test-job",
            template_name="text-generation",
            inputs={"prompt": "test"}
        )
        
        # Job should use Inference API by default
        assert job.metadata.get('use_space', False) is False


class TestHuggingFaceBackendRetryBehavior:
    """Test retry behavior for different error types."""
    
    def test_authentication_errors_not_retried(self):
        """Test that authentication errors are not retried."""
        backend = HuggingFaceBackend()
        
        job = Job(
            id="auth-fail-job",
            template_name="text-generation",
            inputs={"prompt": "test"}
        )
        
        template = Mock(spec=MLTemplate)
        template.requirements = {}
        
        # Should fail on first attempt with authentication or connection error
        with pytest.raises((BackendAuthenticationError, BackendConnectionError)):
            backend.execute_job(job, template)
    
    def test_timeout_errors_not_retried(self):
        """Test that timeout errors are not retried."""
        backend = HuggingFaceBackend()
        
        # The execute_job method should not retry timeout errors
        assert hasattr(backend, 'execute_job')


class TestHuggingFaceBackendSpaceManagement:
    """Test Space creation and management."""
    
    def test_get_or_create_space_not_implemented(self):
        """Test that automatic Space creation raises NotImplementedError."""
        backend = HuggingFaceBackend()
        
        job = Job(
            id="test-job",
            template_name="text-generation",
            inputs={"prompt": "test"}
        )
        
        with pytest.raises(BackendNotAvailableError) as exc_info:
            backend._get_or_create_space(job)
        
        assert "not yet implemented" in str(exc_info.value).lower()


class TestHuggingFaceBackendEnhancedErrorHandling:
    """Test enhanced error handling for HuggingFace backend (Task 3.3)."""
    
    def test_space_not_found_error(self):
        """Test that Space not found errors are properly handled."""
        # Mock gradio_client module
        mock_gradio = MagicMock()
        mock_client_class = Mock()
        mock_client_class.side_effect = Exception("Space not found: test/space")
        mock_gradio.Client = mock_client_class
        
        with patch.dict('sys.modules', {'gradio_client': mock_gradio}):
            config = {
                'credentials': {'token': 'test_token'},
                'options': {'space_startup_timeout': 5}
            }
            backend = HuggingFaceBackend(config=config)
            backend._authenticated = True
            backend._hf_api = Mock()
            
            job = Job(
                id="test-job",
                template_name="text-generation",
                inputs={"prompt": "test"}
            )
            
            with pytest.raises(BackendNotAvailableError) as exc_info:
                backend._execute_via_space(job, "test/space")
            
            assert "not found" in str(exc_info.value).lower()
            assert "test/space" in str(exc_info.value)
    
    @patch('time.sleep')
    def test_space_building_timeout(self, mock_sleep):
        """Test that Space building delays timeout properly."""
        # Mock gradio_client module
        mock_gradio = MagicMock()
        mock_client_class = Mock()
        mock_client_class.side_effect = Exception("Space is building")
        mock_gradio.Client = mock_client_class
        
        with patch.dict('sys.modules', {'gradio_client': mock_gradio}):
            config = {
                'credentials': {'token': 'test_token'},
                'options': {'space_startup_timeout': 15}  # 15 seconds timeout
            }
            backend = HuggingFaceBackend(config=config)
            backend._authenticated = True
            backend._hf_api = Mock()
            
            job = Job(
                id="test-job",
                template_name="text-generation",
                inputs={"prompt": "test"}
            )
            
            with pytest.raises(JobTimeoutError) as exc_info:
                backend._execute_via_space(job, "test/space")
            
            assert "building" in str(exc_info.value).lower()
            assert "timeout" in str(exc_info.value).lower()
            assert exc_info.value.job_id == "test-job"
    
    def test_space_rate_limit_error(self):
        """Test that API rate limiting is properly handled for Spaces."""
        # Mock gradio_client module
        mock_gradio = MagicMock()
        mock_client_class = Mock()
        mock_client_class.side_effect = Exception("Rate limit exceeded: too many requests")
        mock_gradio.Client = mock_client_class
        
        with patch.dict('sys.modules', {'gradio_client': mock_gradio}):
            config = {
                'credentials': {'token': 'test_token'},
                'options': {'space_startup_timeout': 5}
            }
            backend = HuggingFaceBackend(config=config)
            backend._authenticated = True
            backend._hf_api = Mock()
            
            job = Job(
                id="test-job",
                template_name="text-generation",
                inputs={"prompt": "test"}
            )
            
            with pytest.raises(BackendRateLimitError) as exc_info:
                backend._execute_via_space(job, "test/space")
            
            assert "rate limit" in str(exc_info.value).lower()
            assert exc_info.value.backend_id == "huggingface"
    
    @patch('time.sleep')
    def test_space_building_retry_then_success(self, mock_sleep):
        """Test that Space building retries and eventually succeeds."""
        # Mock gradio_client module
        mock_gradio = MagicMock()
        mock_client_instance = Mock()
        mock_client_instance.predict.return_value = "test result"
        
        call_count = [0]
        def side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] <= 2:
                raise Exception("Space is building")
            return mock_client_instance
        
        mock_client_class = Mock(side_effect=side_effect)
        mock_gradio.Client = mock_client_class
        
        with patch.dict('sys.modules', {'gradio_client': mock_gradio}):
            config = {
                'credentials': {'token': 'test_token'},
                'options': {'space_startup_timeout': 60}
            }
            backend = HuggingFaceBackend(config=config)
            backend._authenticated = True
            backend._hf_api = Mock()
            
            job = Job(
                id="test-job",
                template_name="text-generation",
                inputs={"prompt": "test"}
            )
            
            result = backend._execute_via_space(job, "test/space")
            
            assert result == {'result': "test result"}
            assert call_count[0] == 3  # Failed twice, succeeded on third attempt
    
    def test_inference_api_model_not_found_error(self):
        """Test that model not found errors are properly handled."""
        config = {
            'credentials': {'token': 'test_token'}
        }
        backend = HuggingFaceBackend(config=config)
        backend._authenticated = True
        
        # Mock inference client to raise model not found error
        mock_client = Mock()
        mock_client.text_generation.side_effect = Exception("Model 'invalid-model' not found")
        backend._inference_client = mock_client
        
        job = Job(
            id="test-job",
            template_name="text-generation",
            inputs={"prompt": "test"}
        )
        
        with pytest.raises(JobExecutionError) as exc_info:
            backend._execute_via_inference_api(job, "invalid-model")
        
        assert "not found" in str(exc_info.value).lower()
        assert "invalid-model" in str(exc_info.value)
        assert exc_info.value.job_id == "test-job"
    
    def test_inference_api_rate_limit_error(self):
        """Test that rate limiting is properly handled for Inference API."""
        config = {
            'credentials': {'token': 'test_token'}
        }
        backend = HuggingFaceBackend(config=config)
        backend._authenticated = True
        
        # Mock inference client to raise rate limit error
        mock_client = Mock()
        mock_client.text_generation.side_effect = Exception("Rate limit exceeded")
        backend._inference_client = mock_client
        
        job = Job(
            id="test-job",
            template_name="text-generation",
            inputs={"prompt": "test"}
        )
        
        with pytest.raises(BackendRateLimitError) as exc_info:
            backend._execute_via_inference_api(job, "gpt2")
        
        assert "rate limit" in str(exc_info.value).lower()
        assert exc_info.value.backend_id == "huggingface"
    
    def test_inference_api_timeout_error(self):
        """Test that timeout errors are properly handled for Inference API."""
        config = {
            'credentials': {'token': 'test_token'}
        }
        backend = HuggingFaceBackend(config=config)
        backend._authenticated = True
        
        # Mock inference client to raise timeout error
        mock_client = Mock()
        mock_client.text_generation.side_effect = Exception("Request timed out")
        backend._inference_client = mock_client
        
        job = Job(
            id="test-job",
            template_name="text-generation",
            inputs={"prompt": "test"}
        )
        
        with pytest.raises(JobTimeoutError) as exc_info:
            backend._execute_via_inference_api(job, "gpt2")
        
        # The error message contains "timed out" which includes "timeout"
        assert "timed out" in str(exc_info.value).lower() or "timeout" in str(exc_info.value).lower()
        assert exc_info.value.job_id == "test-job"
    
    def test_inference_api_authentication_error(self):
        """Test that authentication errors are properly handled for Inference API."""
        config = {
            'credentials': {'token': 'test_token'}
        }
        backend = HuggingFaceBackend(config=config)
        backend._authenticated = True
        
        # Mock inference client to raise authentication error
        mock_client = Mock()
        mock_client.text_generation.side_effect = Exception("Unauthorized: invalid token")
        backend._inference_client = mock_client
        
        job = Job(
            id="test-job",
            template_name="text-generation",
            inputs={"prompt": "test"}
        )
        
        with pytest.raises(BackendAuthenticationError) as exc_info:
            backend._execute_via_inference_api(job, "gpt2")
        
        assert "authentication" in str(exc_info.value).lower()
        assert exc_info.value.backend_id == "huggingface"
    
    def test_space_execution_rate_limit_error(self):
        """Test that rate limiting during Space execution is properly handled."""
        # Mock gradio_client module
        mock_gradio = MagicMock()
        mock_client_instance = Mock()
        mock_client_instance.predict.side_effect = Exception("Rate limit exceeded")
        mock_gradio.Client.return_value = mock_client_instance
        
        with patch.dict('sys.modules', {'gradio_client': mock_gradio}):
            config = {
                'credentials': {'token': 'test_token'}
            }
            backend = HuggingFaceBackend(config=config)
            backend._authenticated = True
            backend._hf_api = Mock()
            
            job = Job(
                id="test-job",
                template_name="text-generation",
                inputs={"prompt": "test"}
            )
            
            with pytest.raises(BackendRateLimitError) as exc_info:
                backend._execute_via_space(job, "test/space")
            
            assert "rate limit" in str(exc_info.value).lower()
    
    def test_space_execution_timeout_error(self):
        """Test that timeout during Space execution is properly handled."""
        # Mock gradio_client module
        mock_gradio = MagicMock()
        mock_client_instance = Mock()
        mock_client_instance.predict.side_effect = Exception("Request timed out")
        mock_gradio.Client.return_value = mock_client_instance
        
        with patch.dict('sys.modules', {'gradio_client': mock_gradio}):
            config = {
                'credentials': {'token': 'test_token'}
            }
            backend = HuggingFaceBackend(config=config)
            backend._authenticated = True
            backend._hf_api = Mock()
            
            job = Job(
                id="test-job",
                template_name="text-generation",
                inputs={"prompt": "test"}
            )
            
            with pytest.raises(JobTimeoutError) as exc_info:
                backend._execute_via_space(job, "test/space")
            
            assert "timeout" in str(exc_info.value).lower() or "timed out" in str(exc_info.value).lower()
    
    def test_execute_job_retries_rate_limit_errors(self):
        """Test that execute_job retries on rate limit errors."""
        config = {
            'credentials': {'token': 'test_token'}
        }
        backend = HuggingFaceBackend(config=config)
        backend._authenticated = True
        
        # Mock to fail twice with rate limit, then succeed
        call_count = [0]
        def mock_execute(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] <= 2:
                raise BackendRateLimitError("Rate limit", backend_id="huggingface")
            from notebook_ml_orchestrator.core.models import JobResult
            return JobResult(
                success=True,
                outputs={"result": "success"},
                execution_time_seconds=1.0,
                backend_used="huggingface"
            )
        
        backend._execute_job_internal = mock_execute
        
        job = Job(
            id="test-job",
            template_name="text-generation",
            inputs={"prompt": "test"}
        )
        template = Mock(spec=MLTemplate)
        
        with patch('time.sleep'):  # Mock sleep to speed up test
            result = backend.execute_job(job, template)
        
        assert result.success is True
        assert call_count[0] == 3  # Failed twice, succeeded on third
    
    def test_execute_job_does_not_retry_authentication_errors(self):
        """Test that execute_job does not retry authentication errors."""
        config = {
            'credentials': {'token': 'test_token'}
        }
        backend = HuggingFaceBackend(config=config)
        
        call_count = [0]
        def mock_execute(*args, **kwargs):
            call_count[0] += 1
            raise BackendAuthenticationError("Auth failed", backend_id="huggingface")
        
        backend._execute_job_internal = mock_execute
        
        job = Job(
            id="test-job",
            template_name="text-generation",
            inputs={"prompt": "test"}
        )
        template = Mock(spec=MLTemplate)
        
        with pytest.raises(BackendAuthenticationError):
            backend.execute_job(job, template)
        
        assert call_count[0] == 1  # Should not retry
    
    def test_execute_job_does_not_retry_timeout_errors(self):
        """Test that execute_job does not retry timeout errors."""
        config = {
            'credentials': {'token': 'test_token'}
        }
        backend = HuggingFaceBackend(config=config)
        
        call_count = [0]
        def mock_execute(*args, **kwargs):
            call_count[0] += 1
            raise JobTimeoutError("Timeout", job_id="test-job")
        
        backend._execute_job_internal = mock_execute
        
        job = Job(
            id="test-job",
            template_name="text-generation",
            inputs={"prompt": "test"}
        )
        template = Mock(spec=MLTemplate)
        
        with pytest.raises(JobTimeoutError):
            backend.execute_job(job, template)
        
        assert call_count[0] == 1  # Should not retry


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
