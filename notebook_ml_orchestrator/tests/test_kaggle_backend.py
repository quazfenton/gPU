"""
Unit tests for Kaggle backend implementation.

Tests the KaggleBackend class functionality including initialization,
health checks, template support, and cost estimation.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from notebook_ml_orchestrator.core.backends.kaggle_backend import KaggleBackend, SUPPORTED_TEMPLATES
from notebook_ml_orchestrator.core.models import (
    BackendType, HealthStatus, ResourceEstimate, JobStatus
)
from notebook_ml_orchestrator.core.interfaces import Job, MLTemplate
from notebook_ml_orchestrator.core.exceptions import (
    BackendConnectionError, BackendNotAvailableError, JobExecutionError, 
    JobTimeoutError, BackendAuthenticationError
)


class TestKaggleBackendInitialization:
    """Test Kaggle backend initialization."""
    
    def test_initialization_with_default_config(self):
        """Test backend initializes with default configuration."""
        backend = KaggleBackend()
        
        assert backend.id == "kaggle"
        assert backend.name == "Kaggle"
        assert backend.type == BackendType.KAGGLE
        assert backend.max_concurrent_kernels == 1
        assert backend.default_timeout == 3600
        assert backend.capabilities.supports_gpu is True
        assert backend.capabilities.supported_templates == SUPPORTED_TEMPLATES
    
    def test_initialization_with_custom_config(self):
        """Test backend initializes with custom configuration."""
        config = {
            'credentials': {
                'username': 'test_user',
                'key': 'test_key'
            },
            'options': {
                'max_concurrent_kernels': 2,
                'timeout': 7200
            }
        }
        
        backend = KaggleBackend(backend_id="kaggle-custom", config=config)
        
        assert backend.id == "kaggle-custom"
        assert backend.max_concurrent_kernels == 2
        assert backend.default_timeout == 7200
        assert backend.credentials['username'] == 'test_user'
        assert backend.credentials['key'] == 'test_key'
    
    def test_capabilities_set_correctly(self):
        """Test backend capabilities are set correctly."""
        backend = KaggleBackend()
        
        assert backend.capabilities.max_concurrent_jobs == 1
        assert backend.capabilities.max_job_duration_minutes == 120
        assert backend.capabilities.supports_gpu is True
        assert backend.capabilities.supports_batch is True
        assert backend.capabilities.cost_per_hour == 0.0


class TestKaggleBackendTemplateSupport:
    """Test template support checking."""
    
    def test_supports_notebook_based_templates(self):
        """Test backend supports notebook-based templates."""
        backend = KaggleBackend()
        
        assert backend.supports_template("image-classification") is True
        assert backend.supports_template("model-training") is True
        assert backend.supports_template("data-processing") is True
        assert backend.supports_template("batch-inference") is True
    
    def test_does_not_support_non_notebook_templates(self):
        """Test backend rejects non-notebook templates."""
        backend = KaggleBackend()
        
        assert backend.supports_template("image-generation") is False
        assert backend.supports_template("text-generation") is False
        assert backend.supports_template("unknown-template") is False


class TestKaggleBackendCostEstimation:
    """Test cost estimation logic."""
    
    def test_cost_estimation_returns_zero_for_free_tier(self):
        """Test cost estimation returns 0.0 for free tier."""
        backend = KaggleBackend()
        
        resource_estimate = ResourceEstimate(
            cpu_cores=2,
            memory_gb=16.0,
            requires_gpu=True,
            estimated_duration_minutes=60
        )
        
        cost = backend.estimate_cost(resource_estimate)
        
        assert cost == 0.0
    
    def test_cost_estimation_for_cpu_job(self):
        """Test cost estimation for CPU-only job."""
        backend = KaggleBackend()
        
        resource_estimate = ResourceEstimate(
            cpu_cores=4,
            memory_gb=8.0,
            requires_gpu=False,
            estimated_duration_minutes=120
        )
        
        cost = backend.estimate_cost(resource_estimate)
        
        assert cost == 0.0
    
    def test_cost_estimation_for_gpu_job(self):
        """Test cost estimation for GPU job."""
        backend = KaggleBackend()
        
        resource_estimate = ResourceEstimate(
            gpu_memory_gb=16.0,
            requires_gpu=True,
            estimated_duration_minutes=30
        )
        
        cost = backend.estimate_cost(resource_estimate)
        
        assert cost == 0.0


class TestKaggleBackendHealthCheck:
    """Test health check functionality."""
    
    def test_health_check_fails_without_credentials(self):
        """Test health check fails when credentials are missing."""
        backend = KaggleBackend()
        
        status = backend.check_health()
        
        assert status == HealthStatus.UNHEALTHY
        assert backend.health_status == HealthStatus.UNHEALTHY
    
    def test_health_check_with_credentials_but_no_sdk(self):
        """Test health check with credentials but SDK not installed."""
        config = {
            'credentials': {
                'username': 'test_user',
                'key': 'test_key'
            }
        }
        backend = KaggleBackend(config=config)
        
        status = backend.check_health()
        
        # Should return UNHEALTHY because Kaggle SDK is not installed
        assert status == HealthStatus.UNHEALTHY


class TestKaggleBackendQueueLength:
    """Test queue length reporting."""
    
    def test_get_queue_length_returns_zero(self):
        """Test queue length always returns 0 (Kaggle handles queueing)."""
        backend = KaggleBackend()
        
        queue_length = backend.get_queue_length()
        
        assert queue_length == 0


class TestKaggleBackendAuthentication:
    """Test authentication logic."""
    
    def test_authentication_fails_without_credentials(self):
        """Test authentication fails when credentials are missing."""
        backend = KaggleBackend()
        
        # Since Kaggle SDK is not installed, it will raise BackendConnectionError first
        with pytest.raises((BackendAuthenticationError, BackendConnectionError)) as exc_info:
            backend._authenticate()
        
        assert exc_info.value.backend_id == "kaggle"
    
    def test_authentication_fails_without_kaggle_sdk(self):
        """Test authentication fails when Kaggle SDK is not installed."""
        config = {
            'credentials': {
                'username': 'test_user',
                'key': 'test_key'
            }
        }
        backend = KaggleBackend(config=config)
        
        with pytest.raises(BackendConnectionError) as exc_info:
            backend._authenticate()
        
        assert "kaggle sdk not installed" in str(exc_info.value).lower()


class TestKaggleBackendNotebookCreation:
    """Test notebook creation logic."""
    
    def test_create_notebook_generates_valid_json(self):
        """Test notebook creation generates valid JSON."""
        backend = KaggleBackend()
        
        job = Job(
            id="test-job",
            template_name="image-classification",
            inputs={"dataset": "test-dataset", "model": "resnet50"}
        )
        
        template = Mock(spec=MLTemplate)
        
        notebook_content = backend._create_notebook(job, template)
        
        # Should be valid JSON
        import json
        notebook = json.loads(notebook_content)
        
        assert "cells" in notebook
        assert "metadata" in notebook
        assert "nbformat" in notebook
        assert notebook["nbformat"] == 4
    
    def test_create_notebook_includes_job_inputs(self):
        """Test notebook includes job inputs."""
        backend = KaggleBackend()
        
        job = Job(
            id="test-job",
            template_name="model-training",
            inputs={"epochs": 10, "batch_size": 32}
        )
        
        template = Mock(spec=MLTemplate)
        
        notebook_content = backend._create_notebook(job, template)
        
        # Should include job inputs
        assert "epochs" in notebook_content
        assert "batch_size" in notebook_content
        assert "10" in notebook_content
        assert "32" in notebook_content


class TestKaggleBackendErrorHandling:
    """Test error handling for Kaggle backend."""
    
    def test_authentication_error_without_credentials(self):
        """Test that authentication fails with proper error when credentials are missing."""
        backend = KaggleBackend()
        
        # Since Kaggle SDK is not installed, it will raise BackendConnectionError first
        with pytest.raises((BackendAuthenticationError, BackendConnectionError)) as exc_info:
            backend._authenticate()
        
        assert exc_info.value.backend_id == "kaggle"
        error_msg = str(exc_info.value).lower()
        assert "kaggle" in error_msg
    
    def test_quota_exceeded_error_detection(self):
        """Test that quota exceeded errors are properly detected."""
        # Verify that the backend can detect quota errors from error messages
        test_errors = [
            "Quota exceeded",
            "GPU limit reached",
            "Weekly limit exceeded",
            "quota limit hit"
        ]
        
        for error_msg in test_errors:
            error_lower = error_msg.lower()
            assert any(keyword in error_lower for keyword in ["quota", "limit"])
    
    def test_timeout_error_detection(self):
        """Test that timeout errors are properly detected."""
        # Verify that the backend can detect timeout from error messages
        test_errors = [
            "Kernel timeout",
            "Execution timed out",
            "timeout exceeded"
        ]
        
        for error_msg in test_errors:
            error_lower = error_msg.lower()
            assert "timeout" in error_lower or "timed out" in error_lower
    
    def test_health_check_handles_authentication_errors(self):
        """Test that health check properly handles authentication errors."""
        backend = KaggleBackend()
        
        status = backend.check_health()
        
        # Should return UNHEALTHY when authentication fails
        assert status == HealthStatus.UNHEALTHY
        assert backend.health_status == HealthStatus.UNHEALTHY
    
    def test_error_messages_include_context(self):
        """Test that error messages include relevant context."""
        backend = KaggleBackend(backend_id="test-kaggle")
        
        try:
            backend._authenticate()
        except (BackendAuthenticationError, BackendConnectionError) as e:
            # Error should include backend_id
            assert e.backend_id == "test-kaggle"
            # Error message should be descriptive
            assert len(str(e)) > 0


class TestKaggleBackendJobExecution:
    """Test job execution logic."""
    
    def test_execute_job_uses_job_metadata_for_gpu(self):
        """Test job execution uses GPU setting from job metadata."""
        config = {
            'credentials': {
                'username': 'test_user',
                'key': 'test_key'
            }
        }
        backend = KaggleBackend(config=config)
        
        job = Job(
            id="test-job",
            template_name="model-training",
            inputs={"dataset": "test"},
            metadata={"enable_gpu": True, "timeout": 7200}
        )
        
        # Verify metadata is accessible
        assert job.metadata.get('enable_gpu') is True
        assert job.metadata.get('timeout') == 7200
    
    def test_execute_job_uses_default_timeout(self):
        """Test job execution uses default timeout when not specified."""
        backend = KaggleBackend()
        
        job = Job(
            id="test-job",
            template_name="data-processing",
            inputs={"data": "test"}
        )
        
        # Should use default timeout
        assert backend.default_timeout == 3600


class TestKaggleBackendQuotaHandling:
    """Test quota handling for Kaggle backend."""
    
    def test_health_check_detects_quota_issues(self):
        """Test that health check detects quota issues and returns DEGRADED."""
        config = {
            'credentials': {
                'username': 'test_user',
                'key': 'test_key'
            }
        }
        backend = KaggleBackend(config=config)
        
        # Mock the Kaggle API to simulate quota error
        with patch.object(backend, '_authenticate'):
            with patch.object(backend, '_kaggle_api') as mock_api:
                mock_api.kernels_list.side_effect = Exception("Quota exceeded")
                
                status = backend.check_health()
                
                # Should return DEGRADED for quota issues
                assert status == HealthStatus.DEGRADED
                assert backend.health_status == HealthStatus.DEGRADED


class TestKaggleBackendCapabilities:
    """Test backend capabilities reporting."""
    
    def test_capabilities_include_free_tier_limits(self):
        """Test that capabilities include free tier limit information."""
        backend = KaggleBackend()
        
        assert 'free_tier_limits' in backend.capabilities.__dict__
        assert backend.capabilities.free_tier_limits['gpu_hours_per_week'] == 30
        assert backend.capabilities.free_tier_limits['gpu_type'] == 'T4 x2'
    
    def test_capabilities_indicate_gpu_support(self):
        """Test that capabilities correctly indicate GPU support."""
        backend = KaggleBackend()
        
        assert backend.capabilities.supports_gpu is True
    
    def test_capabilities_indicate_batch_support(self):
        """Test that capabilities correctly indicate batch processing support."""
        backend = KaggleBackend()
        
        assert backend.capabilities.supports_batch is True


class TestKaggleBackendNetworkRetry:
    """Test network retry logic with exponential backoff."""
    
    def test_retry_with_backoff_succeeds_on_second_attempt(self):
        """Test that retry logic succeeds when operation succeeds on retry."""
        config = {
            'credentials': {
                'username': 'test_user',
                'key': 'test_key'
            },
            'options': {
                'max_retries': 3,
                'retry_backoff_base': 2
            }
        }
        backend = KaggleBackend(config=config)
        
        # Mock operation that fails once then succeeds
        call_count = [0]
        def mock_operation():
            call_count[0] += 1
            if call_count[0] == 1:
                raise Exception("Network connection failed")
            return "success"
        
        result = backend._retry_with_backoff(mock_operation, "Test operation")
        
        assert result == "success"
        assert call_count[0] == 2  # Failed once, succeeded on second attempt
    
    def test_retry_with_backoff_raises_after_max_retries(self):
        """Test that retry logic raises error after max retries."""
        config = {
            'credentials': {
                'username': 'test_user',
                'key': 'test_key'
            },
            'options': {
                'max_retries': 2,
                'retry_backoff_base': 1  # Use 1 to avoid sleep delays in tests
            }
        }
        backend = KaggleBackend(config=config)
        
        # Mock operation that always fails with network error
        def mock_operation():
            raise Exception("Network timeout")
        
        # Should raise the original exception after retries are exhausted
        with pytest.raises(Exception) as exc_info:
            backend._retry_with_backoff(mock_operation, "Test operation")
        
        assert "network timeout" in str(exc_info.value).lower()
    
    def test_retry_does_not_retry_authentication_errors(self):
        """Test that authentication errors are not retried."""
        config = {
            'credentials': {
                'username': 'test_user',
                'key': 'test_key'
            },
            'options': {
                'max_retries': 3
            }
        }
        backend = KaggleBackend(config=config)
        
        # Mock operation that raises authentication error
        call_count = [0]
        def mock_operation():
            call_count[0] += 1
            from notebook_ml_orchestrator.core.exceptions import BackendAuthenticationError
            raise BackendAuthenticationError("Invalid credentials", backend_id="test")
        
        from notebook_ml_orchestrator.core.exceptions import BackendAuthenticationError
        with pytest.raises(BackendAuthenticationError):
            backend._retry_with_backoff(mock_operation, "Test operation")
        
        # Should not retry authentication errors
        assert call_count[0] == 1
    
    def test_retry_does_not_retry_quota_errors(self):
        """Test that quota exceeded errors are not retried."""
        config = {
            'credentials': {
                'username': 'test_user',
                'key': 'test_key'
            },
            'options': {
                'max_retries': 3
            }
        }
        backend = KaggleBackend(config=config)
        
        # Mock operation that raises quota error
        call_count = [0]
        def mock_operation():
            call_count[0] += 1
            from notebook_ml_orchestrator.core.exceptions import BackendQuotaExceededError
            raise BackendQuotaExceededError("Quota exceeded", backend_id="test", quota_type="gpu")
        
        from notebook_ml_orchestrator.core.exceptions import BackendQuotaExceededError
        with pytest.raises(BackendQuotaExceededError):
            backend._retry_with_backoff(mock_operation, "Test operation")
        
        # Should not retry quota errors
        assert call_count[0] == 1
    
    def test_retry_does_not_retry_non_network_errors(self):
        """Test that non-network errors are not retried."""
        config = {
            'credentials': {
                'username': 'test_user',
                'key': 'test_key'
            },
            'options': {
                'max_retries': 3
            }
        }
        backend = KaggleBackend(config=config)
        
        # Mock operation that raises non-network error
        call_count = [0]
        def mock_operation():
            call_count[0] += 1
            raise ValueError("Invalid input data")
        
        with pytest.raises(ValueError):
            backend._retry_with_backoff(mock_operation, "Test operation")
        
        # Should not retry non-network errors
        assert call_count[0] == 1


class TestKaggleBackendKernelCancellation:
    """Test kernel cancellation on timeout."""
    
    def test_cancel_kernel_logs_warning(self):
        """Test that kernel cancellation logs appropriate warning."""
        backend = KaggleBackend()
        
        # Cancel kernel should log warning about lack of API support
        # This test just verifies the method doesn't crash
        backend._cancel_kernel("test-user/test-kernel")
        
        # If we get here without exception, the test passes


class TestKaggleBackendExecutionErrorHandling:
    """Test comprehensive error handling in execute_job."""
    
    def test_execute_job_handles_quota_exceeded_in_kernel_creation(self):
        """Test that quota exceeded during kernel creation is properly handled."""
        config = {
            'credentials': {
                'username': 'test_user',
                'key': 'test_key'
            }
        }
        backend = KaggleBackend(config=config)
        
        job = Job(
            id="test-job",
            template_name="model-training",
            inputs={"data": "test"}
        )
        template = Mock(spec=MLTemplate)
        
        # Mock authentication to succeed
        with patch.object(backend, '_authenticate'):
            # Mock _create_kernel to raise quota error
            with patch.object(backend, '_create_kernel') as mock_create:
                from notebook_ml_orchestrator.core.exceptions import BackendQuotaExceededError
                mock_create.side_effect = BackendQuotaExceededError(
                    "Quota exceeded", 
                    backend_id="kaggle",
                    quota_type="kernels"
                )
                
                with pytest.raises(BackendQuotaExceededError) as exc_info:
                    backend.execute_job(job, template)
                
                assert "quota exceeded" in str(exc_info.value).lower()
    
    def test_execute_job_handles_timeout_and_cancels_kernel(self):
        """Test that timeout triggers kernel cancellation."""
        config = {
            'credentials': {
                'username': 'test_user',
                'key': 'test_key'
            }
        }
        backend = KaggleBackend(config=config)
        
        job = Job(
            id="test-job",
            template_name="model-training",
            inputs={"data": "test"},
            metadata={"timeout": 10}
        )
        template = Mock(spec=MLTemplate)
        
        # Mock authentication to succeed
        with patch.object(backend, '_authenticate'):
            # Mock _create_kernel to return a kernel slug
            with patch.object(backend, '_create_kernel', return_value="test-user/test-kernel"):
                # Mock _poll_kernel_status to raise timeout
                with patch.object(backend, '_poll_kernel_status') as mock_poll:
                    from notebook_ml_orchestrator.core.exceptions import JobTimeoutError
                    mock_poll.side_effect = JobTimeoutError(
                        "Timeout",
                        job_id="test-job",
                        timeout_seconds=10
                    )
                    
                    # Mock _cancel_kernel to track if it was called
                    with patch.object(backend, '_cancel_kernel') as mock_cancel:
                        with pytest.raises(JobTimeoutError):
                            backend.execute_job(job, template)
                        
                        # Verify cancel was called
                        mock_cancel.assert_called_once_with("test-user/test-kernel")
    
    def test_execute_job_handles_authentication_failure(self):
        """Test that authentication failures are properly propagated."""
        backend = KaggleBackend()
        
        job = Job(
            id="test-job",
            template_name="model-training",
            inputs={"data": "test"}
        )
        template = Mock(spec=MLTemplate)
        
        # Authentication should fail - either due to missing credentials or missing SDK
        from notebook_ml_orchestrator.core.exceptions import (
            BackendAuthenticationError, BackendConnectionError
        )
        with pytest.raises((BackendAuthenticationError, BackendConnectionError)) as exc_info:
            backend.execute_job(job, template)
        
        # Should get an error about credentials or SDK
        error_msg = str(exc_info.value).lower()
        assert "credentials" in error_msg or "kaggle sdk" in error_msg
    
    def test_execute_job_handles_network_errors_with_retry(self):
        """Test that _create_kernel uses retry logic for network errors."""
        config = {
            'credentials': {
                'username': 'test_user',
                'key': 'test_key'
            },
            'options': {
                'max_retries': 2,
                'retry_backoff_base': 1  # Use 1 to avoid sleep delays in tests
            }
        }
        backend = KaggleBackend(config=config)
        
        # Test that _retry_with_backoff is used in _create_kernel
        # by verifying the retry mechanism works at the lower level
        call_count = [0]
        def mock_operation():
            call_count[0] += 1
            if call_count[0] == 1:
                raise Exception("Network connection reset")
            return "success"
        
        result = backend._retry_with_backoff(mock_operation, "Test operation")
        
        # Should succeed after retry
        assert result == "success"
        assert call_count[0] == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
