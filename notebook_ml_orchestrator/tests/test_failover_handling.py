"""
Unit tests for enhanced failover handling in MultiBackendRouter.

Tests the enhanced handle_backend_failure method with:
- Alternative backend routing logic
- Job state preservation
- Comprehensive failover logging

Feature: enhanced-backend-support
Task: 8.1 Add failover handling to MultiBackendRouter
Requirements: 8.1, 8.6, 8.7
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime

from notebook_ml_orchestrator.core.backend_router import MultiBackendRouter
from notebook_ml_orchestrator.core.models import BackendType, HealthStatus, ResourceEstimate
from notebook_ml_orchestrator.core.interfaces import Backend, Job, JobStatus
from notebook_ml_orchestrator.core.exceptions import BackendNotAvailableError


class TestFailoverHandling:
    """Test enhanced failover handling functionality."""
    
    def test_handle_backend_failure_marks_backend_unhealthy(self):
        """Test that failed backend is marked as unhealthy after repeated failures."""
        router = MultiBackendRouter()
        
        # Create backends
        failed_backend = Mock(spec=Backend, id="failed", type=BackendType.MODAL)
        failed_backend.supports_template.return_value = True
        failed_backend.capabilities = Mock(supports_gpu=True, max_job_duration_minutes=60)
        failed_backend.check_health.return_value = HealthStatus.HEALTHY
        failed_backend.get_queue_length.return_value = 0
        
        alternative_backend = Mock(spec=Backend, id="alternative", type=BackendType.KAGGLE)
        alternative_backend.supports_template.return_value = True
        alternative_backend.capabilities = Mock(supports_gpu=True, max_job_duration_minutes=60)
        alternative_backend.check_health.return_value = HealthStatus.HEALTHY
        alternative_backend.get_queue_length.return_value = 0
        
        router.register_backend(failed_backend)
        router.register_backend(alternative_backend)
        
        # Mark both as healthy initially
        router.health_monitor.update_health_status("failed", HealthStatus.HEALTHY)
        router.health_monitor.update_health_status("alternative", HealthStatus.HEALTHY)
        
        job1 = Job(id="job1", template_name="test-template")
        job2 = Job(id="job2", template_name="test-template")
        job3 = Job(id="job3", template_name="test-template")
        
        # Handle first failure - should not mark as unhealthy yet
        result = router.handle_backend_failure("failed", job1)
        assert router.health_monitor.get_job_failure_count("failed") == 1
        # Backend should still be considered healthy after 1 failure
        
        # Handle second failure
        result = router.handle_backend_failure("failed", job2)
        assert router.health_monitor.get_job_failure_count("failed") == 2
        
        # Handle third failure - should now mark as unhealthy (Requirement 8.5)
        result = router.handle_backend_failure("failed", job3)
        assert router.health_monitor.get_job_failure_count("failed") == 3
        
        # Verify failed backend is marked unhealthy after 3 consecutive failures
        assert not router.health_monitor.is_backend_healthy("failed")
        assert result is not None
    
    def test_handle_backend_failure_routes_to_alternative(self):
        """Test that job is routed to alternative backend on failure (Requirement 8.1)."""
        router = MultiBackendRouter()
        
        # Create backends
        failed_backend = Mock(spec=Backend, id="failed", type=BackendType.MODAL)
        failed_backend.supports_template.return_value = True
        failed_backend.capabilities = Mock(supports_gpu=True, max_job_duration_minutes=60)
        
        alternative_backend = Mock(spec=Backend, id="alternative", type=BackendType.KAGGLE)
        alternative_backend.supports_template.return_value = True
        alternative_backend.capabilities = Mock(supports_gpu=True, max_job_duration_minutes=60)
        
        router.register_backend(failed_backend)
        router.register_backend(alternative_backend)
        
        # Mark both as healthy initially
        router.health_monitor.update_health_status("failed", HealthStatus.HEALTHY)
        router.health_monitor.update_health_status("alternative", HealthStatus.HEALTHY)
        
        job = Job(id="job1", template_name="test-template")
        
        # Handle failure
        result = router.handle_backend_failure("failed", job)
        
        # Verify alternative backend is returned
        assert result is not None
        assert result.id == "alternative"
    
    def test_handle_backend_failure_preserves_job_state(self):
        """Test that job state is preserved during failover (Requirement 8.6)."""
        router = MultiBackendRouter()
        
        # Create backends
        failed_backend = Mock(spec=Backend, id="failed", type=BackendType.MODAL)
        failed_backend.supports_template.return_value = True
        failed_backend.capabilities = Mock(supports_gpu=True, max_job_duration_minutes=60)
        
        alternative_backend = Mock(spec=Backend, id="alternative", type=BackendType.KAGGLE)
        alternative_backend.supports_template.return_value = True
        alternative_backend.capabilities = Mock(supports_gpu=True, max_job_duration_minutes=60)
        
        router.register_backend(failed_backend)
        router.register_backend(alternative_backend)
        
        # Mark both as healthy initially
        router.health_monitor.update_health_status("failed", HealthStatus.HEALTHY)
        router.health_monitor.update_health_status("alternative", HealthStatus.HEALTHY)
        
        # Create job with specific state
        job = Job(
            id="job1",
            template_name="test-template",
            inputs={"param1": "value1", "param2": "value2"},
            priority=5,
            retry_count=1,
            metadata={"custom_key": "custom_value"}
        )
        
        # Handle failure
        result = router.handle_backend_failure("failed", job)
        
        # Verify job state is preserved in metadata
        assert 'failover_history' in job.metadata
        assert len(job.metadata['failover_history']) == 1
        
        failover_record = job.metadata['failover_history'][0]
        assert failover_record['failed_backend_id'] == "failed"
        assert failover_record['retry_count'] == 1
        assert 'failure_timestamp' in failover_record
        
        # Verify last_failover metadata is set
        assert 'last_failover' in job.metadata
        assert job.metadata['last_failover']['from_backend'] == "failed"
        assert job.metadata['last_failover']['to_backend'] == "alternative"
        assert 'timestamp' in job.metadata['last_failover']
    
    def test_handle_backend_failure_preserves_original_inputs(self):
        """Test that original job inputs are preserved (Requirement 8.6)."""
        router = MultiBackendRouter()
        
        # Create backends
        failed_backend = Mock(spec=Backend, id="failed", type=BackendType.MODAL)
        failed_backend.supports_template.return_value = True
        failed_backend.capabilities = Mock(supports_gpu=True, max_job_duration_minutes=60)
        
        alternative_backend = Mock(spec=Backend, id="alternative", type=BackendType.KAGGLE)
        alternative_backend.supports_template.return_value = True
        alternative_backend.capabilities = Mock(supports_gpu=True, max_job_duration_minutes=60)
        
        router.register_backend(failed_backend)
        router.register_backend(alternative_backend)
        
        # Mark both as healthy initially
        router.health_monitor.update_health_status("failed", HealthStatus.HEALTHY)
        router.health_monitor.update_health_status("alternative", HealthStatus.HEALTHY)
        
        # Create job with inputs
        original_inputs = {"param1": "value1", "param2": "value2"}
        job = Job(
            id="job1",
            template_name="test-template",
            inputs=original_inputs.copy()
        )
        
        # Handle failure
        router.handle_backend_failure("failed", job)
        
        # Verify original inputs are unchanged
        assert job.inputs == original_inputs
    
    def test_handle_backend_failure_logs_original_and_alternative(self):
        """Test that failover logs both original and alternative backends (Requirement 8.7)."""
        router = MultiBackendRouter()
        
        # Create backends
        failed_backend = Mock(spec=Backend, id="failed", type=BackendType.MODAL)
        failed_backend.supports_template.return_value = True
        failed_backend.capabilities = Mock(supports_gpu=True, max_job_duration_minutes=60)
        
        alternative_backend = Mock(spec=Backend, id="alternative", type=BackendType.KAGGLE)
        alternative_backend.supports_template.return_value = True
        alternative_backend.capabilities = Mock(supports_gpu=True, max_job_duration_minutes=60)
        
        router.register_backend(failed_backend)
        router.register_backend(alternative_backend)
        
        # Mark both as healthy initially
        router.health_monitor.update_health_status("failed", HealthStatus.HEALTHY)
        router.health_monitor.update_health_status("alternative", HealthStatus.HEALTHY)
        
        job = Job(id="job1", template_name="test-template")
        
        # Mock logger to capture log messages
        with patch.object(router.logger, 'warning') as mock_warning, \
             patch.object(router.logger, 'info') as mock_info:
            
            result = router.handle_backend_failure("failed", job)
            
            # Verify warning log for failure
            mock_warning.assert_called_once()
            warning_call = mock_warning.call_args[0][0]
            assert "failed" in warning_call
            assert "job1" in warning_call
            
            # Verify info log for successful failover
            mock_info.assert_called()
            info_calls = [call[0][0] for call in mock_info.call_args_list]
            
            # Find the failover success log
            failover_log = [log for log in info_calls if "Failover successful" in log]
            assert len(failover_log) > 0
            assert "failed" in failover_log[0]
            assert "alternative" in failover_log[0]
    
    def test_handle_backend_failure_returns_none_when_no_alternative(self):
        """Test that None is returned when no alternative backend is available."""
        router = MultiBackendRouter()
        
        # Create only one backend
        failed_backend = Mock(spec=Backend, id="failed", type=BackendType.MODAL)
        failed_backend.supports_template.return_value = True
        failed_backend.capabilities = Mock(supports_gpu=True, max_job_duration_minutes=60)
        
        router.register_backend(failed_backend)
        router.health_monitor.update_health_status("failed", HealthStatus.HEALTHY)
        
        job = Job(id="job1", template_name="test-template")
        
        # Handle failure
        result = router.handle_backend_failure("failed", job)
        
        # Verify None is returned
        assert result is None
    
    def test_handle_backend_failure_logs_no_alternative_error(self):
        """Test that error is logged when no alternative backend is available."""
        router = MultiBackendRouter()
        
        # Create only one backend
        failed_backend = Mock(spec=Backend, id="failed", type=BackendType.MODAL)
        failed_backend.supports_template.return_value = True
        failed_backend.capabilities = Mock(supports_gpu=True, max_job_duration_minutes=60)
        
        router.register_backend(failed_backend)
        router.health_monitor.update_health_status("failed", HealthStatus.HEALTHY)
        
        job = Job(id="job1", template_name="test-template")
        
        # Mock logger to capture error message
        with patch.object(router.logger, 'error') as mock_error:
            result = router.handle_backend_failure("failed", job)
            
            # Verify error log
            mock_error.assert_called_once()
            error_call = mock_error.call_args[0][0]
            assert "no alternative backend available" in error_call.lower()
            assert "job1" in error_call
            assert "failed" in error_call
    
    def test_handle_backend_failure_stores_failure_metadata(self):
        """Test that failure metadata is stored when no alternative is available."""
        router = MultiBackendRouter()
        
        # Create only one backend
        failed_backend = Mock(spec=Backend, id="failed", type=BackendType.MODAL)
        failed_backend.supports_template.return_value = True
        failed_backend.capabilities = Mock(supports_gpu=True, max_job_duration_minutes=60)
        
        router.register_backend(failed_backend)
        router.health_monitor.update_health_status("failed", HealthStatus.HEALTHY)
        
        job = Job(id="job1", template_name="test-template")
        
        # Handle failure
        result = router.handle_backend_failure("failed", job)
        
        # Verify failure metadata is stored
        assert 'last_failover_attempt' in job.metadata
        assert job.metadata['last_failover_attempt']['from_backend'] == "failed"
        assert job.metadata['last_failover_attempt']['failure_reason'] == 'no_alternative_backend'
        assert 'timestamp' in job.metadata['last_failover_attempt']
        assert 'error' in job.metadata['last_failover_attempt']
    
    def test_handle_backend_failure_with_routing_strategy(self):
        """Test that custom routing strategy is used for failover."""
        router = MultiBackendRouter()
        
        # Create backends
        failed_backend = Mock(spec=Backend, id="failed", type=BackendType.MODAL)
        failed_backend.supports_template.return_value = True
        failed_backend.capabilities = Mock(supports_gpu=True, max_job_duration_minutes=60)
        
        alternative_backend = Mock(spec=Backend, id="alternative", type=BackendType.KAGGLE)
        alternative_backend.supports_template.return_value = True
        alternative_backend.estimate_cost.return_value = 0.0
        alternative_backend.capabilities = Mock(supports_gpu=True, max_job_duration_minutes=60)
        
        router.register_backend(failed_backend)
        router.register_backend(alternative_backend)
        
        # Mark both as healthy initially
        router.health_monitor.update_health_status("failed", HealthStatus.HEALTHY)
        router.health_monitor.update_health_status("alternative", HealthStatus.HEALTHY)
        
        job = Job(id="job1", template_name="test-template")
        
        # Handle failure with cost-optimized strategy
        result = router.handle_backend_failure(
            "failed", 
            job, 
            routing_strategy="cost-optimized",
            resource_estimate=ResourceEstimate()
        )
        
        # Verify alternative backend is returned
        assert result is not None
        assert result.id == "alternative"
        
        # Verify routing strategy is stored in metadata
        assert job.metadata['last_failover']['routing_strategy'] == "cost-optimized"
    
    def test_handle_backend_failure_with_resource_estimate(self):
        """Test that resource estimate is passed to alternative routing."""
        router = MultiBackendRouter()
        
        # Create backends with different capabilities
        failed_backend = Mock(spec=Backend, id="failed", type=BackendType.MODAL)
        failed_backend.supports_template.return_value = True
        failed_backend.capabilities = Mock(supports_gpu=True, max_job_duration_minutes=30)
        
        alternative_backend = Mock(spec=Backend, id="alternative", type=BackendType.KAGGLE)
        alternative_backend.supports_template.return_value = True
        alternative_backend.capabilities = Mock(supports_gpu=True, max_job_duration_minutes=120)
        
        router.register_backend(failed_backend)
        router.register_backend(alternative_backend)
        
        # Mark both as healthy initially
        router.health_monitor.update_health_status("failed", HealthStatus.HEALTHY)
        router.health_monitor.update_health_status("alternative", HealthStatus.HEALTHY)
        
        job = Job(id="job1", template_name="test-template")
        
        # Handle failure with resource estimate requiring longer duration
        resource_estimate = ResourceEstimate(
            requires_gpu=True,
            estimated_duration_minutes=60
        )
        
        result = router.handle_backend_failure(
            "failed",
            job,
            resource_estimate=resource_estimate
        )
        
        # Verify alternative backend with sufficient duration is selected
        assert result is not None
        assert result.id == "alternative"
    
    def test_handle_backend_failure_multiple_failovers(self):
        """Test that multiple failovers are tracked in history."""
        router = MultiBackendRouter()
        
        # Create backends
        backend1 = Mock(spec=Backend, id="backend1", type=BackendType.MODAL)
        backend1.supports_template.return_value = True
        backend1.estimate_cost.return_value = 0.0
        backend1.capabilities = Mock(supports_gpu=True, max_job_duration_minutes=60)
        
        backend2 = Mock(spec=Backend, id="backend2", type=BackendType.KAGGLE)
        backend2.supports_template.return_value = True
        backend2.estimate_cost.return_value = 0.0
        backend2.capabilities = Mock(supports_gpu=True, max_job_duration_minutes=60)
        
        backend3 = Mock(spec=Backend, id="backend3", type=BackendType.HUGGINGFACE)
        backend3.supports_template.return_value = True
        backend3.estimate_cost.return_value = 0.0
        backend3.capabilities = Mock(supports_gpu=True, max_job_duration_minutes=60)
        
        router.register_backend(backend1)
        router.register_backend(backend2)
        router.register_backend(backend3)
        
        # Mark all as healthy initially
        router.health_monitor.update_health_status("backend1", HealthStatus.HEALTHY)
        router.health_monitor.update_health_status("backend2", HealthStatus.HEALTHY)
        router.health_monitor.update_health_status("backend3", HealthStatus.HEALTHY)
        
        job = Job(id="job1", template_name="test-template", retry_count=0)
        
        # First failover
        result1 = router.handle_backend_failure("backend1", job)
        assert result1 is not None
        
        # Second failover
        job.retry_count = 1
        result2 = router.handle_backend_failure(result1.id, job)
        assert result2 is not None
        
        # Verify failover history has two entries
        assert 'failover_history' in job.metadata
        assert len(job.metadata['failover_history']) == 2
        assert job.metadata['failover_history'][0]['failed_backend_id'] == "backend1"
        assert job.metadata['failover_history'][1]['failed_backend_id'] == result1.id
    
    def test_handle_backend_failure_excludes_failed_backend_from_routing(self):
        """Test that failed backend is excluded from alternative routing."""
        router = MultiBackendRouter()
        
        # Create three backends
        failed_backend = Mock(spec=Backend, id="failed", type=BackendType.MODAL)
        failed_backend.supports_template.return_value = True
        failed_backend.estimate_cost.return_value = 0.0
        failed_backend.capabilities = Mock(supports_gpu=True, max_job_duration_minutes=60)
        
        backend2 = Mock(spec=Backend, id="backend2", type=BackendType.KAGGLE)
        backend2.supports_template.return_value = True
        backend2.estimate_cost.return_value = 0.0
        backend2.capabilities = Mock(supports_gpu=True, max_job_duration_minutes=60)
        
        backend3 = Mock(spec=Backend, id="backend3", type=BackendType.HUGGINGFACE)
        backend3.supports_template.return_value = True
        backend3.estimate_cost.return_value = 0.0
        backend3.capabilities = Mock(supports_gpu=True, max_job_duration_minutes=60)
        
        router.register_backend(failed_backend)
        router.register_backend(backend2)
        router.register_backend(backend3)
        
        # Mark all as healthy initially
        router.health_monitor.update_health_status("failed", HealthStatus.HEALTHY)
        router.health_monitor.update_health_status("backend2", HealthStatus.HEALTHY)
        router.health_monitor.update_health_status("backend3", HealthStatus.HEALTHY)
        
        job = Job(id="job1", template_name="test-template")
        
        # Handle failure
        result = router.handle_backend_failure("failed", job)
        
        # Verify alternative backend is not the failed one
        assert result is not None
        assert result.id != "failed"
        assert result.id in ["backend2", "backend3"]
