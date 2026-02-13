"""
Unit tests for job failure tracking and health monitoring integration.

Tests the connection between failover logic and health monitoring (Task 8.3):
- Recording consecutive job failures
- Marking backends as unhealthy after repeated job failures
- Resetting failure counts on successful job execution

Requirements:
    - 8.5: Mark backends as unhealthy after consecutive job failures
"""

import pytest
from unittest.mock import Mock, MagicMock
from datetime import datetime

from notebook_ml_orchestrator.core.backend_router import (
    MultiBackendRouter,
    HealthMonitor
)
from notebook_ml_orchestrator.core.models import (
    HealthStatus,
    JobStatus,
    BackendType,
    ResourceEstimate
)
from notebook_ml_orchestrator.core.interfaces import Backend, Job


class TestJobFailureTracking:
    """Test job failure tracking in HealthMonitor."""
    
    def test_record_job_failure_increments_count(self):
        """Test that recording a job failure increments the failure count."""
        monitor = HealthMonitor()
        backend_id = "test-backend"
        
        # Record first failure
        monitor.record_job_failure(backend_id)
        assert monitor.get_job_failure_count(backend_id) == 1
        
        # Record second failure
        monitor.record_job_failure(backend_id)
        assert monitor.get_job_failure_count(backend_id) == 2
    
    def test_record_job_failure_marks_unhealthy_after_three_failures(self):
        """Test that backend is marked unhealthy after 3 consecutive job failures (Requirement 8.5)."""
        monitor = HealthMonitor()
        backend_id = "test-backend"
        
        # Record 3 consecutive failures
        monitor.record_job_failure(backend_id)
        monitor.record_job_failure(backend_id)
        
        # After 2 failures, should not be marked unhealthy yet
        assert monitor.get_job_failure_count(backend_id) == 2
        
        # Third failure should mark as unhealthy
        monitor.record_job_failure(backend_id)
        assert monitor.get_job_failure_count(backend_id) == 3
        
        # Check that health status was updated to UNHEALTHY
        history = monitor.health_history.get(backend_id, [])
        assert len(history) > 0
        assert history[-1]['status'] == HealthStatus.UNHEALTHY
    
    def test_record_job_success_resets_failure_count(self):
        """Test that recording a job success resets the failure count."""
        monitor = HealthMonitor()
        backend_id = "test-backend"
        
        # Record some failures
        monitor.record_job_failure(backend_id)
        monitor.record_job_failure(backend_id)
        assert monitor.get_job_failure_count(backend_id) == 2
        
        # Record success
        monitor.record_job_success(backend_id)
        assert monitor.get_job_failure_count(backend_id) == 0
    
    def test_get_job_failure_count_returns_zero_for_new_backend(self):
        """Test that get_job_failure_count returns 0 for backends with no failures."""
        monitor = HealthMonitor()
        assert monitor.get_job_failure_count("new-backend") == 0
    
    def test_health_metrics_include_job_failure_count(self):
        """Test that health metrics include consecutive job failure count."""
        monitor = HealthMonitor()
        backend_id = "test-backend"
        
        # Record some failures
        monitor.record_job_failure(backend_id)
        monitor.record_job_failure(backend_id)
        
        # Get metrics
        metrics = monitor.get_health_metrics(backend_id)
        assert 'consecutive_job_failures' in metrics
        assert metrics['consecutive_job_failures'] == 2


class TestFailoverHealthIntegration:
    """Test integration between failover and health monitoring."""
    
    def test_handle_backend_failure_records_job_failure(self):
        """Test that handle_backend_failure records job failures."""
        router = MultiBackendRouter()
        
        # Create mock backends
        backend1 = Mock(spec=Backend)
        backend1.id = "backend1"
        backend1.name = "Backend 1"
        backend1.type = BackendType.MODAL
        backend1.supports_template = Mock(return_value=True)
        backend1.check_health = Mock(return_value=HealthStatus.HEALTHY)
        backend1.get_queue_length = Mock(return_value=0)
        backend1.capabilities = Mock(supports_gpu=True, max_job_duration_minutes=300)
        
        backend2 = Mock(spec=Backend)
        backend2.id = "backend2"
        backend2.name = "Backend 2"
        backend2.type = BackendType.HUGGINGFACE
        backend2.supports_template = Mock(return_value=True)
        backend2.check_health = Mock(return_value=HealthStatus.HEALTHY)
        backend2.get_queue_length = Mock(return_value=0)
        backend2.capabilities = Mock(supports_gpu=True, max_job_duration_minutes=300)
        
        router.register_backend(backend1)
        router.register_backend(backend2)
        
        # Create a job
        job = Mock(spec=Job)
        job.id = "job1"
        job.template_name = "test-template"
        job.retry_count = 0
        job.metadata = {}
        job.inputs = {}
        job.priority = 1
        job.status = JobStatus.QUEUED
        
        # Initial failure count should be 0
        assert router.health_monitor.get_job_failure_count("backend1") == 0
        
        # Handle backend failure
        router.handle_backend_failure("backend1", job)
        
        # Failure count should be incremented
        assert router.health_monitor.get_job_failure_count("backend1") == 1
    
    def test_multiple_job_failures_mark_backend_unhealthy(self):
        """Test that multiple consecutive job failures mark backend as unhealthy (Requirement 8.5)."""
        router = MultiBackendRouter()
        
        # Create mock backends
        backend1 = Mock(spec=Backend)
        backend1.id = "backend1"
        backend1.name = "Backend 1"
        backend1.type = BackendType.MODAL
        backend1.supports_template = Mock(return_value=True)
        backend1.check_health = Mock(return_value=HealthStatus.HEALTHY)
        backend1.get_queue_length = Mock(return_value=0)
        backend1.capabilities = Mock(supports_gpu=True, max_job_duration_minutes=300)
        
        backend2 = Mock(spec=Backend)
        backend2.id = "backend2"
        backend2.name = "Backend 2"
        backend2.type = BackendType.HUGGINGFACE
        backend2.supports_template = Mock(return_value=True)
        backend2.check_health = Mock(return_value=HealthStatus.HEALTHY)
        backend2.get_queue_length = Mock(return_value=0)
        backend2.capabilities = Mock(supports_gpu=True, max_job_duration_minutes=300)
        
        router.register_backend(backend1)
        router.register_backend(backend2)
        
        # Create jobs
        job1 = Mock(spec=Job)
        job1.id = "job1"
        job1.template_name = "test-template"
        job1.retry_count = 0
        job1.metadata = {}
        job1.inputs = {}
        job1.priority = 1
        job1.status = JobStatus.QUEUED
        
        job2 = Mock(spec=Job)
        job2.id = "job2"
        job2.template_name = "test-template"
        job2.retry_count = 0
        job2.metadata = {}
        job2.inputs = {}
        job2.priority = 1
        job2.status = JobStatus.QUEUED
        
        job3 = Mock(spec=Job)
        job3.id = "job3"
        job3.template_name = "test-template"
        job3.retry_count = 0
        job3.metadata = {}
        job3.inputs = {}
        job3.priority = 1
        job3.status = JobStatus.QUEUED
        
        # Simulate 3 consecutive job failures
        router.handle_backend_failure("backend1", job1)
        assert router.health_monitor.get_job_failure_count("backend1") == 1
        
        router.handle_backend_failure("backend1", job2)
        assert router.health_monitor.get_job_failure_count("backend1") == 2
        
        router.handle_backend_failure("backend1", job3)
        assert router.health_monitor.get_job_failure_count("backend1") == 3
        
        # Backend should now be marked as unhealthy
        history = router.health_monitor.health_history.get("backend1", [])
        assert len(history) > 0
        assert history[-1]['status'] == HealthStatus.UNHEALTHY
    
    def test_execute_job_with_retry_records_success(self):
        """Test that successful job execution records success and resets failure count."""
        router = MultiBackendRouter()
        
        # Create mock backend
        backend = Mock(spec=Backend)
        backend.id = "backend1"
        backend.name = "Backend 1"
        backend.type = BackendType.MODAL
        backend.execute_job = Mock(return_value={"result": "success"})
        
        router.register_backend(backend)
        
        # Create a job
        job = Mock(spec=Job)
        job.id = "job1"
        job.template_name = "test-template"
        job.retry_count = 0
        job.metadata = {}
        
        # Record some failures first
        router.health_monitor.record_job_failure("backend1")
        router.health_monitor.record_job_failure("backend1")
        assert router.health_monitor.get_job_failure_count("backend1") == 2
        
        # Execute job successfully
        result = router.execute_job_with_retry(job, backend, max_retries=3)
        
        # Failure count should be reset
        assert router.health_monitor.get_job_failure_count("backend1") == 0
        assert result == {"result": "success"}
    
    def test_job_failure_count_independent_of_health_check_failures(self):
        """Test that job failure count is independent of health check failure count."""
        monitor = HealthMonitor()
        backend_id = "test-backend"
        
        # Create mock backend
        backend = Mock(spec=Backend)
        backend.id = backend_id
        backend.check_health = Mock(side_effect=Exception("Health check failed"))
        
        # Record health check failures
        monitor.check_backend_health(backend)
        monitor.check_backend_health(backend)
        
        # Health check failure count should be 2
        assert monitor.failure_counts.get(backend_id, 0) == 2
        
        # Job failure count should still be 0
        assert monitor.get_job_failure_count(backend_id) == 0
        
        # Record job failures
        monitor.record_job_failure(backend_id)
        monitor.record_job_failure(backend_id)
        
        # Job failure count should be 2
        assert monitor.get_job_failure_count(backend_id) == 2
        
        # Health check failure count should still be 2
        assert monitor.failure_counts.get(backend_id, 0) == 2
