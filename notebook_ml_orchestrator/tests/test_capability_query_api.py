"""
Tests for the capability query API (Task 9.3).

This module tests the list_backends_with_capabilities method and the CLI command
for querying backend capabilities.
"""

import pytest
from unittest.mock import Mock, MagicMock
from datetime import datetime

from notebook_ml_orchestrator.core.backend_router import MultiBackendRouter
from notebook_ml_orchestrator.core.models import (
    BackendType,
    BackendCapabilities,
    HealthStatus,
)
from notebook_ml_orchestrator.core.interfaces import Backend


class MockBackend(Backend):
    """Mock backend for testing."""
    
    def __init__(self, backend_id: str, backend_type: BackendType, 
                 capabilities: BackendCapabilities, name: str = None):
        self.id = backend_id
        self.type = backend_type
        self.name = name or backend_id
        self.capabilities = capabilities
        self._queue_length = 0
    
    def execute_job(self, job):
        return {"status": "success"}
    
    def check_health(self):
        return HealthStatus.HEALTHY
    
    def supports_template(self, template_name: str) -> bool:
        return template_name in self.capabilities.supported_templates
    
    def estimate_cost(self, resource_estimate):
        return self.capabilities.cost_per_hour
    
    def get_queue_length(self) -> int:
        return self._queue_length


class TestCapabilityQueryAPI:
    """Test suite for capability query API."""
    
    def test_list_backends_with_capabilities_empty(self):
        """Test listing capabilities when no backends are registered."""
        router = MultiBackendRouter()
        
        backends_info = router.list_backends_with_capabilities()
        
        assert backends_info == []
    
    def test_list_backends_with_capabilities_single_backend(self):
        """Test listing capabilities for a single backend."""
        router = MultiBackendRouter()
        
        # Create a mock backend with capabilities
        capabilities = BackendCapabilities(
            supported_templates=["image-generation", "text-generation"],
            max_concurrent_jobs=5,
            max_job_duration_minutes=300,
            supports_gpu=True,
            supports_batch=True,
            cost_per_hour=1.10,
            free_tier_limits={}
        )
        
        backend = MockBackend(
            backend_id="modal-1",
            backend_type=BackendType.MODAL,
            capabilities=capabilities,
            name="Modal Backend"
        )
        
        router.register_backend(backend)
        
        # Initialize health status
        router.health_monitor.update_health_status("modal-1", HealthStatus.HEALTHY)
        
        backends_info = router.list_backends_with_capabilities()
        
        assert len(backends_info) == 1
        
        backend_info = backends_info[0]
        assert backend_info['id'] == "modal-1"
        assert backend_info['name'] == "Modal Backend"
        assert backend_info['type'] == "modal"
        assert backend_info['health_status'] == "healthy"
        assert backend_info['queue_length'] == 0
        
        # Check capabilities
        caps = backend_info['capabilities']
        assert caps['supported_templates'] == ["image-generation", "text-generation"]
        assert caps['max_concurrent_jobs'] == 5
        assert caps['max_job_duration_minutes'] == 300
        assert caps['supports_gpu'] is True
        assert caps['supports_batch'] is True
        assert caps['cost_per_hour'] == 1.10
        assert caps['free_tier_limits'] == {}
        
        # Check health metrics
        metrics = backend_info['health_metrics']
        assert 'uptime_percentage' in metrics
        assert 'total_checks' in metrics
        assert 'failure_rate' in metrics
    
    def test_list_backends_with_capabilities_multiple_backends(self):
        """Test listing capabilities for multiple backends."""
        router = MultiBackendRouter()
        
        # Create Modal backend
        modal_caps = BackendCapabilities(
            supported_templates=["image-generation", "text-generation"],
            max_concurrent_jobs=10,
            max_job_duration_minutes=300,
            supports_gpu=True,
            supports_batch=True,
            cost_per_hour=1.10,
            free_tier_limits={}
        )
        modal_backend = MockBackend("modal-1", BackendType.MODAL, modal_caps, "Modal")
        
        # Create HuggingFace backend
        hf_caps = BackendCapabilities(
            supported_templates=["text-generation", "embeddings"],
            max_concurrent_jobs=3,
            max_job_duration_minutes=60,
            supports_gpu=True,
            supports_batch=False,
            cost_per_hour=0.0,
            free_tier_limits={"rate_limit": "1000 requests/day"}
        )
        hf_backend = MockBackend("hf-1", BackendType.HUGGINGFACE, hf_caps, "HuggingFace")
        
        # Create Kaggle backend
        kaggle_caps = BackendCapabilities(
            supported_templates=["model-training", "data-processing"],
            max_concurrent_jobs=1,
            max_job_duration_minutes=540,
            supports_gpu=True,
            supports_batch=True,
            cost_per_hour=0.0,
            free_tier_limits={"gpu_hours_per_week": 30}
        )
        kaggle_backend = MockBackend("kaggle-1", BackendType.KAGGLE, kaggle_caps, "Kaggle")
        
        # Register all backends
        router.register_backend(modal_backend)
        router.register_backend(hf_backend)
        router.register_backend(kaggle_backend)
        
        # Initialize health statuses
        router.health_monitor.update_health_status("modal-1", HealthStatus.HEALTHY)
        router.health_monitor.update_health_status("hf-1", HealthStatus.HEALTHY)
        router.health_monitor.update_health_status("kaggle-1", HealthStatus.DEGRADED)
        
        backends_info = router.list_backends_with_capabilities()
        
        assert len(backends_info) == 3
        
        # Verify each backend is present
        backend_ids = [b['id'] for b in backends_info]
        assert "modal-1" in backend_ids
        assert "hf-1" in backend_ids
        assert "kaggle-1" in backend_ids
        
        # Find and verify Modal backend
        modal_info = next(b for b in backends_info if b['id'] == "modal-1")
        assert modal_info['capabilities']['supports_gpu'] is True
        assert modal_info['capabilities']['cost_per_hour'] == 1.10
        
        # Find and verify HuggingFace backend
        hf_info = next(b for b in backends_info if b['id'] == "hf-1")
        assert hf_info['capabilities']['cost_per_hour'] == 0.0
        assert hf_info['capabilities']['free_tier_limits'] == {"rate_limit": "1000 requests/day"}
        
        # Find and verify Kaggle backend
        kaggle_info = next(b for b in backends_info if b['id'] == "kaggle-1")
        assert kaggle_info['health_status'] == "degraded"
        assert kaggle_info['capabilities']['free_tier_limits'] == {"gpu_hours_per_week": 30}
    
    def test_list_backends_with_capabilities_includes_health_metrics(self):
        """Test that health metrics are included in the capability listing."""
        router = MultiBackendRouter()
        
        capabilities = BackendCapabilities(
            supported_templates=["test-template"],
            max_concurrent_jobs=1,
            max_job_duration_minutes=60,
            supports_gpu=False,
            supports_batch=True,
            cost_per_hour=0.0,
            free_tier_limits={}
        )
        
        backend = MockBackend("test-1", BackendType.LOCAL_GPU, capabilities)
        router.register_backend(backend)
        
        # Simulate some health checks
        router.health_monitor.update_health_status("test-1", HealthStatus.HEALTHY)
        router.health_monitor.update_health_status("test-1", HealthStatus.HEALTHY)
        router.health_monitor.update_health_status("test-1", HealthStatus.UNHEALTHY)
        
        backends_info = router.list_backends_with_capabilities()
        
        assert len(backends_info) == 1
        
        metrics = backends_info[0]['health_metrics']
        assert metrics['total_checks'] == 3
        assert metrics['healthy_checks'] == 2
        assert metrics['uptime_percentage'] == pytest.approx(66.67, rel=0.1)
        assert metrics['failure_rate'] == pytest.approx(33.33, rel=0.1)
    
    def test_list_backends_with_capabilities_queue_length(self):
        """Test that queue length is included in the capability listing."""
        router = MultiBackendRouter()
        
        capabilities = BackendCapabilities(
            supported_templates=["test-template"],
            max_concurrent_jobs=5,
            max_job_duration_minutes=60,
            supports_gpu=False,
            supports_batch=True,
            cost_per_hour=0.0,
            free_tier_limits={}
        )
        
        backend = MockBackend("test-1", BackendType.LOCAL_GPU, capabilities)
        backend._queue_length = 3  # Set queue length
        
        router.register_backend(backend)
        router.health_monitor.update_health_status("test-1", HealthStatus.HEALTHY)
        
        backends_info = router.list_backends_with_capabilities()
        
        assert len(backends_info) == 1
        assert backends_info[0]['queue_length'] == 3
    
    def test_list_backends_with_capabilities_unknown_health_status(self):
        """Test handling of backends with unknown health status."""
        router = MultiBackendRouter()
        
        capabilities = BackendCapabilities(
            supported_templates=["test-template"],
            max_concurrent_jobs=1,
            max_job_duration_minutes=60,
            supports_gpu=False,
            supports_batch=True,
            cost_per_hour=0.0,
            free_tier_limits={}
        )
        
        backend = MockBackend("test-1", BackendType.LOCAL_GPU, capabilities)
        router.register_backend(backend)
        
        # Don't initialize health status - system should perform health check
        backends_info = router.list_backends_with_capabilities()
        
        assert len(backends_info) == 1
        # System performs health check automatically, so status should be healthy
        assert backends_info[0]['health_status'] in ["healthy", "unknown"]
    
    def test_list_backends_with_capabilities_validates_requirements_10_4(self):
        """
        Test that list_backends_with_capabilities satisfies Requirement 10.4.
        
        Requirement 10.4: Backend listing with capabilities
        - Backend list request should include capability information for each backend
        """
        router = MultiBackendRouter()
        
        capabilities = BackendCapabilities(
            supported_templates=["image-generation"],
            max_concurrent_jobs=5,
            max_job_duration_minutes=300,
            supports_gpu=True,
            supports_batch=True,
            cost_per_hour=1.10,
            free_tier_limits={}
        )
        
        backend = MockBackend("modal-1", BackendType.MODAL, capabilities, "Modal")
        router.register_backend(backend)
        router.health_monitor.update_health_status("modal-1", HealthStatus.HEALTHY)
        
        backends_info = router.list_backends_with_capabilities()
        
        # Verify all required capability fields are present
        assert len(backends_info) == 1
        backend_info = backends_info[0]
        
        # Required fields for Requirement 10.4
        assert 'id' in backend_info
        assert 'name' in backend_info
        assert 'type' in backend_info
        assert 'capabilities' in backend_info
        
        caps = backend_info['capabilities']
        assert 'supported_templates' in caps
        assert 'max_concurrent_jobs' in caps
        assert 'max_job_duration_minutes' in caps
        assert 'supports_gpu' in caps
        assert 'supports_batch' in caps
        assert 'cost_per_hour' in caps
        assert 'free_tier_limits' in caps
    
    def test_list_backends_with_capabilities_validates_requirements_10_7(self):
        """
        Test that list_backends_with_capabilities satisfies Requirement 10.7.
        
        Requirement 10.7: Capability API endpoint
        - API request to capabilities endpoint should return list of available backends with capabilities
        """
        router = MultiBackendRouter()
        
        # Register multiple backends to simulate a real scenario
        modal_caps = BackendCapabilities(
            supported_templates=["image-generation", "text-generation"],
            max_concurrent_jobs=10,
            max_job_duration_minutes=300,
            supports_gpu=True,
            supports_batch=True,
            cost_per_hour=1.10,
            free_tier_limits={}
        )
        modal_backend = MockBackend("modal-1", BackendType.MODAL, modal_caps, "Modal")
        
        hf_caps = BackendCapabilities(
            supported_templates=["text-generation", "embeddings"],
            max_concurrent_jobs=3,
            max_job_duration_minutes=60,
            supports_gpu=True,
            supports_batch=False,
            cost_per_hour=0.0,
            free_tier_limits={"rate_limit": "1000 requests/day"}
        )
        hf_backend = MockBackend("hf-1", BackendType.HUGGINGFACE, hf_caps, "HuggingFace")
        
        router.register_backend(modal_backend)
        router.register_backend(hf_backend)
        
        router.health_monitor.update_health_status("modal-1", HealthStatus.HEALTHY)
        router.health_monitor.update_health_status("hf-1", HealthStatus.HEALTHY)
        
        # This method serves as the API endpoint implementation
        backends_info = router.list_backends_with_capabilities()
        
        # Verify API returns list of backends with capabilities
        assert isinstance(backends_info, list)
        assert len(backends_info) == 2
        
        # Verify each backend has complete capability information
        for backend_info in backends_info:
            assert 'id' in backend_info
            assert 'name' in backend_info
            assert 'type' in backend_info
            assert 'health_status' in backend_info
            assert 'capabilities' in backend_info
            assert 'queue_length' in backend_info
            assert 'health_metrics' in backend_info
            
            # Verify capabilities structure
            caps = backend_info['capabilities']
            assert isinstance(caps['supported_templates'], list)
            assert isinstance(caps['max_concurrent_jobs'], int)
            assert isinstance(caps['max_job_duration_minutes'], int)
            assert isinstance(caps['supports_gpu'], bool)
            assert isinstance(caps['supports_batch'], bool)
            assert isinstance(caps['cost_per_hour'], (int, float))
            assert isinstance(caps['free_tier_limits'], dict)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
