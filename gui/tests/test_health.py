"""Tests for health check module.

This module tests the health check functionality including:
- Component health checks
- Overall health status determination
- Health check response formatting
"""

import pytest
from unittest.mock import MagicMock, patch
import time

from gui.health import (
    ComponentStatus,
    ComponentHealth,
    HealthCheckResponse,
    HealthChecker,
    create_health_check_handler
)
from notebook_ml_orchestrator.core.models import HealthStatus, JobStatus, WorkflowStatus


class TestComponentHealth:
    """Test ComponentHealth dataclass."""
    
    def test_component_health_creation(self):
        """Test creating ComponentHealth."""
        health = ComponentHealth(
            name="Test Component",
            status=ComponentStatus.HEALTHY,
            message="All good"
        )
        
        assert health.name == "Test Component"
        assert health.status == ComponentStatus.HEALTHY
        assert health.message == "All good"
        assert health.details is None
    
    def test_component_health_with_details(self):
        """Test ComponentHealth with details."""
        health = ComponentHealth(
            name="Test Component",
            status=ComponentStatus.HEALTHY,
            details={'count': 42}
        )
        
        assert health.details == {'count': 42}


class TestHealthCheckResponse:
    """Test HealthCheckResponse dataclass."""
    
    def test_health_check_response_creation(self):
        """Test creating HealthCheckResponse."""
        components = {
            'test': ComponentHealth(
                name="Test",
                status=ComponentStatus.HEALTHY
            )
        }
        
        response = HealthCheckResponse(
            status=ComponentStatus.HEALTHY,
            timestamp=time.time(),
            version="1.0.0",
            components=components
        )
        
        assert response.status == ComponentStatus.HEALTHY
        assert response.version == "1.0.0"
        assert 'test' in response.components
    
    def test_health_check_response_to_dict(self):
        """Test converting HealthCheckResponse to dictionary."""
        components = {
            'test': ComponentHealth(
                name="Test",
                status=ComponentStatus.HEALTHY,
                message="OK",
                details={'count': 5}
            )
        }
        
        response = HealthCheckResponse(
            status=ComponentStatus.HEALTHY,
            timestamp=1234567890.0,
            version="1.0.0",
            components=components,
            uptime=100.5
        )
        
        result = response.to_dict()
        
        assert result['status'] == 'healthy'
        assert result['timestamp'] == 1234567890.0
        assert result['version'] == "1.0.0"
        assert result['uptime'] == 100.5
        assert 'test' in result['components']
        assert result['components']['test']['status'] == 'healthy'
        assert result['components']['test']['message'] == "OK"
        assert result['components']['test']['details'] == {'count': 5}


class TestHealthChecker:
    """Test HealthChecker class."""
    
    def test_health_checker_initialization(self):
        """Test HealthChecker initialization."""
        job_queue = MagicMock()
        backend_router = MagicMock()
        
        checker = HealthChecker(
            job_queue=job_queue,
            backend_router=backend_router,
            version="1.0.0"
        )
        
        assert checker.job_queue == job_queue
        assert checker.backend_router == backend_router
        assert checker.version == "1.0.0"
        assert checker.start_time > 0
    
    def test_check_health_all_healthy(self):
        """Test health check with all components healthy."""
        # Mock job queue
        job_queue = MagicMock()
        job_queue.get_queue_statistics.return_value = {
            'total': 10,
            'pending': 2,
            'running': 3
        }
        
        # Mock backend router
        backend_router = MagicMock()
        backend_router.get_backend_status.return_value = {
            'backend1': HealthStatus.HEALTHY,
            'backend2': HealthStatus.HEALTHY
        }
        
        # Mock workflow engine
        workflow_engine = MagicMock()
        workflow_engine.list_executions.return_value = []
        
        # Mock template registry
        template_registry = MagicMock()
        template_registry.list_templates.return_value = ['template1', 'template2']
        
        checker = HealthChecker(
            job_queue=job_queue,
            backend_router=backend_router,
            workflow_engine=workflow_engine,
            template_registry=template_registry,
            version="1.0.0"
        )
        
        response = checker.check_health()
        
        assert response.status == ComponentStatus.HEALTHY
        assert response.version == "1.0.0"
        assert 'job_queue' in response.components
        assert 'backend_router' in response.components
        assert 'workflow_engine' in response.components
        assert 'template_registry' in response.components
        assert response.uptime is not None
    
    def test_check_health_degraded_backend(self):
        """Test health check with degraded backend."""
        # Mock backend router with one unhealthy backend
        backend_router = MagicMock()
        backend_router.get_backend_status.return_value = {
            'backend1': HealthStatus.HEALTHY,
            'backend2': HealthStatus.UNHEALTHY
        }
        
        checker = HealthChecker(
            backend_router=backend_router,
            version="1.0.0"
        )
        
        response = checker.check_health()
        
        # Overall status should be degraded
        assert response.status == ComponentStatus.DEGRADED
        assert response.components['backend_router'].status == ComponentStatus.DEGRADED
    
    def test_check_health_all_backends_unhealthy(self):
        """Test health check with all backends unhealthy."""
        # Mock backend router with all unhealthy backends
        backend_router = MagicMock()
        backend_router.get_backend_status.return_value = {
            'backend1': HealthStatus.UNHEALTHY,
            'backend2': HealthStatus.UNHEALTHY
        }
        
        checker = HealthChecker(
            backend_router=backend_router,
            version="1.0.0"
        )
        
        response = checker.check_health()
        
        # Overall status should be unhealthy
        assert response.status == ComponentStatus.UNHEALTHY
        assert response.components['backend_router'].status == ComponentStatus.UNHEALTHY
    
    def test_check_health_job_queue_error(self):
        """Test health check with job queue error."""
        # Mock job queue that raises exception
        job_queue = MagicMock()
        job_queue.get_queue_statistics.side_effect = Exception("Database error")
        
        checker = HealthChecker(
            job_queue=job_queue,
            version="1.0.0"
        )
        
        response = checker.check_health()
        
        # Overall status should be unhealthy
        assert response.status == ComponentStatus.UNHEALTHY
        assert response.components['job_queue'].status == ComponentStatus.UNHEALTHY
        assert "error" in response.components['job_queue'].message.lower()
    
    def test_check_health_workflow_engine_error(self):
        """Test health check with workflow engine error."""
        # Mock workflow engine that raises exception
        workflow_engine = MagicMock()
        workflow_engine.list_executions.side_effect = Exception("Engine error")
        
        checker = HealthChecker(
            workflow_engine=workflow_engine,
            version="1.0.0"
        )
        
        response = checker.check_health()
        
        # Overall status should be unhealthy
        assert response.status == ComponentStatus.UNHEALTHY
        assert response.components['workflow_engine'].status == ComponentStatus.UNHEALTHY
    
    def test_check_health_template_registry_error(self):
        """Test health check with template registry error."""
        # Mock template registry that raises exception
        template_registry = MagicMock()
        template_registry.list_templates.side_effect = Exception("Registry error")
        
        checker = HealthChecker(
            template_registry=template_registry,
            version="1.0.0"
        )
        
        response = checker.check_health()
        
        # Overall status should be unhealthy
        assert response.status == ComponentStatus.UNHEALTHY
        assert response.components['template_registry'].status == ComponentStatus.UNHEALTHY
    
    def test_check_health_no_components(self):
        """Test health check with no components."""
        checker = HealthChecker(version="1.0.0")
        
        response = checker.check_health()
        
        # Overall status should be unknown
        assert response.status == ComponentStatus.UNKNOWN
        assert len(response.components) == 0
    
    def test_check_health_uptime_calculation(self):
        """Test that uptime is calculated correctly."""
        checker = HealthChecker(version="1.0.0")
        
        # Wait a bit
        time.sleep(0.1)
        
        response = checker.check_health()
        
        # Uptime should be at least 0.1 seconds
        assert response.uptime >= 0.1


class TestCreateHealthCheckHandler:
    """Test create_health_check_handler function."""
    
    def test_create_handler(self):
        """Test creating health check handler."""
        # Mock health checker
        checker = MagicMock()
        mock_response = MagicMock()
        mock_response.to_dict.return_value = {'status': 'healthy'}
        checker.check_health.return_value = mock_response
        
        handler = create_health_check_handler(checker)
        
        # Handler should be callable
        assert callable(handler)
        
        # Call handler
        result = handler()
        
        # Should return dictionary
        assert isinstance(result, dict)
        assert result['status'] == 'healthy'
        
        # Should have called check_health
        checker.check_health.assert_called_once()
    
    def test_handler_returns_dict(self):
        """Test that handler returns dictionary."""
        job_queue = MagicMock()
        job_queue.get_queue_statistics.return_value = {'total': 5}
        
        checker = HealthChecker(
            job_queue=job_queue,
            version="1.0.0"
        )
        
        handler = create_health_check_handler(checker)
        result = handler()
        
        # Should be a dictionary
        assert isinstance(result, dict)
        assert 'status' in result
        assert 'timestamp' in result
        assert 'version' in result
        assert 'components' in result
