"""
Unit tests for BackendMonitorService.

Tests backend monitoring functionality including status retrieval,
health checks, and backend details.
"""

import pytest
from datetime import datetime
from unittest.mock import Mock, MagicMock

from gui.services.backend_monitor_service import BackendMonitorService
from notebook_ml_orchestrator.core.models import HealthStatus, BackendCapabilities
from notebook_ml_orchestrator.core.interfaces import Backend


class TestBackendMonitorService:
    """Test suite for BackendMonitorService."""
    
    @pytest.fixture
    def mock_backend_router(self):
        """Create a mock backend router."""
        router = Mock()
        router.health_monitor = Mock()
        router.cost_optimizer = Mock()
        router.health_monitor.health_history = {}
        return router
    
    @pytest.fixture
    def mock_backend(self):
        """Create a mock backend."""
        backend = Mock(spec=Backend)
        backend.id = "test_backend"
        backend.capabilities = BackendCapabilities(
            supported_templates=["template1", "template2"],
            max_concurrent_jobs=5,
            max_job_duration_minutes=60,
            supports_gpu=True,
            supports_batch=False,
            cost_per_hour=1.5,
            free_tier_limits=None
        )
        return backend
    
    @pytest.fixture
    def service(self, mock_backend_router):
        """Create BackendMonitorService instance."""
        return BackendMonitorService(mock_backend_router)
    
    def test_initialization(self, service, mock_backend_router):
        """Test service initialization."""
        assert service.backend_router == mock_backend_router
        assert hasattr(service, 'logger')
    
    def test_get_backends_status_empty(self, service, mock_backend_router):
        """Test getting status when no backends are registered."""
        mock_backend_router.list_backends.return_value = []
        mock_backend_router.get_backend_status.return_value = {}
        
        result = service.get_backends_status()
        
        assert result == []
        mock_backend_router.list_backends.assert_called_once()
    
    def test_get_backends_status_single_healthy(self, service, mock_backend_router, mock_backend):
        """Test getting status for a single healthy backend."""
        mock_backend_router.list_backends.return_value = [mock_backend]
        mock_backend_router.get_backend_status.return_value = {
            "test_backend": HealthStatus.HEALTHY
        }
        mock_backend_router.health_monitor.get_health_metrics.return_value = {
            'uptime_percentage': 95.0,
            'total_checks': 100,
            'healthy_checks': 95,
            'failure_rate': 5.0,
            'last_check': datetime.now(),
            'consecutive_failures': 0,
            'consecutive_job_failures': 0
        }
        mock_backend_router.cost_optimizer.get_total_cost.return_value = 10.5
        
        result = service.get_backends_status()
        
        assert len(result) == 1
        assert result[0]['name'] == 'test_backend'
        assert result[0]['status'] == 'healthy'
        assert result[0]['uptime_percentage'] == 95.0
        assert result[0]['cost_total'] == 10.5
        assert result[0]['capabilities']['supports_gpu'] is True
    
    def test_get_backends_status_unhealthy_with_error(self, service, mock_backend_router, mock_backend):
        """Test getting status for an unhealthy backend with error history."""
        mock_backend_router.list_backends.return_value = [mock_backend]
        mock_backend_router.get_backend_status.return_value = {
            "test_backend": HealthStatus.UNHEALTHY
        }
        mock_backend_router.health_monitor.get_health_metrics.return_value = {
            'uptime_percentage': 50.0,
            'total_checks': 10,
            'healthy_checks': 5,
            'failure_rate': 50.0,
            'last_check': datetime.now(),
            'consecutive_failures': 3,
            'consecutive_job_failures': 2
        }
        mock_backend_router.health_monitor.health_history = {
            'test_backend': [
                {'status': HealthStatus.HEALTHY, 'timestamp': datetime.now()},
                {'status': HealthStatus.UNHEALTHY, 'timestamp': datetime.now()}
            ]
        }
        mock_backend_router.cost_optimizer.get_total_cost.return_value = 5.0
        
        result = service.get_backends_status()
        
        assert len(result) == 1
        assert result[0]['name'] == 'test_backend'
        assert result[0]['status'] == 'unhealthy'
        assert result[0]['last_error'] is not None
        assert 'unhealthy' in result[0]['last_error'].lower()
    
    def test_get_backend_details_success(self, service, mock_backend_router, mock_backend):
        """Test getting detailed backend information."""
        mock_backend_router.get_backend.return_value = mock_backend
        mock_backend_router.get_backend_status.return_value = {
            "test_backend": HealthStatus.HEALTHY
        }
        mock_backend_router.health_monitor.get_health_metrics.return_value = {
            'uptime_percentage': 98.5,
            'total_checks': 200,
            'healthy_checks': 197,
            'failure_rate': 1.5,
            'last_check': datetime.now(),
            'consecutive_failures': 0,
            'consecutive_job_failures': 0
        }
        mock_backend_router.health_monitor.health_history = {}
        mock_backend_router.cost_optimizer.get_total_cost.return_value = 25.75
        
        result = service.get_backend_details("test_backend")
        
        assert result['name'] == 'test_backend'
        assert result['status'] == 'healthy'
        assert result['health_metrics']['uptime_percentage'] == 98.5
        assert result['health_metrics']['total_checks'] == 200
        assert result['capabilities']['supports_gpu'] is True
        assert result['cost_metrics']['total_cost'] == 25.75
        assert result['configuration_status'] == 'configured'
    
    def test_get_backend_details_not_found(self, service, mock_backend_router):
        """Test getting details for non-existent backend."""
        mock_backend_router.get_backend.return_value = None
        
        with pytest.raises(ValueError, match="Backend 'nonexistent' not found"):
            service.get_backend_details("nonexistent")
    
    def test_get_backend_details_with_error_history(self, service, mock_backend_router, mock_backend):
        """Test getting details for backend with error history."""
        error_timestamp = datetime.now()
        mock_backend_router.get_backend.return_value = mock_backend
        mock_backend_router.get_backend_status.return_value = {
            "test_backend": HealthStatus.DEGRADED
        }
        mock_backend_router.health_monitor.get_health_metrics.return_value = {
            'uptime_percentage': 75.0,
            'total_checks': 100,
            'healthy_checks': 75,
            'failure_rate': 25.0,
            'last_check': datetime.now(),
            'consecutive_failures': 2,
            'consecutive_job_failures': 1
        }
        mock_backend_router.health_monitor.health_history = {
            'test_backend': [
                {'status': HealthStatus.HEALTHY, 'timestamp': datetime.now()},
                {'status': HealthStatus.DEGRADED, 'timestamp': error_timestamp}
            ]
        }
        mock_backend_router.cost_optimizer.get_total_cost.return_value = 15.0
        
        result = service.get_backend_details("test_backend")
        
        assert result['status'] == 'degraded'
        assert result['health_metrics']['last_error'] is not None
        assert 'degraded' in result['health_metrics']['last_error'].lower()
        assert result['health_metrics']['last_error_timestamp'] is not None
    
    def test_trigger_health_check_healthy(self, service, mock_backend_router, mock_backend):
        """Test triggering health check for healthy backend."""
        mock_backend_router.get_backend.return_value = mock_backend
        mock_backend_router.health_monitor.check_backend_health.return_value = HealthStatus.HEALTHY
        
        result = service.trigger_health_check("test_backend")
        
        assert result['backend_name'] == 'test_backend'
        assert result['status'] == 'healthy'
        assert 'timestamp' in result
        assert 'healthy' in result['message'].lower()
        mock_backend_router.health_monitor.check_backend_health.assert_called_once_with(mock_backend)
    
    def test_trigger_health_check_unhealthy(self, service, mock_backend_router, mock_backend):
        """Test triggering health check for unhealthy backend."""
        mock_backend_router.get_backend.return_value = mock_backend
        mock_backend_router.health_monitor.check_backend_health.return_value = HealthStatus.UNHEALTHY
        
        result = service.trigger_health_check("test_backend")
        
        assert result['backend_name'] == 'test_backend'
        assert result['status'] == 'unhealthy'
        assert 'unhealthy' in result['message'].lower()
    
    def test_trigger_health_check_degraded(self, service, mock_backend_router, mock_backend):
        """Test triggering health check for degraded backend."""
        mock_backend_router.get_backend.return_value = mock_backend
        mock_backend_router.health_monitor.check_backend_health.return_value = HealthStatus.DEGRADED
        
        result = service.trigger_health_check("test_backend")
        
        assert result['backend_name'] == 'test_backend'
        assert result['status'] == 'degraded'
        assert 'degraded' in result['message'].lower()
    
    def test_trigger_health_check_not_found(self, service, mock_backend_router):
        """Test triggering health check for non-existent backend."""
        mock_backend_router.get_backend.return_value = None
        
        with pytest.raises(ValueError, match="Backend 'nonexistent' not found"):
            service.trigger_health_check("nonexistent")
    
    def test_get_backends_status_multiple_backends(self, service, mock_backend_router):
        """Test getting status for multiple backends with different health states."""
        backend1 = Mock(spec=Backend)
        backend1.id = "backend1"
        backend1.capabilities = BackendCapabilities(
            supported_templates=["template1"],
            max_concurrent_jobs=3,
            max_job_duration_minutes=30,
            supports_gpu=False,
            supports_batch=True,
            cost_per_hour=0.5,
            free_tier_limits=None
        )
        
        backend2 = Mock(spec=Backend)
        backend2.id = "backend2"
        backend2.capabilities = BackendCapabilities(
            supported_templates=["template2"],
            max_concurrent_jobs=10,
            max_job_duration_minutes=120,
            supports_gpu=True,
            supports_batch=False,
            cost_per_hour=2.0,
            free_tier_limits=None
        )
        
        mock_backend_router.list_backends.return_value = [backend1, backend2]
        mock_backend_router.get_backend_status.return_value = {
            "backend1": HealthStatus.HEALTHY,
            "backend2": HealthStatus.UNHEALTHY
        }
        
        def get_metrics(backend_id):
            if backend_id == "backend1":
                return {
                    'uptime_percentage': 100.0,
                    'total_checks': 50,
                    'healthy_checks': 50,
                    'failure_rate': 0.0,
                    'last_check': datetime.now(),
                    'consecutive_failures': 0,
                    'consecutive_job_failures': 0
                }
            else:
                return {
                    'uptime_percentage': 60.0,
                    'total_checks': 50,
                    'healthy_checks': 30,
                    'failure_rate': 40.0,
                    'last_check': datetime.now(),
                    'consecutive_failures': 5,
                    'consecutive_job_failures': 3
                }
        
        mock_backend_router.health_monitor.get_health_metrics.side_effect = get_metrics
        mock_backend_router.cost_optimizer.get_total_cost.side_effect = lambda x: 5.0 if x == "backend1" else 20.0
        
        result = service.get_backends_status()
        
        assert len(result) == 2
        assert result[0]['name'] == 'backend1'
        assert result[0]['status'] == 'healthy'
        assert result[1]['name'] == 'backend2'
        assert result[1]['status'] == 'unhealthy'
