"""
Unit tests for BackendStatusTab component.
"""

import pytest
from unittest.mock import Mock
from datetime import datetime
from gui.components.backend_status_tab import BackendStatusTab
from gui.services.backend_monitor_service import BackendMonitorService


@pytest.fixture
def mock_backend_monitor():
    """Create mock backend monitor service."""
    service = Mock(spec=BackendMonitorService)
    
    # Mock get_backends_status
    service.get_backends_status = Mock(return_value=[
        {
            'name': 'colab',
            'status': 'healthy',
            'uptime_percentage': 95.5,
            'avg_response_time': 1.2,
            'jobs_executed': 42,
            'last_health_check': datetime.now(),
            'last_error': None,
            'capabilities': {
                'supported_templates': ['template1', 'template2'],
                'max_concurrent_jobs': 5,
                'max_job_duration_minutes': 60,
                'supports_gpu': True,
                'supports_batch': False,
                'cost_per_hour': 0.5
            },
            'cost_total': 12.50
        },
        {
            'name': 'kaggle',
            'status': 'degraded',
            'uptime_percentage': 75.0,
            'avg_response_time': 2.5,
            'jobs_executed': 15,
            'last_health_check': datetime.now(),
            'last_error': 'Backend marked as degraded',
            'capabilities': {
                'supported_templates': ['template1'],
                'max_concurrent_jobs': 3,
                'max_job_duration_minutes': 30,
                'supports_gpu': False,
                'supports_batch': True,
                'cost_per_hour': 0.0
            },
            'cost_total': 0.0
        }
    ])
    
    # Mock get_backend_details
    service.get_backend_details = Mock(return_value={
        'name': 'colab',
        'status': 'healthy',
        'health_metrics': {
            'uptime_percentage': 95.5,
            'total_checks': 100,
            'healthy_checks': 95,
            'failure_rate': 5.0,
            'last_check': datetime.now().isoformat(),
            'consecutive_failures': 0,
            'consecutive_job_failures': 0,
            'last_error': None,
            'last_error_timestamp': None
        },
        'capabilities': {
            'supported_templates': ['template1', 'template2'],
            'max_concurrent_jobs': 5,
            'max_job_duration_minutes': 60,
            'supports_gpu': True,
            'supports_batch': False,
            'cost_per_hour': 0.5,
            'free_tier_limits': None
        },
        'cost_metrics': {
            'total_cost': 12.50
        },
        'configuration_status': 'configured'
    })
    
    # Mock trigger_health_check
    service.trigger_health_check = Mock(return_value={
        'backend_name': 'colab',
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'message': 'Backend colab is healthy'
    })
    
    return service


@pytest.fixture
def backend_status_tab(mock_backend_monitor):
    """Create BackendStatusTab instance."""
    return BackendStatusTab(mock_backend_monitor)


class TestBackendStatusTab:
    """Test suite for BackendStatusTab."""
    
    def test_initialization(self, backend_status_tab):
        """Test that BackendStatusTab initializes correctly."""
        assert backend_status_tab.backend_monitor is not None
    
    def test_get_initial_backends(self, backend_status_tab, mock_backend_monitor):
        """Test retrieving initial backend list."""
        df = backend_status_tab._get_initial_backends()
        
        assert len(df) == 2
        assert 'colab' in df['Backend'].values
        assert 'kaggle' in df['Backend'].values
        mock_backend_monitor.get_backends_status.assert_called_once()
    
    def test_backends_to_dataframe(self, backend_status_tab):
        """Test converting backend list to DataFrame."""
        backends = [
            {
                'name': 'test_backend',
                'status': 'healthy',
                'uptime_percentage': 99.9,
                'avg_response_time': 0.5,
                'jobs_executed': 100
            }
        ]
        
        df = backend_status_tab._backends_to_dataframe(backends)
        
        assert len(df) == 1
        assert df.iloc[0]['Backend'] == 'test_backend'
        assert df.iloc[0]['Status'] == 'healthy'
        assert df.iloc[0]['Uptime %'] == 99.9
        assert df.iloc[0]['Avg Response Time'] == 0.5
        assert df.iloc[0]['Jobs Executed'] == 100
    
    def test_backends_to_dataframe_empty(self, backend_status_tab):
        """Test converting empty backend list to DataFrame."""
        df = backend_status_tab._backends_to_dataframe([])
        
        assert len(df) == 0
        assert list(df.columns) == ["Backend", "Status", "Uptime %", "Avg Response Time", "Jobs Executed"]
    
    def test_on_refresh_backends(self, backend_status_tab, mock_backend_monitor):
        """Test refreshing backend list."""
        df = backend_status_tab.on_refresh_backends()
        
        assert len(df) == 2
        mock_backend_monitor.get_backends_status.assert_called()
    
    def test_on_load_backend_details_success(self, backend_status_tab, mock_backend_monitor):
        """Test loading backend details successfully."""
        backend_name = 'colab'
        
        details, metrics_md = backend_status_tab.on_load_backend_details(backend_name)
        
        assert details['name'] == 'colab'
        assert details['status'] == 'healthy'
        assert 'Health Status' in metrics_md
        assert 'Uptime Metrics' in metrics_md
        assert 'Capabilities' in metrics_md
        assert 'Cost Tracking' in metrics_md
        mock_backend_monitor.get_backend_details.assert_called_once_with('colab')
    
    def test_on_load_backend_details_empty_name(self, backend_status_tab):
        """Test loading backend details with empty name."""
        details, metrics_md = backend_status_tab.on_load_backend_details('')
        
        assert details == {}
        assert 'Select a backend' in metrics_md
    
    def test_on_load_backend_details_not_found(self, backend_status_tab, mock_backend_monitor):
        """Test loading backend details for non-existent backend."""
        mock_backend_monitor.get_backend_details.side_effect = ValueError("Backend 'invalid' not found")
        
        details, metrics_md = backend_status_tab.on_load_backend_details('invalid')
        
        assert details == {}
        assert 'not found' in metrics_md
    
    def test_on_trigger_health_check_success(self, backend_status_tab, mock_backend_monitor):
        """Test triggering manual health check successfully."""
        backend_name = 'colab'
        
        message, updated_table = backend_status_tab.on_trigger_health_check(backend_name)
        
        assert 'Health check completed' in message
        assert 'colab' in message
        assert 'healthy' in message
        assert len(updated_table) == 2  # Should refresh the table
        mock_backend_monitor.trigger_health_check.assert_called_once_with('colab')
    
    def test_on_trigger_health_check_empty_name(self, backend_status_tab):
        """Test triggering health check with empty name."""
        message, updated_table = backend_status_tab.on_trigger_health_check('')
        
        assert 'select a backend' in message.lower()
    
    def test_on_trigger_health_check_not_found(self, backend_status_tab, mock_backend_monitor):
        """Test triggering health check for non-existent backend."""
        mock_backend_monitor.trigger_health_check.side_effect = ValueError("Backend 'invalid' not found")
        
        message, updated_table = backend_status_tab.on_trigger_health_check('invalid')
        
        assert 'not found' in message
    
    def test_format_health_metrics(self, backend_status_tab):
        """Test formatting health metrics as markdown."""
        backend_details = {
            'status': 'healthy',
            'health_metrics': {
                'uptime_percentage': 95.5,
                'total_checks': 100,
                'healthy_checks': 95,
                'failure_rate': 4.5,
                'last_check': '2024-01-01T12:00:00',
                'consecutive_failures': 0,
                'consecutive_job_failures': 0,
                'last_error': 'Test error',
                'last_error_timestamp': '2024-01-01T11:00:00'
            },
            'capabilities': {
                'supports_gpu': True,
                'supports_batch': False,
                'max_concurrent_jobs': 5,
                'max_job_duration_minutes': 60,
                'cost_per_hour': 0.5
            },
            'cost_metrics': {
                'total_cost': 12.50
            },
            'configuration_status': 'configured'
        }
        
        metrics_md = backend_status_tab._format_health_metrics(backend_details)
        
        assert 'Health Status' in metrics_md
        assert 'healthy' in metrics_md
        assert '95.5' in metrics_md
        assert 'Uptime Metrics' in metrics_md
        assert 'Capabilities' in metrics_md
        assert 'GPU Support:**' in metrics_md or 'GPU Support: Yes' in metrics_md
        assert 'Cost Tracking' in metrics_md
        assert '$12.50' in metrics_md
        assert 'Last Error' in metrics_md
        assert 'Test error' in metrics_md
