"""
Unit tests for Google Colab backend implementation.

Tests the ColabBackend class functionality including initialization,
health checks, template support, and cost estimation.
"""

import pytest
import sys
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

# Mock Google API modules before importing ColabBackend
sys.modules['google'] = MagicMock()
sys.modules['google.oauth2'] = MagicMock()
sys.modules['google.oauth2.credentials'] = MagicMock()
sys.modules['google.auth'] = MagicMock()
sys.modules['google.auth.transport'] = MagicMock()
sys.modules['google.auth.transport.requests'] = MagicMock()
sys.modules['googleapiclient'] = MagicMock()
sys.modules['googleapiclient.discovery'] = MagicMock()
sys.modules['googleapiclient.http'] = MagicMock()

from notebook_ml_orchestrator.core.backends.colab_backend import ColabBackend, SUPPORTED_TEMPLATES
from notebook_ml_orchestrator.core.models import (
    BackendType, HealthStatus, ResourceEstimate, JobStatus
)
from notebook_ml_orchestrator.core.interfaces import Job, MLTemplate
from notebook_ml_orchestrator.core.exceptions import (
    BackendConnectionError, BackendNotAvailableError, JobExecutionError, 
    JobTimeoutError, BackendAuthenticationError
)


class TestColabBackendInitialization:
    """Test Colab backend initialization."""
    
    def test_initialization_with_default_config(self):
        """Test backend initializes with default configuration."""
        backend = ColabBackend()
        
        assert backend.id == "colab"
        assert backend.name == "Google Colab"
        assert backend.type == BackendType.COLAB
        assert backend.default_timeout == 3600
        assert backend.enable_gpu is True
        assert backend.drive_folder == 'orchestrator_jobs'
        assert backend.capabilities.supports_gpu is True
        assert backend.capabilities.supported_templates == SUPPORTED_TEMPLATES
    
    def test_initialization_with_custom_config(self):
        """Test backend initializes with custom configuration."""
        config = {
            'credentials': {
                'client_id': 'test_client_id',
                'client_secret': 'test_client_secret',
                'refresh_token': 'test_refresh_token'
            },
            'options': {
                'timeout': 7200,
                'enable_gpu': False,
                'drive_folder': 'custom_folder'
            }
        }
        
        backend = ColabBackend(backend_id="colab-custom", config=config)
        
        assert backend.id == "colab-custom"
        assert backend.default_timeout == 7200
        assert backend.enable_gpu is False
        assert backend.drive_folder == 'custom_folder'
        assert backend.credentials['client_id'] == 'test_client_id'
        assert backend.credentials['client_secret'] == 'test_client_secret'
        assert backend.credentials['refresh_token'] == 'test_refresh_token'
    
    def test_capabilities_set_correctly(self):
        """Test backend capabilities are set correctly."""
        backend = ColabBackend()
        
        assert backend.capabilities.supports_gpu is True
        assert backend.capabilities.max_concurrent_jobs == 1
        assert backend.capabilities.max_job_duration_minutes == 720
        assert backend.capabilities.supports_batch is False
        assert backend.capabilities.cost_per_hour == 0.0
        assert 'model-training' in backend.capabilities.supported_templates
        assert 'data-processing' in backend.capabilities.supported_templates


class TestColabBackendAuthentication:
    """Test Colab backend authentication."""
    
    @patch('google.auth.transport.requests.Request')
    @patch('googleapiclient.discovery.build')
    @patch('google.oauth2.credentials.Credentials')
    def test_authentication_success(self, mock_credentials, mock_build, mock_request):
        """Test successful OAuth authentication."""
        config = {
            'credentials': {
                'client_id': 'test_client_id',
                'client_secret': 'test_client_secret',
                'refresh_token': 'test_refresh_token'
            }
        }
        
        # Mock credentials and Drive service
        mock_creds_instance = MagicMock()
        mock_credentials.return_value = mock_creds_instance
        
        mock_drive_service = MagicMock()
        mock_drive_service.files().list().execute.return_value = {'files': []}
        mock_build.return_value = mock_drive_service
        
        backend = ColabBackend(config=config)
        backend._authenticate()
        
        assert backend._authenticated is True
        mock_creds_instance.refresh.assert_called_once()
        mock_build.assert_called_once_with('drive', 'v3', credentials=mock_creds_instance)
    
    def test_authentication_missing_credentials(self):
        """Test authentication fails with missing credentials."""
        backend = ColabBackend()
        
        with pytest.raises(BackendAuthenticationError) as exc_info:
            backend._authenticate()
        
        assert "not configured" in str(exc_info.value)
    
    @patch('google.oauth2.credentials.Credentials')
    def test_authentication_token_refresh_failure(self, mock_credentials):
        """Test authentication fails when token refresh fails."""
        config = {
            'credentials': {
                'client_id': 'test_client_id',
                'client_secret': 'test_client_secret',
                'refresh_token': 'invalid_token'
            }
        }
        
        mock_creds_instance = MagicMock()
        mock_creds_instance.refresh.side_effect = Exception("Invalid grant")
        mock_credentials.return_value = mock_creds_instance
        
        backend = ColabBackend(config=config)
        
        with pytest.raises(BackendAuthenticationError) as exc_info:
            backend._authenticate()
        
        assert "Invalid credentials" in str(exc_info.value)
    
    @patch('google.auth.transport.requests.Request')
    @patch('googleapiclient.discovery.build')
    @patch('google.oauth2.credentials.Credentials')
    def test_token_expiration_handling(self, mock_credentials, mock_build, mock_request):
        """Test OAuth token expiration is handled with refresh."""
        config = {
            'credentials': {
                'client_id': 'test_client_id',
                'client_secret': 'test_client_secret',
                'refresh_token': 'test_refresh_token'
            }
        }
        
        mock_creds_instance = MagicMock()
        mock_credentials.return_value = mock_creds_instance
        
        mock_drive_service = MagicMock()
        mock_drive_service.files().list().execute.return_value = {'files': []}
        mock_build.return_value = mock_drive_service
        
        backend = ColabBackend(config=config)
        
        # Simulate token expiration error
        error = Exception("Token expired")
        result = backend._handle_token_expiration(error)
        
        assert result is True
        assert backend._authenticated is True


class TestColabBackendHealthCheck:
    """Test Colab backend health checks."""
    
    @patch('google.auth.transport.requests.Request')
    @patch('googleapiclient.discovery.build')
    @patch('google.oauth2.credentials.Credentials')
    def test_health_check_healthy(self, mock_credentials, mock_build, mock_request):
        """Test health check returns HEALTHY when authentication succeeds."""
        config = {
            'credentials': {
                'client_id': 'test_client_id',
                'client_secret': 'test_client_secret',
                'refresh_token': 'test_refresh_token'
            }
        }
        
        mock_creds_instance = MagicMock()
        mock_credentials.return_value = mock_creds_instance
        
        mock_drive_service = MagicMock()
        mock_drive_service.files().list().execute.return_value = {'files': []}
        mock_build.return_value = mock_drive_service
        
        backend = ColabBackend(config=config)
        status = backend.check_health()
        
        assert status == HealthStatus.HEALTHY
        assert backend.health_status == HealthStatus.HEALTHY
    
    def test_health_check_unhealthy_missing_credentials(self):
        """Test health check returns UNHEALTHY with missing credentials."""
        backend = ColabBackend()
        status = backend.check_health()
        
        assert status == HealthStatus.UNHEALTHY
        assert backend.health_status == HealthStatus.UNHEALTHY
    
    @patch('google.auth.transport.requests.Request')
    @patch('googleapiclient.discovery.build')
    @patch('google.oauth2.credentials.Credentials')
    def test_health_check_degraded_quota_exceeded(self, mock_credentials, mock_build, mock_request):
        """Test health check returns DEGRADED when Drive quota is exceeded during file listing."""
        config = {
            'credentials': {
                'client_id': 'test_client_id',
                'client_secret': 'test_client_secret',
                'refresh_token': 'test_refresh_token'
            }
        }
        
        mock_creds_instance = MagicMock()
        mock_credentials.return_value = mock_creds_instance
        
        mock_drive_service = MagicMock()
        # First call (during authentication) succeeds, second call (during health check) fails with quota error
        mock_drive_service.files().list().execute.side_effect = [
            {'files': []},  # Authentication succeeds
            Exception("Storage quota exceeded")  # Health check fails
        ]
        mock_build.return_value = mock_drive_service
        
        backend = ColabBackend(config=config)
        status = backend.check_health()
        
        assert status == HealthStatus.DEGRADED
        assert backend.health_status == HealthStatus.DEGRADED


class TestColabBackendTemplateSupport:
    """Test Colab backend template support."""
    
    def test_supports_template_valid(self):
        """Test supports_template returns True for supported templates."""
        backend = ColabBackend()
        
        assert backend.supports_template('model-training') is True
        assert backend.supports_template('data-processing') is True
        assert backend.supports_template('image-classification') is True
        assert backend.supports_template('text-generation') is True
        assert backend.supports_template('embeddings') is True
    
    def test_supports_template_invalid(self):
        """Test supports_template returns False for unsupported templates."""
        backend = ColabBackend()
        
        assert backend.supports_template('unsupported-template') is False
        assert backend.supports_template('batch-inference') is False


class TestColabBackendCostEstimation:
    """Test Colab backend cost estimation."""
    
    def test_estimate_cost_returns_zero(self):
        """Test estimate_cost returns 0.0 for free tier."""
        backend = ColabBackend()
        
        resource_estimate = ResourceEstimate(
            cpu_cores=2,
            memory_gb=8,
            requires_gpu=True,
            estimated_duration_minutes=60
        )
        
        cost = backend.estimate_cost(resource_estimate)
        
        assert cost == 0.0
    
    def test_estimate_cost_with_gpu(self):
        """Test estimate_cost returns 0.0 even with GPU requirements."""
        backend = ColabBackend()
        
        resource_estimate = ResourceEstimate(
            cpu_cores=4,
            memory_gb=16,
            requires_gpu=True,
            gpu_memory_gb=16.0,
            estimated_duration_minutes=120
        )
        
        cost = backend.estimate_cost(resource_estimate)
        
        assert cost == 0.0


class TestColabBackendQueueLength:
    """Test Colab backend queue length."""
    
    def test_get_queue_length_returns_zero(self):
        """Test get_queue_length returns 0 for Colab."""
        backend = ColabBackend()
        
        queue_length = backend.get_queue_length()
        
        assert queue_length == 0


class TestColabBackendJobExecution:
    """Test Colab backend job execution."""
    
    @patch('googleapiclient.http.MediaInMemoryUpload')
    @patch('google.auth.transport.requests.Request')
    @patch('googleapiclient.discovery.build')
    @patch('google.oauth2.credentials.Credentials')
    def test_execute_job_success(self, mock_credentials, mock_build, mock_request, mock_media):
        """Test successful job execution."""
        config = {
            'credentials': {
                'client_id': 'test_client_id',
                'client_secret': 'test_client_secret',
                'refresh_token': 'test_refresh_token'
            }
        }
        
        mock_creds_instance = MagicMock()
        mock_credentials.return_value = mock_creds_instance
        
        mock_drive_service = MagicMock()
        mock_drive_service.files().list().execute.return_value = {'files': []}
        mock_drive_service.files().create().execute.return_value = {'id': 'test_notebook_id'}
        mock_build.return_value = mock_drive_service
        
        backend = ColabBackend(config=config)
        
        job = Job(
            id='test_job_id',
            template_name='model-training',
            inputs={'data': 'test_data'}
        )
        
        template = Mock(spec=MLTemplate)
        
        result = backend.execute_job(job, template)
        
        assert result.success is True
        assert result.backend_used == 'colab'
        assert 'notebook_id' in result.metadata
        assert result.metadata['notebook_id'] == 'test_notebook_id'
    
    def test_execute_job_authentication_failure(self):
        """Test job execution fails with authentication error."""
        backend = ColabBackend()
        
        job = Job(
            id='test_job_id',
            template_name='model-training',
            inputs={'data': 'test_data'}
        )
        
        template = Mock(spec=MLTemplate)
        
        with pytest.raises(BackendAuthenticationError):
            backend.execute_job(job, template)
    
    @patch('google.auth.transport.requests.Request')
    @patch('googleapiclient.discovery.build')
    @patch('google.oauth2.credentials.Credentials')
    def test_execute_job_drive_quota_exceeded(self, mock_credentials, mock_build, mock_request):
        """Test job execution fails when Drive quota is exceeded."""
        config = {
            'credentials': {
                'client_id': 'test_client_id',
                'client_secret': 'test_client_secret',
                'refresh_token': 'test_refresh_token'
            }
        }
        
        mock_creds_instance = MagicMock()
        mock_credentials.return_value = mock_creds_instance
        
        mock_drive_service = MagicMock()
        mock_drive_service.files().list().execute.return_value = {'files': []}
        mock_drive_service.files().create().execute.side_effect = Exception("Storage quota exceeded")
        mock_build.return_value = mock_drive_service
        
        backend = ColabBackend(config=config)
        
        job = Job(
            id='test_job_id',
            template_name='model-training',
            inputs={'data': 'test_data'}
        )
        
        template = Mock(spec=MLTemplate)
        
        with pytest.raises(BackendNotAvailableError) as exc_info:
            backend.execute_job(job, template)
        
        assert "quota" in str(exc_info.value).lower()
    
    @patch('google.auth.transport.requests.Request')
    @patch('googleapiclient.discovery.build')
    @patch('google.oauth2.credentials.Credentials')
    def test_execute_job_gpu_unavailable(self, mock_credentials, mock_build, mock_request):
        """Test job execution fails when GPU is unavailable."""
        config = {
            'credentials': {
                'client_id': 'test_client_id',
                'client_secret': 'test_client_secret',
                'refresh_token': 'test_refresh_token'
            }
        }
        
        mock_creds_instance = MagicMock()
        mock_credentials.return_value = mock_creds_instance
        
        mock_drive_service = MagicMock()
        mock_drive_service.files().list().execute.return_value = {'files': []}
        mock_drive_service.files().create().execute.side_effect = Exception("GPU not available")
        mock_build.return_value = mock_drive_service
        
        backend = ColabBackend(config=config)
        
        job = Job(
            id='test_job_id',
            template_name='model-training',
            inputs={'data': 'test_data'},
            metadata={'enable_gpu': True}
        )
        
        template = Mock(spec=MLTemplate)
        
        with pytest.raises(BackendNotAvailableError) as exc_info:
            backend.execute_job(job, template)
        
        assert "gpu" in str(exc_info.value).lower()
    
    @patch('google.auth.transport.requests.Request')
    @patch('googleapiclient.discovery.build')
    @patch('google.oauth2.credentials.Credentials')
    def test_execute_job_runtime_disconnection(self, mock_credentials, mock_build, mock_request):
        """Test job execution handles runtime disconnection."""
        config = {
            'credentials': {
                'client_id': 'test_client_id',
                'client_secret': 'test_client_secret',
                'refresh_token': 'test_refresh_token'
            }
        }
        
        mock_creds_instance = MagicMock()
        mock_credentials.return_value = mock_creds_instance
        
        mock_drive_service = MagicMock()
        mock_drive_service.files().list().execute.return_value = {'files': []}
        mock_drive_service.files().create().execute.side_effect = Exception("Runtime disconnected")
        mock_build.return_value = mock_drive_service
        
        backend = ColabBackend(config=config)
        
        job = Job(
            id='test_job_id',
            template_name='model-training',
            inputs={'data': 'test_data'}
        )
        
        template = Mock(spec=MLTemplate)
        
        with pytest.raises(BackendConnectionError) as exc_info:
            backend.execute_job(job, template)
        
        assert "disconnect" in str(exc_info.value).lower()


class TestColabBackendNotebookCreation:
    """Test Colab backend notebook creation."""
    
    def test_create_notebook_structure(self):
        """Test notebook is created with correct structure."""
        backend = ColabBackend()
        
        job = Job(
            id='test_job_id',
            template_name='model-training',
            inputs={'data': 'test_data', 'epochs': 10}
        )
        
        template = Mock(spec=MLTemplate)
        
        notebook_content = backend._create_notebook(job, template, enable_gpu=True)
        
        import json
        notebook = json.loads(notebook_content)
        
        assert notebook['nbformat'] == 4
        assert notebook['nbformat_minor'] == 0
        assert 'cells' in notebook
        assert len(notebook['cells']) > 0
        assert notebook['metadata']['accelerator'] == 'GPU'
        
        # Check for GPU check cell
        cell_sources = [' '.join(cell['source']) for cell in notebook['cells']]
        assert any('torch.cuda.is_available()' in source for source in cell_sources)
    
    def test_create_notebook_without_gpu(self):
        """Test notebook is created without GPU configuration."""
        backend = ColabBackend()
        
        job = Job(
            id='test_job_id',
            template_name='data-processing',
            inputs={'data': 'test_data'}
        )
        
        template = Mock(spec=MLTemplate)
        
        notebook_content = backend._create_notebook(job, template, enable_gpu=False)
        
        import json
        notebook = json.loads(notebook_content)
        
        assert notebook['metadata']['accelerator'] == 'None'
   
