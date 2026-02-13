"""
Unit tests for JobSubmissionTab component.
"""

import pytest
from unittest.mock import Mock, MagicMock
from gui.components.job_submission_tab import JobSubmissionTab
from gui.services.job_service import JobService
from gui.services.template_service import TemplateService


@pytest.fixture
def mock_job_service():
    """Create mock job service."""
    service = Mock(spec=JobService)
    service.submit_job = Mock(return_value="job-123")
    return service


@pytest.fixture
def mock_template_service():
    """Create mock template service."""
    service = Mock(spec=TemplateService)
    service.get_templates = Mock(return_value=[
        {'name': 'test_template', 'category': 'test', 'description': 'Test template'}
    ])
    service.get_template_metadata = Mock(return_value={
        'name': 'test_template',
        'category': 'test',
        'description': 'Test template',
        'version': '1.0.0',
        'inputs': [
            {
                'name': 'input1',
                'type': 'text',
                'description': 'Test input',
                'required': True,
                'default': None,
                'options': None
            }
        ],
        'outputs': [
            {
                'name': 'output1',
                'type': 'text',
                'description': 'Test output'
            }
        ],
        'gpu_required': False,
        'gpu_type': None,
        'memory_mb': 512,
        'timeout_sec': 300,
        'supported_backends': ['colab', 'kaggle']
    })
    return service


@pytest.fixture
def job_submission_tab(mock_job_service, mock_template_service):
    """Create JobSubmissionTab instance."""
    return JobSubmissionTab(mock_job_service, mock_template_service)


class TestJobSubmissionTab:
    """Test suite for JobSubmissionTab."""
    
    def test_initialization(self, job_submission_tab):
        """Test that JobSubmissionTab initializes correctly."""
        assert job_submission_tab.job_service is not None
        assert job_submission_tab.template_service is not None
    
    def test_get_template_choices(self, job_submission_tab, mock_template_service):
        """Test retrieving template choices for dropdown."""
        choices = job_submission_tab._get_template_choices()
        
        assert len(choices) == 1
        assert choices[0] == 'test_template'
        mock_template_service.get_templates.assert_called_once()
    
    def test_get_backend_choices(self, job_submission_tab):
        """Test retrieving backend choices for dropdown."""
        choices = job_submission_tab._get_backend_choices()
        
        # Should always include "auto" as first option
        assert len(choices) > 0
        assert choices[0] == 'auto'
        
        # Should include common backend types
        assert any(backend in choices for backend in ['colab', 'kaggle', 'huggingface', 'modal'])
    
    def test_generate_template_docs(self, job_submission_tab, mock_template_service):
        """Test generating template documentation."""
        metadata = mock_template_service.get_template_metadata('test_template')
        docs = job_submission_tab._generate_template_docs(metadata)
        
        assert 'test_template' in docs
        assert 'Test template' in docs
        assert 'input1' in docs
        assert 'output1' in docs
        assert 'colab, kaggle' in docs
    
    def test_validate_inputs_from_metadata_success(self, job_submission_tab, mock_template_service):
        """Test input validation with valid inputs."""
        metadata = mock_template_service.get_template_metadata('test_template')
        inputs = {'input1': 'test value'}
        
        errors = job_submission_tab._validate_inputs_from_metadata(metadata, inputs)
        
        assert len(errors) == 0
    
    def test_validate_inputs_from_metadata_missing_required(self, job_submission_tab, mock_template_service):
        """Test input validation with missing required field."""
        metadata = mock_template_service.get_template_metadata('test_template')
        inputs = {}
        
        errors = job_submission_tab._validate_inputs_from_metadata(metadata, inputs)
        
        assert len(errors) > 0
        assert any('input1' in err and 'missing' in err.lower() for err in errors)
    
    def test_on_submit_job_success(self, job_submission_tab, mock_job_service):
        """Test successful job submission."""
        template_name = 'test_template'
        backend = 'colab'
        routing_strategy = 'cost-optimized'
        inputs = {'input1': 'test value'}
        
        job_id_output, status_message, loading_indicator = job_submission_tab.on_submit_job(
            template_name, backend, routing_strategy, inputs
        )
        
        # Verify job service was called
        mock_job_service.submit_job.assert_called_once_with(
            template_name='test_template',
            inputs={'input1': 'test value'},
            backend='colab',
            routing_strategy='cost-optimized'
        )
    
    def test_on_submit_job_no_template(self, job_submission_tab):
        """Test job submission without template selection."""
        job_id_output, status_message, loading_indicator = job_submission_tab.on_submit_job(
            None, 'auto', 'cost-optimized', {}
        )
        
        # Should return error message
        assert status_message.value is not None
        assert 'Error' in status_message.value or 'select a template' in status_message.value.lower()
    
    def test_on_submit_job_with_auto_backend(self, job_submission_tab, mock_job_service):
        """Test job submission with automatic backend routing."""
        template_name = 'test_template'
        backend = 'auto'
        routing_strategy = 'cost-optimized'
        inputs = {'input1': 'test value'}
        
        job_id_output, status_message, loading_indicator = job_submission_tab.on_submit_job(
            template_name, backend, routing_strategy, inputs
        )
        
        # Verify job service was called with backend=None for automatic routing
        mock_job_service.submit_job.assert_called_once_with(
            template_name='test_template',
            inputs={'input1': 'test value'},
            backend=None,  # "auto" should be converted to None
            routing_strategy='cost-optimized'
        )
    
    def test_on_submit_job_with_specific_backend(self, job_submission_tab, mock_job_service):
        """Test job submission with specific backend selection."""
        template_name = 'test_template'
        backend = 'kaggle'
        routing_strategy = 'round-robin'
        inputs = {'input1': 'test value'}
        
        job_id_output, status_message, loading_indicator = job_submission_tab.on_submit_job(
            template_name, backend, routing_strategy, inputs
        )
        
        # Verify job service was called with the specific backend
        mock_job_service.submit_job.assert_called_once_with(
            template_name='test_template',
            inputs={'input1': 'test value'},
            backend='kaggle',
            routing_strategy='round-robin'
        )
