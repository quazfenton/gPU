"""
Tests for Orchestrator job submission with template integration.

This module tests the integration of templates with the job submission flow,
including template validation, resource estimation, and backend routing.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from notebook_ml_orchestrator.orchestrator import Orchestrator
from notebook_ml_orchestrator.core.interfaces import Job, Backend
from notebook_ml_orchestrator.core.models import (
    BackendType, HealthStatus, ResourceEstimate, JobResult, JobStatus
)
from notebook_ml_orchestrator.core.exceptions import (
    TemplateNotFoundError, JobValidationError, BackendNotAvailableError
)
from templates.base import Template, InputField, OutputField, RouteType


class MockTemplate(Template):
    """Mock template for testing."""
    
    name = "mock-template"
    category = "Test"
    description = "A mock template for testing"
    version = "1.0.0"
    
    inputs = [
        InputField(name="text", type="text", description="Input text", required=True),
        InputField(name="number", type="number", description="Input number", required=False, default=42)
    ]
    
    outputs = [
        OutputField(name="result", type="text", description="Output result")
    ]
    
    routing = [RouteType.LOCAL, RouteType.MODAL]
    gpu_required = False
    memory_mb = 512
    timeout_sec = 60
    
    def run(self, **kwargs):
        return {"result": f"Processed: {kwargs.get('text', '')}"}


class TestOrchestratorJobSubmission:
    """Test suite for orchestrator job submission with templates."""
    
    @pytest.fixture
    def orchestrator(self, tmp_path):
        """Create orchestrator instance with mocked components."""
        db_path = str(tmp_path / "test.db")
        
        with patch('notebook_ml_orchestrator.orchestrator.TemplateRegistry') as mock_registry_class:
            # Create mock registry instance
            mock_registry = Mock()
            mock_registry.discover_templates.return_value = 1
            mock_registry.get_registry_stats.return_value = {
                'templates_by_category': {'Test': 1},
                'failed_templates': 0,
                'failed_template_list': []
            }
            mock_registry_class.return_value = mock_registry
            
            orchestrator = Orchestrator(db_path=db_path, templates_dir="templates")
            orchestrator.template_registry = mock_registry
            
            yield orchestrator
            
            orchestrator.shutdown()
    
    def test_submit_job_with_valid_template(self, orchestrator):
        """Test submitting a job with a valid template."""
        # Setup mock template
        mock_template = MockTemplate()
        orchestrator.template_registry.get_template.return_value = mock_template
        orchestrator.template_registry.list_templates.return_value = [mock_template]
        
        # Setup mock backend
        mock_backend = Mock(spec=Backend)
        mock_backend.id = "test-backend"
        mock_backend.type = BackendType.MODAL
        
        # Mock the backend router's route_job method
        with patch.object(orchestrator.backend_router, 'route_job', return_value=mock_backend):
            # Submit job
            job_id = orchestrator.submit_job(
                template_name="mock-template",
                inputs={"text": "hello world"},
                user_id="test-user"
            )
            
            # Verify job was submitted
            assert job_id is not None
            
            # Verify template was retrieved
            orchestrator.template_registry.get_template.assert_called_once_with("mock-template")
            
            # Verify backend routing was called
            orchestrator.backend_router.route_job.assert_called_once()
    
    def test_submit_job_with_template_not_found(self, orchestrator):
        """Test submitting a job with non-existent template."""
        orchestrator.template_registry.get_template.return_value = None
        orchestrator.template_registry.list_templates.return_value = []
        
        with pytest.raises(TemplateNotFoundError) as exc_info:
            orchestrator.submit_job(
                template_name="non-existent-template",
                inputs={"text": "hello"},
                user_id="test-user"
            )
        
        assert "non-existent-template" in str(exc_info.value)
        assert "not found in registry" in str(exc_info.value)
    
    def test_submit_job_with_invalid_inputs(self, orchestrator):
        """Test submitting a job with invalid inputs."""
        mock_template = MockTemplate()
        orchestrator.template_registry.get_template.return_value = mock_template
        
        # Missing required field 'text'
        with pytest.raises(JobValidationError) as exc_info:
            orchestrator.submit_job(
                template_name="mock-template",
                inputs={"number": 123},  # Missing required 'text' field
                user_id="test-user"
            )
        
        assert "Input validation failed" in str(exc_info.value)
    
    def test_submit_job_with_wrong_input_type(self, orchestrator):
        """Test submitting a job with wrong input type."""
        mock_template = MockTemplate()
        orchestrator.template_registry.get_template.return_value = mock_template
        
        # Wrong type for 'text' field
        with pytest.raises(JobValidationError) as exc_info:
            orchestrator.submit_job(
                template_name="mock-template",
                inputs={"text": 123},  # Should be string
                user_id="test-user"
            )
        
        assert "Input validation failed" in str(exc_info.value)
    
    def test_submit_job_estimates_resources_from_template(self, orchestrator):
        """Test that job submission estimates resources from template."""
        mock_template = MockTemplate()
        orchestrator.template_registry.get_template.return_value = mock_template
        
        mock_backend = Mock(spec=Backend)
        mock_backend.id = "test-backend"
        mock_backend.type = BackendType.MODAL
        
        # Capture the resource_estimate passed to route_job
        captured_estimate = None
        def capture_route_job(job, routing_strategy=None, resource_estimate=None):
            nonlocal captured_estimate
            captured_estimate = resource_estimate
            return mock_backend
        
        with patch.object(orchestrator.backend_router, 'route_job', side_effect=capture_route_job):
            # Submit job
            orchestrator.submit_job(
                template_name="mock-template",
                inputs={"text": "hello"},
                user_id="test-user"
            )
            
            # Verify resource estimate was created
            assert captured_estimate is not None
            assert isinstance(captured_estimate, ResourceEstimate)
            assert captured_estimate.requires_gpu == False
            assert captured_estimate.memory_gb == 0.5  # 512 MB = 0.5 GB
            assert captured_estimate.estimated_duration_minutes == 1  # 60 sec = 1 min
    
    def test_submit_job_with_no_suitable_backend(self, orchestrator):
        """Test submitting a job when no suitable backend is available."""
        mock_template = MockTemplate()
        orchestrator.template_registry.get_template.return_value = mock_template
        
        # Mock backend router to raise BackendNotAvailableError
        with patch.object(orchestrator.backend_router, 'route_job', side_effect=BackendNotAvailableError(
            "No suitable backend available",
            ["mock-template"]
        )):
            with pytest.raises(BackendNotAvailableError):
                orchestrator.submit_job(
                    template_name="mock-template",
                    inputs={"text": "hello"},
                    user_id="test-user"
                )
    
    def test_submit_job_stores_resource_estimate_in_metadata(self, orchestrator):
        """Test that resource estimate is stored in job metadata."""
        mock_template = MockTemplate()
        orchestrator.template_registry.get_template.return_value = mock_template
        
        mock_backend = Mock(spec=Backend)
        mock_backend.id = "test-backend"
        mock_backend.type = BackendType.MODAL
        
        # Capture the job submitted to queue
        captured_job = None
        original_submit = orchestrator.job_queue.submit_job
        def capture_submit_job(job):
            nonlocal captured_job
            captured_job = job
            return original_submit(job)
        
        with patch.object(orchestrator.backend_router, 'route_job', return_value=mock_backend):
            with patch.object(orchestrator.job_queue, 'submit_job', side_effect=capture_submit_job):
                # Submit job
                orchestrator.submit_job(
                    template_name="mock-template",
                    inputs={"text": "hello"},
                    user_id="test-user"
                )
                
                # Verify job metadata contains resource estimate
                assert captured_job is not None
                assert 'resource_estimate' in captured_job.metadata
                assert captured_job.metadata['resource_estimate']['requires_gpu'] == False
                assert captured_job.metadata['resource_estimate']['memory_gb'] == 0.5
    
    def test_get_template(self, orchestrator):
        """Test getting a template by name."""
        mock_template = MockTemplate()
        orchestrator.template_registry.get_template.return_value = mock_template
        
        template = orchestrator.get_template("mock-template")
        
        assert template == mock_template
        orchestrator.template_registry.get_template.assert_called_once_with("mock-template")
    
    def test_list_templates(self, orchestrator):
        """Test listing templates."""
        mock_template = MockTemplate()
        orchestrator.template_registry.list_templates.return_value = [mock_template]
        
        templates = orchestrator.list_templates()
        
        assert len(templates) == 1
        assert templates[0] == mock_template
        orchestrator.template_registry.list_templates.assert_called_once_with(None)
    
    def test_list_templates_by_category(self, orchestrator):
        """Test listing templates filtered by category."""
        mock_template = MockTemplate()
        orchestrator.template_registry.list_templates.return_value = [mock_template]
        
        templates = orchestrator.list_templates(category="Test")
        
        assert len(templates) == 1
        orchestrator.template_registry.list_templates.assert_called_once_with("Test")
    
    def test_get_template_metadata(self, orchestrator):
        """Test getting template metadata."""
        metadata = {
            "name": "mock-template",
            "category": "Test",
            "description": "A mock template",
            "version": "1.0.0"
        }
        orchestrator.template_registry.get_template_metadata.return_value = metadata
        
        result = orchestrator.get_template_metadata("mock-template")
        
        assert result == metadata
        orchestrator.template_registry.get_template_metadata.assert_called_once_with("mock-template")
