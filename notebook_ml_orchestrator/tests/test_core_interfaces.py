"""
Unit tests for core interfaces and data models.

This module tests the fundamental data structures and interfaces
of the orchestration system.
"""

import pytest
from datetime import datetime

from notebook_ml_orchestrator.core.interfaces import Job, Workflow, WorkflowExecution, BatchJob
from notebook_ml_orchestrator.core.models import (
    JobStatus, WorkflowStatus, BatchStatus, BackendType, HealthStatus,
    ResourceEstimate, JobResult, BatchProgress, BatchItem, WorkflowDefinition
)


class TestJob:
    """Test Job data structure."""
    
    def test_job_creation(self):
        """Test basic job creation."""
        job = Job(
            user_id="test-user",
            template_name="test-template",
            inputs={"key": "value"}
        )
        
        assert job.user_id == "test-user"
        assert job.template_name == "test-template"
        assert job.inputs == {"key": "value"}
        assert job.status == JobStatus.QUEUED
        assert job.retry_count == 0
        assert job.priority == 0
        assert isinstance(job.created_at, datetime)
        assert job.id is not None and len(job.id) > 0
    
    def test_job_with_custom_values(self):
        """Test job creation with custom values."""
        custom_time = datetime.now()
        job = Job(
            id="custom-id",
            user_id="user-123",
            template_name="custom-template",
            inputs={"param1": 42, "param2": "test"},
            status=JobStatus.RUNNING,
            priority=5,
            retry_count=2,
            created_at=custom_time
        )
        
        assert job.id == "custom-id"
        assert job.user_id == "user-123"
        assert job.template_name == "custom-template"
        assert job.inputs == {"param1": 42, "param2": "test"}
        assert job.status == JobStatus.RUNNING
        assert job.priority == 5
        assert job.retry_count == 2
        assert job.created_at == custom_time
    
    def test_job_metadata(self):
        """Test job metadata handling."""
        job = Job(
            user_id="test-user",
            template_name="test-template",
            inputs={},
            metadata={"custom_field": "custom_value", "number": 123}
        )
        
        assert job.metadata["custom_field"] == "custom_value"
        assert job.metadata["number"] == 123


class TestWorkflow:
    """Test Workflow data structure."""
    
    def test_workflow_creation(self):
        """Test basic workflow creation."""
        definition = WorkflowDefinition(
            steps=[{"name": "step1", "template": "test-template"}],
            connections=[{"from": "step1", "to": "step2"}]
        )
        
        workflow = Workflow(
            name="Test Workflow",
            description="A test workflow",
            user_id="test-user",
            definition=definition
        )
        
        assert workflow.name == "Test Workflow"
        assert workflow.description == "A test workflow"
        assert workflow.user_id == "test-user"
        assert workflow.definition == definition
        assert workflow.version == 1
        assert isinstance(workflow.created_at, datetime)
        assert isinstance(workflow.updated_at, datetime)
        assert workflow.id is not None and len(workflow.id) > 0
    
    def test_workflow_execution_creation(self):
        """Test workflow execution creation."""
        execution = WorkflowExecution(
            workflow_id="workflow-123",
            inputs={"start_param": "value"},
            status=WorkflowStatus.RUNNING
        )
        
        assert execution.workflow_id == "workflow-123"
        assert execution.inputs == {"start_param": "value"}
        assert execution.status == WorkflowStatus.RUNNING
        assert execution.outputs == {}
        assert isinstance(execution.started_at, datetime)
        assert execution.id is not None and len(execution.id) > 0


class TestBatchJob:
    """Test BatchJob data structure."""
    
    def test_batch_job_creation(self):
        """Test basic batch job creation."""
        items = [
            BatchItem(inputs={"param": "value1"}),
            BatchItem(inputs={"param": "value2"}),
            BatchItem(inputs={"param": "value3"})
        ]
        
        batch = BatchJob(
            user_id="test-user",
            template_name="test-template",
            items=items
        )
        
        assert batch.user_id == "test-user"
        assert batch.template_name == "test-template"
        assert len(batch.items) == 3
        assert batch.status == BatchStatus.QUEUED
        assert isinstance(batch.created_at, datetime)
        assert batch.id is not None and len(batch.id) > 0
    
    def test_batch_progress_calculation(self):
        """Test batch progress calculation."""
        progress = BatchProgress(total_items=10)
        progress.completed_items = 7
        progress.failed_items = 2
        progress.running_items = 1
        
        assert progress.total_items == 10
        assert progress.completed_items == 7
        assert progress.failed_items == 2
        assert progress.running_items == 1
        assert progress.queued_items == 0
        assert progress.completion_percentage == 70.0
    
    def test_batch_progress_empty(self):
        """Test batch progress with no items."""
        progress = BatchProgress(total_items=0)
        assert progress.completion_percentage == 0.0


class TestResourceEstimate:
    """Test ResourceEstimate data structure."""
    
    def test_resource_estimate_defaults(self):
        """Test resource estimate with default values."""
        estimate = ResourceEstimate()
        
        assert estimate.cpu_cores == 1
        assert estimate.memory_gb == 1.0
        assert estimate.gpu_memory_gb == 0.0
        assert estimate.estimated_duration_minutes == 5
        assert estimate.requires_gpu is False
        assert estimate.requires_internet is True
    
    def test_resource_estimate_custom(self):
        """Test resource estimate with custom values."""
        estimate = ResourceEstimate(
            cpu_cores=4,
            memory_gb=8.0,
            gpu_memory_gb=12.0,
            estimated_duration_minutes=30,
            requires_gpu=True,
            requires_internet=False
        )
        
        assert estimate.cpu_cores == 4
        assert estimate.memory_gb == 8.0
        assert estimate.gpu_memory_gb == 12.0
        assert estimate.estimated_duration_minutes == 30
        assert estimate.requires_gpu is True
        assert estimate.requires_internet is False


class TestJobResult:
    """Test JobResult data structure."""
    
    def test_job_result_success(self):
        """Test successful job result."""
        result = JobResult(
            success=True,
            outputs={"result": "success", "value": 42},
            execution_time_seconds=10.5,
            backend_used="test-backend"
        )
        
        assert result.success is True
        assert result.outputs == {"result": "success", "value": 42}
        assert result.execution_time_seconds == 10.5
        assert result.backend_used == "test-backend"
        assert result.error_message is None
    
    def test_job_result_failure(self):
        """Test failed job result."""
        result = JobResult(
            success=False,
            error_message="Something went wrong",
            execution_time_seconds=5.0
        )
        
        assert result.success is False
        assert result.error_message == "Something went wrong"
        assert result.execution_time_seconds == 5.0
        assert result.outputs == {}


class TestEnums:
    """Test enum values and behavior."""
    
    def test_job_status_values(self):
        """Test JobStatus enum values."""
        assert JobStatus.QUEUED.value == "queued"
        assert JobStatus.RUNNING.value == "running"
        assert JobStatus.COMPLETED.value == "completed"
        assert JobStatus.FAILED.value == "failed"
        assert JobStatus.CANCELLED.value == "cancelled"
        assert JobStatus.RETRYING.value == "retrying"
    
    def test_workflow_status_values(self):
        """Test WorkflowStatus enum values."""
        assert WorkflowStatus.CREATED.value == "created"
        assert WorkflowStatus.RUNNING.value == "running"
        assert WorkflowStatus.PAUSED.value == "paused"
        assert WorkflowStatus.COMPLETED.value == "completed"
        assert WorkflowStatus.FAILED.value == "failed"
        assert WorkflowStatus.CANCELLED.value == "cancelled"
    
    def test_backend_type_values(self):
        """Test BackendType enum values."""
        assert BackendType.LOCAL_GPU.value == "local_gpu"
        assert BackendType.MODAL.value == "modal"
        assert BackendType.HUGGINGFACE.value == "huggingface"
        assert BackendType.KAGGLE.value == "kaggle"
        assert BackendType.COLAB.value == "colab"
    
    def test_health_status_values(self):
        """Test HealthStatus enum values."""
        assert HealthStatus.HEALTHY.value == "healthy"
        assert HealthStatus.DEGRADED.value == "degraded"
        assert HealthStatus.UNHEALTHY.value == "unhealthy"
        assert HealthStatus.UNKNOWN.value == "unknown"