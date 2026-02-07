"""
Pytest configuration and fixtures for the Notebook ML Orchestrator tests.

This module provides common fixtures and configuration for all tests.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any

from notebook_ml_orchestrator.core.database import DatabaseManager
from notebook_ml_orchestrator.core.job_queue import JobQueueManager
from notebook_ml_orchestrator.core.backend_router import MultiBackendRouter
from notebook_ml_orchestrator.core.workflow_engine import WorkflowEngine
from notebook_ml_orchestrator.core.batch_processor import BatchProcessor
from notebook_ml_orchestrator.core.interfaces import Job, MLTemplate, Backend
from notebook_ml_orchestrator.core.models import (
    JobStatus, BackendType, HealthStatus, ResourceEstimate, JobResult,
    BackendCapabilities
)


@pytest.fixture
def temp_db_path():
    """Provide a temporary database path for testing."""
    temp_dir = tempfile.mkdtemp()
    db_path = Path(temp_dir) / "test_orchestrator.db"
    yield str(db_path)
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def db_manager(temp_db_path):
    """Provide a database manager instance for testing."""
    return DatabaseManager(temp_db_path)


@pytest.fixture
def job_queue(temp_db_path):
    """Provide a job queue manager instance for testing."""
    return JobQueueManager(temp_db_path)


@pytest.fixture
def backend_router():
    """Provide a backend router instance for testing."""
    return MultiBackendRouter()


@pytest.fixture
def workflow_engine():
    """Provide a workflow engine instance for testing."""
    return WorkflowEngine()


@pytest.fixture
def batch_processor():
    """Provide a batch processor instance for testing."""
    return BatchProcessor()


@pytest.fixture
def sample_job():
    """Provide a sample job for testing."""
    return Job(
        id="test-job-1",
        user_id="test-user",
        template_name="test-template",
        inputs={"input1": "value1", "input2": 42},
        status=JobStatus.QUEUED,
        priority=1
    )


@pytest.fixture
def sample_jobs():
    """Provide multiple sample jobs for testing."""
    jobs = []
    for i in range(5):
        job = Job(
            id=f"test-job-{i+1}",
            user_id="test-user",
            template_name="test-template",
            inputs={"input1": f"value{i+1}", "input2": i+1},
            status=JobStatus.QUEUED,
            priority=i % 3  # Mix of priorities
        )
        jobs.append(job)
    return jobs


class MockTemplate(MLTemplate):
    """Mock template for testing."""
    
    def __init__(self, name: str = "mock-template", category: str = "test"):
        super().__init__(name, category, "Mock template for testing")
        self.parameters = {
            "input1": {"type": "string", "required": True},
            "input2": {"type": "integer", "required": False, "default": 0}
        }
    
    def validate_inputs(self, inputs: Dict[str, Any]) -> bool:
        """Validate inputs against template requirements."""
        if "input1" not in inputs:
            return False
        if not isinstance(inputs["input1"], str):
            return False
        if "input2" in inputs and not isinstance(inputs["input2"], int):
            return False
        return True
    
    def estimate_resources(self, inputs: Dict[str, Any]) -> ResourceEstimate:
        """Estimate compute requirements."""
        return ResourceEstimate(
            cpu_cores=1,
            memory_gb=2.0,
            gpu_memory_gb=0.0,
            estimated_duration_minutes=5,
            requires_gpu=False
        )
    
    def execute(self, inputs: Dict[str, Any], backend: Backend) -> JobResult:
        """Execute the template."""
        return JobResult(
            success=True,
            outputs={"result": f"Processed {inputs.get('input1', 'unknown')}"},
            execution_time_seconds=1.0,
            backend_used=backend.id if backend else None
        )
    
    def get_schema(self) -> Dict[str, Any]:
        """Return JSON schema for input validation."""
        return {
            "type": "object",
            "properties": {
                "input1": {"type": "string"},
                "input2": {"type": "integer"}
            },
            "required": ["input1"]
        }


class MockBackend(Backend):
    """Mock backend for testing."""
    
    def __init__(self, backend_id: str = "mock-backend", name: str = "Mock Backend"):
        super().__init__(backend_id, name, BackendType.LOCAL_GPU)
        self.capabilities = BackendCapabilities(
            supported_templates=["mock-template", "test-template"],
            max_concurrent_jobs=2,
            supports_gpu=False,
            cost_per_hour=0.0
        )
        self.health_status = HealthStatus.HEALTHY
        self._queue_length = 0
    
    def execute_job(self, job: Job, template: MLTemplate) -> JobResult:
        """Execute a job on this backend."""
        return JobResult(
            success=True,
            outputs={"result": f"Job {job.id} executed on {self.id}"},
            execution_time_seconds=2.0,
            backend_used=self.id
        )
    
    def check_health(self) -> HealthStatus:
        """Check backend health."""
        return self.health_status
    
    def get_queue_length(self) -> int:
        """Get queue length."""
        return self._queue_length
    
    def supports_template(self, template_name: str) -> bool:
        """Check if template is supported."""
        return template_name in self.capabilities.supported_templates
    
    def estimate_cost(self, resource_estimate: ResourceEstimate) -> float:
        """Estimate execution cost."""
        return 0.0  # Free for testing


@pytest.fixture
def mock_template():
    """Provide a mock template for testing."""
    return MockTemplate()


@pytest.fixture
def mock_backend():
    """Provide a mock backend for testing."""
    return MockBackend()


@pytest.fixture
def mock_backends():
    """Provide multiple mock backends for testing."""
    backends = []
    for i in range(3):
        backend = MockBackend(
            backend_id=f"mock-backend-{i+1}",
            name=f"Mock Backend {i+1}"
        )
        backends.append(backend)
    return backends


# Hypothesis settings for property-based testing
from hypothesis import settings, Verbosity

# Configure Hypothesis for thorough testing
settings.register_profile("default", max_examples=100, verbosity=Verbosity.normal)
settings.register_profile("ci", max_examples=200, verbosity=Verbosity.verbose)
settings.register_profile("dev", max_examples=50, verbosity=Verbosity.normal)

# Load the appropriate profile
settings.load_profile("default")