"""
Core abstract base classes and interfaces for the Notebook ML Orchestrator.

This module defines the fundamental interfaces that all components must implement,
providing a consistent API across the entire system.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
import uuid

from .models import (
    JobStatus, WorkflowStatus, BatchStatus, BackendType, HealthStatus,
    ResourceEstimate, JobResult, BatchProgress, BatchItem, 
    WorkflowDefinition, BackendCapabilities
)


@dataclass
class Job:
    """Core job data structure."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = ""
    template_name: str = ""
    inputs: Dict[str, Any] = field(default_factory=dict)
    status: JobStatus = JobStatus.QUEUED
    backend_id: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[JobResult] = None
    error: Optional[str] = None
    retry_count: int = 0
    priority: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Workflow:
    """Core workflow data structure."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    user_id: str = ""
    definition: WorkflowDefinition = field(default_factory=WorkflowDefinition)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    version: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowExecution:
    """Workflow execution instance."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    workflow_id: str = ""
    status: WorkflowStatus = WorkflowStatus.CREATED
    inputs: Dict[str, Any] = field(default_factory=dict)
    outputs: Dict[str, Any] = field(default_factory=dict)
    current_step: str = ""
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BatchJob:
    """Batch job data structure."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = ""
    template_name: str = ""
    items: List[BatchItem] = field(default_factory=list)
    status: BatchStatus = BatchStatus.QUEUED
    progress: BatchProgress = field(default_factory=lambda: BatchProgress(0))
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class MLTemplate(ABC):
    """Abstract base class for ML templates."""
    
    def __init__(self, name: str, category: str, description: str):
        self.name = name
        self.category = category
        self.description = description
        self.parameters = {}
        self.requirements = {}
    
    @abstractmethod
    def validate_inputs(self, inputs: Dict[str, Any]) -> bool:
        """
        Validate input parameters against template requirements.
        
        Args:
            inputs: Dictionary of input parameters
            
        Returns:
            True if inputs are valid, False otherwise
        """
        pass
    
    @abstractmethod
    def estimate_resources(self, inputs: Dict[str, Any]) -> ResourceEstimate:
        """
        Estimate compute requirements for the job.
        
        Args:
            inputs: Dictionary of input parameters
            
        Returns:
            ResourceEstimate object with estimated requirements
        """
        pass
    
    @abstractmethod
    def execute(self, inputs: Dict[str, Any], backend: 'Backend') -> JobResult:
        """
        Execute the ML operation on specified backend.
        
        Args:
            inputs: Dictionary of input parameters
            backend: Backend instance to execute on
            
        Returns:
            JobResult with execution results
        """
        pass
    
    @abstractmethod
    def get_schema(self) -> Dict[str, Any]:
        """
        Return JSON schema for input validation.
        
        Returns:
            JSON schema dictionary
        """
        pass


class Backend(ABC):
    """Abstract base class for compute backends."""
    
    def __init__(self, backend_id: str, name: str, backend_type: BackendType):
        self.id = backend_id
        self.name = name
        self.type = backend_type
        self.capabilities = BackendCapabilities()
        self.health_status = HealthStatus.UNKNOWN
        self.last_health_check = datetime.now()
    
    @abstractmethod
    def execute_job(self, job: Job, template: MLTemplate) -> JobResult:
        """
        Execute a job on this backend.
        
        Args:
            job: Job to execute
            template: Template instance for the job
            
        Returns:
            JobResult with execution results
        """
        pass
    
    @abstractmethod
    def check_health(self) -> HealthStatus:
        """
        Check the health status of this backend.
        
        Returns:
            Current health status
        """
        pass
    
    @abstractmethod
    def get_queue_length(self) -> int:
        """
        Get the current queue length for this backend.
        
        Returns:
            Number of jobs in queue
        """
        pass
    
    @abstractmethod
    def supports_template(self, template_name: str) -> bool:
        """
        Check if this backend supports a specific template.
        
        Args:
            template_name: Name of the template
            
        Returns:
            True if supported, False otherwise
        """
        pass
    
    @abstractmethod
    def estimate_cost(self, resource_estimate: ResourceEstimate) -> float:
        """
        Estimate the cost for executing a job with given resources.
        
        Args:
            resource_estimate: Resource requirements
            
        Returns:
            Estimated cost in USD
        """
        pass


class JobQueueInterface(ABC):
    """Interface for job queue management."""
    
    @abstractmethod
    def submit_job(self, job: Job) -> str:
        """Submit a new job to the queue."""
        pass
    
    @abstractmethod
    def get_next_job(self, backend_capabilities: List[str]) -> Optional[Job]:
        """Get the next job suitable for the given backend."""
        pass
    
    @abstractmethod
    def update_job_status(self, job_id: str, status: JobStatus, result: Any = None):
        """Update job status and store results."""
        pass
    
    @abstractmethod
    def get_job(self, job_id: str) -> Optional[Job]:
        """Retrieve a job by ID."""
        pass
    
    @abstractmethod
    def get_job_history(self, user_id: str, limit: int = 100) -> List[Job]:
        """Retrieve job history for a user."""
        pass


class BackendRouterInterface(ABC):
    """Interface for backend routing."""
    
    @abstractmethod
    def register_backend(self, backend: Backend):
        """Register a new compute backend."""
        pass
    
    @abstractmethod
    def route_job(self, job: Job) -> Backend:
        """Select optimal backend for job execution."""
        pass
    
    @abstractmethod
    def get_backend_status(self) -> Dict[str, HealthStatus]:
        """Get current status of all backends."""
        pass


class WorkflowEngineInterface(ABC):
    """Interface for workflow execution."""
    
    @abstractmethod
    def create_workflow(self, definition: WorkflowDefinition) -> Workflow:
        """Create a new workflow from definition."""
        pass
    
    @abstractmethod
    def execute_workflow(self, workflow_id: str, inputs: Dict[str, Any]) -> WorkflowExecution:
        """Execute a workflow with given inputs."""
        pass
    
    @abstractmethod
    def get_workflow_status(self, execution_id: str) -> WorkflowStatus:
        """Get current workflow execution status."""
        pass


class BatchProcessorInterface(ABC):
    """Interface for batch processing."""
    
    @abstractmethod
    def submit_batch(self, template: MLTemplate, inputs: List[Dict[str, Any]]) -> BatchJob:
        """Submit a batch of jobs for processing."""
        pass
    
    @abstractmethod
    def track_batch_progress(self, batch_id: str) -> BatchProgress:
        """Track progress of batch execution."""
        pass