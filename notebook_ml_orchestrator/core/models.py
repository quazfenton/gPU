"""
Core data models and enums for the Notebook ML Orchestrator.

This module defines the fundamental data structures, enums, and types
used throughout the orchestration system.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
import uuid


class JobStatus(Enum):
    """Job execution status enumeration."""
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"


class WorkflowStatus(Enum):
    """Workflow execution status enumeration."""
    CREATED = "created"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class BatchStatus(Enum):
    """Batch job status enumeration."""
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    PARTIALLY_FAILED = "partially_failed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class BackendType(Enum):
    """Backend type enumeration."""
    LOCAL_GPU = "local_gpu"
    MODAL = "modal"
    HUGGINGFACE = "huggingface"
    KAGGLE = "kaggle"
    COLAB = "colab"


class HealthStatus(Enum):
    """Backend health status enumeration."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class ResourceEstimate:
    """Resource requirements estimate for a job."""
    cpu_cores: int = 1
    memory_gb: float = 1.0
    gpu_memory_gb: float = 0.0
    estimated_duration_minutes: int = 5
    requires_gpu: bool = False
    requires_internet: bool = True


@dataclass
class JobResult:
    """Result of a job execution."""
    success: bool
    outputs: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    execution_time_seconds: float = 0.0
    backend_used: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BatchProgress:
    """Progress tracking for batch jobs."""
    total_items: int
    completed_items: int = 0
    failed_items: int = 0
    running_items: int = 0
    queued_items: int = 0
    
    @property
    def completion_percentage(self) -> float:
        """Calculate completion percentage."""
        if self.total_items == 0:
            return 0.0
        return (self.completed_items / self.total_items) * 100.0


@dataclass
class BatchItem:
    """Individual item within a batch job."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    inputs: Dict[str, Any] = field(default_factory=dict)
    status: JobStatus = JobStatus.QUEUED
    result: Optional[JobResult] = None
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


@dataclass
class WorkflowDefinition:
    """Definition of a workflow structure."""
    steps: List[Dict[str, Any]] = field(default_factory=list)
    connections: List[Dict[str, str]] = field(default_factory=list)
    conditions: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BackendCapabilities:
    """Capabilities and limits of a backend."""
    supported_templates: List[str] = field(default_factory=list)
    max_concurrent_jobs: int = 1
    max_job_duration_minutes: int = 60
    supports_gpu: bool = False
    supports_batch: bool = True
    cost_per_hour: float = 0.0
    free_tier_limits: Dict[str, Any] = field(default_factory=dict)