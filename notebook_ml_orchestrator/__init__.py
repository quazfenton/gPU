"""
Notebook ML Orchestrator

A comprehensive ML orchestration platform that leverages free notebook platforms 
(Colab, Kaggle, Modal free tier, HF Spaces) to provide a unified GUI with template 
library, Zapier-style workflow automation, persistent job queuing, multi-backend 
routing, and batch processing capabilities for ML pipelines.
"""

__version__ = "0.1.0"
__author__ = "Notebook ML Orchestrator Team"

from .core.interfaces import MLTemplate, Backend, Job, Workflow
from .core.job_queue import JobQueueManager
from .core.backend_router import MultiBackendRouter
from .core.workflow_engine import WorkflowEngine
from .core.batch_processor import BatchProcessor

__all__ = [
    "MLTemplate",
    "Backend", 
    "Job",
    "Workflow",
    "JobQueueManager",
    "MultiBackendRouter", 
    "WorkflowEngine",
    "BatchProcessor",
]