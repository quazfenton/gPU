"""Health check endpoint for GUI application.

This module provides health check functionality to monitor the status
of the GUI application and its components.

Requirements:
    - 11.6: Provide health check endpoint for monitoring
"""

import time
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import logging

from notebook_ml_orchestrator.core.interfaces import (
    JobQueueInterface,
    BackendRouterInterface,
    WorkflowEngineInterface
)
from notebook_ml_orchestrator.core.template_registry import TemplateRegistry


logger = logging.getLogger('gui.health')


class ComponentStatus(str, Enum):
    """Status of a component."""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DEGRADED = "degraded"
    UNKNOWN = "unknown"


@dataclass
class ComponentHealth:
    """Health status of a component.
    
    Attributes:
        name: Component name
        status: Health status
        message: Optional status message
        details: Optional additional details
    """
    name: str
    status: ComponentStatus
    message: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


@dataclass
class HealthCheckResponse:
    """Health check response.
    
    Attributes:
        status: Overall system status
        timestamp: Unix timestamp of health check
        version: Application version
        components: Health status of individual components
        uptime: Application uptime in seconds (if available)
    """
    status: ComponentStatus
    timestamp: float
    version: str
    components: Dict[str, ComponentHealth]
    uptime: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            'status': self.status.value,
            'timestamp': self.timestamp,
            'version': self.version,
            'components': {
                name: {
                    'name': comp.name,
                    'status': comp.status.value,
                    'message': comp.message,
                    'details': comp.details
                }
                for name, comp in self.components.items()
            }
        }
        if self.uptime is not None:
            result['uptime'] = self.uptime
        return result


class HealthChecker:
    """Health checker for GUI application.
    
    This class performs health checks on the GUI application and its
    components, including the job queue, backend router, workflow engine,
    and template registry.
    
    Requirements:
        - 11.6: Provide health check endpoint for monitoring
    """
    
    def __init__(
        self,
        job_queue: Optional[JobQueueInterface] = None,
        backend_router: Optional[BackendRouterInterface] = None,
        workflow_engine: Optional[WorkflowEngineInterface] = None,
        template_registry: Optional[TemplateRegistry] = None,
        version: str = "unknown"
    ):
        """Initialize the health checker.
        
        Args:
            job_queue: Job queue instance (optional)
            backend_router: Backend router instance (optional)
            workflow_engine: Workflow engine instance (optional)
            template_registry: Template registry instance (optional)
            version: Application version
        """
        self.job_queue = job_queue
        self.backend_router = backend_router
        self.workflow_engine = workflow_engine
        self.template_registry = template_registry
        self.version = version
        self.start_time = time.time()
    
    def check_health(self) -> HealthCheckResponse:
        """Perform health check on all components.
        
        Returns:
            HealthCheckResponse with overall status and component details
            
        Requirements:
            - 11.6: Return system status and component health
        """
        logger.debug("Performing health check")
        
        components = {}
        
        # Check job queue
        if self.job_queue:
            components['job_queue'] = self._check_job_queue()
        
        # Check backend router
        if self.backend_router:
            components['backend_router'] = self._check_backend_router()
        
        # Check workflow engine
        if self.workflow_engine:
            components['workflow_engine'] = self._check_workflow_engine()
        
        # Check template registry
        if self.template_registry:
            components['template_registry'] = self._check_template_registry()
        
        # Determine overall status
        overall_status = self._determine_overall_status(components)
        
        # Calculate uptime
        uptime = time.time() - self.start_time
        
        response = HealthCheckResponse(
            status=overall_status,
            timestamp=time.time(),
            version=self.version,
            components=components,
            uptime=uptime
        )
        
        logger.debug(f"Health check complete: {overall_status.value}")
        return response
    
    def _check_job_queue(self) -> ComponentHealth:
        """Check job queue health.
        
        Returns:
            ComponentHealth for job queue
        """
        try:
            # Try to get queue statistics
            stats = self.job_queue.get_queue_statistics()
            
            return ComponentHealth(
                name="Job Queue",
                status=ComponentStatus.HEALTHY,
                message="Job queue is operational",
                details={
                    'total_jobs': stats.get('total', 0),
                    'pending_jobs': stats.get('pending', 0),
                    'running_jobs': stats.get('running', 0)
                }
            )
        except Exception as e:
            logger.error(f"Job queue health check failed: {e}")
            return ComponentHealth(
                name="Job Queue",
                status=ComponentStatus.UNHEALTHY,
                message=f"Job queue error: {str(e)}"
            )
    
    def _check_backend_router(self) -> ComponentHealth:
        """Check backend router health.
        
        Returns:
            ComponentHealth for backend router
        """
        try:
            # Try to get backend status
            backend_status = self.backend_router.get_backend_status()
            
            # Count healthy backends
            healthy_count = sum(
                1 for status in backend_status.values()
                if status.name == 'HEALTHY'
            )
            total_count = len(backend_status)
            
            # Determine status based on healthy backend ratio
            if healthy_count == 0:
                status = ComponentStatus.UNHEALTHY
                message = "No healthy backends available"
            elif healthy_count < total_count:
                status = ComponentStatus.DEGRADED
                message = f"{healthy_count}/{total_count} backends healthy"
            else:
                status = ComponentStatus.HEALTHY
                message = "All backends healthy"
            
            return ComponentHealth(
                name="Backend Router",
                status=status,
                message=message,
                details={
                    'total_backends': total_count,
                    'healthy_backends': healthy_count
                }
            )
        except Exception as e:
            logger.error(f"Backend router health check failed: {e}")
            return ComponentHealth(
                name="Backend Router",
                status=ComponentStatus.UNHEALTHY,
                message=f"Backend router error: {str(e)}"
            )
    
    def _check_workflow_engine(self) -> ComponentHealth:
        """Check workflow engine health.
        
        Returns:
            ComponentHealth for workflow engine
        """
        try:
            # Try to list executions (basic connectivity check)
            executions = self.workflow_engine.list_executions()
            
            # Count running executions
            running_count = sum(
                1 for exec in executions
                if exec.status.name == 'RUNNING'
            )
            
            return ComponentHealth(
                name="Workflow Engine",
                status=ComponentStatus.HEALTHY,
                message="Workflow engine is operational",
                details={
                    'total_executions': len(executions),
                    'running_executions': running_count
                }
            )
        except Exception as e:
            logger.error(f"Workflow engine health check failed: {e}")
            return ComponentHealth(
                name="Workflow Engine",
                status=ComponentStatus.UNHEALTHY,
                message=f"Workflow engine error: {str(e)}"
            )
    
    def _check_template_registry(self) -> ComponentHealth:
        """Check template registry health.
        
        Returns:
            ComponentHealth for template registry
        """
        try:
            # Try to list templates
            templates = self.template_registry.list_templates()
            
            return ComponentHealth(
                name="Template Registry",
                status=ComponentStatus.HEALTHY,
                message="Template registry is operational",
                details={
                    'total_templates': len(templates)
                }
            )
        except Exception as e:
            logger.error(f"Template registry health check failed: {e}")
            return ComponentHealth(
                name="Template Registry",
                status=ComponentStatus.UNHEALTHY,
                message=f"Template registry error: {str(e)}"
            )
    
    def _determine_overall_status(
        self,
        components: Dict[str, ComponentHealth]
    ) -> ComponentStatus:
        """Determine overall system status from component statuses.
        
        Rules:
        - If any component is UNHEALTHY, overall is UNHEALTHY
        - If any component is DEGRADED, overall is DEGRADED
        - If all components are HEALTHY, overall is HEALTHY
        - If no components, overall is UNKNOWN
        
        Args:
            components: Dictionary of component health statuses
            
        Returns:
            Overall system status
        """
        if not components:
            return ComponentStatus.UNKNOWN
        
        statuses = [comp.status for comp in components.values()]
        
        if ComponentStatus.UNHEALTHY in statuses:
            return ComponentStatus.UNHEALTHY
        elif ComponentStatus.DEGRADED in statuses:
            return ComponentStatus.DEGRADED
        elif all(s == ComponentStatus.HEALTHY for s in statuses):
            return ComponentStatus.HEALTHY
        else:
            return ComponentStatus.UNKNOWN


def create_health_check_handler(health_checker: HealthChecker):
    """Create a health check handler function.
    
    This function creates a handler that can be used as a Gradio API endpoint
    or FastAPI endpoint to provide health check information.
    
    Args:
        health_checker: HealthChecker instance
        
    Returns:
        Handler function that returns health check response
        
    Example:
        >>> health_checker = HealthChecker(job_queue, backend_router, ...)
        >>> handler = create_health_check_handler(health_checker)
        >>> response = handler()
        >>> print(response['status'])  # 'healthy'
    """
    def handler() -> Dict[str, Any]:
        """Handle health check request.
        
        Returns:
            Health check response as dictionary
        """
        response = health_checker.check_health()
        return response.to_dict()
    
    return handler
