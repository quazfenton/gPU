"""
Backend monitor service for GUI interface.

This module provides business logic for backend health monitoring and status tracking
through the GUI interface.
"""

from typing import Any, Dict, List, Optional
from datetime import datetime

from notebook_ml_orchestrator.core.interfaces import BackendRouterInterface
from notebook_ml_orchestrator.core.models import HealthStatus
from notebook_ml_orchestrator.core.logging_config import LoggerMixin


class BackendMonitorService(LoggerMixin):
    """Service for backend monitoring."""
    
    def __init__(self, backend_router: BackendRouterInterface):
        """
        Initialize backend monitor service.
        
        Args:
            backend_router: Backend router instance
        """
        self.backend_router = backend_router
        self.logger.info("BackendMonitorService initialized")
    
    def get_backends_status(self) -> List[Dict[str, Any]]:
        """
        Retrieve status for all backends.
        
        Returns:
            List of dictionaries with backend status information:
            - name: Backend name/ID
            - status: Health status (healthy, unhealthy, degraded, unknown)
            - uptime_percentage: Percentage of successful health checks
            - avg_response_time: Average response time (placeholder, not yet implemented)
            - jobs_executed: Number of jobs executed (placeholder, not yet implemented)
            - last_health_check: Timestamp of last health check
            - last_error: Last error message if unhealthy
            - capabilities: Backend capabilities
            - cost_total: Total cost tracked for this backend
            
        Requirements:
            - 5.1: Display all registered backends with health status indicators
            - 10.5: Integration with Backend_Router health monitoring
        """
        backends_status = []
        
        # Get all backends from router
        backends = self.backend_router.list_backends()
        
        # Get health status for all backends
        health_statuses = self.backend_router.get_backend_status()
        
        for backend in backends:
            backend_id = backend.id
            
            # Get health metrics from health monitor
            health_metrics = self.backend_router.health_monitor.get_health_metrics(backend_id)
            
            # Get current health status
            current_status = health_statuses.get(backend_id, HealthStatus.UNKNOWN)
            
            # Get last error from health history
            last_error = None
            health_history = self.backend_router.health_monitor.health_history.get(backend_id, [])
            if health_history:
                # Find most recent unhealthy status with error
                for entry in reversed(health_history):
                    if entry['status'] in [HealthStatus.UNHEALTHY, HealthStatus.DEGRADED]:
                        # Error information would be stored in backend or metadata
                        # For now, we'll indicate the status
                        last_error = f"Backend marked as {entry['status'].value}"
                        break
            
            # Calculate jobs executed from job queue
            jobs_executed = self._get_backend_jobs_executed(backend_id)
            
            # Calculate average response time from job execution history
            avg_response_time = self._get_backend_avg_response_time(backend_id)

            backends_status.append({
                'name': backend_id,
                'status': current_status.value,
                'uptime_percentage': health_metrics.get('uptime_percentage', 0.0),
                'avg_response_time': avg_response_time,
                'jobs_executed': jobs_executed,
                'last_health_check': health_metrics.get('last_check'),
                'last_error': last_error,
                'capabilities': {
                    'supported_templates': backend.capabilities.supported_templates,
                    'max_concurrent_jobs': backend.capabilities.max_concurrent_jobs,
                    'max_job_duration_minutes': backend.capabilities.max_job_duration_minutes,
                    'supports_gpu': backend.capabilities.supports_gpu,
                    'supports_batch': backend.capabilities.supports_batch,
                    'cost_per_hour': backend.capabilities.cost_per_hour,
                },
                'cost_total': cost_total
            })
        
        self.logger.debug(f"Retrieved status for {len(backends_status)} backends")
        
        return backends_status
    
    def get_backend_details(self, backend_name: str) -> Dict[str, Any]:
        """
        Retrieve detailed backend information.
        
        Args:
            backend_name: Backend name/ID
            
        Returns:
            Dictionary with detailed backend information:
            - name: Backend name/ID
            - status: Current health status
            - health_metrics: Detailed health metrics including:
                - uptime_percentage: Percentage of successful health checks
                - total_checks: Total number of health checks performed
                - healthy_checks: Number of successful health checks
                - failure_rate: Percentage of failed health checks
                - last_check: Timestamp of last health check
                - consecutive_failures: Number of consecutive health check failures
                - consecutive_job_failures: Number of consecutive job failures
            - capabilities: Backend capabilities including:
                - supported_templates: List of supported template names
                - max_concurrent_jobs: Maximum concurrent jobs
                - max_job_duration_minutes: Maximum job duration
                - supports_gpu: GPU support flag
                - supports_batch: Batch processing support
                - cost_per_hour: Cost per hour
            - cost_metrics: Cost tracking information:
                - total_cost: Total cost for this backend
            - configuration_status: Backend configuration status
            
        Raises:
            ValueError: If backend not found
            
        Requirements:
            - 5.3: Display health metrics (uptime, response time, failure rate)
            - 5.4: Display backend capabilities
            - 5.6: Display cost tracking metrics
            - 5.9: Display backend configuration status
        """
        # Get backend from router
        backend = self.backend_router.get_backend(backend_name)
        
        if not backend:
            raise ValueError(f"Backend '{backend_name}' not found")
        
        # Get health status
        health_statuses = self.backend_router.get_backend_status()
        current_status = health_statuses.get(backend_name, HealthStatus.UNKNOWN)
        
        # Get health metrics
        health_metrics = self.backend_router.health_monitor.get_health_metrics(backend_name)
        
        # Get cost information
        cost_total = self.backend_router.cost_optimizer.get_total_cost(backend_name)
        
        # Get last error from health history
        last_error = None
        last_error_timestamp = None
        health_history = self.backend_router.health_monitor.health_history.get(backend_name, [])
        if health_history:
            for entry in reversed(health_history):
                if entry['status'] in [HealthStatus.UNHEALTHY, HealthStatus.DEGRADED]:
                    last_error = f"Backend marked as {entry['status'].value}"
                    last_error_timestamp = entry['timestamp']
                    break
        
        details = {
            'name': backend_name,
            'status': current_status.value,
            'health_metrics': {
                'uptime_percentage': health_metrics.get('uptime_percentage', 0.0),
                'total_checks': health_metrics.get('total_checks', 0),
                'healthy_checks': health_metrics.get('healthy_checks', 0),
                'failure_rate': health_metrics.get('failure_rate', 0.0),
                'last_check': health_metrics.get('last_check').isoformat() if health_metrics.get('last_check') else None,
                'consecutive_failures': health_metrics.get('consecutive_failures', 0),
                'consecutive_job_failures': health_metrics.get('consecutive_job_failures', 0),
                'last_error': last_error,
                'last_error_timestamp': last_error_timestamp.isoformat() if last_error_timestamp else None
            },
            'capabilities': {
                'supported_templates': backend.capabilities.supported_templates,
                'max_concurrent_jobs': backend.capabilities.max_concurrent_jobs,
                'max_job_duration_minutes': backend.capabilities.max_job_duration_minutes,
                'supports_gpu': backend.capabilities.supports_gpu,
                'supports_batch': backend.capabilities.supports_batch,
                'cost_per_hour': backend.capabilities.cost_per_hour,
                'free_tier_limits': backend.capabilities.free_tier_limits
            },
            'cost_metrics': {
                'total_cost': cost_total
            },
            'configuration_status': 'configured'  # Placeholder - credential validation not yet implemented
        }
        
        self.logger.debug(f"Retrieved details for backend: {backend_name}")

        return details

    def _get_backend_jobs_executed(self, backend_id: str) -> int:
        """
        Get number of jobs executed by a backend.
        
        Args:
            backend_id: Backend identifier
            
        Returns:
            Number of jobs executed
        """
        try:
            # Get job queue statistics
            if hasattr(self.backend_router, 'job_queue'):
                stats = self.backend_router.job_queue.get_queue_statistics()
                # Count completed jobs (this is approximate - would need backend tracking)
                return stats.get('completed', 0)
        except Exception as e:
            self.logger.debug(f"Could not get jobs executed for {backend_id}: {e}")
        
        return 0

    def _get_backend_avg_response_time(self, backend_id: str) -> float:
        """
        Get average response time for a backend.
        
        Args:
            backend_id: Backend identifier
            
        Returns:
            Average response time in seconds
        """
        try:
            # Get health metrics which may include response time
            health_metrics = self.backend_router.health_monitor.get_health_metrics(backend_id)
            
            # If response time is tracked, use it
            if 'avg_response_time' in health_metrics:
                return health_metrics['avg_response_time']
        except Exception as e:
            self.logger.debug(f"Could not get response time for {backend_id}: {e}")
        
        # Return 0.0 if not available
        return 0.0

    def trigger_health_check(self, backend_name: str) -> Dict[str, Any]:
        """
        Manually trigger health check for a backend.
        
        Args:
            backend_name: Backend name/ID
            
        Returns:
            Dictionary with health check result:
            - backend_name: Backend name/ID
            - status: Health status after check
            - timestamp: Timestamp of health check
            - message: Status message
            
        Raises:
            ValueError: If backend not found
            
        Requirements:
            - 5.7: Provide manual health check triggers for each backend
        """
        # Get backend from router
        backend = self.backend_router.get_backend(backend_name)
        
        if not backend:
            raise ValueError(f"Backend '{backend_name}' not found")
        
        self.logger.info(f"Triggering manual health check for backend: {backend_name}")
        
        # Perform health check
        health_status = self.backend_router.health_monitor.check_backend_health(backend)
        
        # Get timestamp
        timestamp = datetime.now()
        
        # Generate status message
        if health_status == HealthStatus.HEALTHY:
            message = f"Backend {backend_name} is healthy"
        elif health_status == HealthStatus.DEGRADED:
            message = f"Backend {backend_name} is degraded (consecutive failures detected)"
        elif health_status == HealthStatus.UNHEALTHY:
            message = f"Backend {backend_name} is unhealthy"
        else:
            message = f"Backend {backend_name} status is unknown"
        
        result = {
            'backend_name': backend_name,
            'status': health_status.value,
            'timestamp': timestamp.isoformat(),
            'message': message
        }
        
        self.logger.info(
            f"Health check completed for {backend_name}: status={health_status.value}"
        )
        
        return result
