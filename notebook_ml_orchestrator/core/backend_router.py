"""
Multi-backend routing system for the Notebook ML Orchestrator.

This module implements intelligent job routing across multiple compute backends
with optimization for cost, performance, and availability.
"""

from typing import Dict, List, Optional
import threading
from datetime import datetime, timedelta

from .interfaces import Backend, Job, BackendRouterInterface
from .models import HealthStatus, ResourceEstimate
from .exceptions import BackendNotAvailableError, BackendConnectionError
from .logging_config import LoggerMixin


class LoadBalancer:
    """Load balancing algorithms for backend selection."""
    
    @staticmethod
    def round_robin(backends: List[Backend]) -> Backend:
        """Simple round-robin selection."""
        # Implementation will be added in task 4.2
        pass
    
    @staticmethod
    def least_loaded(backends: List[Backend]) -> Backend:
        """Select backend with least load."""
        # Implementation will be added in task 4.2
        pass
    
    @staticmethod
    def weighted_random(backends: List[Backend], weights: Dict[str, float]) -> Backend:
        """Weighted random selection based on performance."""
        # Implementation will be added in task 4.2
        pass


class CostOptimizer:
    """Cost optimization for backend selection."""
    
    def __init__(self):
        self.cost_history = {}
        self.performance_metrics = {}
    
    def calculate_cost_efficiency(self, backend: Backend, resource_estimate: ResourceEstimate) -> float:
        """Calculate cost efficiency score for a backend."""
        # Implementation will be added in task 4.2
        pass
    
    def get_cheapest_backend(self, backends: List[Backend], resource_estimate: ResourceEstimate) -> Backend:
        """Get the most cost-effective backend."""
        # Implementation will be added in task 4.2
        pass


class HealthMonitor:
    """Backend health monitoring and status tracking."""
    
    def __init__(self):
        self.health_history = {}
        self.last_check_times = {}
        self._lock = threading.RLock()
    
    def check_backend_health(self, backend: Backend) -> HealthStatus:
        """Check health of a specific backend."""
        # Implementation will be added in task 4.1
        pass
    
    def update_health_status(self, backend_id: str, status: HealthStatus):
        """Update health status for a backend."""
        with self._lock:
            self.health_history[backend_id] = {
                'status': status,
                'timestamp': datetime.now()
            }
            self.last_check_times[backend_id] = datetime.now()
    
    def is_backend_healthy(self, backend_id: str) -> bool:
        """Check if backend is healthy."""
        with self._lock:
            health_info = self.health_history.get(backend_id)
            if not health_info:
                return False
            
            # Consider backend unhealthy if not checked recently
            if datetime.now() - health_info['timestamp'] > timedelta(minutes=5):
                return False
            
            return health_info['status'] == HealthStatus.HEALTHY


class MultiBackendRouter(BackendRouterInterface, LoggerMixin):
    """Intelligent multi-backend routing system."""
    
    def __init__(self):
        """Initialize the multi-backend router."""
        self.backends: Dict[str, Backend] = {}
        self.load_balancer = LoadBalancer()
        self.cost_optimizer = CostOptimizer()
        self.health_monitor = HealthMonitor()
        self._lock = threading.RLock()
        
        self.logger.info("Multi-backend router initialized")
    
    def register_backend(self, backend: Backend):
        """
        Register a new compute backend.
        
        Args:
            backend: Backend instance to register
        """
        with self._lock:
            self.backends[backend.id] = backend
            self.logger.info(f"Backend {backend.id} ({backend.type.value}) registered")
    
    def unregister_backend(self, backend_id: str) -> bool:
        """
        Unregister a backend.
        
        Args:
            backend_id: Backend ID to unregister
            
        Returns:
            True if successful, False if backend not found
        """
        with self._lock:
            if backend_id in self.backends:
                del self.backends[backend_id]
                self.logger.info(f"Backend {backend_id} unregistered")
                return True
            return False
    
    def route_job(self, job: Job) -> Backend:
        """
        Select optimal backend for job execution.

        Args:
            job: Job to route

        Returns:
            Selected backend

        Raises:
            BackendNotAvailableError: If no suitable backend is available
        """
        with self._lock:
            # Get healthy backends that support the job template
            suitable_backends = []

            for backend in self.backends.values():
                # Check if backend supports the template first (cheaper check)
                if not backend.supports_template(job.template_name):
                    continue

                # Check health status and update if no history exists
                if not self.health_monitor.is_backend_healthy(backend.id):
                    # Attempt to check health if no history exists
                    health_status = self.health_monitor.check_backend_health(backend)
                    self.health_monitor.update_health_status(backend.id, health_status)
                    if health_status != HealthStatus.HEALTHY:
                        continue

                suitable_backends.append(backend)

            if not suitable_backends:
                raise BackendNotAvailableError(
                    f"No suitable backend available for template {job.template_name}",
                    [job.template_name]
                )
            
            # For now, return the first suitable backend
            # Full routing logic will be implemented in task 4.2
            selected_backend = suitable_backends[0]
            
            self.logger.info(f"Job {job.id} routed to backend {selected_backend.id}")
            return selected_backend
    
    def get_backend_status(self) -> Dict[str, HealthStatus]:
        """
        Get current status of all backends.
        
        Returns:
            Dictionary mapping backend IDs to health status
        """
        with self._lock:
            status = {}
            for backend_id, backend in self.backends.items():
                # Check health if not checked recently
                if not self.health_monitor.is_backend_healthy(backend_id):
                    try:
                        health = backend.check_health()
                        self.health_monitor.update_health_status(backend_id, health)
                    except Exception as e:
                        self.logger.error(f"Health check failed for backend {backend_id}: {e}")
                        health = HealthStatus.UNHEALTHY
                        self.health_monitor.update_health_status(backend_id, health)
                
                status[backend_id] = self.health_monitor.health_history.get(
                    backend_id, {'status': HealthStatus.UNKNOWN}
                )['status']
            
            return status
    
    def handle_backend_failure(self, backend_id: str, job: Job):
        """
        Handle backend failures with failover.
        
        Args:
            backend_id: Failed backend ID
            job: Job that failed
        """
        self.logger.warning(f"Backend {backend_id} failed for job {job.id}")
        
        # Mark backend as unhealthy
        self.health_monitor.update_health_status(backend_id, HealthStatus.UNHEALTHY)
        
        # Try to route job to alternative backend
        try:
            alternative_backend = self.route_job(job)
            self.logger.info(f"Job {job.id} failed over to backend {alternative_backend.id}")
        except BackendNotAvailableError:
            self.logger.error(f"No alternative backend available for job {job.id}")
    
    def get_backend(self, backend_id: str) -> Optional[Backend]:
        """
        Get backend by ID.
        
        Args:
            backend_id: Backend ID
            
        Returns:
            Backend instance or None if not found
        """
        return self.backends.get(backend_id)
    
    def list_backends(self) -> List[Backend]:
        """
        List all registered backends.
        
        Returns:
            List of all backends
        """
        return list(self.backends.values())
    
    def get_routing_statistics(self) -> Dict:
        """
        Get routing statistics.
        
        Returns:
            Dictionary with routing statistics
        """
        with self._lock:
            stats = {
                'total_backends': len(self.backends),
                'healthy_backends': 0,
                'backends_by_type': {},
                'backend_details': {}
            }
            
            for backend_id, backend in self.backends.items():
                backend_type = backend.type.value
                if backend_type not in stats['backends_by_type']:
                    stats['backends_by_type'][backend_type] = 0
                stats['backends_by_type'][backend_type] += 1
                
                is_healthy = self.health_monitor.is_backend_healthy(backend_id)
                if is_healthy:
                    stats['healthy_backends'] += 1
                
                stats['backend_details'][backend_id] = {
                    'name': backend.name,
                    'type': backend_type,
                    'healthy': is_healthy,
                    'queue_length': backend.get_queue_length()
                }
            
            return stats