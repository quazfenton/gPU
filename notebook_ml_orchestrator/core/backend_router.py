"""
Multi-backend routing system for the Notebook ML Orchestrator.

This module implements intelligent job routing across multiple compute backends
with optimization for cost, performance, and availability.
"""

from typing import Dict, List, Optional, Any
import threading
from datetime import datetime, timedelta

from .interfaces import Backend, Job, BackendRouterInterface
from .models import HealthStatus, ResourceEstimate
from .exceptions import BackendNotAvailableError, BackendConnectionError
from .logging_config import LoggerMixin


class LoadBalancer:
    """Load balancing algorithms for backend selection."""

    def __init__(self):
        """Initialize the load balancer with state tracking."""
        self.round_robin_index = 0

    def round_robin(self, backends: List[Backend]) -> Backend:
        """
        Simple round-robin selection with index tracking.

        Args:
            backends: List of available backends

        Returns:
            Selected backend

        Raises:
            BackendNotAvailableError: If no backends available
        """
        if not backends:
            raise BackendNotAvailableError("No backends available")

        backend = backends[self.round_robin_index % len(backends)]
        self.round_robin_index += 1
        return backend

    def least_loaded(self, backends: List[Backend]) -> Backend:
        """
        Select backend with least load using get_queue_length.

        Args:
            backends: List of available backends

        Returns:
            Backend with shortest queue

        Raises:
            BackendNotAvailableError: If no backends available
        """
        if not backends:
            raise BackendNotAvailableError("No backends available")

        return min(backends, key=lambda b: b.get_queue_length())

    def weighted_random(self, backends: List[Backend], weights: Dict[str, float]) -> Backend:
        """
        Weighted random selection based on performance weights.

        Args:
            backends: List of available backends
            weights: Dictionary mapping backend IDs to weight values

        Returns:
            Randomly selected backend based on weights

        Raises:
            BackendNotAvailableError: If no backends available
        """
        if not backends:
            raise BackendNotAvailableError("No backends available")

        import random

        # Get weights for each backend, default to 1.0 if not specified
        backend_weights = [weights.get(b.id, 1.0) for b in backends]

        # Use random.choices for weighted selection
        return random.choices(backends, weights=backend_weights)[0]



class CostOptimizer:
    """Cost optimization for backend selection."""
    
    def __init__(self):
        self.cost_history = {}
        self.performance_metrics = {}
    
    def calculate_cost_efficiency(self, backend: Backend, resource_estimate: ResourceEstimate) -> float:
        """
        Calculate cost efficiency score for a backend.
        
        Args:
            backend: Backend to evaluate
            resource_estimate: Resource requirements
            
        Returns:
            Cost per hour (lower is better)
        """
        cost = backend.estimate_cost(resource_estimate)
        duration = resource_estimate.estimated_duration_minutes / 60.0
        
        # Cost per hour
        if duration > 0:
            return cost / duration
        return cost
    
    def get_cheapest_backend(self, backends: List[Backend], resource_estimate: ResourceEstimate) -> Backend:
        """
        Get the most cost-effective backend with free-tier preference.
        
        Args:
            backends: List of available backends
            resource_estimate: Resource requirements
            
        Returns:
            Most cost-effective backend
            
        Raises:
            BackendNotAvailableError: If no backends available
        """
        if not backends:
            raise BackendNotAvailableError("No backends available")
        
        import random
        
        # Prefer free tier backends
        free_backends = [b for b in backends if b.estimate_cost(resource_estimate) == 0.0]
        if free_backends:
            # Distribute load among free backends
            return random.choice(free_backends)
        
        # Otherwise, select cheapest paid backend
        return min(backends, key=lambda b: b.estimate_cost(resource_estimate))
    
    def track_cost(self, backend_id: str, cost: float):
        """
        Track cost history for a backend.
        
        Args:
            backend_id: Backend identifier
            cost: Cost to record
        """
        if backend_id not in self.cost_history:
            self.cost_history[backend_id] = []
        
        self.cost_history[backend_id].append({
            'cost': cost,
            'timestamp': datetime.now()
        })
    
    def get_total_cost(self, backend_id: Optional[str] = None) -> float:
        """
        Get total cost for a backend or all backends.
        
        Args:
            backend_id: Optional backend ID (None for all backends)
            
        Returns:
            Total cost
        """
        if backend_id:
            history = self.cost_history.get(backend_id, [])
            return sum(entry['cost'] for entry in history)
        
        # Sum across all backends
        total = 0.0
        for history in self.cost_history.values():
            total += sum(entry['cost'] for entry in history)
        return total


class HealthMonitor(LoggerMixin):
    """Backend health monitoring and status tracking."""
    
    def __init__(self):
        self.health_history = {}
        self.last_check_times = {}
        self.failure_counts = {}  # Track consecutive health check failures
        self.job_failure_counts = {}  # Track consecutive job execution failures
        self._lock = threading.RLock()

    
    def check_backend_health(self, backend: Backend) -> HealthStatus:
        """
        Check health of a specific backend.
        
        Args:
            backend: Backend to check
            
        Returns:
            Current health status
        """
        try:
            # Call the backend's check_health method
            status = backend.check_health()
            
            # Update health status
            self.update_health_status(backend.id, status)
            
            # Reset failure count on success
            if status == HealthStatus.HEALTHY:
                with self._lock:
                    self.failure_counts[backend.id] = 0
            else:
                # Increment failure count
                with self._lock:
                    self.failure_counts[backend.id] = self.failure_counts.get(backend.id, 0) + 1
                    
                    # Mark as degraded after 3 consecutive failures
                    if self.failure_counts[backend.id] >= 3:
                        status = HealthStatus.DEGRADED
                        self.update_health_status(backend.id, status)
            
            return status
            
        except Exception as e:
            # On exception, mark as unhealthy
            with self._lock:
                self.failure_counts[backend.id] = self.failure_counts.get(backend.id, 0) + 1
                
                # Mark as degraded after 3 consecutive failures
                if self.failure_counts[backend.id] >= 3:
                    status = HealthStatus.DEGRADED
                else:
                    status = HealthStatus.UNHEALTHY
                    
                self.update_health_status(backend.id, status)
            
            return status
    
    def update_health_status(self, backend_id: str, status: HealthStatus):
        """
        Update health status for a backend.
        
        Args:
            backend_id: Backend identifier
            status: New health status
        """
        with self._lock:
            if backend_id not in self.health_history:
                self.health_history[backend_id] = []
            
            # Store health check result with timestamp
            self.health_history[backend_id].append({
                'status': status,
                'timestamp': datetime.now()
            })
            
            self.last_check_times[backend_id] = datetime.now()
    
    def is_backend_healthy(self, backend_id: str) -> bool:
        """
        Check if backend is healthy.
        
        Args:
            backend_id: Backend identifier
            
        Returns:
            True if healthy, False otherwise
        """
        with self._lock:
            if backend_id not in self.health_history or not self.health_history[backend_id]:
                return False
            
            # Get most recent health check
            latest_check = self.health_history[backend_id][-1]
            
            # Consider backend unhealthy if not checked recently
            if datetime.now() - latest_check['timestamp'] > timedelta(minutes=5):
                return False
            
            return latest_check['status'] == HealthStatus.HEALTHY
    
    def should_check_health(self, backend_id: str, interval_seconds: int = 300) -> bool:
        """
        Determine if a health check is needed.
        
        Args:
            backend_id: Backend identifier
            interval_seconds: Check interval in seconds (default 5 minutes)
            
        Returns:
            True if health check is needed
        """
        last_check = self.last_check_times.get(backend_id)
        if not last_check:
            return True
        return (datetime.now() - last_check).total_seconds() > interval_seconds
    
    def get_health_metrics(self, backend_id: str) -> Dict[str, Any]:
        """
        Get health metrics for a backend.
        
        Args:
            backend_id: Backend identifier
            
        Returns:
            Dictionary with health metrics
        """
        with self._lock:
            history = self.health_history.get(backend_id, [])
            
            if not history:
                return {
                    'uptime_percentage': 0.0,
                    'total_checks': 0,
                    'healthy_checks': 0,
                    'failure_rate': 0.0,
                    'consecutive_failures': self.failure_counts.get(backend_id, 0),
                    'consecutive_job_failures': self.job_failure_counts.get(backend_id, 0)
                }
            
            total_checks = len(history)
            healthy_checks = sum(1 for entry in history if entry['status'] == HealthStatus.HEALTHY)
            
            uptime_percentage = (healthy_checks / total_checks) * 100.0 if total_checks > 0 else 0.0
            failure_rate = ((total_checks - healthy_checks) / total_checks) * 100.0 if total_checks > 0 else 0.0
            
            return {
                'uptime_percentage': uptime_percentage,
                'total_checks': total_checks,
                'healthy_checks': healthy_checks,
                'failure_rate': failure_rate,
                'last_check': self.last_check_times.get(backend_id),
                'consecutive_failures': self.failure_counts.get(backend_id, 0),
                'consecutive_job_failures': self.job_failure_counts.get(backend_id, 0)
            }
    
    def record_job_failure(self, backend_id: str):
        """
        Record a job execution failure for a backend.
        
        This method tracks consecutive job failures and marks backends as unhealthy
        after repeated failures, implementing Requirement 8.5.
        
        Args:
            backend_id: Backend identifier
            
        Requirements:
            - 8.5: Mark backends as unhealthy after consecutive job failures
        """
        with self._lock:
            # Increment consecutive job failure count
            self.job_failure_counts[backend_id] = self.job_failure_counts.get(backend_id, 0) + 1
            
            consecutive_failures = self.job_failure_counts[backend_id]
            
            # Mark backend as unhealthy after 3 consecutive job failures
            if consecutive_failures >= 3:
                self.update_health_status(backend_id, HealthStatus.UNHEALTHY)
                self.logger.warning(
                    f"Backend {backend_id} marked as UNHEALTHY after {consecutive_failures} "
                    f"consecutive job failures"
                )
            elif consecutive_failures >= 2:
                # Log warning after 2 failures
                self.logger.warning(
                    f"Backend {backend_id} has {consecutive_failures} consecutive job failures"
                )
    
    def record_job_success(self, backend_id: str):
        """
        Record a successful job execution for a backend.
        
        This method resets the consecutive job failure count when a job succeeds.
        
        Args:
            backend_id: Backend identifier
        """
        with self._lock:
            # Reset consecutive job failure count on success
            if backend_id in self.job_failure_counts and self.job_failure_counts[backend_id] > 0:
                previous_failures = self.job_failure_counts[backend_id]
                self.job_failure_counts[backend_id] = 0
                self.logger.info(
                    f"Backend {backend_id} job failure count reset after successful execution "
                    f"(was {previous_failures})"
                )
    
    def get_job_failure_count(self, backend_id: str) -> int:
        """
        Get the current consecutive job failure count for a backend.
        
        Args:
            backend_id: Backend identifier
            
        Returns:
            Number of consecutive job failures
        """
        with self._lock:
            return self.job_failure_counts.get(backend_id, 0)


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
    
    @staticmethod
    def template_to_resource_estimate(template) -> ResourceEstimate:
        """
        Convert template resource requirements to ResourceEstimate.
        
        This method extracts resource requirements from a template and creates
        a ResourceEstimate object for backend routing decisions.
        
        Args:
            template: Template instance with resource requirements
            
        Returns:
            ResourceEstimate object with template's resource requirements
            
        Requirements:
            - 8.1: Extract GPU requirements from templates
            - 8.2: Extract memory requirements from templates
            - 8.3: Extract timeout requirements from templates
        """
        # Extract GPU requirements
        requires_gpu = getattr(template, 'gpu_required', False)
        gpu_type = getattr(template, 'gpu_type', None)
        
        # Estimate GPU memory based on GPU type
        gpu_memory_gb = 0.0
        if requires_gpu and gpu_type:
            # Map GPU types to typical memory sizes
            gpu_memory_map = {
                'T4': 16.0,
                'A10G': 24.0,
                'A100': 40.0,
            }
            gpu_memory_gb = gpu_memory_map.get(gpu_type, 16.0)
        
        # Extract memory requirements (convert MB to GB)
        memory_mb = getattr(template, 'memory_mb', 512)
        memory_gb = memory_mb / 1024.0
        
        # Extract timeout (convert seconds to minutes)
        timeout_sec = getattr(template, 'timeout_sec', 300)
        estimated_duration_minutes = timeout_sec // 60
        
        return ResourceEstimate(
            cpu_cores=1,  # Default to 1 CPU core
            memory_gb=memory_gb,
            gpu_memory_gb=gpu_memory_gb,
            estimated_duration_minutes=estimated_duration_minutes,
            requires_gpu=requires_gpu,
            requires_internet=True  # Most ML templates need internet
        )
    
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
    
    def route_job(self, job: Job, routing_strategy: str = "cost-optimized", 
                  resource_estimate: Optional[ResourceEstimate] = None) -> Backend:
        """
        Select optimal backend for job execution with full routing logic.
        
        This method implements comprehensive capability validation:
        - Validates job requirements against backend capabilities (Requirement 10.5)
        - Filters backends based on GPU requirements (Requirement 10.6)
        - Excludes backends that don't meet resource requirements
        - Excludes unhealthy backends from routing decisions
        
        Args:
            job: Job to route
            routing_strategy: Strategy to use ("round-robin", "least-loaded", "cost-optimized")
            resource_estimate: Optional resource requirements for the job
            
        Returns:
            Selected backend
            
        Raises:
            BackendNotAvailableError: If no suitable backend is available
            
        Requirements:
            - 10.5: Capability-based validation before routing
            - 10.6: GPU capability filtering
        """
        with self._lock:
            # Get healthy backends that support the job template
            suitable_backends = []
            available_backend_ids = []
            excluded_backends = {}  # Track exclusion reasons for debugging
            
            for backend in self.backends.values():
                available_backend_ids.append(backend.id)
                
                # Check if backend is healthy
                is_healthy = self.health_monitor.is_backend_healthy(backend.id)
                
                # Perform health check if needed
                if self.health_monitor.should_check_health(backend.id):
                    health_status = self.health_monitor.check_backend_health(backend)
                    is_healthy = health_status == HealthStatus.HEALTHY
                
                # Exclude unhealthy backends
                if not is_healthy:
                    excluded_backends[backend.id] = "unhealthy"
                    self.logger.debug(f"Backend {backend.id} excluded: unhealthy")
                    continue
                
                # Check template support
                if not backend.supports_template(job.template_name):
                    excluded_backends[backend.id] = f"template '{job.template_name}' not supported"
                    self.logger.debug(f"Backend {backend.id} excluded: template not supported")
                    continue
                
                # Validate resource requirements against backend capabilities (Requirement 10.5)
                if resource_estimate:
                    # Check GPU requirement (Requirement 10.6)
                    if resource_estimate.requires_gpu and not backend.capabilities.supports_gpu:
                        excluded_backends[backend.id] = "no GPU support"
                        self.logger.debug(
                            f"Backend {backend.id} excluded: job requires GPU but backend doesn't support it"
                        )
                        continue
                    
                    # Check duration limits
                    if resource_estimate.estimated_duration_minutes > backend.capabilities.max_job_duration_minutes:
                        excluded_backends[backend.id] = (
                            f"duration {resource_estimate.estimated_duration_minutes}min exceeds "
                            f"limit {backend.capabilities.max_job_duration_minutes}min"
                        )
                        self.logger.debug(
                            f"Backend {backend.id} excluded: estimated duration "
                            f"{resource_estimate.estimated_duration_minutes}min exceeds "
                            f"max {backend.capabilities.max_job_duration_minutes}min"
                        )
                        continue
                    
                    # Check memory requirements
                    if resource_estimate.memory_gb > 0:
                        # Backends should have sufficient memory (basic validation)
                        # Note: BackendCapabilities doesn't currently have memory limits,
                        # but we log this for future enhancement
                        self.logger.debug(
                            f"Backend {backend.id}: job requires {resource_estimate.memory_gb}GB memory"
                        )
                    
                    # Check GPU memory requirements
                    if resource_estimate.gpu_memory_gb > 0 and not backend.capabilities.supports_gpu:
                        excluded_backends[backend.id] = (
                            f"requires {resource_estimate.gpu_memory_gb}GB GPU memory but no GPU support"
                        )
                        self.logger.debug(
                            f"Backend {backend.id} excluded: job requires "
                            f"{resource_estimate.gpu_memory_gb}GB GPU memory but backend has no GPU"
                        )
                        continue
                    
                    # Check CPU requirements
                    if resource_estimate.cpu_cores > 0:
                        # Log CPU requirements for monitoring
                        self.logger.debug(
                            f"Backend {backend.id}: job requires {resource_estimate.cpu_cores} CPU cores"
                        )
                    
                    # Check internet requirement
                    if resource_estimate.requires_internet:
                        # Most backends support internet, but log for completeness
                        self.logger.debug(
                            f"Backend {backend.id}: job requires internet access"
                        )
                
                # Backend passed all validation checks
                suitable_backends.append(backend)
                self.logger.debug(
                    f"Backend {backend.id} is suitable for job {job.id} (template: {job.template_name})"
                )
            
            if not suitable_backends:
                # Provide detailed error message with exclusion reasons
                error_details = [
                    f"No suitable backend available for job {job.id} (template: {job.template_name})"
                ]
                
                if excluded_backends:
                    error_details.append("Exclusion reasons:")
                    for backend_id, reason in excluded_backends.items():
                        error_details.append(f"  - {backend_id}: {reason}")
                
                if resource_estimate:
                    error_details.append("Job requirements:")
                    error_details.append(f"  - GPU required: {resource_estimate.requires_gpu}")
                    error_details.append(f"  - Duration: {resource_estimate.estimated_duration_minutes} minutes")
                    if resource_estimate.gpu_memory_gb > 0:
                        error_details.append(f"  - GPU memory: {resource_estimate.gpu_memory_gb} GB")
                    if resource_estimate.memory_gb > 0:
                        error_details.append(f"  - Memory: {resource_estimate.memory_gb} GB")
                    if resource_estimate.cpu_cores > 0:
                        error_details.append(f"  - CPU cores: {resource_estimate.cpu_cores}")
                
                error_message = "\n".join(error_details)
                self.logger.error(error_message)
                
                raise BackendNotAvailableError(
                    f"No suitable backend available for template {job.template_name}. "
                    f"Checked {len(available_backend_ids)} backends, all excluded. "
                    f"See logs for details.",
                    [job.template_name]
                )
            
            # Select backend based on routing strategy
            if routing_strategy == "round-robin":
                selected_backend = self.load_balancer.round_robin(suitable_backends)
            elif routing_strategy == "least-loaded":
                selected_backend = self.load_balancer.least_loaded(suitable_backends)
            elif routing_strategy == "cost-optimized":
                # Use default resource estimate if not provided
                if not resource_estimate:
                    resource_estimate = ResourceEstimate()
                selected_backend = self.cost_optimizer.get_cheapest_backend(suitable_backends, resource_estimate)
            else:
                # Default to round-robin
                selected_backend = self.load_balancer.round_robin(suitable_backends)
            
            # Log routing decision
            decision_factors = {
                'strategy': routing_strategy,
                'available_backends': available_backend_ids,
                'suitable_backends': [b.id for b in suitable_backends],
                'excluded_backends': excluded_backends,
                'selected_backend': selected_backend.id,
                'template': job.template_name
            }
            
            if resource_estimate:
                decision_factors['resource_requirements'] = {
                    'requires_gpu': resource_estimate.requires_gpu,
                    'estimated_duration_minutes': resource_estimate.estimated_duration_minutes,
                    'memory_gb': resource_estimate.memory_gb,
                    'gpu_memory_gb': resource_estimate.gpu_memory_gb,
                    'cpu_cores': resource_estimate.cpu_cores,
                    'requires_internet': resource_estimate.requires_internet
                }
            
            self.logger.info(
                f"Job {job.id} routed to backend {selected_backend.id} "
                f"using {routing_strategy} strategy. "
                f"Suitable: {len(suitable_backends)}/{len(self.backends)} backends "
                f"(excluded: {len(excluded_backends)})"
            )
            self.logger.debug(f"Routing decision: {decision_factors}")
            
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
                # Use the health monitor's method consistently
                health = self.health_monitor.check_backend_health(backend)
                self.health_monitor.update_health_status(backend_id, health)
                
                status[backend_id] = health

            return status
    
    def handle_backend_failure(self, backend_id: str, job: Job, 
                              routing_strategy: str = "cost-optimized",
                              resource_estimate: Optional[ResourceEstimate] = None) -> Optional[Backend]:
        """
        Handle backend failures with failover, state preservation, and comprehensive logging.

        This method implements automatic failover by:
        1. Marking the failed backend as unhealthy
        2. Preserving job state and inputs for failover
        3. Attempting to route to an alternative backend
        4. Logging the original failure and alternative backend selection

        Args:
            backend_id: Failed backend ID
            job: Job that failed
            routing_strategy: Strategy to use for alternative routing (default: "cost-optimized")
            resource_estimate: Optional resource requirements for the job

        Returns:
            Alternative backend if available, None otherwise
            
        Requirements:
            - 8.1: Automatic failover to alternative backend
            - 8.6: Job state preservation across failover
            - 8.7: Failover logging with original and alternative backends
        """
        self.logger.warning(
            f"Backend failure detected: backend_id={backend_id}, job_id={job.id}, "
            f"template={job.template_name}, retry_count={job.retry_count}"
        )

        # Record job failure and update health status based on consecutive failures (Requirement 8.5)
        self.health_monitor.record_job_failure(backend_id)

        # Preserve job state for failover (Requirements 8.6)
        # Store original backend information in metadata for tracking
        if 'failover_history' not in job.metadata:
            job.metadata['failover_history'] = []
        
        job.metadata['failover_history'].append({
            'failed_backend_id': backend_id,
            'failure_timestamp': datetime.now().isoformat(),
            'retry_count': job.retry_count,
            'job_status': job.status.value if hasattr(job.status, 'value') else str(job.status)
        })
        
        # Preserve original inputs and state
        preserved_state = {
            'original_inputs': job.inputs.copy(),
            'original_template': job.template_name,
            'original_priority': job.priority,
            'original_metadata': {k: v for k, v in job.metadata.items() if k != 'failover_history'}
        }
        
        self.logger.debug(f"Job state preserved for failover: job_id={job.id}, state={preserved_state}")

        # Try to route job to alternative backend (Requirements 8.1)
        try:
            # Exclude the failed backend from routing by temporarily removing it
            with self._lock:
                failed_backend = self.backends.get(backend_id)
                if failed_backend:
                    # Temporarily remove failed backend to prevent re-selection
                    self.backends.pop(backend_id, None)
                    
                    try:
                        alternative_backend = self.route_job(
                            job, 
                            routing_strategy=routing_strategy,
                            resource_estimate=resource_estimate
                        )
                        
                        # Comprehensive failover logging (Requirements 8.7)
                        self.logger.info(
                            f"Failover successful: job_id={job.id}, "
                            f"original_backend={backend_id}, "
                            f"alternative_backend={alternative_backend.id}, "
                            f"template={job.template_name}, "
                            f"routing_strategy={routing_strategy}, "
                            f"retry_count={job.retry_count}"
                        )
                        
                        # Store failover success in metadata
                        job.metadata['last_failover'] = {
                            'from_backend': backend_id,
                            'to_backend': alternative_backend.id,
                            'timestamp': datetime.now().isoformat(),
                            'routing_strategy': routing_strategy
                        }
                        
                        return alternative_backend
                        
                    finally:
                        # Restore the failed backend to the registry (still marked unhealthy)
                        self.backends[backend_id] = failed_backend
                else:
                    # Backend not found in registry
                    self.logger.error(f"Failed backend {backend_id} not found in registry")
                    alternative_backend = self.route_job(
                        job,
                        routing_strategy=routing_strategy,
                        resource_estimate=resource_estimate
                    )
                    
                    self.logger.info(
                        f"Failover successful (backend not in registry): job_id={job.id}, "
                        f"original_backend={backend_id}, "
                        f"alternative_backend={alternative_backend.id}"
                    )
                    
                    return alternative_backend
                    
        except BackendNotAvailableError as e:
            # No alternative backend available
            self.logger.error(
                f"Failover failed - no alternative backend available: "
                f"job_id={job.id}, original_backend={backend_id}, "
                f"template={job.template_name}, error={str(e)}"
            )
            
            # Store failover failure in metadata
            job.metadata['last_failover_attempt'] = {
                'from_backend': backend_id,
                'timestamp': datetime.now().isoformat(),
                'failure_reason': 'no_alternative_backend',
                'error': str(e)
            }
            
            return None
    
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
    
    def list_backends_with_capabilities(self) -> List[Dict[str, Any]]:
        """
        List all registered backends with their capability information.
        
        This method provides comprehensive capability information for each backend,
        including supported templates, GPU support, duration limits, and cost information.
        
        Returns:
            List of dictionaries, each containing:
            - id: Backend identifier
            - name: Backend name
            - type: Backend type (e.g., "modal", "huggingface")
            - health_status: Current health status
            - capabilities: BackendCapabilities object with:
                - supported_templates: List of template names
                - max_concurrent_jobs: Maximum concurrent job limit
                - max_job_duration_minutes: Maximum job duration
                - supports_gpu: GPU support flag
                - supports_batch: Batch processing support
                - cost_per_hour: Estimated cost per hour
                - free_tier_limits: Free tier limitations (if applicable)
            - queue_length: Current queue length
            - health_metrics: Health metrics (uptime, failure rate, etc.)
            
        Requirements:
            - 10.4: Backend listing with capabilities
            - 10.7: Capability API endpoint support
        """
        with self._lock:
            backends_info = []
            
            for backend_id, backend in self.backends.items():
                # Get current health status
                is_healthy = self.health_monitor.is_backend_healthy(backend_id)
                
                # Perform health check if needed
                if self.health_monitor.should_check_health(backend_id):
                    health_status = self.health_monitor.check_backend_health(backend)
                else:
                    # Get most recent health status from history
                    history = self.health_monitor.health_history.get(backend_id, [])
                    if history:
                        health_status = history[-1]['status']
                    else:
                        health_status = HealthStatus.UNKNOWN
                
                # Get health metrics
                health_metrics = self.health_monitor.get_health_metrics(backend_id)
                
                # Build backend info dictionary
                backend_info = {
                    'id': backend.id,
                    'name': backend.name,
                    'type': backend.type.value,
                    'health_status': health_status.value,
                    'capabilities': {
                        'supported_templates': backend.capabilities.supported_templates,
                        'max_concurrent_jobs': backend.capabilities.max_concurrent_jobs,
                        'max_job_duration_minutes': backend.capabilities.max_job_duration_minutes,
                        'supports_gpu': backend.capabilities.supports_gpu,
                        'supports_batch': backend.capabilities.supports_batch,
                        'cost_per_hour': backend.capabilities.cost_per_hour,
                        'free_tier_limits': backend.capabilities.free_tier_limits,
                    },
                    'queue_length': backend.get_queue_length(),
                    'health_metrics': health_metrics,
                }
                
                backends_info.append(backend_info)
            
            self.logger.debug(
                f"Listed {len(backends_info)} backends with capabilities"
            )
            
            return backends_info
    
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
    
    def execute_job_with_retry(self, job: Job, backend: Backend,
                               routing_strategy: str = "cost-optimized",
                               resource_estimate: Optional[ResourceEstimate] = None,
                               max_retries: Optional[int] = None) -> Any:
        """
        Execute a job with automatic retry logic and exponential backoff.

        This method implements:
        1. Retry counter tracking per job execution
        2. Exponential backoff calculation between retries
        3. Retry limit enforcement (default 3, configurable)
        4. Handling of no alternative backend scenario

        Args:
            job: Job to execute
            backend: Backend to execute on
            routing_strategy: Strategy for failover routing (default: "cost-optimized")
            resource_estimate: Optional resource requirements
            max_retries: Maximum retry attempts (default from config: 3)

        Returns:
            Job execution result

        Raises:
            JobExecutionError: If job fails after all retries exhausted
            BackendNotAvailableError: If no alternative backend available for failover

        Requirements:
            - 8.2: Handle no alternative backend scenario
            - 8.3: Retry limit enforcement (default 3)
            - 8.4: Exponential backoff calculation
        """
        import time
        from notebook_ml_orchestrator.config import get_config
        from .exceptions import JobExecutionError

        # Get retry configuration
        config = get_config()
        if max_retries is None:
            max_retries = config.job_queue.max_retries

        base_delay = config.job_queue.base_retry_delay
        exponential_base = config.job_queue.exponential_base
        max_delay = config.job_queue.max_retry_delay

        current_backend = backend
        last_error = None

        # Retry loop with exponential backoff
        while job.retry_count <= max_retries:
            try:
                self.logger.info(
                    f"Executing job {job.id} on backend {current_backend.id} "
                    f"(attempt {job.retry_count + 1}/{max_retries + 1})"
                )

                # Execute the job on the current backend
                result = current_backend.execute_job(job)

                # Success - record job success and return result
                self.health_monitor.record_job_success(current_backend.id)
                
                self.logger.info(
                    f"Job {job.id} completed successfully on backend {current_backend.id} "
                    f"after {job.retry_count} retries"
                )

                return result

            except Exception as e:
                last_error = e
                job.retry_count += 1

                self.logger.warning(
                    f"Job {job.id} failed on backend {current_backend.id}: {str(e)} "
                    f"(attempt {job.retry_count}/{max_retries + 1})"
                )

                # Check if we've exhausted retries
                if job.retry_count > max_retries:
                    self.logger.error(
                        f"Job {job.id} failed after {max_retries + 1} attempts. "
                        f"Last error: {str(e)}"
                    )
                    raise JobExecutionError(
                        f"Job {job.id} failed after {max_retries + 1} attempts: {str(e)}",
                        job_id=job.id
                    )

                # Try to failover to alternative backend
                alternative_backend = self.handle_backend_failure(
                    current_backend.id,
                    job,
                    routing_strategy=routing_strategy,
                    resource_estimate=resource_estimate
                )

                if alternative_backend is None:
                    # No alternative backend available (Requirement 8.2)
                    self.logger.error(
                        f"No alternative backend available for job {job.id} after failure. "
                        f"Retry count: {job.retry_count}/{max_retries + 1}"
                    )

                    # Calculate exponential backoff delay (Requirement 8.4)
                    delay = min(base_delay * (exponential_base ** (job.retry_count - 1)), max_delay)

                    self.logger.info(
                        f"Retrying job {job.id} on same backend {current_backend.id} "
                        f"after {delay:.2f}s backoff delay"
                    )

                    # Wait before retrying on the same backend
                    time.sleep(delay)

                    # Continue with same backend
                    continue
                else:
                    # Switch to alternative backend
                    self.logger.info(
                        f"Switching job {job.id} from backend {current_backend.id} "
                        f"to alternative backend {alternative_backend.id}"
                    )
                    current_backend = alternative_backend

                    # Calculate exponential backoff delay (Requirement 8.4)
                    delay = min(base_delay * (exponential_base ** (job.retry_count - 1)), max_delay)

                    self.logger.info(
                        f"Retrying job {job.id} on alternative backend {alternative_backend.id} "
                        f"after {delay:.2f}s backoff delay"
                    )

                    # Wait before retrying on alternative backend
                    time.sleep(delay)

                    # Continue with alternative backend
                    continue

        # Should not reach here, but handle edge case
        raise JobExecutionError(
            f"Job {job.id} failed after {max_retries + 1} attempts: {str(last_error)}",
            job_id=job.id
        )

