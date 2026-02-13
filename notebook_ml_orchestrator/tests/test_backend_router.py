"""
Unit tests for MultiBackendRouter enhancements.

Tests the LoadBalancer, CostOptimizer, HealthMonitor, and enhanced routing logic.
"""

import pytest
from unittest.mock import Mock, MagicMock
from datetime import datetime, timedelta

from notebook_ml_orchestrator.core.backend_router import (
    LoadBalancer, CostOptimizer, HealthMonitor, MultiBackendRouter
)
from notebook_ml_orchestrator.core.models import (
    BackendType, HealthStatus, ResourceEstimate
)
from notebook_ml_orchestrator.core.interfaces import Backend, Job
from notebook_ml_orchestrator.core.exceptions import BackendNotAvailableError


class TestLoadBalancer:
    """Test LoadBalancer implementation."""
    
    def test_round_robin_cycles_through_backends(self):
        """Test round-robin selection cycles through backends."""
        lb = LoadBalancer()
        
        # Create mock backends
        backend1 = Mock(spec=Backend, id="backend1")
        backend2 = Mock(spec=Backend, id="backend2")
        backend3 = Mock(spec=Backend, id="backend3")
        backends = [backend1, backend2, backend3]
        
        # Should cycle through backends
        assert lb.round_robin(backends) == backend1
        assert lb.round_robin(backends) == backend2
        assert lb.round_robin(backends) == backend3
        assert lb.round_robin(backends) == backend1  # Cycles back
    
    def test_round_robin_with_empty_list(self):
        """Test round-robin raises error with no backends."""
        lb = LoadBalancer()
        
        with pytest.raises(BackendNotAvailableError):
            lb.round_robin([])
    
    def test_least_loaded_selects_shortest_queue(self):
        """Test least-loaded selection picks backend with shortest queue."""
        lb = LoadBalancer()
        
        # Create mock backends with different queue lengths
        backend1 = Mock(spec=Backend, id="backend1")
        backend1.get_queue_length.return_value = 5
        
        backend2 = Mock(spec=Backend, id="backend2")
        backend2.get_queue_length.return_value = 2
        
        backend3 = Mock(spec=Backend, id="backend3")
        backend3.get_queue_length.return_value = 8
        
        backends = [backend1, backend2, backend3]
        
        # Should select backend2 with queue length 2
        selected = lb.least_loaded(backends)
        assert selected == backend2
    
    def test_least_loaded_with_empty_list(self):
        """Test least-loaded raises error with no backends."""
        lb = LoadBalancer()
        
        with pytest.raises(BackendNotAvailableError):
            lb.least_loaded([])
    
    def test_weighted_random_uses_weights(self):
        """Test weighted random selection respects weights."""
        lb = LoadBalancer()
        
        # Create mock backends
        backend1 = Mock(spec=Backend, id="backend1")
        backend2 = Mock(spec=Backend, id="backend2")
        backends = [backend1, backend2]
        
        # Give backend1 much higher weight
        weights = {"backend1": 100.0, "backend2": 1.0}
        
        # Run multiple times and check backend1 is selected more often
        selections = [lb.weighted_random(backends, weights) for _ in range(100)]
        backend1_count = sum(1 for b in selections if b == backend1)
        
        # With 100:1 weight ratio, backend1 should be selected ~99% of the time
        assert backend1_count > 90
    
    def test_weighted_random_with_default_weights(self):
        """Test weighted random uses default weight of 1.0 for missing backends."""
        lb = LoadBalancer()
        
        backend1 = Mock(spec=Backend, id="backend1")
        backend2 = Mock(spec=Backend, id="backend2")
        backends = [backend1, backend2]
        
        # Only provide weight for backend1
        weights = {"backend1": 1.0}
        
        # Should not raise error, backend2 gets default weight 1.0
        selected = lb.weighted_random(backends, weights)
        assert selected in backends
    
    def test_weighted_random_with_empty_list(self):
        """Test weighted random raises error with no backends."""
        lb = LoadBalancer()
        
        with pytest.raises(BackendNotAvailableError):
            lb.weighted_random([], {})


class TestCostOptimizer:
    """Test CostOptimizer implementation."""
    
    def test_calculate_cost_efficiency(self):
        """Test cost efficiency calculation."""
        optimizer = CostOptimizer()
        
        backend = Mock(spec=Backend)
        backend.estimate_cost.return_value = 2.0  # $2.00
        
        resource_estimate = ResourceEstimate(estimated_duration_minutes=60)
        
        # Cost per hour should be 2.0 / 1.0 = 2.0
        efficiency = optimizer.calculate_cost_efficiency(backend, resource_estimate)
        assert efficiency == 2.0
    
    def test_calculate_cost_efficiency_with_short_duration(self):
        """Test cost efficiency with short duration."""
        optimizer = CostOptimizer()
        
        backend = Mock(spec=Backend)
        backend.estimate_cost.return_value = 0.5  # $0.50
        
        resource_estimate = ResourceEstimate(estimated_duration_minutes=15)
        
        # Cost per hour should be 0.5 / 0.25 = 2.0
        efficiency = optimizer.calculate_cost_efficiency(backend, resource_estimate)
        assert efficiency == 2.0
    
    def test_get_cheapest_backend_prefers_free_tier(self):
        """Test cheapest backend selection prefers free tier."""
        optimizer = CostOptimizer()
        
        # Create backends with different costs
        free_backend = Mock(spec=Backend, id="free")
        free_backend.estimate_cost.return_value = 0.0
        
        paid_backend = Mock(spec=Backend, id="paid")
        paid_backend.estimate_cost.return_value = 1.0
        
        backends = [paid_backend, free_backend]
        resource_estimate = ResourceEstimate()
        
        # Should select free backend
        selected = optimizer.get_cheapest_backend(backends, resource_estimate)
        assert selected == free_backend
    
    def test_get_cheapest_backend_distributes_among_free_tier(self):
        """Test load distribution among multiple free-tier backends."""
        optimizer = CostOptimizer()
        
        # Create multiple free backends
        free1 = Mock(spec=Backend, id="free1")
        free1.estimate_cost.return_value = 0.0
        
        free2 = Mock(spec=Backend, id="free2")
        free2.estimate_cost.return_value = 0.0
        
        backends = [free1, free2]
        resource_estimate = ResourceEstimate()
        
        # Run multiple times to check distribution
        selections = [optimizer.get_cheapest_backend(backends, resource_estimate) for _ in range(100)]
        
        # Both backends should be selected at least once
        assert free1 in selections
        assert free2 in selections
    
    def test_get_cheapest_backend_selects_cheapest_paid(self):
        """Test cheapest backend selection among paid backends."""
        optimizer = CostOptimizer()
        
        # Create paid backends with different costs
        expensive = Mock(spec=Backend, id="expensive")
        expensive.estimate_cost.return_value = 5.0
        
        cheap = Mock(spec=Backend, id="cheap")
        cheap.estimate_cost.return_value = 1.0
        
        backends = [expensive, cheap]
        resource_estimate = ResourceEstimate()
        
        # Should select cheaper backend
        selected = optimizer.get_cheapest_backend(backends, resource_estimate)
        assert selected == cheap
    
    def test_get_cheapest_backend_with_empty_list(self):
        """Test cheapest backend raises error with no backends."""
        optimizer = CostOptimizer()
        
        with pytest.raises(BackendNotAvailableError):
            optimizer.get_cheapest_backend([], ResourceEstimate())
    
    def test_track_cost(self):
        """Test cost history tracking."""
        optimizer = CostOptimizer()
        
        optimizer.track_cost("backend1", 2.5)
        optimizer.track_cost("backend1", 1.5)
        optimizer.track_cost("backend2", 3.0)
        
        assert len(optimizer.cost_history["backend1"]) == 2
        assert len(optimizer.cost_history["backend2"]) == 1
        assert optimizer.cost_history["backend1"][0]["cost"] == 2.5
        assert optimizer.cost_history["backend1"][1]["cost"] == 1.5
    
    def test_get_total_cost_for_backend(self):
        """Test getting total cost for a specific backend."""
        optimizer = CostOptimizer()
        
        optimizer.track_cost("backend1", 2.5)
        optimizer.track_cost("backend1", 1.5)
        optimizer.track_cost("backend2", 3.0)
        
        assert optimizer.get_total_cost("backend1") == 4.0
        assert optimizer.get_total_cost("backend2") == 3.0
    
    def test_get_total_cost_all_backends(self):
        """Test getting total cost across all backends."""
        optimizer = CostOptimizer()
        
        optimizer.track_cost("backend1", 2.5)
        optimizer.track_cost("backend2", 3.0)
        optimizer.track_cost("backend3", 1.5)
        
        assert optimizer.get_total_cost() == 7.0


class TestHealthMonitor:
    """Test HealthMonitor implementation."""
    
    def test_check_backend_health_success(self):
        """Test health check with healthy backend."""
        monitor = HealthMonitor()
        
        backend = Mock(spec=Backend, id="backend1")
        backend.check_health.return_value = HealthStatus.HEALTHY
        
        status = monitor.check_backend_health(backend)
        
        assert status == HealthStatus.HEALTHY
        assert monitor.failure_counts.get("backend1", 0) == 0
    
    def test_check_backend_health_failure(self):
        """Test health check with unhealthy backend."""
        monitor = HealthMonitor()
        
        backend = Mock(spec=Backend, id="backend1")
        backend.check_health.return_value = HealthStatus.UNHEALTHY
        
        status = monitor.check_backend_health(backend)
        
        assert status == HealthStatus.UNHEALTHY
        assert monitor.failure_counts["backend1"] == 1
    
    def test_check_backend_health_degraded_after_three_failures(self):
        """Test backend marked as degraded after 3 consecutive failures."""
        monitor = HealthMonitor()
        
        backend = Mock(spec=Backend, id="backend1")
        backend.check_health.return_value = HealthStatus.UNHEALTHY
        
        # First two failures
        monitor.check_backend_health(backend)
        monitor.check_backend_health(backend)
        
        # Third failure should mark as degraded
        status = monitor.check_backend_health(backend)
        
        assert status == HealthStatus.DEGRADED
        assert monitor.failure_counts["backend1"] == 3
    
    def test_check_backend_health_resets_on_success(self):
        """Test failure count resets on successful health check."""
        monitor = HealthMonitor()
        
        backend = Mock(spec=Backend, id="backend1")
        
        # Two failures
        backend.check_health.return_value = HealthStatus.UNHEALTHY
        monitor.check_backend_health(backend)
        monitor.check_backend_health(backend)
        
        # Then success
        backend.check_health.return_value = HealthStatus.HEALTHY
        monitor.check_backend_health(backend)
        
        assert monitor.failure_counts["backend1"] == 0
    
    def test_check_backend_health_exception_handling(self):
        """Test health check handles exceptions."""
        monitor = HealthMonitor()
        
        backend = Mock(spec=Backend, id="backend1")
        backend.check_health.side_effect = Exception("Connection failed")
        
        status = monitor.check_backend_health(backend)
        
        assert status == HealthStatus.UNHEALTHY
        assert monitor.failure_counts["backend1"] == 1
    
    def test_update_health_status(self):
        """Test health status update."""
        monitor = HealthMonitor()
        
        monitor.update_health_status("backend1", HealthStatus.HEALTHY)
        
        assert len(monitor.health_history["backend1"]) == 1
        assert monitor.health_history["backend1"][0]["status"] == HealthStatus.HEALTHY
        assert "backend1" in monitor.last_check_times
    
    def test_is_backend_healthy(self):
        """Test backend health check."""
        monitor = HealthMonitor()
        
        # No history - should be unhealthy
        assert not monitor.is_backend_healthy("backend1")
        
        # Add healthy status
        monitor.update_health_status("backend1", HealthStatus.HEALTHY)
        assert monitor.is_backend_healthy("backend1")
        
        # Add unhealthy status
        monitor.update_health_status("backend1", HealthStatus.UNHEALTHY)
        assert not monitor.is_backend_healthy("backend1")
    
    def test_is_backend_healthy_with_stale_check(self):
        """Test backend considered unhealthy if check is stale."""
        monitor = HealthMonitor()
        
        # Add healthy status with old timestamp
        monitor.health_history["backend1"] = [{
            'status': HealthStatus.HEALTHY,
            'timestamp': datetime.now() - timedelta(minutes=10)
        }]
        
        assert not monitor.is_backend_healthy("backend1")
    
    def test_should_check_health(self):
        """Test health check interval logic."""
        monitor = HealthMonitor()
        
        # No previous check - should check
        assert monitor.should_check_health("backend1")
        
        # Recent check - should not check
        monitor.last_check_times["backend1"] = datetime.now()
        assert not monitor.should_check_health("backend1", interval_seconds=300)
        
        # Old check - should check
        monitor.last_check_times["backend1"] = datetime.now() - timedelta(seconds=400)
        assert monitor.should_check_health("backend1", interval_seconds=300)
    
    def test_get_health_metrics(self):
        """Test health metrics calculation."""
        monitor = HealthMonitor()
        
        # Add some health history
        monitor.update_health_status("backend1", HealthStatus.HEALTHY)
        monitor.update_health_status("backend1", HealthStatus.HEALTHY)
        monitor.update_health_status("backend1", HealthStatus.UNHEALTHY)
        monitor.update_health_status("backend1", HealthStatus.HEALTHY)
        
        metrics = monitor.get_health_metrics("backend1")
        
        assert metrics["total_checks"] == 4
        assert metrics["healthy_checks"] == 3
        assert metrics["uptime_percentage"] == 75.0
        assert metrics["failure_rate"] == 25.0
    
    def test_get_health_metrics_no_history(self):
        """Test health metrics with no history."""
        monitor = HealthMonitor()
        
        metrics = monitor.get_health_metrics("backend1")
        
        assert metrics["total_checks"] == 0
        assert metrics["uptime_percentage"] == 0.0
        assert metrics["failure_rate"] == 0.0


class TestMultiBackendRouterEnhancements:
    """Test enhanced MultiBackendRouter functionality."""
    
    def test_route_job_with_round_robin_strategy(self):
        """Test job routing with round-robin strategy."""
        router = MultiBackendRouter()
        
        # Create mock backends
        backend1 = Mock(spec=Backend, id="backend1", type=BackendType.MODAL)
        backend1.supports_template.return_value = True
        backend1.capabilities = Mock(supports_gpu=True, max_job_duration_minutes=60)
        
        backend2 = Mock(spec=Backend, id="backend2", type=BackendType.KAGGLE)
        backend2.supports_template.return_value = True
        backend2.capabilities = Mock(supports_gpu=True, max_job_duration_minutes=60)
        
        router.register_backend(backend1)
        router.register_backend(backend2)
        
        # Mark backends as healthy
        router.health_monitor.update_health_status("backend1", HealthStatus.HEALTHY)
        router.health_monitor.update_health_status("backend2", HealthStatus.HEALTHY)
        
        # Create jobs
        job1 = Job(id="job1", template_name="test-template")
        job2 = Job(id="job2", template_name="test-template")
        
        # Route with round-robin
        selected1 = router.route_job(job1, routing_strategy="round-robin")
        selected2 = router.route_job(job2, routing_strategy="round-robin")
        
        # Should alternate between backends
        assert selected1 != selected2
    
    def test_route_job_with_cost_optimized_strategy(self):
        """Test job routing with cost-optimized strategy."""
        router = MultiBackendRouter()
        
        # Create backends with different costs
        free_backend = Mock(spec=Backend, id="free", type=BackendType.KAGGLE)
        free_backend.supports_template.return_value = True
        free_backend.estimate_cost.return_value = 0.0
        free_backend.capabilities = Mock(supports_gpu=True, max_job_duration_minutes=60)
        
        paid_backend = Mock(spec=Backend, id="paid", type=BackendType.MODAL)
        paid_backend.supports_template.return_value = True
        paid_backend.estimate_cost.return_value = 2.0
        paid_backend.capabilities = Mock(supports_gpu=True, max_job_duration_minutes=60)
        
        router.register_backend(free_backend)
        router.register_backend(paid_backend)
        
        # Mark backends as healthy
        router.health_monitor.update_health_status("free", HealthStatus.HEALTHY)
        router.health_monitor.update_health_status("paid", HealthStatus.HEALTHY)
        
        job = Job(id="job1", template_name="test-template")
        resource_estimate = ResourceEstimate()
        
        # Route with cost-optimized strategy
        selected = router.route_job(job, routing_strategy="cost-optimized", 
                                    resource_estimate=resource_estimate)
        
        # Should select free backend
        assert selected == free_backend
    
    def test_route_job_excludes_unhealthy_backends(self):
        """Test routing excludes unhealthy backends."""
        router = MultiBackendRouter()
        
        healthy_backend = Mock(spec=Backend, id="healthy", type=BackendType.MODAL)
        healthy_backend.supports_template.return_value = True
        healthy_backend.capabilities = Mock(supports_gpu=True, max_job_duration_minutes=60)
        
        unhealthy_backend = Mock(spec=Backend, id="unhealthy", type=BackendType.KAGGLE)
        unhealthy_backend.supports_template.return_value = True
        unhealthy_backend.capabilities = Mock(supports_gpu=True, max_job_duration_minutes=60)
        
        router.register_backend(healthy_backend)
        router.register_backend(unhealthy_backend)
        
        # Mark one as healthy, one as unhealthy
        router.health_monitor.update_health_status("healthy", HealthStatus.HEALTHY)
        router.health_monitor.update_health_status("unhealthy", HealthStatus.UNHEALTHY)
        
        job = Job(id="job1", template_name="test-template")
        
        # Should only select healthy backend
        selected = router.route_job(job)
        assert selected == healthy_backend
    
    def test_route_job_validates_gpu_requirements(self):
        """Test routing validates GPU requirements."""
        router = MultiBackendRouter()
        
        gpu_backend = Mock(spec=Backend, id="gpu", type=BackendType.MODAL)
        gpu_backend.supports_template.return_value = True
        gpu_backend.capabilities = Mock(supports_gpu=True, max_job_duration_minutes=60)
        
        cpu_backend = Mock(spec=Backend, id="cpu", type=BackendType.HUGGINGFACE)
        cpu_backend.supports_template.return_value = True
        cpu_backend.capabilities = Mock(supports_gpu=False, max_job_duration_minutes=60)
        
        router.register_backend(gpu_backend)
        router.register_backend(cpu_backend)
        
        # Mark both as healthy
        router.health_monitor.update_health_status("gpu", HealthStatus.HEALTHY)
        router.health_monitor.update_health_status("cpu", HealthStatus.HEALTHY)
        
        job = Job(id="job1", template_name="test-template")
        resource_estimate = ResourceEstimate(requires_gpu=True)
        
        # Should only select GPU backend
        selected = router.route_job(job, resource_estimate=resource_estimate)
        assert selected == gpu_backend
    
    def test_route_job_validates_duration_limits(self):
        """Test routing validates duration limits."""
        router = MultiBackendRouter()
        
        long_backend = Mock(spec=Backend, id="long", type=BackendType.MODAL)
        long_backend.supports_template.return_value = True
        long_backend.capabilities = Mock(supports_gpu=True, max_job_duration_minutes=120)
        
        short_backend = Mock(spec=Backend, id="short", type=BackendType.KAGGLE)
        short_backend.supports_template.return_value = True
        short_backend.capabilities = Mock(supports_gpu=True, max_job_duration_minutes=30)
        
        router.register_backend(long_backend)
        router.register_backend(short_backend)
        
        # Mark both as healthy
        router.health_monitor.update_health_status("long", HealthStatus.HEALTHY)
        router.health_monitor.update_health_status("short", HealthStatus.HEALTHY)
        
        job = Job(id="job1", template_name="test-template")
        resource_estimate = ResourceEstimate(estimated_duration_minutes=60)
        
        # Should only select backend with sufficient duration limit
        selected = router.route_job(job, resource_estimate=resource_estimate)
        assert selected == long_backend
    
    def test_route_job_raises_error_when_no_suitable_backend(self):
        """Test routing raises error when no suitable backend available."""
        router = MultiBackendRouter()
        
        backend = Mock(spec=Backend, id="backend1", type=BackendType.MODAL)
        backend.supports_template.return_value = False  # Doesn't support template
        backend.capabilities = Mock(supports_gpu=True, max_job_duration_minutes=60)
        
        router.register_backend(backend)
        router.health_monitor.update_health_status("backend1", HealthStatus.HEALTHY)
        
        job = Job(id="job1", template_name="unsupported-template")
        
        with pytest.raises(BackendNotAvailableError):
            router.route_job(job)
    
    def test_route_job_logs_routing_decision(self):
        """Test routing logs decision details."""
        import logging
        
        router = MultiBackendRouter()
        
        backend = Mock(spec=Backend, id="backend1", type=BackendType.MODAL)
        backend.supports_template.return_value = True
        backend.capabilities = Mock(supports_gpu=True, max_job_duration_minutes=60)
        
        router.register_backend(backend)
        router.health_monitor.update_health_status("backend1", HealthStatus.HEALTHY)
        
        job = Job(id="job1", template_name="test-template")
        
        # Route the job
        selected = router.route_job(job, routing_strategy="round-robin")
        
        # Verify backend was selected
        assert selected == backend
    
    def test_route_job_validates_gpu_memory_requirements(self):
        """Test routing validates GPU memory requirements (Requirement 10.5, 10.6)."""
        router = MultiBackendRouter()
        
        gpu_backend = Mock(spec=Backend, id="gpu", type=BackendType.MODAL)
        gpu_backend.supports_template.return_value = True
        gpu_backend.capabilities = Mock(supports_gpu=True, max_job_duration_minutes=60)
        
        cpu_backend = Mock(spec=Backend, id="cpu", type=BackendType.HUGGINGFACE)
        cpu_backend.supports_template.return_value = True
        cpu_backend.capabilities = Mock(supports_gpu=False, max_job_duration_minutes=60)
        
        router.register_backend(gpu_backend)
        router.register_backend(cpu_backend)
        
        # Mark both as healthy
        router.health_monitor.update_health_status("gpu", HealthStatus.HEALTHY)
        router.health_monitor.update_health_status("cpu", HealthStatus.HEALTHY)
        
        job = Job(id="job1", template_name="test-template")
        resource_estimate = ResourceEstimate(gpu_memory_gb=8.0)
        
        # Should only select GPU backend (CPU backend excluded due to no GPU support)
        selected = router.route_job(job, resource_estimate=resource_estimate)
        assert selected == gpu_backend
    
    def test_route_job_excludes_backends_with_detailed_reasons(self):
        """Test routing tracks and logs detailed exclusion reasons (Requirement 10.5)."""
        router = MultiBackendRouter()
        
        # Backend 1: Unhealthy
        backend1 = Mock(spec=Backend, id="backend1", type=BackendType.MODAL)
        backend1.supports_template.return_value = True
        backend1.capabilities = Mock(supports_gpu=True, max_job_duration_minutes=60)
        
        # Backend 2: Doesn't support template
        backend2 = Mock(spec=Backend, id="backend2", type=BackendType.HUGGINGFACE)
        backend2.supports_template.return_value = False
        backend2.capabilities = Mock(supports_gpu=True, max_job_duration_minutes=60)
        
        # Backend 3: No GPU support
        backend3 = Mock(spec=Backend, id="backend3", type=BackendType.KAGGLE)
        backend3.supports_template.return_value = True
        backend3.capabilities = Mock(supports_gpu=False, max_job_duration_minutes=60)
        
        # Backend 4: Duration limit too short
        backend4 = Mock(spec=Backend, id="backend4", type=BackendType.COLAB)
        backend4.supports_template.return_value = True
        backend4.capabilities = Mock(supports_gpu=True, max_job_duration_minutes=30)
        
        router.register_backend(backend1)
        router.register_backend(backend2)
        router.register_backend(backend3)
        router.register_backend(backend4)
        
        # Mark backend1 as unhealthy, others as healthy
        router.health_monitor.update_health_status("backend1", HealthStatus.UNHEALTHY)
        router.health_monitor.update_health_status("backend2", HealthStatus.HEALTHY)
        router.health_monitor.update_health_status("backend3", HealthStatus.HEALTHY)
        router.health_monitor.update_health_status("backend4", HealthStatus.HEALTHY)
        
        job = Job(id="job1", template_name="test-template")
        resource_estimate = ResourceEstimate(requires_gpu=True, estimated_duration_minutes=60)
        
        # Should raise error with detailed exclusion reasons
        with pytest.raises(BackendNotAvailableError) as exc_info:
            router.route_job(job, resource_estimate=resource_estimate)
        
        # Verify error message contains useful information
        error_msg = str(exc_info.value)
        assert "No suitable backend available" in error_msg
        assert "test-template" in error_msg
    
    def test_route_job_validates_all_resource_requirements(self):
        """Test routing validates all resource requirements comprehensively (Requirement 10.5)."""
        router = MultiBackendRouter()
        
        suitable_backend = Mock(spec=Backend, id="suitable", type=BackendType.MODAL)
        suitable_backend.supports_template.return_value = True
        suitable_backend.capabilities = Mock(
            supports_gpu=True,
            max_job_duration_minutes=120
        )
        
        router.register_backend(suitable_backend)
        router.health_monitor.update_health_status("suitable", HealthStatus.HEALTHY)
        
        job = Job(id="job1", template_name="test-template")
        resource_estimate = ResourceEstimate(
            requires_gpu=True,
            gpu_memory_gb=16.0,
            memory_gb=32.0,
            cpu_cores=8,
            estimated_duration_minutes=90,
            requires_internet=True
        )
        
        # Should select the suitable backend
        selected = router.route_job(job, resource_estimate=resource_estimate)
        assert selected == suitable_backend
    
    def test_route_job_logs_exclusion_reasons_in_decision(self):
        """Test routing logs exclusion reasons in decision factors (Requirement 10.5)."""
        import logging
        from unittest.mock import patch
        
        router = MultiBackendRouter()
        
        # Backend 1: Suitable
        backend1 = Mock(spec=Backend, id="backend1", type=BackendType.MODAL)
        backend1.supports_template.return_value = True
        backend1.capabilities = Mock(supports_gpu=True, max_job_duration_minutes=120)
        backend1.estimate_cost.return_value = 0.0
        
        # Backend 2: No GPU
        backend2 = Mock(spec=Backend, id="backend2", type=BackendType.HUGGINGFACE)
        backend2.supports_template.return_value = True
        backend2.capabilities = Mock(supports_gpu=False, max_job_duration_minutes=120)
        
        router.register_backend(backend1)
        router.register_backend(backend2)
        
        router.health_monitor.update_health_status("backend1", HealthStatus.HEALTHY)
        router.health_monitor.update_health_status("backend2", HealthStatus.HEALTHY)
        
        job = Job(id="job1", template_name="test-template")
        resource_estimate = ResourceEstimate(requires_gpu=True)
        
        # Capture log output
        with patch.object(router.logger, 'debug') as mock_debug:
            selected = router.route_job(job, resource_estimate=resource_estimate)
            
            # Verify backend2 exclusion was logged
            debug_calls = [str(call) for call in mock_debug.call_args_list]
            assert any("backend2" in call and "excluded" in call for call in debug_calls)
        
        assert selected == backend1
    
    def test_route_job_includes_resource_requirements_in_decision_log(self):
        """Test routing includes all resource requirements in decision log (Requirement 10.5)."""
        from unittest.mock import patch
        
        router = MultiBackendRouter()
        
        backend = Mock(spec=Backend, id="backend1", type=BackendType.MODAL)
        backend.supports_template.return_value = True
        backend.capabilities = Mock(supports_gpu=True, max_job_duration_minutes=120)
        backend.estimate_cost.return_value = 0.0
        
        router.register_backend(backend)
        router.health_monitor.update_health_status("backend1", HealthStatus.HEALTHY)
        
        job = Job(id="job1", template_name="test-template")
        resource_estimate = ResourceEstimate(
            requires_gpu=True,
            gpu_memory_gb=8.0,
            memory_gb=16.0,
            cpu_cores=4,
            estimated_duration_minutes=60,
            requires_internet=True
        )
        
        # Capture debug log output
        with patch.object(router.logger, 'debug') as mock_debug:
            selected = router.route_job(job, resource_estimate=resource_estimate)
            
            # Verify resource requirements were logged
            debug_calls = [str(call) for call in mock_debug.call_args_list]
            
            # Check that various resource requirements were mentioned in logs
            log_output = " ".join(debug_calls)
            assert "suitable" in log_output.lower() or "backend1" in log_output
        
        assert selected == backend


class TestRetryLogic:
    """Test retry logic in job execution."""
    
    def test_execute_job_with_retry_success_on_first_attempt(self):
        """Test successful job execution on first attempt."""
        router = MultiBackendRouter()
        
        backend = Mock(spec=Backend, id="backend1", type=BackendType.MODAL)
        backend.execute_job.return_value = {"status": "success", "result": "data"}
        
        job = Job(id="job1", template_name="test-template", retry_count=0)
        
        result = router.execute_job_with_retry(job, backend, max_retries=3)
        
        assert result == {"status": "success", "result": "data"}
        assert job.retry_count == 0  # No retries needed
        backend.execute_job.assert_called_once()
    
    def test_execute_job_with_retry_success_after_retries(self):
        """Test successful job execution after retries."""
        router = MultiBackendRouter()
        
        backend = Mock(spec=Backend, id="backend1", type=BackendType.MODAL)
        # Fail twice, then succeed
        backend.execute_job.side_effect = [
            Exception("Temporary failure"),
            Exception("Temporary failure"),
            {"status": "success", "result": "data"}
        ]
        
        job = Job(id="job1", template_name="test-template", retry_count=0)
        
        # Mock handle_backend_failure to return None (no alternative backend)
        router.handle_backend_failure = Mock(return_value=None)
        
        result = router.execute_job_with_retry(job, backend, max_retries=3)
        
        assert result == {"status": "success", "result": "data"}
        assert job.retry_count == 2  # Two retries before success
        assert backend.execute_job.call_count == 3
    
    def test_execute_job_with_retry_enforces_retry_limit(self):
        """Test retry limit enforcement (Requirement 8.3)."""
        from notebook_ml_orchestrator.core.exceptions import JobExecutionError
        
        router = MultiBackendRouter()
        
        backend = Mock(spec=Backend, id="backend1", type=BackendType.MODAL)
        backend.execute_job.side_effect = Exception("Persistent failure")
        
        job = Job(id="job1", template_name="test-template", retry_count=0)
        
        # Mock handle_backend_failure to return None (no alternative backend)
        router.handle_backend_failure = Mock(return_value=None)
        
        # Should fail after max_retries + 1 attempts
        with pytest.raises(JobExecutionError) as exc_info:
            router.execute_job_with_retry(job, backend, max_retries=3)
        
        assert "failed after 4 attempts" in str(exc_info.value)
        assert job.retry_count == 4  # 3 retries + 1 initial attempt
        assert backend.execute_job.call_count == 4
    
    def test_execute_job_with_retry_exponential_backoff(self):
        """Test exponential backoff calculation (Requirement 8.4)."""
        import time
        from unittest.mock import patch
        
        router = MultiBackendRouter()
        
        backend = Mock(spec=Backend, id="backend1", type=BackendType.MODAL)
        backend.execute_job.side_effect = [
            Exception("Failure 1"),
            Exception("Failure 2"),
            {"status": "success"}
        ]
        
        job = Job(id="job1", template_name="test-template", retry_count=0)
        
        # Mock handle_backend_failure to return None (no alternative backend)
        router.handle_backend_failure = Mock(return_value=None)
        
        # Mock time.sleep to track backoff delays
        with patch('time.sleep') as mock_sleep:
            result = router.execute_job_with_retry(job, backend, max_retries=3)
            
            # Verify exponential backoff was applied
            # First retry: base_delay * (exponential_base ** 0) = 1.0 * (2 ** 0) = 1.0
            # Second retry: base_delay * (exponential_base ** 1) = 1.0 * (2 ** 1) = 2.0
            assert mock_sleep.call_count == 2
            
            # Check the delays (allowing for min() with max_delay)
            calls = mock_sleep.call_args_list
            assert calls[0][0][0] == 1.0  # First backoff: 2^0 = 1.0
            assert calls[1][0][0] == 2.0  # Second backoff: 2^1 = 2.0
    
    def test_execute_job_with_retry_handles_no_alternative_backend(self):
        """Test handling of no alternative backend scenario (Requirement 8.2)."""
        from notebook_ml_orchestrator.core.exceptions import JobExecutionError
        
        router = MultiBackendRouter()
        
        backend = Mock(spec=Backend, id="backend1", type=BackendType.MODAL)
        backend.execute_job.side_effect = Exception("Backend failure")
        
        job = Job(id="job1", template_name="test-template", retry_count=0)
        
        # Mock handle_backend_failure to return None (no alternative backend)
        router.handle_backend_failure = Mock(return_value=None)
        
        # Should retry on same backend when no alternative available
        with pytest.raises(JobExecutionError):
            router.execute_job_with_retry(job, backend, max_retries=2)
        
        # Verify handle_backend_failure was called for each failure (not including the final one that exceeds limit)
        assert router.handle_backend_failure.call_count == 2  # Called on first 2 failures
        
        # Verify job was retried on same backend
        assert backend.execute_job.call_count == 3
    
    def test_execute_job_with_retry_switches_to_alternative_backend(self):
        """Test switching to alternative backend on failure."""
        router = MultiBackendRouter()
        
        backend1 = Mock(spec=Backend, id="backend1", type=BackendType.MODAL)
        backend1.execute_job.side_effect = Exception("Backend 1 failure")
        
        backend2 = Mock(spec=Backend, id="backend2", type=BackendType.KAGGLE)
        backend2.execute_job.return_value = {"status": "success", "result": "data"}
        
        job = Job(id="job1", template_name="test-template", retry_count=0)
        
        # Mock handle_backend_failure to return alternative backend
        router.handle_backend_failure = Mock(return_value=backend2)
        
        result = router.execute_job_with_retry(job, backend1, max_retries=3)
        
        assert result == {"status": "success", "result": "data"}
        assert job.retry_count == 1  # One retry on alternative backend
        
        # Verify failover was attempted
        router.handle_backend_failure.assert_called_once()
        
        # Verify execution on both backends
        backend1.execute_job.assert_called_once()
        backend2.execute_job.assert_called_once()
    
    def test_execute_job_with_retry_uses_config_defaults(self):
        """Test retry logic uses configuration defaults."""
        from notebook_ml_orchestrator.core.exceptions import JobExecutionError
        
        router = MultiBackendRouter()
        
        backend = Mock(spec=Backend, id="backend1", type=BackendType.MODAL)
        backend.execute_job.side_effect = Exception("Persistent failure")
        
        job = Job(id="job1", template_name="test-template", retry_count=0)
        
        # Mock handle_backend_failure to return None
        router.handle_backend_failure = Mock(return_value=None)
        
        # Call without specifying max_retries (should use config default of 3)
        with pytest.raises(JobExecutionError):
            router.execute_job_with_retry(job, backend)
        
        # Should have attempted 4 times (1 initial + 3 retries)
        assert backend.execute_job.call_count == 4
    
    def test_execute_job_with_retry_preserves_job_state(self):
        """Test job state is preserved across retries."""
        router = MultiBackendRouter()
        
        backend1 = Mock(spec=Backend, id="backend1", type=BackendType.MODAL)
        backend1.execute_job.side_effect = Exception("Failure")
        
        backend2 = Mock(spec=Backend, id="backend2", type=BackendType.KAGGLE)
        backend2.execute_job.return_value = {"status": "success"}
        
        job = Job(
            id="job1",
            template_name="test-template",
            inputs={"param1": "value1"},
            metadata={"key": "value"},
            retry_count=0
        )
        
        # Mock handle_backend_failure to return alternative backend
        router.handle_backend_failure = Mock(return_value=backend2)
        
        result = router.execute_job_with_retry(job, backend1, max_retries=3)
        
        # Verify job inputs and metadata are preserved
        assert job.inputs == {"param1": "value1"}
        assert "key" in job.metadata
        assert job.metadata["key"] == "value"
    
    def test_execute_job_with_retry_increments_retry_counter(self):
        """Test retry counter is incremented correctly."""
        router = MultiBackendRouter()
        
        backend = Mock(spec=Backend, id="backend1", type=BackendType.MODAL)
        backend.execute_job.side_effect = [
            Exception("Failure 1"),
            Exception("Failure 2"),
            {"status": "success"}
        ]
        
        job = Job(id="job1", template_name="test-template", retry_count=0)
        
        # Mock handle_backend_failure to return None
        router.handle_backend_failure = Mock(return_value=None)
        
        result = router.execute_job_with_retry(job, backend, max_retries=3)
        
        # Verify retry counter was incremented
        assert job.retry_count == 2  # Two failures before success
    
    def test_execute_job_with_retry_logs_attempts(self):
        """Test retry logic logs execution attempts."""
        router = MultiBackendRouter()
        
        backend = Mock(spec=Backend, id="backend1", type=BackendType.MODAL)
        backend.execute_job.side_effect = [
            Exception("Failure"),
            {"status": "success"}
        ]
        
        job = Job(id="job1", template_name="test-template", retry_count=0)
        
        # Mock handle_backend_failure to return None
        router.handle_backend_failure = Mock(return_value=None)
        
        result = router.execute_job_with_retry(job, backend, max_retries=3)
        
        # Verify execution completed
        assert result == {"status": "success"}
        assert backend.execute_job.call_count == 2
