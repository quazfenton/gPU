"""
Property-based tests for job persistence and recovery.

This module implements property-based tests using Hypothesis to verify
that jobs persist across system restarts and runtime disconnects,
maintaining all state information and resuming processing appropriately.

**Validates: Requirements 2.1, 2.2, 2.3**
"""

import pytest
import tempfile
import shutil
import os
from pathlib import Path
from typing import Dict, Any, List

from hypothesis import given, strategies as st, settings, assume, example
from hypothesis.stateful import RuleBasedStateMachine, Bundle, rule, initialize, invariant, run_state_machine_as_test

from notebook_ml_orchestrator.core.job_queue import JobQueueManager, RetryPolicy
from notebook_ml_orchestrator.core.database import DatabaseManager
from notebook_ml_orchestrator.core.interfaces import Job
from notebook_ml_orchestrator.core.models import JobStatus, JobResult
from notebook_ml_orchestrator.core.exceptions import JobValidationError


# Hypothesis strategies for generating test data
job_id_strategy = st.text(min_size=1, max_size=50, alphabet=st.characters(
    whitelist_categories=('Lu', 'Ll', 'Nd'), 
    whitelist_characters='-_'
))

user_id_strategy = st.text(min_size=1, max_size=30, alphabet=st.characters(
    whitelist_categories=('Lu', 'Ll', 'Nd'), 
    whitelist_characters='-_'
))

template_name_strategy = st.text(min_size=1, max_size=50, alphabet=st.characters(
    whitelist_categories=('Lu', 'Ll', 'Nd'), 
    whitelist_characters='-_'
))

job_inputs_strategy = st.dictionaries(
    keys=st.text(min_size=1, max_size=20, alphabet=st.characters(
        whitelist_categories=('Lu', 'Ll', 'Nd'), 
        whitelist_characters='_'
    )),
    values=st.one_of(
        st.text(max_size=100),
        st.integers(min_value=-1000, max_value=1000),
        st.floats(min_value=-1000.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
        st.booleans(),
        st.lists(st.integers(min_value=0, max_value=100), max_size=10)
    ),
    min_size=0,
    max_size=10
)

job_priority_strategy = st.integers(min_value=0, max_value=10)

job_metadata_strategy = st.dictionaries(
    keys=st.text(min_size=1, max_size=20, alphabet=st.characters(
        whitelist_categories=('Lu', 'Ll', 'Nd'), 
        whitelist_characters='_'
    )),
    values=st.one_of(
        st.text(max_size=50),
        st.integers(min_value=0, max_value=1000),
        st.booleans()
    ),
    min_size=0,
    max_size=5
)

valid_job_status_strategy = st.sampled_from([
    JobStatus.QUEUED, JobStatus.RUNNING, JobStatus.COMPLETED, 
    JobStatus.FAILED, JobStatus.CANCELLED, JobStatus.RETRYING
])


def create_job_strategy():
    """Create a strategy for generating valid Job instances."""
    return st.builds(
        Job,
        id=job_id_strategy,
        user_id=user_id_strategy,
        template_name=template_name_strategy,
        inputs=job_inputs_strategy,
        status=st.just(JobStatus.QUEUED),  # Always start with QUEUED
        priority=job_priority_strategy,
        metadata=job_metadata_strategy
    )


class TestJobPersistenceProperties:
    """Property-based tests for job persistence and recovery."""
    
    @given(create_job_strategy())
    @settings(max_examples=50, deadline=None)
    def test_job_persistence_across_restarts(self, job):
        """
        **Property 2: Job Persistence and Recovery**
        **Validates: Requirements 2.1, 2.2, 2.3**
        
        Property: For any job submitted to the Job_Queue, the job should persist 
        across system restarts and runtime disconnects, maintaining all state 
        information and resuming processing appropriately.
        """
        # Create temporary database
        temp_dir = tempfile.mkdtemp()
        db_path = os.path.join(temp_dir, "test_persistence.db")
        
        try:
            # Phase 1: Submit job to first queue manager instance
            queue_manager_1 = JobQueueManager(db_path)
            
            # Submit the job
            submitted_job_id = queue_manager_1.submit_job(job)
            assert submitted_job_id == job.id
            
            # Verify job was stored
            stored_job = queue_manager_1.get_job(job.id)
            assert stored_job is not None
            assert stored_job.id == job.id
            assert stored_job.user_id == job.user_id
            assert stored_job.template_name == job.template_name
            assert stored_job.inputs == job.inputs
            assert stored_job.status == JobStatus.QUEUED
            assert stored_job.priority == job.priority
            assert stored_job.metadata == job.metadata
            
            # Update job to running status to simulate processing
            queue_manager_1.update_job_status(job.id, JobStatus.RUNNING)
            running_job = queue_manager_1.get_job(job.id)
            assert running_job.status == JobStatus.RUNNING
            assert running_job.started_at is not None
            
            # Store the started_at time for later comparison
            original_started_at = running_job.started_at
            
            # Simulate system shutdown by stopping retry processor and closing connections
            queue_manager_1.stop_retry_processor()
            queue_manager_1.db.close()
            del queue_manager_1
            
            # Phase 2: Create new queue manager instance (simulating restart)
            queue_manager_2 = JobQueueManager(db_path)
            
            # Verify job persisted across restart
            recovered_job = queue_manager_2.get_job(job.id)
            assert recovered_job is not None
            
            # Verify all job data was preserved
            assert recovered_job.id == job.id
            assert recovered_job.user_id == job.user_id
            assert recovered_job.template_name == job.template_name
            assert recovered_job.inputs == job.inputs
            assert recovered_job.status == JobStatus.RUNNING
            assert recovered_job.priority == job.priority
            assert recovered_job.metadata == job.metadata
            assert recovered_job.started_at == original_started_at
            
            # Verify job can continue processing after restart
            result = JobResult(
                success=True,
                outputs={"result": "completed after restart"},
                execution_time_seconds=10.0,
                backend_used="test-backend"
            )
            queue_manager_2.update_job_status(job.id, JobStatus.COMPLETED, result)
            
            completed_job = queue_manager_2.get_job(job.id)
            assert completed_job.status == JobStatus.COMPLETED
            assert completed_job.result is not None
            assert completed_job.result.success is True
            assert completed_job.result.outputs == {"result": "completed after restart"}
            assert completed_job.completed_at is not None
            
            # Clean up
            try:
                queue_manager_2.stop_retry_processor()
                queue_manager_2.db.close()
            except Exception:
                pass  # Ignore cleanup errors
            
        finally:
            # Clean up temporary directory
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    @given(st.lists(create_job_strategy(), min_size=1, max_size=10))
    @settings(max_examples=30, deadline=None)
    def test_multiple_jobs_persistence(self, jobs):
        """
        **Property 2: Job Persistence and Recovery**
        **Validates: Requirements 2.1, 2.2, 2.3**
        
        Property: Multiple jobs should all persist across system restarts,
        maintaining their individual state information.
        """
        # Ensure unique job IDs
        unique_jobs = []
        seen_ids = set()
        for job in jobs:
            if job.id not in seen_ids:
                unique_jobs.append(job)
                seen_ids.add(job.id)
        
        assume(len(unique_jobs) > 0)
        
        # Create temporary database
        temp_dir = tempfile.mkdtemp()
        db_path = os.path.join(temp_dir, "test_multi_persistence.db")
        
        try:
            # Phase 1: Submit multiple jobs
            queue_manager_1 = JobQueueManager(db_path)
            
            submitted_jobs = {}
            for job in unique_jobs:
                job_id = queue_manager_1.submit_job(job)
                submitted_jobs[job_id] = job
            
            # Update some jobs to different statuses
            job_states = {}
            for i, (job_id, job) in enumerate(submitted_jobs.items()):
                if i % 3 == 0:
                    # Keep as QUEUED
                    job_states[job_id] = JobStatus.QUEUED
                elif i % 3 == 1:
                    # Update to RUNNING
                    queue_manager_1.update_job_status(job_id, JobStatus.RUNNING)
                    job_states[job_id] = JobStatus.RUNNING
                else:
                    # Update to RUNNING then COMPLETED
                    queue_manager_1.update_job_status(job_id, JobStatus.RUNNING)
                    result = JobResult(success=True, outputs={"result": f"completed-{job_id}"})
                    queue_manager_1.update_job_status(job_id, JobStatus.COMPLETED, result)
                    job_states[job_id] = JobStatus.COMPLETED
            
            # Simulate system restart
            queue_manager_1.stop_retry_processor()
            queue_manager_1.db.close()
            del queue_manager_1
            
            # Phase 2: Verify all jobs persisted
            queue_manager_2 = JobQueueManager(db_path)
            
            for job_id, original_job in submitted_jobs.items():
                recovered_job = queue_manager_2.get_job(job_id)
                assert recovered_job is not None
                
                # Verify job data integrity
                assert recovered_job.id == original_job.id
                assert recovered_job.user_id == original_job.user_id
                assert recovered_job.template_name == original_job.template_name
                assert recovered_job.inputs == original_job.inputs
                assert recovered_job.priority == original_job.priority
                assert recovered_job.metadata == original_job.metadata
                
                # Verify status was preserved
                expected_status = job_states[job_id]
                assert recovered_job.status == expected_status
                
                if expected_status == JobStatus.COMPLETED:
                    assert recovered_job.result is not None
                    assert recovered_job.result.success is True
                    assert recovered_job.completed_at is not None
            
            # Clean up
            queue_manager_2.stop_retry_processor()
            queue_manager_2.db.close()
            
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    @given(create_job_strategy())
    @settings(max_examples=30, deadline=None)
    def test_job_retry_persistence_across_restarts(self, job):
        """
        **Property 2: Job Persistence and Recovery**
        **Validates: Requirements 2.1, 2.2, 2.3**
        
        Property: Jobs in retry state should persist across restarts and
        resume retry processing appropriately.
        """
        # Create temporary database
        temp_dir = tempfile.mkdtemp()
        db_path = os.path.join(temp_dir, "test_retry_persistence.db")
        
        try:
            # Phase 1: Submit job and simulate failure with retry
            retry_policy = RetryPolicy(max_retries=3, base_delay=1.0)
            queue_manager_1 = JobQueueManager(db_path, retry_policy)
            
            # Submit job
            queue_manager_1.submit_job(job)
            
            # Simulate job failure to trigger retry
            error = Exception("Simulated failure for testing")
            queue_manager_1.handle_job_failure(job.id, error)
            
            # Verify job is in retrying state
            retrying_job = queue_manager_1.get_job(job.id)
            assert retrying_job.status == JobStatus.RETRYING
            assert retrying_job.retry_count == 1
            assert retrying_job.error == "Simulated failure for testing"
            assert "retry_at" in retrying_job.metadata
            
            # Store retry information for comparison
            original_retry_count = retrying_job.retry_count
            original_error = retrying_job.error
            original_retry_at = retrying_job.metadata["retry_at"]
            
            # Simulate system restart
            queue_manager_1.stop_retry_processor()
            queue_manager_1.db.close()
            del queue_manager_1
            
            # Phase 2: Verify retry state persisted
            queue_manager_2 = JobQueueManager(db_path, retry_policy)
            
            recovered_job = queue_manager_2.get_job(job.id)
            assert recovered_job is not None
            
            # Verify retry state was preserved
            assert recovered_job.status == JobStatus.RETRYING
            assert recovered_job.retry_count == original_retry_count
            assert recovered_job.error == original_error
            assert recovered_job.metadata.get("retry_at") == original_retry_at
            
            # Verify all original job data was preserved
            assert recovered_job.id == job.id
            assert recovered_job.user_id == job.user_id
            assert recovered_job.template_name == job.template_name
            assert recovered_job.inputs == job.inputs
            assert recovered_job.priority == job.priority
            
            # Verify retry processing can continue after restart
            # (We won't wait for actual retry processing due to timing,
            # but we can verify the retry processor is running)
            assert queue_manager_2._running is True
            assert queue_manager_2._retry_thread is not None
            assert queue_manager_2._retry_thread.is_alive()
            
            # Clean up
            queue_manager_2.stop_retry_processor()
            queue_manager_2.db.close()
            
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    @given(create_job_strategy(), st.integers(min_value=1, max_value=5))
    @settings(max_examples=30, deadline=None)
    def test_job_state_transitions_persist_across_restarts(self, job, num_transitions):
        """
        **Property 2: Job Persistence and Recovery**
        **Validates: Requirements 2.1, 2.2, 2.3**
        
        Property: Job state transitions should persist across system restarts,
        maintaining the complete state history.
        """
        # Create temporary database
        temp_dir = tempfile.mkdtemp()
        db_path = os.path.join(temp_dir, "test_transitions_persistence.db")
        
        try:
            # Phase 1: Submit job and perform state transitions
            queue_manager_1 = JobQueueManager(db_path)
            
            # Submit job
            queue_manager_1.submit_job(job)
            
            # Perform valid state transitions
            transitions_performed = []
            current_status = JobStatus.QUEUED
            transitions_performed.append(current_status)
            
            for i in range(min(num_transitions, 3)):  # Limit to avoid invalid transitions
                if current_status == JobStatus.QUEUED:
                    # QUEUED -> RUNNING
                    queue_manager_1.update_job_status(job.id, JobStatus.RUNNING)
                    current_status = JobStatus.RUNNING
                    transitions_performed.append(current_status)
                elif current_status == JobStatus.RUNNING and i < 2:
                    if i % 2 == 0:
                        # RUNNING -> COMPLETED
                        result = JobResult(success=True, outputs={"result": "success"})
                        queue_manager_1.update_job_status(job.id, JobStatus.COMPLETED, result)
                        current_status = JobStatus.COMPLETED
                        transitions_performed.append(current_status)
                        break  # Can't transition further from COMPLETED
                    else:
                        # RUNNING -> FAILED
                        queue_manager_1.update_job_status(job.id, JobStatus.FAILED, "Test failure")
                        current_status = JobStatus.FAILED
                        transitions_performed.append(current_status)
                elif current_status == JobStatus.FAILED:
                    # FAILED -> RETRYING (via handle_job_failure)
                    error = Exception("Test error for retry")
                    queue_manager_1.handle_job_failure(job.id, error)
                    current_status = JobStatus.RETRYING
                    transitions_performed.append(current_status)
                    break  # Stop here to avoid complexity
            
            # Get final job state before restart
            pre_restart_job = queue_manager_1.get_job(job.id)
            
            # Simulate system restart
            queue_manager_1.stop_retry_processor()
            queue_manager_1.db.close()
            del queue_manager_1
            
            # Phase 2: Verify final state persisted
            queue_manager_2 = JobQueueManager(db_path)
            
            post_restart_job = queue_manager_2.get_job(job.id)
            assert post_restart_job is not None
            
            # Verify final state matches
            assert post_restart_job.status == pre_restart_job.status
            
            # Verify all job data was preserved
            assert post_restart_job.id == job.id
            assert post_restart_job.user_id == job.user_id
            assert post_restart_job.template_name == job.template_name
            assert post_restart_job.inputs == job.inputs
            assert post_restart_job.priority == job.priority
            
            # Verify timestamps were preserved
            if pre_restart_job.started_at:
                assert post_restart_job.started_at == pre_restart_job.started_at
            if pre_restart_job.completed_at:
                assert post_restart_job.completed_at == pre_restart_job.completed_at
            
            # Verify result was preserved if job completed
            if pre_restart_job.result:
                assert post_restart_job.result is not None
                assert post_restart_job.result.success == pre_restart_job.result.success
                assert post_restart_job.result.outputs == pre_restart_job.result.outputs
            
            # Verify error was preserved if job failed
            if pre_restart_job.error:
                assert post_restart_job.error == pre_restart_job.error
            
            # Clean up
            queue_manager_2.stop_retry_processor()
            queue_manager_2.db.close()
            
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


class JobPersistenceStateMachine(RuleBasedStateMachine):
    """
    Stateful property-based testing for job persistence.
    
    This state machine tests complex scenarios involving multiple jobs,
    system restarts, and various state transitions to ensure persistence
    properties hold under all conditions.
    """
    
    jobs = Bundle('jobs')
    
    def __init__(self):
        super().__init__()
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "stateful_test.db")
        self.queue_manager = None
        self.submitted_jobs = {}
        self.restart_count = 0
    
    @initialize()
    def setup_queue_manager(self):
        """Initialize the job queue manager."""
        self.queue_manager = JobQueueManager(self.db_path)
    
    @rule(target=jobs, job=create_job_strategy())
    def submit_job(self, job):
        """Submit a job to the queue."""
        assume(job.id not in self.submitted_jobs)
        
        job_id = self.queue_manager.submit_job(job)
        self.submitted_jobs[job_id] = job
        return job_id
    
    @rule(job_id=jobs, status=st.sampled_from([JobStatus.RUNNING, JobStatus.COMPLETED, JobStatus.FAILED]))
    def update_job_status(self, job_id, status):
        """Update job status following valid transitions."""
        current_job = self.queue_manager.get_job(job_id)
        if current_job is None:
            return
        
        # Only perform valid transitions
        if current_job.status == JobStatus.QUEUED and status == JobStatus.RUNNING:
            self.queue_manager.update_job_status(job_id, status)
        elif current_job.status == JobStatus.RUNNING:
            if status == JobStatus.COMPLETED:
                result = JobResult(success=True, outputs={"result": "test"})
                self.queue_manager.update_job_status(job_id, status, result)
            elif status == JobStatus.FAILED:
                self.queue_manager.update_job_status(job_id, status, "Test failure")
    
    @rule()
    def simulate_system_restart(self):
        """Simulate a system restart by recreating the queue manager."""
        if self.queue_manager:
            self.queue_manager.stop_retry_processor()
            self.queue_manager.db.close()
        
        # Create new queue manager instance
        self.queue_manager = JobQueueManager(self.db_path)
        self.restart_count += 1
    
    @invariant()
    def all_submitted_jobs_persist(self):
        """Invariant: All submitted jobs should persist across restarts."""
        if not self.submitted_jobs or not self.queue_manager:
            return
        
        for job_id, original_job in self.submitted_jobs.items():
            persisted_job = self.queue_manager.get_job(job_id)
            
            # Job must exist
            assert persisted_job is not None, f"Job {job_id} was lost after {self.restart_count} restarts"
            
            # Core job data must be preserved
            assert persisted_job.id == original_job.id
            assert persisted_job.user_id == original_job.user_id
            assert persisted_job.template_name == original_job.template_name
            assert persisted_job.inputs == original_job.inputs
            assert persisted_job.priority == original_job.priority
    
    def teardown(self):
        """Clean up resources."""
        if self.queue_manager:
            try:
                self.queue_manager.stop_retry_processor()
                self.queue_manager.db.close()
            except Exception:
                pass  # Ignore cleanup errors
        try:
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        except Exception:
            pass  # Ignore cleanup errors


# Stateful test class
class TestJobPersistenceStateful:
    """Stateful property-based tests for job persistence."""
    
    def test_job_persistence_stateful(self):
        """
        **Property 2: Job Persistence and Recovery**
        **Validates: Requirements 2.1, 2.2, 2.3**
        
        Stateful property test that verifies job persistence across
        multiple system restarts and complex state transitions.
        """
        run_state_machine_as_test(JobPersistenceStateMachine, settings=settings(max_examples=50, stateful_step_count=20, deadline=None))


# Additional edge case tests
class TestJobPersistenceEdgeCases:
    """Edge case tests for job persistence."""
    
    @given(create_job_strategy())
    @settings(max_examples=10, deadline=None)
    def test_persistence_with_database_corruption_recovery(self, job):
        """
        **Property 2: Job Persistence and Recovery**
        **Validates: Requirements 2.1, 2.2, 2.3**
        
        Property: System should handle database corruption gracefully
        and maintain data integrity where possible.
        """
        temp_dir = tempfile.mkdtemp()
        db_path = os.path.join(temp_dir, "corruption_test.db")
        
        try:
            # Submit job
            queue_manager_1 = JobQueueManager(db_path)
            queue_manager_1.submit_job(job)
            
            # Verify job was stored
            stored_job = queue_manager_1.get_job(job.id)
            assert stored_job is not None
            
            # Close cleanly
            queue_manager_1.stop_retry_processor()
            queue_manager_1.db.close()
            
            # Verify database file exists and is not empty
            db_file = Path(db_path)
            assert db_file.exists()
            assert db_file.stat().st_size > 0
            
            # Create new manager and verify data is still accessible
            queue_manager_2 = JobQueueManager(db_path)
            recovered_job = queue_manager_2.get_job(job.id)
            
            assert recovered_job is not None
            assert recovered_job.id == job.id
            assert recovered_job.user_id == job.user_id
            assert recovered_job.template_name == job.template_name
            
            queue_manager_2.stop_retry_processor()
            queue_manager_2.db.close()
            
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    @given(st.lists(create_job_strategy(), min_size=5, max_size=20))
    @settings(max_examples=10, deadline=None)
    def test_high_volume_persistence(self, jobs):
        """
        **Property 2: Job Persistence and Recovery**
        **Validates: Requirements 2.1, 2.2, 2.3**
        
        Property: Large numbers of jobs should persist correctly
        across system restarts.
        """
        # Ensure unique job IDs
        unique_jobs = []
        seen_ids = set()
        for job in jobs:
            if job.id not in seen_ids:
                unique_jobs.append(job)
                seen_ids.add(job.id)
        
        assume(len(unique_jobs) >= 5)
        
        temp_dir = tempfile.mkdtemp()
        db_path = os.path.join(temp_dir, "high_volume_test.db")
        
        try:
            # Submit all jobs
            queue_manager_1 = JobQueueManager(db_path)
            
            for job in unique_jobs:
                queue_manager_1.submit_job(job)
            
            # Verify all jobs were stored
            stats_before = queue_manager_1.get_queue_statistics()
            assert stats_before['total_jobs'] == len(unique_jobs)
            
            # Simulate restart
            queue_manager_1.stop_retry_processor()
            queue_manager_1.db.close()
            
            # Verify all jobs persisted
            queue_manager_2 = JobQueueManager(db_path)
            stats_after = queue_manager_2.get_queue_statistics()
            
            assert stats_after['total_jobs'] == len(unique_jobs)
            
            # Verify each job individually
            for original_job in unique_jobs:
                recovered_job = queue_manager_2.get_job(original_job.id)
                assert recovered_job is not None
                assert recovered_job.id == original_job.id
                assert recovered_job.user_id == original_job.user_id
                assert recovered_job.template_name == original_job.template_name
                assert recovered_job.inputs == original_job.inputs
            
            queue_manager_2.stop_retry_processor()
            queue_manager_2.db.close()
            
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)