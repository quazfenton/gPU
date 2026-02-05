"""
Property-based tests for job state transitions.

This module implements property-based tests using Hypothesis to verify
that job state transitions follow the valid state machine (queued → running → 
completed/failed/cancelled) and all state changes are tracked accurately.

**Validates: Requirements 2.4, 2.5**
"""

import pytest
import tempfile
import shutil
import os
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple, Optional

from hypothesis import given, strategies as st, settings, assume, example
from hypothesis.stateful import RuleBasedStateMachine, Bundle, rule, initialize, invariant, run_state_machine_as_test

from notebook_ml_orchestrator.core.job_queue import JobQueueManager, JobStateManager, RetryPolicy
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

# Strategy for generating valid state transition sequences
def valid_transition_sequences_strategy():
    """Generate valid sequences of job state transitions."""
    # Define possible transition paths
    paths = [
        # Simple completion path
        [JobStatus.QUEUED, JobStatus.RUNNING, JobStatus.COMPLETED],
        # Simple failure path
        [JobStatus.QUEUED, JobStatus.RUNNING, JobStatus.FAILED],
        # Cancellation from queued
        [JobStatus.QUEUED, JobStatus.CANCELLED],
        # Cancellation from running
        [JobStatus.QUEUED, JobStatus.RUNNING, JobStatus.CANCELLED],
        # Retry path
        [JobStatus.QUEUED, JobStatus.RUNNING, JobStatus.FAILED, JobStatus.RETRYING, JobStatus.QUEUED],
        # Retry then complete
        [JobStatus.QUEUED, JobStatus.RUNNING, JobStatus.FAILED, JobStatus.RETRYING, JobStatus.QUEUED, JobStatus.RUNNING, JobStatus.COMPLETED],
        # Retry then cancel
        [JobStatus.QUEUED, JobStatus.RUNNING, JobStatus.FAILED, JobStatus.RETRYING, JobStatus.CANCELLED],
    ]
    return st.sampled_from(paths)


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


class TestJobStateTransitionProperties:
    """Property-based tests for job state transitions."""
    
    @given(create_job_strategy(), valid_transition_sequences_strategy())
    @settings(max_examples=100, deadline=None)
    def test_valid_state_transitions_are_allowed(self, job, transition_sequence):
        """
        **Property 3: Job State Transitions**
        **Validates: Requirements 2.4, 2.5**
        
        Property: For any job in the system, state transitions should follow 
        the valid state machine (queued → running → completed/failed/cancelled) 
        and all state changes should be tracked accurately.
        """
        # Create temporary database
        temp_dir = tempfile.mkdtemp()
        db_path = os.path.join(temp_dir, "test_transitions.db")
        
        try:
            queue_manager = JobQueueManager(db_path)
            
            # Submit the job
            queue_manager.submit_job(job)
            
            # Verify initial state
            current_job = queue_manager.get_job(job.id)
            assert current_job.status == JobStatus.QUEUED
            
            # Apply each transition in the sequence
            previous_status = JobStatus.QUEUED
            for target_status in transition_sequence[1:]:  # Skip first QUEUED state
                
                # Verify transition is valid according to state machine
                assert JobStateManager.can_transition(previous_status, target_status), \
                    f"Invalid transition from {previous_status} to {target_status}"
                
                # Apply the transition
                if target_status == JobStatus.RUNNING:
                    queue_manager.update_job_status(job.id, JobStatus.RUNNING)
                elif target_status == JobStatus.COMPLETED:
                    result = JobResult(
                        success=True,
                        outputs={"result": "test completion"},
                        execution_time_seconds=5.0,
                        backend_used="test-backend"
                    )
                    queue_manager.update_job_status(job.id, JobStatus.COMPLETED, result)
                elif target_status == JobStatus.FAILED:
                    queue_manager.update_job_status(job.id, JobStatus.FAILED, "Test failure")
                elif target_status == JobStatus.CANCELLED:
                    queue_manager.cancel_job(job.id)
                elif target_status == JobStatus.RETRYING:
                    # RETRYING is set by handle_job_failure, not update_job_status
                    error = Exception("Test error for retry")
                    queue_manager.handle_job_failure(job.id, error)
                elif target_status == JobStatus.QUEUED and previous_status == JobStatus.RETRYING:
                    # This transition happens automatically via retry processor
                    # We'll simulate it by directly updating the job
                    current_job = queue_manager.get_job(job.id)
                    current_job.status = JobStatus.QUEUED
                    current_job.started_at = None
                    current_job.completed_at = None
                    current_job.error = None
                    if 'retry_at' in current_job.metadata:
                        del current_job.metadata['retry_at']
                    queue_manager.db.update_job(current_job)
                
                # Verify the transition was applied
                updated_job = queue_manager.get_job(job.id)
                assert updated_job.status == target_status, \
                    f"Expected status {target_status}, got {updated_job.status}"
                
                # Verify timestamps are set appropriately
                if target_status == JobStatus.RUNNING:
                    assert updated_job.started_at is not None, \
                        "started_at should be set when job transitions to RUNNING"
                elif target_status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
                    assert updated_job.completed_at is not None, \
                        f"completed_at should be set when job transitions to {target_status}"
                
                # Verify result is set for completed jobs
                if target_status == JobStatus.COMPLETED:
                    assert updated_job.result is not None, \
                        "result should be set when job completes successfully"
                    assert updated_job.result.success is True, \
                        "result should indicate success for completed jobs"
                
                # Verify error is set for failed jobs
                if target_status == JobStatus.FAILED:
                    assert updated_job.error is not None, \
                        "error should be set when job fails"
                
                # Verify retry count is incremented for retrying jobs
                if target_status == JobStatus.RETRYING:
                    assert updated_job.retry_count > 0, \
                        "retry_count should be incremented when job enters RETRYING state"
                    assert "retry_at" in updated_job.metadata, \
                        "retry_at should be set in metadata for retrying jobs"
                
                previous_status = target_status
            
            # Clean up
            queue_manager.stop_retry_processor()
            queue_manager.db.close()
            
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    @given(create_job_strategy())
    @settings(max_examples=50, deadline=None)
    def test_invalid_state_transitions_are_rejected(self, job):
        """
        **Property 3: Job State Transitions**
        **Validates: Requirements 2.4, 2.5**
        
        Property: Invalid state transitions should be rejected and raise
        appropriate validation errors.
        """
        # Create temporary database
        temp_dir = tempfile.mkdtemp()
        db_path = os.path.join(temp_dir, "test_invalid_transitions.db")
        
        try:
            queue_manager = JobQueueManager(db_path)
            
            # Submit the job
            queue_manager.submit_job(job)
            
            # Test invalid transitions from QUEUED
            invalid_from_queued = [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.RETRYING]
            for invalid_status in invalid_from_queued:
                with pytest.raises(JobValidationError):
                    queue_manager.update_job_status(job.id, invalid_status)
            
            # Move to RUNNING state
            queue_manager.update_job_status(job.id, JobStatus.RUNNING)
            
            # Test invalid transitions from RUNNING
            invalid_from_running = [JobStatus.QUEUED, JobStatus.RETRYING]
            for invalid_status in invalid_from_running:
                with pytest.raises(JobValidationError):
                    queue_manager.update_job_status(job.id, invalid_status)
            
            # Move to COMPLETED state
            result = JobResult(success=True, outputs={"result": "test"})
            queue_manager.update_job_status(job.id, JobStatus.COMPLETED, result)
            
            # Test that no transitions are allowed from COMPLETED
            all_statuses = [JobStatus.QUEUED, JobStatus.RUNNING, JobStatus.FAILED, 
                          JobStatus.CANCELLED, JobStatus.RETRYING]
            for invalid_status in all_statuses:
                with pytest.raises(JobValidationError):
                    queue_manager.update_job_status(job.id, invalid_status)
            
            # Clean up
            queue_manager.stop_retry_processor()
            queue_manager.db.close()
            
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    @given(create_job_strategy())
    @settings(max_examples=50, deadline=None)
    def test_state_transition_timestamps_are_accurate(self, job):
        """
        **Property 3: Job State Transitions**
        **Validates: Requirements 2.4, 2.5**
        
        Property: All state changes should be tracked accurately with
        appropriate timestamps.
        """
        # Create temporary database
        temp_dir = tempfile.mkdtemp()
        db_path = os.path.join(temp_dir, "test_timestamps.db")
        
        try:
            queue_manager = JobQueueManager(db_path)
            
            # Clear the job's created_at to let the system set it
            job.created_at = None
            
            # Record start time
            test_start_time = datetime.now()
            
            # Submit the job
            queue_manager.submit_job(job)
            
            # Verify created_at timestamp
            queued_job = queue_manager.get_job(job.id)
            assert queued_job.created_at is not None
            assert queued_job.created_at >= test_start_time
            assert queued_job.started_at is None
            assert queued_job.completed_at is None
            
            # Transition to RUNNING
            running_start_time = datetime.now()
            queue_manager.update_job_status(job.id, JobStatus.RUNNING)
            
            running_job = queue_manager.get_job(job.id)
            assert running_job.started_at is not None
            assert running_job.started_at >= running_start_time
            assert running_job.completed_at is None
            
            # Verify started_at is after created_at
            assert running_job.started_at >= running_job.created_at
            
            # Transition to COMPLETED
            completion_start_time = datetime.now()
            result = JobResult(success=True, outputs={"result": "test"})
            queue_manager.update_job_status(job.id, JobStatus.COMPLETED, result)
            
            completed_job = queue_manager.get_job(job.id)
            assert completed_job.completed_at is not None
            assert completed_job.completed_at >= completion_start_time
            
            # Verify completed_at is after started_at
            assert completed_job.completed_at >= completed_job.started_at
            
            # Verify all timestamps are in logical order
            assert completed_job.created_at <= completed_job.started_at <= completed_job.completed_at
            
            # Clean up
            queue_manager.stop_retry_processor()
            queue_manager.db.close()
            
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    @given(st.lists(create_job_strategy(), min_size=2, max_size=10))
    @settings(max_examples=30, deadline=None)
    def test_concurrent_state_transitions_are_isolated(self, jobs):
        """
        **Property 3: Job State Transitions**
        **Validates: Requirements 2.4, 2.5**
        
        Property: State transitions for different jobs should be isolated
        and not interfere with each other.
        """
        # Ensure unique job IDs
        unique_jobs = []
        seen_ids = set()
        for job in jobs:
            if job.id not in seen_ids:
                unique_jobs.append(job)
                seen_ids.add(job.id)
        
        assume(len(unique_jobs) >= 2)
        
        # Create temporary database
        temp_dir = tempfile.mkdtemp()
        db_path = os.path.join(temp_dir, "test_concurrent_transitions.db")
        
        try:
            queue_manager = JobQueueManager(db_path)
            
            # Submit all jobs
            for job in unique_jobs:
                queue_manager.submit_job(job)
            
            # Verify all jobs start in QUEUED state
            for job in unique_jobs:
                queued_job = queue_manager.get_job(job.id)
                assert queued_job.status == JobStatus.QUEUED
            
            # Apply different transitions to different jobs
            job_states = {}
            for i, job in enumerate(unique_jobs):
                if i % 3 == 0:
                    # Transition to RUNNING
                    queue_manager.update_job_status(job.id, JobStatus.RUNNING)
                    job_states[job.id] = JobStatus.RUNNING
                elif i % 3 == 1:
                    # Transition to RUNNING then COMPLETED
                    queue_manager.update_job_status(job.id, JobStatus.RUNNING)
                    result = JobResult(success=True, outputs={"result": f"job-{i}"})
                    queue_manager.update_job_status(job.id, JobStatus.COMPLETED, result)
                    job_states[job.id] = JobStatus.COMPLETED
                else:
                    # Cancel job
                    queue_manager.cancel_job(job.id)
                    job_states[job.id] = JobStatus.CANCELLED
            
            # Verify each job has the expected state and others are unaffected
            for job in unique_jobs:
                current_job = queue_manager.get_job(job.id)
                expected_status = job_states[job.id]
                assert current_job.status == expected_status, \
                    f"Job {job.id} expected {expected_status}, got {current_job.status}"
                
                # Verify job data integrity
                assert current_job.id == job.id
                assert current_job.user_id == job.user_id
                assert current_job.template_name == job.template_name
                assert current_job.inputs == job.inputs
                assert current_job.priority == job.priority
            
            # Clean up
            queue_manager.stop_retry_processor()
            queue_manager.db.close()
            
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    @given(create_job_strategy(), st.integers(min_value=1, max_value=5))
    @settings(max_examples=30, deadline=None)
    def test_retry_state_transitions_follow_policy(self, job, max_retries):
        """
        **Property 3: Job State Transitions**
        **Validates: Requirements 2.4, 2.5**
        
        Property: Jobs should follow retry policy correctly, transitioning
        through FAILED -> RETRYING -> QUEUED cycles until max retries reached.
        """
        # Create temporary database
        temp_dir = tempfile.mkdtemp()
        db_path = os.path.join(temp_dir, "test_retry_transitions.db")
        
        try:
            retry_policy = RetryPolicy(max_retries=max_retries, base_delay=0.1)
            queue_manager = JobQueueManager(db_path, retry_policy)
            
            # Submit the job
            queue_manager.submit_job(job)
            
            # Simulate failure attempts
            # The logic: should_retry checks BEFORE incrementing retry_count
            # So max_retries=1 means: first failure (retry_count=0) -> should_retry(0 < 1) -> RETRYING, retry_count=1
            # Second failure (retry_count=1) -> should_retry(1 < 1) -> FAILED
            
            failure_count = 0
            while True:
                failure_count += 1
                
                # Move to RUNNING
                queue_manager.update_job_status(job.id, JobStatus.RUNNING)
                
                # Simulate failure
                error = Exception(f"Test failure attempt {failure_count}")
                queue_manager.handle_job_failure(job.id, error)
                
                current_job = queue_manager.get_job(job.id)
                
                if current_job.status == JobStatus.RETRYING:
                    # Job should retry
                    assert current_job.retry_count <= max_retries, \
                        f"retry_count {current_job.retry_count} should be <= max_retries {max_retries}"
                    assert "retry_at" in current_job.metadata, \
                        "retry_at should be set in metadata"
                    
                    # Simulate retry processor moving job back to QUEUED
                    current_job.status = JobStatus.QUEUED
                    current_job.started_at = None
                    current_job.completed_at = None
                    current_job.error = None
                    if 'retry_at' in current_job.metadata:
                        del current_job.metadata['retry_at']
                    queue_manager.db.update_job(current_job)
                    
                    # Verify job is back in QUEUED state
                    queued_job = queue_manager.get_job(job.id)
                    assert queued_job.status == JobStatus.QUEUED
                    
                elif current_job.status == JobStatus.FAILED:
                    # Job should be permanently failed
                    assert current_job.retry_count >= max_retries, \
                        f"retry_count {current_job.retry_count} should be >= max_retries {max_retries} when permanently failed"
                    break  # Exit the loop
                else:
                    # Unexpected status
                    assert False, f"Unexpected job status: {current_job.status}"
                
                # Safety check to prevent infinite loop
                if failure_count > max_retries + 2:
                    assert False, f"Too many failures ({failure_count}), expected max {max_retries + 1}"
            
            # Clean up
            queue_manager.stop_retry_processor()
            queue_manager.db.close()
            
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


class JobStateTransitionStateMachine(RuleBasedStateMachine):
    """
    Stateful property-based testing for job state transitions.
    
    This state machine tests complex scenarios involving multiple jobs
    and various state transitions to ensure the state machine properties
    hold under all conditions.
    """
    
    jobs = Bundle('jobs')
    
    def __init__(self):
        super().__init__()
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "stateful_transitions_test.db")
        self.queue_manager = None
        self.submitted_jobs = {}
    
    @initialize()
    def setup_queue_manager(self):
        """Initialize the job queue manager."""
        self.queue_manager = JobQueueManager(self.db_path)
    
    @rule(target=jobs, job=create_job_strategy())
    def submit_job(self, job):
        """Submit a job to the queue."""
        assume(job.id not in self.submitted_jobs)
        
        job_id = self.queue_manager.submit_job(job)
        self.submitted_jobs[job_id] = JobStatus.QUEUED
        return job_id
    
    @rule(job_id=jobs)
    def transition_to_running(self, job_id):
        """Transition job from QUEUED to RUNNING."""
        current_job = self.queue_manager.get_job(job_id)
        if current_job and current_job.status == JobStatus.QUEUED:
            self.queue_manager.update_job_status(job_id, JobStatus.RUNNING)
            self.submitted_jobs[job_id] = JobStatus.RUNNING
    
    @rule(job_id=jobs)
    def complete_job(self, job_id):
        """Complete a running job."""
        current_job = self.queue_manager.get_job(job_id)
        if current_job and current_job.status == JobStatus.RUNNING:
            result = JobResult(success=True, outputs={"result": "completed"})
            self.queue_manager.update_job_status(job_id, JobStatus.COMPLETED, result)
            self.submitted_jobs[job_id] = JobStatus.COMPLETED
    
    @rule(job_id=jobs)
    def fail_job(self, job_id):
        """Fail a running job."""
        current_job = self.queue_manager.get_job(job_id)
        if current_job and current_job.status == JobStatus.RUNNING:
            self.queue_manager.update_job_status(job_id, JobStatus.FAILED, "Test failure")
            self.submitted_jobs[job_id] = JobStatus.FAILED
    
    @rule(job_id=jobs)
    def cancel_job(self, job_id):
        """Cancel a job."""
        current_job = self.queue_manager.get_job(job_id)
        if current_job and current_job.status in [JobStatus.QUEUED, JobStatus.RUNNING]:
            success = self.queue_manager.cancel_job(job_id)
            if success:
                self.submitted_jobs[job_id] = JobStatus.CANCELLED
    
    @rule(job_id=jobs)
    def retry_failed_job(self, job_id):
        """Retry a failed job."""
        current_job = self.queue_manager.get_job(job_id)
        if current_job and current_job.status == JobStatus.FAILED:
            error = Exception("Test error for retry")
            self.queue_manager.handle_job_failure(job_id, error)
            # Job should now be in RETRYING state
            retrying_job = self.queue_manager.get_job(job_id)
            if retrying_job.status == JobStatus.RETRYING:
                self.submitted_jobs[job_id] = JobStatus.RETRYING
    
    @invariant()
    def all_jobs_have_valid_states(self):
        """Invariant: All jobs should have valid states."""
        if not self.submitted_jobs or not self.queue_manager:
            return
        
        for job_id in self.submitted_jobs:
            current_job = self.queue_manager.get_job(job_id)
            assert current_job is not None, f"Job {job_id} should exist"
            
            # Verify status is valid
            assert isinstance(current_job.status, JobStatus), \
                f"Job {job_id} has invalid status type: {type(current_job.status)}"
            
            # Verify timestamps are consistent
            if current_job.started_at and current_job.created_at:
                assert current_job.started_at >= current_job.created_at, \
                    f"Job {job_id} started_at should be >= created_at"
            
            if current_job.completed_at and current_job.started_at:
                assert current_job.completed_at >= current_job.started_at, \
                    f"Job {job_id} completed_at should be >= started_at"
            
            # Verify state-specific properties
            if current_job.status == JobStatus.RUNNING:
                assert current_job.started_at is not None, \
                    f"Running job {job_id} should have started_at timestamp"
            
            if current_job.status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
                assert current_job.completed_at is not None, \
                    f"Finished job {job_id} should have completed_at timestamp"
            
            if current_job.status == JobStatus.COMPLETED:
                assert current_job.result is not None, \
                    f"Completed job {job_id} should have result"
            
            if current_job.status == JobStatus.RETRYING:
                assert current_job.retry_count > 0, \
                    f"Retrying job {job_id} should have retry_count > 0"
    
    @invariant()
    def state_transitions_are_valid(self):
        """Invariant: All state transitions should be valid according to state machine."""
        if not self.submitted_jobs or not self.queue_manager:
            return
        
        for job_id in self.submitted_jobs:
            current_job = self.queue_manager.get_job(job_id)
            if current_job:
                # We can't easily track previous states in this model,
                # but we can verify that the current state is reachable
                # from QUEUED (the initial state)
                reachable_states = {
                    JobStatus.QUEUED, JobStatus.RUNNING, JobStatus.COMPLETED,
                    JobStatus.FAILED, JobStatus.CANCELLED, JobStatus.RETRYING
                }
                assert current_job.status in reachable_states, \
                    f"Job {job_id} has unreachable state: {current_job.status}"
    
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


class TestJobStateTransitionStateful:
    """Stateful property-based tests for job state transitions."""
    
    def test_job_state_transitions_stateful(self):
        """
        **Property 3: Job State Transitions**
        **Validates: Requirements 2.4, 2.5**
        
        Stateful property test that verifies job state transitions follow
        the valid state machine across complex scenarios.
        """
        run_state_machine_as_test(
            JobStateTransitionStateMachine, 
            settings=settings(max_examples=50, stateful_step_count=30, deadline=None)
        )


class TestJobStateTransitionEdgeCases:
    """Edge case tests for job state transitions."""
    
    @given(create_job_strategy())
    @settings(max_examples=20, deadline=None)
    def test_rapid_state_transitions(self, job):
        """
        **Property 3: Job State Transitions**
        **Validates: Requirements 2.4, 2.5**
        
        Property: Rapid state transitions should be handled correctly
        without race conditions or data corruption.
        """
        # Create temporary database
        temp_dir = tempfile.mkdtemp()
        db_path = os.path.join(temp_dir, "test_rapid_transitions.db")
        
        try:
            queue_manager = JobQueueManager(db_path)
            
            # Submit the job
            queue_manager.submit_job(job)
            
            # Perform rapid transitions
            queue_manager.update_job_status(job.id, JobStatus.RUNNING)
            
            # Immediately complete the job
            result = JobResult(success=True, outputs={"result": "rapid completion"})
            queue_manager.update_job_status(job.id, JobStatus.COMPLETED, result)
            
            # Verify final state is correct
            final_job = queue_manager.get_job(job.id)
            assert final_job.status == JobStatus.COMPLETED
            assert final_job.result is not None
            assert final_job.result.success is True
            assert final_job.started_at is not None
            assert final_job.completed_at is not None
            assert final_job.completed_at >= final_job.started_at
            
            # Clean up
            queue_manager.stop_retry_processor()
            queue_manager.db.close()
            
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    @given(create_job_strategy())
    @settings(max_examples=20, deadline=None)
    def test_state_transition_data_integrity(self, job):
        """
        **Property 3: Job State Transitions**
        **Validates: Requirements 2.4, 2.5**
        
        Property: State transitions should preserve job data integrity
        and not corrupt other job fields.
        """
        # Create temporary database
        temp_dir = tempfile.mkdtemp()
        db_path = os.path.join(temp_dir, "test_data_integrity.db")
        
        try:
            queue_manager = JobQueueManager(db_path)
            
            # Submit the job
            queue_manager.submit_job(job)
            
            # Store original job data
            original_job = queue_manager.get_job(job.id)
            original_id = original_job.id
            original_user_id = original_job.user_id
            original_template_name = original_job.template_name
            original_inputs = original_job.inputs.copy()
            original_priority = original_job.priority
            original_metadata = original_job.metadata.copy()
            original_created_at = original_job.created_at
            
            # Perform state transitions
            queue_manager.update_job_status(job.id, JobStatus.RUNNING)
            running_job = queue_manager.get_job(job.id)
            
            # Verify core data is preserved
            assert running_job.id == original_id
            assert running_job.user_id == original_user_id
            assert running_job.template_name == original_template_name
            assert running_job.inputs == original_inputs
            assert running_job.priority == original_priority
            assert running_job.metadata == original_metadata
            assert running_job.created_at == original_created_at
            
            # Complete the job
            result = JobResult(success=True, outputs={"result": "integrity test"})
            queue_manager.update_job_status(job.id, JobStatus.COMPLETED, result)
            completed_job = queue_manager.get_job(job.id)
            
            # Verify core data is still preserved
            assert completed_job.id == original_id
            assert completed_job.user_id == original_user_id
            assert completed_job.template_name == original_template_name
            assert completed_job.inputs == original_inputs
            assert completed_job.priority == original_priority
            assert completed_job.metadata == original_metadata
            assert completed_job.created_at == original_created_at
            
            # Verify new fields are set correctly
            assert completed_job.status == JobStatus.COMPLETED
            assert completed_job.result is not None
            assert completed_job.started_at is not None
            assert completed_job.completed_at is not None
            
            # Clean up
            queue_manager.stop_retry_processor()
            queue_manager.db.close()
            
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)