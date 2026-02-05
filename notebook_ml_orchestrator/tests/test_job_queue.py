"""
Unit tests for job queue management functionality.

This module tests job submission, retrieval, status updates,
and retry mechanisms.
"""

import pytest
import time
from datetime import datetime, timedelta

from notebook_ml_orchestrator.core.job_queue import JobQueueManager, RetryPolicy, JobStateManager
from notebook_ml_orchestrator.core.interfaces import Job
from notebook_ml_orchestrator.core.models import JobStatus, JobResult
from notebook_ml_orchestrator.core.exceptions import JobValidationError, DatabaseError


class TestRetryPolicy:
    """Test RetryPolicy functionality."""
    
    def test_default_retry_policy(self):
        """Test default retry policy settings."""
        policy = RetryPolicy()
        
        assert policy.max_retries == 3
        assert policy.base_delay == 1.0
        assert policy.max_delay == 300.0
        assert policy.exponential_base == 2.0
    
    def test_should_retry(self):
        """Test retry decision logic."""
        policy = RetryPolicy(max_retries=2)
        
        job = Job(id="test", user_id="user", template_name="template", inputs={})
        
        # Should retry when under limit
        job.retry_count = 0
        assert policy.should_retry(job) is True
        
        job.retry_count = 1
        assert policy.should_retry(job) is True
        
        # Should not retry when at limit
        job.retry_count = 2
        assert policy.should_retry(job) is False
        
        job.retry_count = 3
        assert policy.should_retry(job) is False
    
    def test_retry_delay_calculation(self):
        """Test retry delay calculation."""
        policy = RetryPolicy(base_delay=2.0, exponential_base=2.0, max_delay=10.0)
        
        # Test exponential backoff
        assert policy.get_retry_delay(0) == 2.0  # 2.0 * 2^0
        assert policy.get_retry_delay(1) == 4.0  # 2.0 * 2^1
        assert policy.get_retry_delay(2) == 8.0  # 2.0 * 2^2
        
        # Test max delay cap
        assert policy.get_retry_delay(10) == 10.0  # Capped at max_delay


class TestJobStateManager:
    """Test JobStateManager functionality."""
    
    def test_valid_transitions(self):
        """Test valid job state transitions."""
        # QUEUED -> RUNNING
        assert JobStateManager.can_transition(JobStatus.QUEUED, JobStatus.RUNNING) is True
        
        # RUNNING -> COMPLETED
        assert JobStateManager.can_transition(JobStatus.RUNNING, JobStatus.COMPLETED) is True
        
        # RUNNING -> FAILED
        assert JobStateManager.can_transition(JobStatus.RUNNING, JobStatus.FAILED) is True
        
        # FAILED -> RETRYING
        assert JobStateManager.can_transition(JobStatus.FAILED, JobStatus.RETRYING) is True
        
        # RETRYING -> QUEUED
        assert JobStateManager.can_transition(JobStatus.RETRYING, JobStatus.QUEUED) is True
    
    def test_invalid_transitions(self):
        """Test invalid job state transitions."""
        # COMPLETED -> RUNNING (invalid)
        assert JobStateManager.can_transition(JobStatus.COMPLETED, JobStatus.RUNNING) is False
        
        # CANCELLED -> RUNNING (invalid)
        assert JobStateManager.can_transition(JobStatus.CANCELLED, JobStatus.RUNNING) is False
        
        # QUEUED -> COMPLETED (invalid - must go through RUNNING)
        assert JobStateManager.can_transition(JobStatus.QUEUED, JobStatus.COMPLETED) is False
    
    def test_validate_transition(self):
        """Test transition validation."""
        job = Job(id="test", user_id="user", template_name="template", inputs={})
        job.status = JobStatus.QUEUED
        
        # Valid transition should pass
        assert JobStateManager.validate_transition(job, JobStatus.RUNNING) is True
        
        # Invalid transition should raise exception
        with pytest.raises(JobValidationError):
            JobStateManager.validate_transition(job, JobStatus.COMPLETED)


class TestJobQueueManager:
    """Test JobQueueManager functionality."""
    
    def test_job_submission(self, job_queue, sample_job):
        """Test job submission to queue."""
        job_id = job_queue.submit_job(sample_job)
        
        assert job_id == sample_job.id
        assert sample_job.status == JobStatus.QUEUED
        assert sample_job.created_at is not None
        
        # Verify job was stored
        retrieved_job = job_queue.get_job(job_id)
        assert retrieved_job is not None
        assert retrieved_job.id == sample_job.id
    
    def test_job_submission_validation(self, job_queue):
        """Test job submission validation."""
        # Test missing job ID
        job = Job(user_id="user", template_name="template", inputs={})
        job.id = ""  # Empty ID
        
        with pytest.raises(JobValidationError):
            job_queue.submit_job(job)
        
        # Test missing template name
        job = Job(id="test", user_id="user", inputs={})
        job.template_name = ""
        
        with pytest.raises(JobValidationError):
            job_queue.submit_job(job)
        
        # Test missing user ID
        job = Job(id="test", template_name="template", inputs={})
        job.user_id = ""
        
        with pytest.raises(JobValidationError):
            job_queue.submit_job(job)
    
    def test_get_next_job(self, job_queue, sample_jobs):
        """Test getting next job from queue."""
        # Submit jobs
        for job in sample_jobs:
            job_queue.submit_job(job)
        
        # Get next job with matching capabilities
        next_job = job_queue.get_next_job(["test-template"])
        
        assert next_job is not None
        assert next_job.template_name == "test-template"
        assert next_job.status == JobStatus.RUNNING
        assert next_job.started_at is not None
    
    def test_get_next_job_no_match(self, job_queue, sample_jobs):
        """Test getting next job with no matching capabilities."""
        # Submit jobs
        for job in sample_jobs:
            job_queue.submit_job(job)
        
        # Try to get job with non-matching capabilities
        next_job = job_queue.get_next_job(["non-existent-template"])
        
        assert next_job is None
    
    def test_get_next_job_wildcard(self, job_queue, sample_jobs):
        """Test getting next job with wildcard capabilities."""
        # Submit jobs
        for job in sample_jobs:
            job_queue.submit_job(job)
        
        # Get job with wildcard capability
        next_job = job_queue.get_next_job(["*"])
        
        assert next_job is not None
        assert next_job.status == JobStatus.RUNNING
    
    def test_update_job_status(self, job_queue, sample_job):
        """Test job status updates."""
        # Submit job
        job_queue.submit_job(sample_job)
        
        # Update to running
        job_queue.update_job_status(sample_job.id, JobStatus.RUNNING)
        
        updated_job = job_queue.get_job(sample_job.id)
        assert updated_job.status == JobStatus.RUNNING
        
        # Update to completed with result
        result = JobResult(success=True, outputs={"result": "success"})
        job_queue.update_job_status(sample_job.id, JobStatus.COMPLETED, result)
        
        completed_job = job_queue.get_job(sample_job.id)
        assert completed_job.status == JobStatus.COMPLETED
        assert completed_job.completed_at is not None
        assert completed_job.result is not None
        assert completed_job.result.success is True
    
    def test_handle_job_failure(self, job_queue, sample_job):
        """Test job failure handling."""
        # Submit job
        job_queue.submit_job(sample_job)
        
        # Handle failure
        error = Exception("Test error")
        job_queue.handle_job_failure(sample_job.id, error)
        
        failed_job = job_queue.get_job(sample_job.id)
        assert failed_job.status == JobStatus.RETRYING  # Should retry first
        assert failed_job.error == "Test error"
        assert failed_job.retry_count == 1
        assert "retry_at" in failed_job.metadata
    
    def test_handle_job_failure_max_retries(self, job_queue):
        """Test job failure handling when max retries exceeded."""
        # Create job with max retries already reached
        job = Job(
            id="max-retry-job",
            user_id="test-user",
            template_name="test-template",
            inputs={},
            retry_count=3  # At max retries
        )
        job_queue.submit_job(job)
        
        # Handle failure
        error = Exception("Final error")
        job_queue.handle_job_failure(job.id, error)
        
        failed_job = job_queue.get_job(job.id)
        assert failed_job.status == JobStatus.FAILED  # Should be permanently failed
        assert failed_job.error == "Final error"
    
    def test_cancel_job(self, job_queue, sample_job):
        """Test job cancellation."""
        # Submit job
        job_queue.submit_job(sample_job)
        
        # Cancel job
        success = job_queue.cancel_job(sample_job.id)
        assert success is True
        
        cancelled_job = job_queue.get_job(sample_job.id)
        assert cancelled_job.status == JobStatus.CANCELLED
        assert cancelled_job.completed_at is not None
    
    def test_cancel_completed_job(self, job_queue, sample_job):
        """Test cancelling already completed job."""
        # Submit and complete job
        job_queue.submit_job(sample_job)
        job_queue.update_job_status(sample_job.id, JobStatus.RUNNING)
        job_queue.update_job_status(sample_job.id, JobStatus.COMPLETED)
        
        # Try to cancel completed job
        success = job_queue.cancel_job(sample_job.id)
        assert success is False  # Cannot cancel completed job
    
    def test_get_job_history(self, job_queue, sample_jobs):
        """Test retrieving job history for a user."""
        # Submit jobs for different users
        for i, job in enumerate(sample_jobs):
            if i < 3:
                job.user_id = "user1"
            else:
                job.user_id = "user2"
            job_queue.submit_job(job)
        
        # Get history for user1
        user1_history = job_queue.get_job_history("user1")
        assert len(user1_history) == 3
        for job in user1_history:
            assert job.user_id == "user1"
        
        # Get history for user2
        user2_history = job_queue.get_job_history("user2")
        assert len(user2_history) == 2
        for job in user2_history:
            assert job.user_id == "user2"
    
    def test_queue_statistics(self, job_queue, sample_jobs):
        """Test queue statistics."""
        # Submit jobs first (they will all be QUEUED initially)
        for job in sample_jobs:
            job_queue.submit_job(job)
        
        # Update jobs to different statuses following valid transitions
        # Job 0: Keep as QUEUED
        # Job 1: QUEUED -> RUNNING
        job_queue.update_job_status(sample_jobs[1].id, JobStatus.RUNNING)
        # Job 2: QUEUED -> RUNNING -> COMPLETED
        job_queue.update_job_status(sample_jobs[2].id, JobStatus.RUNNING)
        job_queue.update_job_status(sample_jobs[2].id, JobStatus.COMPLETED)
        # Job 3: QUEUED -> RUNNING -> FAILED
        job_queue.update_job_status(sample_jobs[3].id, JobStatus.RUNNING)
        job_queue.update_job_status(sample_jobs[3].id, JobStatus.FAILED)
        # Job 4: Keep as QUEUED
        
        stats = job_queue.get_queue_statistics()
        
        assert stats['total_jobs'] == 5
        assert stats['by_status']['queued'] == 2
        assert stats['by_status']['running'] == 1
        assert stats['by_status']['completed'] == 1
        assert stats['by_status']['failed'] == 1
        assert stats['queue_length'] == 2
        assert stats['running_jobs'] == 1
    
    def test_cleanup_old_jobs(self, job_queue):
        """Test cleanup of old jobs."""
        # Create old completed job
        old_job = Job(
            id="old-job",
            user_id="test-user",
            template_name="test-template",
            inputs={}
        )
        old_job.created_at = datetime.now() - timedelta(days=35)
        job_queue.submit_job(old_job)
        # Update to completed status (QUEUED -> RUNNING -> COMPLETED)
        job_queue.update_job_status(old_job.id, JobStatus.RUNNING)
        job_queue.update_job_status(old_job.id, JobStatus.COMPLETED)
        
        # Create recent job
        recent_job = Job(
            id="recent-job",
            user_id="test-user",
            template_name="test-template",
            inputs={}
        )
        job_queue.submit_job(recent_job)
        # Update to completed status (QUEUED -> RUNNING -> COMPLETED)
        job_queue.update_job_status(recent_job.id, JobStatus.RUNNING)
        job_queue.update_job_status(recent_job.id, JobStatus.COMPLETED)
        
        # Cleanup old jobs
        deleted_count = job_queue.cleanup_old_jobs(30)
        
        # Should delete the old job but keep the recent one
        assert deleted_count == 1
        assert job_queue.get_job("old-job") is None
        assert job_queue.get_job("recent-job") is not None