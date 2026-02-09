"""
Job queue management for the Notebook ML Orchestrator.

This module implements persistent job queuing with SQLite storage,
retry mechanisms, and priority-based scheduling.
"""

import time
from datetime import datetime, timedelta
from typing import Any, List, Optional
import threading

from .interfaces import Job, JobQueueInterface
from .models import JobStatus, JobResult
from .database import DatabaseManager
from .exceptions import JobError, JobValidationError, DatabaseError
from .logging_config import LoggerMixin


class RetryPolicy:
    """Configurable retry policy for failed jobs."""
    
    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 300.0,
        exponential_base: float = 2.0
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
    
    def should_retry(self, job: Job) -> bool:
        """Check if job should be retried."""
        return job.retry_count < self.max_retries
    
    def get_retry_delay(self, retry_count: int) -> float:
        """Calculate delay before retry."""
        delay = self.base_delay * (self.exponential_base ** retry_count)
        return min(delay, self.max_delay)


class JobStateManager:
    """Manages job state transitions and validation."""
    
    VALID_TRANSITIONS = {
        JobStatus.QUEUED: [JobStatus.RUNNING, JobStatus.CANCELLED],
        JobStatus.RUNNING: [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED],
        JobStatus.COMPLETED: [],
        JobStatus.FAILED: [JobStatus.RETRYING, JobStatus.CANCELLED],
        JobStatus.CANCELLED: [],
        JobStatus.RETRYING: [JobStatus.QUEUED, JobStatus.CANCELLED]
    }
    
    @classmethod
    def can_transition(cls, from_status: JobStatus, to_status: JobStatus) -> bool:
        """Check if status transition is valid."""
        return to_status in cls.VALID_TRANSITIONS.get(from_status, [])
    
    @classmethod
    def validate_transition(cls, job: Job, new_status: JobStatus) -> bool:
        """Validate and perform status transition."""
        if not cls.can_transition(job.status, new_status):
            raise JobValidationError(
                f"Invalid status transition from {job.status} to {new_status}",
                {'job_id': job.id, 'current_status': job.status.value, 'new_status': new_status.value}
            )
        return True


class JobQueueManager(JobQueueInterface, LoggerMixin):
    """Manages persistent job queue with SQLite storage."""
    
    def __init__(self, db_path: str = "orchestrator.db", retry_policy: RetryPolicy = None):
        """
        Initialize job queue manager.
        
        Args:
            db_path: Path to SQLite database
            retry_policy: Retry policy for failed jobs
        """
        self.db = DatabaseManager(db_path)
        self.retry_policy = retry_policy or RetryPolicy()
        self._lock = threading.RLock()
        self._running = False
        self._retry_thread = None
        
        # Start retry processing thread
        self.start_retry_processor()
    
    def submit_job(self, job: Job) -> str:
        """
        Submit a new job to the queue.
        
        Args:
            job: Job to submit
            
        Returns:
            Job ID
            
        Raises:
            JobValidationError: If job validation fails
            DatabaseError: If database operation fails
        """
        with self._lock:
            # Validate job
            if not job.id:
                raise JobValidationError("Job ID is required")
            if not job.template_name:
                raise JobValidationError("Template name is required")
            if not job.user_id:
                raise JobValidationError("User ID is required")
            
            # Set initial status and timestamp (only if not already set)
            job.status = JobStatus.QUEUED
            if job.created_at is None:
                job.created_at = datetime.now()
            
            # Insert into database
            if not self.db.insert_job(job):
                raise DatabaseError(f"Failed to insert job {job.id}")
            
            self.logger.info(f"Job {job.id} submitted to queue")
            return job.id
    
    def get_next_job(self, backend_capabilities: List[str]) -> Optional[Job]:
        """
        Get the next job suitable for the given backend.
        
        Args:
            backend_capabilities: List of backend capabilities
            
        Returns:
            Next job to execute or None if no suitable job available
        """
        with self._lock:
            # Get queued jobs ordered by priority and creation time
            queued_jobs = self.db.get_jobs_by_status(JobStatus.QUEUED, limit=50)
            
            for job in queued_jobs:
                # Check if backend supports this template
                if job.template_name in backend_capabilities or '*' in backend_capabilities:
                    # Mark job as running
                    job.status = JobStatus.RUNNING
                    job.started_at = datetime.now()
                    
                    if self.db.update_job(job):
                        self.logger.info(f"Job {job.id} assigned for execution")
                        return job
                    else:
                        self.logger.error(f"Failed to update job {job.id} status to running")
            
            return None
    
    def update_job_status(self, job_id: str, status: JobStatus, result: Any = None):
        """
        Update job status and store results.
        
        Args:
            job_id: Job ID to update
            status: New job status
            result: Job result (for completed jobs)

        Raises:
            JobValidationError: If status transition is invalid
            DatabaseError: If database operation fails
        """
        with self._lock:
            job = self.db.get_job(job_id)
            if not job:
                raise JobValidationError(f"Job {job_id} not found")

            # Validate status transition
            JobStateManager.validate_transition(job, status)

            # Update job
            job.status = status
            if status == JobStatus.RUNNING:
                job.started_at = datetime.now()
            elif status == JobStatus.COMPLETED or (status == JobStatus.FAILED and not self.retry_policy.should_retry(job)):
                job.completed_at = datetime.now()
                if result:
                    job.result = result
            elif status == JobStatus.FAILED:
                # Consolidate error message assignment for failed jobs
                if isinstance(result, Exception) or isinstance(result, str):
                    job.error = str(result)
                elif result is not None:
                    job.error = str(result) # Fallback for other types of results

                if not self.retry_policy.should_retry(job):
                    job.completed_at = datetime.now()
                    # job.error is already set
                else:
                    job.retry_count += 1
                    job.retry_at = datetime.now() + timedelta(
                        seconds=self.retry_policy.get_retry_delay(job.retry_count)
                    )
                    # job.error is already set
                    job.status = JobStatus.RETRYING
                    status = job.status # Update local status variable for logging
            elif status == JobStatus.RETRYING:
                job.retry_count += 1
                job.retry_at = datetime.now() + timedelta(
                    seconds=self.retry_policy.get_retry_delay(job.retry_count)
                )
            elif isinstance(result, str): # This block handles string results if not caught by FAILED status
                job.error = result
            
            if not self.db.update_job(job):
                raise DatabaseError(f"Failed to update job {job_id}")
            
            self.logger.info(f"Job {job_id} status updated to {status}")
    
    def handle_job_failure(self, job_id: str, error: Exception):
        """
        Handle job failures with retry logic.
        
        Args:
            job_id: Failed job ID
            error: Exception that caused the failure
        """
        with self._lock:
            job = self.db.get_job(job_id)
            if not job:
                self.logger.error(f"Cannot handle failure for unknown job {job_id}")
                return
            
            job.error = str(error)
            job.completed_at = datetime.now()
            
            if self.retry_policy.should_retry(job):
                # Schedule for retry
                job.status = JobStatus.RETRYING
                job.retry_count += 1
                
                # Calculate retry delay
                retry_delay = self.retry_policy.get_retry_delay(job.retry_count)
                retry_time = datetime.now() + timedelta(seconds=retry_delay)
                job.metadata['retry_at'] = retry_time.isoformat()
                
                self.logger.info(f"Job {job_id} scheduled for retry {job.retry_count}/{self.retry_policy.max_retries} in {retry_delay}s")
            else:
                # Max retries exceeded
                job.status = JobStatus.FAILED
                self.logger.error(f"Job {job_id} failed permanently after {job.retry_count} retries: {error}")
            
            self.db.update_job(job)
    
    def get_job(self, job_id: str) -> Optional[Job]:
        """
        Retrieve a job by ID.
        
        Args:
            job_id: Job ID to retrieve
            
        Returns:
            Job instance or None if not found
        """
        return self.db.get_job(job_id)
    
    def get_job_history(self, user_id: str, limit: int = 100) -> List[Job]:
        """
        Retrieve job history for a user.
        
        Args:
            user_id: User ID
            limit: Maximum number of jobs to return
            
        Returns:
            List of jobs for the user
        """
        return self.db.get_user_jobs(user_id, limit)

    def cancel_job(self, job_id: str) -> bool:
        """
        Cancel a job.

        Args:
            job_id: Job ID to cancel

        Returns:
            True if successful, False otherwise
        """
        try:
            with self._lock:
                job = self.db.get_job(job_id)
                if not job:
                    return False

                if job.status in [JobStatus.COMPLETED, JobStatus.CANCELLED]:
                    return False  # Cannot cancel finished or already cancelled jobs

                if job.status == JobStatus.FAILED:
                    # Validate state transition before cancelling a failed job
                    JobStateManager.validate_transition(job, JobStatus.CANCELLED)

                job.status = JobStatus.CANCELLED
                job.completed_at = datetime.now()
                success = self.db.update_job(job) # Moved inside the lock
                if success:
                    self.logger.info(f"Job {job_id} cancelled")
                return success
        except Exception as e:
            self.logger.error(f"Failed to cancel job {job_id}: {e}")
            return False
    
    def get_queue_statistics(self) -> dict:
        """
        Get queue statistics.
        
        Returns:
            Dictionary with queue statistics
        """
        stats = self.db.get_job_statistics()
        return {
            'total_jobs': sum(stats.values()),
            'by_status': stats,
            'queue_length': stats.get('queued', 0),
            'running_jobs': stats.get('running', 0)
        }

    def start_retry_processor(self):
        """Start the retry processing thread."""
        if self._retry_thread and self._retry_thread.is_alive():
            return

        self._running = True
        self._retry_thread = threading.Thread(target=self._process_retries, daemon=True)
        self._retry_thread.start()
        self.logger.info("Retry processor started")

    def stop_retry_processor(self):
        """Stop the retry processing thread."""
        self._running = False
        if self._retry_thread:
            self._retry_thread.join(timeout=5.0)
        self.logger.info("Retry processor stopped")

    def _process_retries(self):
        """Process jobs scheduled for retry."""
        while self._running:
            try:
                with self._lock:
                    # Get jobs in retrying status
                    retrying_jobs = self.db.get_jobs_by_status(JobStatus.RETRYING, limit=100)
            
                    for job in retrying_jobs: # Moved inside the lock
                        retry_at_str = job.metadata.get('retry_at')
                        if retry_at_str:
                            retry_at = datetime.fromisoformat(retry_at_str)
                            if datetime.now() >= retry_at:
                                # Time to retry
                                job.status = JobStatus.QUEUED
                                job.started_at = None
                                job.completed_at = None
                                job.error = None
                                
                                if 'retry_at' in job.metadata:
                                    del job.metadata['retry_at']
                                
                                if self.db.update_job(job):
                                    self.logger.info(f"Job {job.id} moved back to queue for retry")
                
                # Sleep before next check (outside the lock, but inside the try)
                time.sleep(10)
                
            except Exception as e:
                self.logger.error(f"Error in retry processor: {e}")
                time.sleep(30)  # Wait longer on error
    
    def cleanup_old_jobs(self, days_old: int = 30) -> int:
        """
        Clean up old completed jobs.
        
        Args:
            days_old: Number of days after which to delete completed jobs
            
        Returns:
            Number of jobs deleted
        """
        deleted_count = self.db.cleanup_old_jobs(days_old)
        if deleted_count > 0:
            self.logger.info(f"Cleaned up {deleted_count} old jobs")
        return deleted_count
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        self.stop_retry_processor()