"""
Unit tests for database management functionality.

This module tests SQLite database operations, job persistence,
and data integrity.
"""

import pytest
from datetime import datetime, timedelta

from notebook_ml_orchestrator.core.database import DatabaseManager
from notebook_ml_orchestrator.core.interfaces import Job
from notebook_ml_orchestrator.core.models import JobStatus, JobResult


class TestDatabaseManager:
    """Test DatabaseManager functionality."""
    
    def test_database_creation(self, temp_db_path):
        """Test database file creation and table setup."""
        db = DatabaseManager(temp_db_path)
        
        # Check that database file was created
        assert db.db_path.exists()
        
        # Check that tables were created
        with db.get_cursor() as cursor:
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            
            expected_tables = ['jobs', 'workflows', 'workflow_executions', 'backends', 'batch_jobs']
            for table in expected_tables:
                assert table in tables
    
    def test_insert_job(self, db_manager, sample_job):
        """Test job insertion."""
        success = db_manager.insert_job(sample_job)
        assert success is True
        
        # Verify job was inserted
        retrieved_job = db_manager.get_job(sample_job.id)
        assert retrieved_job is not None
        assert retrieved_job.id == sample_job.id
        assert retrieved_job.user_id == sample_job.user_id
        assert retrieved_job.template_name == sample_job.template_name
        assert retrieved_job.inputs == sample_job.inputs
        assert retrieved_job.status == sample_job.status
    
    def test_update_job(self, db_manager, sample_job):
        """Test job updates."""
        # Insert job first
        db_manager.insert_job(sample_job)
        
        # Update job
        sample_job.status = JobStatus.RUNNING
        sample_job.started_at = datetime.now()
        sample_job.backend_id = "test-backend"
        
        success = db_manager.update_job(sample_job)
        assert success is True
        
        # Verify update
        retrieved_job = db_manager.get_job(sample_job.id)
        assert retrieved_job.status == JobStatus.RUNNING
        assert retrieved_job.started_at is not None
        assert retrieved_job.backend_id == "test-backend"
    
    def test_update_job_with_result(self, db_manager, sample_job):
        """Test job update with result."""
        # Insert job first
        db_manager.insert_job(sample_job)
        
        # Update with result
        result = JobResult(
            success=True,
            outputs={"result": "test output"},
            execution_time_seconds=5.0,
            backend_used="test-backend"
        )
        
        sample_job.status = JobStatus.COMPLETED
        sample_job.completed_at = datetime.now()
        sample_job.result = result
        
        success = db_manager.update_job(sample_job)
        assert success is True
        
        # Verify result was stored
        retrieved_job = db_manager.get_job(sample_job.id)
        assert retrieved_job.result is not None
        assert retrieved_job.result.success is True
        assert retrieved_job.result.outputs == {"result": "test output"}
        assert retrieved_job.result.execution_time_seconds == 5.0
    
    def test_get_jobs_by_status(self, db_manager, sample_jobs):
        """Test retrieving jobs by status."""
        # Insert jobs with different statuses
        for i, job in enumerate(sample_jobs):
            if i < 2:
                job.status = JobStatus.QUEUED
            elif i < 4:
                job.status = JobStatus.RUNNING
            else:
                job.status = JobStatus.COMPLETED
            db_manager.insert_job(job)
        
        # Test queued jobs
        queued_jobs = db_manager.get_jobs_by_status(JobStatus.QUEUED)
        assert len(queued_jobs) == 2
        for job in queued_jobs:
            assert job.status == JobStatus.QUEUED
        
        # Test running jobs
        running_jobs = db_manager.get_jobs_by_status(JobStatus.RUNNING)
        assert len(running_jobs) == 2
        for job in running_jobs:
            assert job.status == JobStatus.RUNNING
        
        # Test completed jobs
        completed_jobs = db_manager.get_jobs_by_status(JobStatus.COMPLETED)
        assert len(completed_jobs) == 1
        for job in completed_jobs:
            assert job.status == JobStatus.COMPLETED
    
    def test_get_user_jobs(self, db_manager, sample_jobs):
        """Test retrieving jobs for a specific user."""
        # Insert jobs for different users
        for i, job in enumerate(sample_jobs):
            if i < 3:
                job.user_id = "user1"
            else:
                job.user_id = "user2"
            db_manager.insert_job(job)
        
        # Test user1 jobs
        user1_jobs = db_manager.get_user_jobs("user1")
        assert len(user1_jobs) == 3
        for job in user1_jobs:
            assert job.user_id == "user1"
        
        # Test user2 jobs
        user2_jobs = db_manager.get_user_jobs("user2")
        assert len(user2_jobs) == 2
        for job in user2_jobs:
            assert job.user_id == "user2"
        
        # Test non-existent user
        no_jobs = db_manager.get_user_jobs("nonexistent")
        assert len(no_jobs) == 0
    
    def test_job_priority_ordering(self, db_manager):
        """Test that jobs are ordered by priority and creation time."""
        # Create jobs with different priorities
        jobs = []
        for i in range(5):
            job = Job(
                id=f"priority-job-{i}",
                user_id="test-user",
                template_name="test-template",
                inputs={},
                priority=i % 3,  # Priorities: 0, 1, 2, 0, 1
                status=JobStatus.QUEUED
            )
            jobs.append(job)
            db_manager.insert_job(job)
        
        # Get queued jobs (should be ordered by priority DESC, then created_at ASC)
        queued_jobs = db_manager.get_jobs_by_status(JobStatus.QUEUED)
        
        # Check that higher priority jobs come first
        priorities = [job.priority for job in queued_jobs]
        assert priorities == sorted(priorities, reverse=True)
    
    def test_get_job_statistics(self, db_manager, sample_jobs):
        """Test job statistics retrieval."""
        # Insert jobs with different statuses
        statuses = [JobStatus.QUEUED, JobStatus.RUNNING, JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.QUEUED]
        for job, status in zip(sample_jobs, statuses):
            job.status = status
            db_manager.insert_job(job)
        
        stats = db_manager.get_job_statistics()
        
        assert stats['queued'] == 2
        assert stats['running'] == 1
        assert stats['completed'] == 1
        assert stats['failed'] == 1
    
    def test_cleanup_old_jobs(self, db_manager):
        """Test cleanup of old completed jobs."""
        # Create old completed jobs
        old_time = datetime.now() - timedelta(days=35)
        recent_time = datetime.now() - timedelta(days=5)
        
        old_job = Job(
            id="old-job",
            user_id="test-user",
            template_name="test-template",
            inputs={},
            status=JobStatus.COMPLETED,
            created_at=old_time
        )
        
        recent_job = Job(
            id="recent-job",
            user_id="test-user",
            template_name="test-template",
            inputs={},
            status=JobStatus.COMPLETED,
            created_at=recent_time
        )
        
        running_job = Job(
            id="running-job",
            user_id="test-user",
            template_name="test-template",
            inputs={},
            status=JobStatus.RUNNING,
            created_at=old_time
        )
        
        db_manager.insert_job(old_job)
        db_manager.insert_job(recent_job)
        db_manager.insert_job(running_job)
        
        # Cleanup jobs older than 30 days
        deleted_count = db_manager.cleanup_old_jobs(30)
        
        # Should delete only the old completed job
        assert deleted_count == 1
        
        # Verify correct jobs remain
        assert db_manager.get_job("old-job") is None
        assert db_manager.get_job("recent-job") is not None
        assert db_manager.get_job("running-job") is not None
    
    def test_concurrent_access(self, db_manager, sample_jobs):
        """Test concurrent database access."""
        import threading
        
        def insert_jobs(jobs_subset):
            for job in jobs_subset:
                db_manager.insert_job(job)
        
        # Split jobs into two groups for concurrent insertion
        mid = len(sample_jobs) // 2
        group1 = sample_jobs[:mid]
        group2 = sample_jobs[mid:]
        
        # Create threads for concurrent insertion
        thread1 = threading.Thread(target=insert_jobs, args=(group1,))
        thread2 = threading.Thread(target=insert_jobs, args=(group2,))
        
        # Start threads
        thread1.start()
        thread2.start()
        
        # Wait for completion
        thread1.join()
        thread2.join()
        
        # Verify all jobs were inserted
        for job in sample_jobs:
            retrieved_job = db_manager.get_job(job.id)
            assert retrieved_job is not None
            assert retrieved_job.id == job.id
    
    def test_database_error_handling(self, db_manager):
        """Test database error handling."""
        # Test inserting job with duplicate ID
        job1 = Job(
            id="duplicate-id",
            user_id="user1",
            template_name="template1",
            inputs={}
        )
        
        job2 = Job(
            id="duplicate-id",
            user_id="user2",
            template_name="template2",
            inputs={}
        )
        
        # First insertion should succeed
        success1 = db_manager.insert_job(job1)
        assert success1 is True
        
        # Second insertion with same ID should fail
        success2 = db_manager.insert_job(job2)
        assert success2 is False
    
    def test_json_serialization(self, db_manager):
        """Test JSON serialization of complex data."""
        complex_inputs = {
            "string_param": "test",
            "number_param": 42,
            "list_param": [1, 2, 3],
            "dict_param": {"nested": "value"},
            "bool_param": True
        }
        
        job = Job(
            id="complex-job",
            user_id="test-user",
            template_name="test-template",
            inputs=complex_inputs,
            metadata={"custom": {"nested": {"data": "value"}}}
        )
        
        # Insert and retrieve
        db_manager.insert_job(job)
        retrieved_job = db_manager.get_job("complex-job")
        
        # Verify complex data was preserved
        assert retrieved_job.inputs == complex_inputs
        assert retrieved_job.metadata == {"custom": {"nested": {"data": "value"}}}