"""Tests for JobService."""

import pytest
from datetime import datetime
from unittest.mock import Mock, MagicMock

from gui.services.job_service import JobService
from notebook_ml_orchestrator.core.interfaces import Job
from notebook_ml_orchestrator.core.models import JobStatus, JobResult


class TestJobService:
    """Test suite for JobService."""
    
    @pytest.fixture
    def mock_job_queue(self):
        """Create mock job queue."""
        return Mock()
    
    @pytest.fixture
    def mock_backend_router(self):
        """Create mock backend router."""
        return Mock()
    
    @pytest.fixture
    def job_service(self, mock_job_queue, mock_backend_router):
        """Create JobService instance with mocks."""
        return JobService(mock_job_queue, mock_backend_router)
    
    def test_submit_job_success(self, job_service, mock_job_queue):
        """Test successful job submission."""
        # Setup
        mock_job_queue.submit_job.return_value = "job-123"
        
        # Execute
        job_id = job_service.submit_job(
            template_name="test_template",
            inputs={"param1": "value1"},
            backend="backend-1",
            user_id="user-1",
            priority=5,
            routing_strategy="round-robin"
        )
        
        # Verify
        assert job_id == "job-123"
        mock_job_queue.submit_job.assert_called_once()
        
        # Check the job that was submitted
        submitted_job = mock_job_queue.submit_job.call_args[0][0]
        assert submitted_job.template_name == "test_template"
        assert submitted_job.inputs == {"param1": "value1"}
        assert submitted_job.backend_id == "backend-1"
        assert submitted_job.user_id == "user-1"
        assert submitted_job.priority == 5
        assert submitted_job.status == JobStatus.QUEUED
        assert submitted_job.metadata.get("routing_strategy") == "round-robin"
    
    def test_submit_job_without_backend(self, job_service, mock_job_queue):
        """Test job submission without explicit backend selection."""
        # Setup
        mock_job_queue.submit_job.return_value = "job-456"
        
        # Execute
        job_id = job_service.submit_job(
            template_name="test_template",
            inputs={"param1": "value1"}
        )
        
        # Verify
        assert job_id == "job-456"
        submitted_job = mock_job_queue.submit_job.call_args[0][0]
        assert submitted_job.backend_id is None  # No backend specified
        assert submitted_job.metadata.get("routing_strategy") == "cost-optimized"  # Default strategy
    
    def test_submit_job_invalid_template(self, job_service):
        """Test job submission with invalid template name."""
        with pytest.raises(ValueError, match="Template name is required"):
            job_service.submit_job(template_name="", inputs={})
    
    def test_submit_job_invalid_inputs(self, job_service):
        """Test job submission with invalid inputs."""
        with pytest.raises(ValueError, match="Inputs dictionary is required"):
            job_service.submit_job(template_name="test", inputs=None)
    
    def test_get_job_status_completed(self, job_service, mock_job_queue):
        """Test retrieving status of a completed job."""
        # Setup
        job = Job(
            id="job-123",
            user_id="user-1",
            template_name="test_template",
            inputs={"param1": "value1"},
            status=JobStatus.COMPLETED,
            backend_id="backend-1",
            created_at=datetime(2024, 1, 1, 10, 0, 0),
            started_at=datetime(2024, 1, 1, 10, 1, 0),
            completed_at=datetime(2024, 1, 1, 10, 5, 0),
            result=JobResult(
                success=True,
                outputs={"result": "output_value"},
                execution_time_seconds=240.0,
                backend_used="backend-1"
            ),
            retry_count=0,
            priority=5
        )
        mock_job_queue.get_job.return_value = job
        
        # Execute
        status = job_service.get_job_status("job-123")
        
        # Verify
        assert status['job_id'] == "job-123"
        assert status['template'] == "test_template"
        assert status['status'] == "completed"
        assert status['backend'] == "backend-1"
        assert status['duration'] == 240.0  # 4 minutes
        assert status['result']['success'] is True
        assert status['result']['outputs'] == {"result": "output_value"}
        assert status['retry_count'] == 0
        assert status['priority'] == 5
    
    def test_get_job_status_running(self, job_service, mock_job_queue):
        """Test retrieving status of a running job."""
        # Setup
        job = Job(
            id="job-456",
            user_id="user-1",
            template_name="test_template",
            inputs={"param1": "value1"},
            status=JobStatus.RUNNING,
            backend_id="backend-2",
            created_at=datetime(2024, 1, 1, 10, 0, 0),
            started_at=datetime(2024, 1, 1, 10, 1, 0),
            retry_count=1
        )
        mock_job_queue.get_job.return_value = job
        
        # Execute
        status = job_service.get_job_status("job-456")
        
        # Verify
        assert status['job_id'] == "job-456"
        assert status['status'] == "running"
        assert status['duration'] is None  # Not completed yet
        assert status['result'] is None
        assert status['retry_count'] == 1
    
    def test_get_job_status_not_found(self, job_service, mock_job_queue):
        """Test retrieving status of non-existent job."""
        # Setup
        mock_job_queue.get_job.return_value = None
        
        # Execute & Verify
        with pytest.raises(ValueError, match="Job job-999 not found"):
            job_service.get_job_status("job-999")
    
    def test_get_jobs_no_filters(self, job_service, mock_job_queue):
        """Test retrieving jobs without filters."""
        # Setup
        jobs = [
            Job(
                id=f"job-{i}",
                user_id="default_user",
                template_name=f"template-{i}",
                inputs={},
                status=JobStatus.COMPLETED,
                created_at=datetime(2024, 1, i, 10, 0, 0),
                started_at=datetime(2024, 1, i, 10, 1, 0),
                completed_at=datetime(2024, 1, i, 10, 5, 0)
            )
            for i in range(1, 4)
        ]
        mock_job_queue.get_job_history.return_value = jobs
        
        # Execute
        result = job_service.get_jobs()
        
        # Verify
        assert result['total_count'] == 3
        assert result['page'] == 1
        assert result['page_size'] == 50
        assert result['total_pages'] == 1
        assert result['has_next'] == False
        assert result['has_prev'] == False
        assert len(result['jobs']) == 3
        assert result['jobs'][0]['job_id'] == "job-3"  # Most recent first (desc order)
        assert result['jobs'][1]['job_id'] == "job-2"
        assert result['jobs'][2]['job_id'] == "job-1"
    
    def test_get_jobs_with_status_filter(self, job_service, mock_job_queue):
        """Test retrieving jobs filtered by status."""
        # Setup
        jobs = [
            Job(id="job-1", user_id="default_user", template_name="t1", 
                inputs={}, status=JobStatus.COMPLETED, created_at=datetime(2024, 1, 1, 10, 0, 0)),
            Job(id="job-2", user_id="default_user", template_name="t2", 
                inputs={}, status=JobStatus.RUNNING, created_at=datetime(2024, 1, 2, 10, 0, 0)),
            Job(id="job-3", user_id="default_user", template_name="t3", 
                inputs={}, status=JobStatus.COMPLETED, created_at=datetime(2024, 1, 3, 10, 0, 0)),
        ]
        mock_job_queue.get_job_history.return_value = jobs
        
        # Execute
        result = job_service.get_jobs(filters={'status': 'completed'})
        
        # Verify
        assert result['total_count'] == 2
        assert len(result['jobs']) == 2
        assert all(job['status'] == 'completed' for job in result['jobs'])
    
    def test_get_jobs_with_template_filter(self, job_service, mock_job_queue):
        """Test retrieving jobs filtered by template."""
        # Setup
        jobs = [
            Job(id="job-1", user_id="default_user", template_name="template-a", 
                inputs={}, status=JobStatus.COMPLETED, created_at=datetime(2024, 1, 1, 10, 0, 0)),
            Job(id="job-2", user_id="default_user", template_name="template-b", 
                inputs={}, status=JobStatus.COMPLETED, created_at=datetime(2024, 1, 2, 10, 0, 0)),
            Job(id="job-3", user_id="default_user", template_name="template-a", 
                inputs={}, status=JobStatus.COMPLETED, created_at=datetime(2024, 1, 3, 10, 0, 0)),
        ]
        mock_job_queue.get_job_history.return_value = jobs
        
        # Execute
        result = job_service.get_jobs(filters={'template': 'template-a'})
        
        # Verify
        assert result['total_count'] == 2
        assert len(result['jobs']) == 2
        assert all(job['template'] == 'template-a' for job in result['jobs'])
    
    def test_get_jobs_with_sorting(self, job_service, mock_job_queue):
        """Test retrieving jobs with custom sorting."""
        # Setup
        jobs = [
            Job(id="job-1", user_id="default_user", template_name="t1", 
                inputs={}, status=JobStatus.COMPLETED, 
                created_at=datetime(2024, 1, 1, 10, 0, 0),
                started_at=datetime(2024, 1, 1, 10, 0, 0),
                completed_at=datetime(2024, 1, 1, 10, 2, 0)),  # 2 min duration
            Job(id="job-2", user_id="default_user", template_name="t2", 
                inputs={}, status=JobStatus.COMPLETED,
                created_at=datetime(2024, 1, 2, 10, 0, 0),
                started_at=datetime(2024, 1, 2, 10, 0, 0),
                completed_at=datetime(2024, 1, 2, 10, 5, 0)),  # 5 min duration
            Job(id="job-3", user_id="default_user", template_name="t3", 
                inputs={}, status=JobStatus.COMPLETED,
                created_at=datetime(2024, 1, 3, 10, 0, 0),
                started_at=datetime(2024, 1, 3, 10, 0, 0),
                completed_at=datetime(2024, 1, 3, 10, 1, 0)),  # 1 min duration
        ]
        mock_job_queue.get_job_history.return_value = jobs
        
        # Execute - sort by duration ascending
        result = job_service.get_jobs(filters={'sort_by': 'duration', 'sort_order': 'asc'})
        
        # Verify
        assert result['total_count'] == 3
        assert len(result['jobs']) == 3
        assert result['jobs'][0]['job_id'] == "job-3"  # 1 min
        assert result['jobs'][1]['job_id'] == "job-1"  # 2 min
        assert result['jobs'][2]['job_id'] == "job-2"  # 5 min
    
    def test_get_jobs_with_pagination(self, job_service, mock_job_queue):
        """Test retrieving jobs with pagination."""
        # Setup - create 25 jobs
        jobs = [
            Job(
                id=f"job-{i:03d}",
                user_id="default_user",
                template_name=f"template-{i}",
                inputs={},
                status=JobStatus.COMPLETED,
                created_at=datetime(2024, 1, 1, 10, i, 0)
            )
            for i in range(1, 26)
        ]
        mock_job_queue.get_job_history.return_value = jobs
        
        # Execute - get first page with page_size=10
        result_page1 = job_service.get_jobs(filters={'page': 1, 'page_size': 10})
        
        # Verify page 1
        assert result_page1['total_count'] == 25
        assert result_page1['page'] == 1
        assert result_page1['page_size'] == 10
        assert result_page1['total_pages'] == 3
        assert result_page1['has_next'] == True
        assert result_page1['has_prev'] == False
        assert len(result_page1['jobs']) == 10
        assert result_page1['jobs'][0]['job_id'] == "job-025"  # Most recent first
        
        # Execute - get second page
        result_page2 = job_service.get_jobs(filters={'page': 2, 'page_size': 10})
        
        # Verify page 2
        assert result_page2['page'] == 2
        assert result_page2['has_next'] == True
        assert result_page2['has_prev'] == True
        assert len(result_page2['jobs']) == 10
        assert result_page2['jobs'][0]['job_id'] == "job-015"
        
        # Execute - get last page
        result_page3 = job_service.get_jobs(filters={'page': 3, 'page_size': 10})
        
        # Verify page 3
        assert result_page3['page'] == 3
        assert result_page3['has_next'] == False
        assert result_page3['has_prev'] == True
        assert len(result_page3['jobs']) == 5  # Only 5 jobs on last page
        assert result_page3['jobs'][0]['job_id'] == "job-005"

    
    def test_get_job_results_success(self, job_service, mock_job_queue):
        """Test retrieving results of a successful job."""
        # Setup
        job = Job(
            id="job-123",
            user_id="user-1",
            template_name="test_template",
            inputs={},
            status=JobStatus.COMPLETED,
            result=JobResult(
                success=True,
                outputs={"result": "output_value"},
                execution_time_seconds=120.0,
                backend_used="backend-1",
                metadata={"info": "test"}
            )
        )
        mock_job_queue.get_job.return_value = job
        
        # Execute
        result = job_service.get_job_results("job-123")
        
        # Verify
        assert result['success'] is True
        assert result['outputs'] == {"result": "output_value"}
        assert result['execution_time_seconds'] == 120.0
        assert result['backend_used'] == "backend-1"
        assert result['metadata'] == {"info": "test"}
    
    def test_get_job_results_failed(self, job_service, mock_job_queue):
        """Test retrieving results of a failed job."""
        # Setup
        job = Job(
            id="job-456",
            user_id="user-1",
            template_name="test_template",
            inputs={},
            status=JobStatus.FAILED,
            error="Execution failed",
            result=None
        )
        mock_job_queue.get_job.return_value = job
        
        # Execute
        result = job_service.get_job_results("job-456")
        
        # Verify
        assert result['success'] is False
        assert result['error_message'] == "Execution failed"
    
    def test_get_job_results_not_completed(self, job_service, mock_job_queue):
        """Test retrieving results of a job that hasn't completed."""
        # Setup
        job = Job(
            id="job-789",
            user_id="user-1",
            template_name="test_template",
            inputs={},
            status=JobStatus.RUNNING
        )
        mock_job_queue.get_job.return_value = job
        
        # Execute & Verify
        with pytest.raises(ValueError, match="has not completed yet"):
            job_service.get_job_results("job-789")
    
    def test_get_job_logs_with_metadata(self, job_service, mock_job_queue):
        """Test retrieving job logs from metadata with pagination."""
        # Setup
        job = Job(
            id="job-123",
            user_id="user-1",
            template_name="test_template",
            inputs={},
            status=JobStatus.COMPLETED,
            metadata={"logs": "Log line 1\nLog line 2\nLog line 3"}
        )
        mock_job_queue.get_job.return_value = job
        
        # Execute
        log_data = job_service.get_job_logs("job-123")
        
        # Verify
        assert "Log line 1" in log_data['logs']
        assert "Log line 2" in log_data['logs']
        assert "Log line 3" in log_data['logs']
        assert log_data['total_lines'] == 3
        assert log_data['start_line'] == 0
        assert log_data['end_line'] == 3
        assert log_data['has_more'] is False
    
    def test_get_job_logs_pagination(self, job_service, mock_job_queue):
        """Test retrieving job logs with pagination."""
        # Setup - create logs with many lines
        log_lines = [f"Log line {i}" for i in range(100)]
        job = Job(
            id="job-123",
            user_id="user-1",
            template_name="test_template",
            inputs={},
            status=JobStatus.COMPLETED,
            metadata={"logs": "\n".join(log_lines)}
        )
        mock_job_queue.get_job.return_value = job
        
        # Execute - get first page
        log_data = job_service.get_job_logs("job-123", start_line=0, max_lines=10)
        
        # Verify first page
        assert "Log line 0" in log_data['logs']
        assert "Log line 9" in log_data['logs']
        assert "Log line 10" not in log_data['logs']
        assert log_data['total_lines'] == 100
        assert log_data['start_line'] == 0
        assert log_data['end_line'] == 10
        assert log_data['has_more'] is True
        
        # Execute - get second page
        log_data = job_service.get_job_logs("job-123", start_line=10, max_lines=10)
        
        # Verify second page
        assert "Log line 10" in log_data['logs']
        assert "Log line 19" in log_data['logs']
        assert "Log line 9" not in log_data['logs']
        assert log_data['start_line'] == 10
        assert log_data['end_line'] == 20
        assert log_data['has_more'] is True
        
        # Execute - get last page
        log_data = job_service.get_job_logs("job-123", start_line=90, max_lines=20)
        
        # Verify last page
        assert "Log line 90" in log_data['logs']
        assert "Log line 99" in log_data['logs']
        assert log_data['start_line'] == 90
        assert log_data['end_line'] == 100
        assert log_data['has_more'] is False
    
    def test_get_job_logs_generated(self, job_service, mock_job_queue):
        """Test retrieving generated job logs when no logs in metadata."""
        # Setup
        job = Job(
            id="job-123",
            user_id="user-1",
            template_name="test_template",
            inputs={},
            status=JobStatus.COMPLETED,
            backend_id="backend-1",
            created_at=datetime(2024, 1, 1, 10, 0, 0),
            started_at=datetime(2024, 1, 1, 10, 1, 0),
            completed_at=datetime(2024, 1, 1, 10, 5, 0),
            retry_count=2,
            error="Some error"
        )
        mock_job_queue.get_job.return_value = job
        
        # Execute
        log_data = job_service.get_job_logs("job-123")
        
        # Verify
        logs = log_data['logs']
        assert "Job ID: job-123" in logs
        assert "Template: test_template" in logs
        assert "Status: completed" in logs
        assert "Backend: backend-1" in logs
        assert "Retry Count: 2" in logs
        assert "Error: Some error" in logs
        assert log_data['total_lines'] > 0
    
    def test_get_job_logs_not_found(self, job_service, mock_job_queue):
        """Test retrieving logs for non-existent job."""
        # Setup
        mock_job_queue.get_job.return_value = None
        
        # Execute & Verify
        with pytest.raises(ValueError, match="Job job-999 not found"):
            job_service.get_job_logs("job-999")
