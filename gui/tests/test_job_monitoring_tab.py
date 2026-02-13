"""
Unit tests for JobMonitoringTab component.
"""

import pytest
import pandas as pd
from unittest.mock import Mock, MagicMock
from gui.components.job_monitoring_tab import JobMonitoringTab
from gui.services.job_service import JobService


@pytest.fixture
def mock_job_service():
    """Create mock job service."""
    service = Mock(spec=JobService)
    
    # Mock get_jobs to return sample jobs with pagination
    service.get_jobs = Mock(return_value={
        'jobs': [
            {
                'job_id': 'job-123',
                'template': 'test_template',
                'status': 'running',
                'backend': 'colab',
                'created_at': '2024-01-01T10:00:00',
                'duration': 120.5
            },
            {
                'job_id': 'job-456',
                'template': 'another_template',
                'status': 'completed',
                'backend': 'kaggle',
                'created_at': '2024-01-01T09:00:00',
                'duration': 300.0
            }
        ],
        'total_count': 2,
        'page': 1,
        'page_size': 50,
        'total_pages': 1,
        'has_next': False,
        'has_prev': False
    })
    
    # Mock get_job_status
    service.get_job_status = Mock(return_value={
        'job_id': 'job-123',
        'template': 'test_template',
        'status': 'running',
        'backend': 'colab',
        'created_at': '2024-01-01T10:00:00',
        'inputs': {'input1': 'value1'},
        'outputs': None,
        'error': None
    })
    
    # Mock get_job_logs
    service.get_job_logs = Mock(return_value={
        'logs': "Job started\nProcessing...\n",
        'start_line': 0,
        'end_line': 2,
        'total_lines': 2,
        'has_more': False
    })
    
    # Mock get_job_results
    service.get_job_results = Mock(return_value={
        'success': True,
        'outputs': {'result': 'success'}
    })
    
    return service


@pytest.fixture
def job_monitoring_tab(mock_job_service):
    """Create JobMonitoringTab instance."""
    return JobMonitoringTab(mock_job_service)


class TestJobMonitoringTab:
    """Test suite for JobMonitoringTab."""
    
    def test_initialization(self, job_monitoring_tab):
        """Test that JobMonitoringTab initializes correctly."""
        assert job_monitoring_tab.job_service is not None
    
    def test_get_template_choices(self, job_monitoring_tab, mock_job_service):
        """Test retrieving unique template names from jobs."""
        choices = job_monitoring_tab._get_template_choices()
        
        assert len(choices) == 2
        assert 'test_template' in choices
        assert 'another_template' in choices
        mock_job_service.get_jobs.assert_called_once()
    
    def test_get_template_choices_error_handling(self, job_monitoring_tab, mock_job_service):
        """Test error handling when retrieving template choices fails."""
        mock_job_service.get_jobs.side_effect = Exception("Database error")
        
        choices = job_monitoring_tab._get_template_choices()
        
        assert choices == []
    
    def test_get_empty_dataframe(self, job_monitoring_tab):
        """Test getting empty dataframe with correct columns."""
        df = job_monitoring_tab._get_empty_dataframe()
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
        assert list(df.columns) == ["Job ID", "Template", "Status", "Backend", "Submitted", "Duration"]
    
    def test_on_refresh_jobs_no_filters(self, job_monitoring_tab, mock_job_service):
        """Test refreshing job list without filters."""
        df, page_info, prev_btn, next_btn, page = job_monitoring_tab.on_refresh_jobs("all", "all", "all", 1, "50")
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert df.iloc[0]["Job ID"] == "job-123"
        assert df.iloc[1]["Job ID"] == "job-456"
        assert "Page 1 of 1" in page_info
        mock_job_service.get_jobs.assert_called_once_with({
            'page': 1,
            'page_size': 50
        })
    
    def test_on_refresh_jobs_with_status_filter(self, job_monitoring_tab, mock_job_service):
        """Test refreshing job list with status filter."""
        df, page_info, prev_btn, next_btn, page = job_monitoring_tab.on_refresh_jobs("running", "all", "all", 1, "50")
        
        mock_job_service.get_jobs.assert_called_once_with({
            'page': 1,
            'page_size': 50,
            'status': 'running'
        })
    
    def test_on_refresh_jobs_with_template_filter(self, job_monitoring_tab, mock_job_service):
        """Test refreshing job list with template filter."""
        df, page_info, prev_btn, next_btn, page = job_monitoring_tab.on_refresh_jobs("all", "test_template", "all", 1, "50")
        
        mock_job_service.get_jobs.assert_called_once_with({
            'page': 1,
            'page_size': 50,
            'template': 'test_template'
        })
    
    def test_on_refresh_jobs_with_backend_filter(self, job_monitoring_tab, mock_job_service):
        """Test refreshing job list with backend filter."""
        df, page_info, prev_btn, next_btn, page = job_monitoring_tab.on_refresh_jobs("all", "all", "colab", 1, "50")
        
        mock_job_service.get_jobs.assert_called_once_with({
            'page': 1,
            'page_size': 50,
            'backend': 'colab'
        })
    
    def test_on_refresh_jobs_with_multiple_filters(self, job_monitoring_tab, mock_job_service):
        """Test refreshing job list with multiple filters."""
        df, page_info, prev_btn, next_btn, page = job_monitoring_tab.on_refresh_jobs("running", "test_template", "colab", 1, "50")
        
        mock_job_service.get_jobs.assert_called_once_with({
            'page': 1,
            'page_size': 50,
            'status': 'running',
            'template': 'test_template',
            'backend': 'colab'
        })
    
    def test_on_refresh_jobs_empty_result(self, job_monitoring_tab, mock_job_service):
        """Test refreshing job list when no jobs match filters."""
        mock_job_service.get_jobs.return_value = {
            'jobs': [],
            'total_count': 0,
            'page': 1,
            'page_size': 50,
            'total_pages': 1,
            'has_next': False,
            'has_prev': False
        }
        
        df, page_info, prev_btn, next_btn, page = job_monitoring_tab.on_refresh_jobs("all", "all", "all", 1, "50")
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
    
    def test_on_refresh_jobs_error_handling(self, job_monitoring_tab, mock_job_service):
        """Test error handling when refreshing jobs fails."""
        mock_job_service.get_jobs.side_effect = Exception("Database error")
        
        df, page_info, prev_btn, next_btn, page = job_monitoring_tab.on_refresh_jobs("all", "all", "all", 1, "50")
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
        assert "Error" in page_info
    
    def test_on_load_job_details_success(self, job_monitoring_tab, mock_job_service):
        """Test loading job details successfully."""
        result = job_monitoring_tab.on_load_job_details("job-123", 0, 1000)
        
        job_details, job_logs, log_page_info, prev_btn, next_btn, start_line, current_job_id, job_results, download_btn1, download_btn2 = result
        
        assert job_details['job_id'] == 'job-123'
        assert job_logs == "Job started\nProcessing...\n"
        assert "Lines 1-2 of 2" in log_page_info
        assert current_job_id == "job-123"
        mock_job_service.get_job_status.assert_called_once_with("job-123")
        mock_job_service.get_job_logs.assert_called_once_with("job-123", 0, 1000)
    
    def test_on_load_job_details_empty_job_id(self, job_monitoring_tab):
        """Test loading job details with empty job ID."""
        result = job_monitoring_tab.on_load_job_details("", 0, 1000)
        
        job_details, job_logs, log_page_info, prev_btn, next_btn, start_line, current_job_id, job_results, download_btn1, download_btn2 = result
        
        assert job_details == {}
        assert "Please enter a job ID" in job_logs
        assert current_job_id == ""
    
    def test_on_load_job_details_completed_job(self, job_monitoring_tab, mock_job_service):
        """Test loading details for a completed job."""
        mock_job_service.get_job_status.return_value = {
            'job_id': 'job-456',
            'status': 'completed',
            'template': 'test_template',
            'backend': 'colab',
            'created_at': '2024-01-01T10:00:00',
            'inputs': {},
            'outputs': {'result': 'success'},
            'error': None
        }
        
        result = job_monitoring_tab.on_load_job_details("job-456", 0, 1000)
        
        job_details, job_logs, log_page_info, prev_btn, next_btn, start_line, current_job_id, job_results, download_btn1, download_btn2 = result
        
        assert job_details['status'] == 'completed'
        assert job_results['success'] is True
        mock_job_service.get_job_results.assert_called_once_with("job-456")
    
    def test_on_load_job_details_failed_job(self, job_monitoring_tab, mock_job_service):
        """Test loading details for a failed job."""
        mock_job_service.get_job_status.return_value = {
            'job_id': 'job-789',
            'status': 'failed',
            'template': 'test_template',
            'backend': 'colab',
            'created_at': '2024-01-01T10:00:00',
            'inputs': {},
            'outputs': None,
            'error': 'Backend unavailable'
        }
        
        result = job_monitoring_tab.on_load_job_details("job-789", 0, 1000)
        
        job_details, job_logs, log_page_info, prev_btn, next_btn, start_line, current_job_id, job_results, download_btn1, download_btn2 = result
        
        assert job_details['status'] == 'failed'
        assert job_details['error'] == 'Backend unavailable'
    
    def test_on_load_job_details_error_handling(self, job_monitoring_tab, mock_job_service):
        """Test error handling when loading job details fails."""
        mock_job_service.get_job_status.side_effect = Exception("Job not found")
        
        result = job_monitoring_tab.on_load_job_details("job-999", 0, 1000)
        
        job_details, job_logs, log_page_info, prev_btn, next_btn, start_line, current_job_id, job_results, download_btn1, download_btn2 = result
        
        assert job_details == {}
        assert "Error loading job" in job_logs
    
    def test_format_datetime_valid(self, job_monitoring_tab):
        """Test formatting valid datetime string."""
        result = job_monitoring_tab._format_datetime("2024-01-01T10:30:45")
        
        assert "2024-01-01" in result
        assert "10:30:45" in result
    
    def test_format_datetime_none(self, job_monitoring_tab):
        """Test formatting None datetime."""
        result = job_monitoring_tab._format_datetime(None)
        
        assert result == "N/A"
    
    def test_format_datetime_invalid(self, job_monitoring_tab):
        """Test formatting invalid datetime string."""
        result = job_monitoring_tab._format_datetime("invalid-date")
        
        assert result == "invalid-date"
    
    def test_format_duration_seconds(self, job_monitoring_tab):
        """Test formatting duration in seconds."""
        result = job_monitoring_tab._format_duration(45.5)
        
        assert result == "45.5s"
    
    def test_format_duration_minutes(self, job_monitoring_tab):
        """Test formatting duration in minutes."""
        result = job_monitoring_tab._format_duration(120.0)
        
        assert result == "2.0m"
    
    def test_format_duration_hours(self, job_monitoring_tab):
        """Test formatting duration in hours."""
        result = job_monitoring_tab._format_duration(7200.0)
        
        assert result == "2.0h"
    
    def test_format_duration_none(self, job_monitoring_tab):
        """Test formatting None duration."""
        result = job_monitoring_tab._format_duration(None)
        
        assert result == "N/A"
    
    def test_websocket_client_code_generation(self, job_monitoring_tab):
        """Test that WebSocket client code is generated correctly."""
        code = job_monitoring_tab._get_websocket_client_code()
        
        # Verify essential WebSocket functionality is present
        assert '<script>' in code
        assert 'WebSocket' in code
        assert 'connectWebSocket' in code
        assert 'handleJobStatusChanged' in code
        assert 'handleJobCompleted' in code
        assert 'handleJobFailed' in code
        assert 'job.status_changed' in code
        assert 'job.completed' in code
        assert 'job.failed' in code
        assert 'wss:' in code or 'ws:' in code  # Protocol strings in template literal
        assert '7861' in code  # WebSocket port
    
    def test_websocket_client_handles_reconnection(self, job_monitoring_tab):
        """Test that WebSocket client code includes reconnection logic."""
        code = job_monitoring_tab._get_websocket_client_code()
        
        assert 'reconnectAttempts' in code
        assert 'maxReconnectAttempts' in code
        assert 'reconnectDelay' in code
        assert 'onclose' in code
    
    def test_websocket_client_handles_notifications(self, job_monitoring_tab):
        """Test that WebSocket client code includes notification functionality."""
        code = job_monitoring_tab._get_websocket_client_code()
        
        assert 'showNotification' in code
        assert 'success' in code
        assert 'error' in code
    
    def test_websocket_client_updates_ui(self, job_monitoring_tab):
        """Test that WebSocket client code includes UI update logic."""
        code = job_monitoring_tab._get_websocket_client_code()
        
        # Should trigger refresh button click
        assert 'Refresh' in code
        assert 'click()' in code
        
        # Should update job details if viewing the updated job
        assert 'Load Details' in code
        assert 'selectedJobIdInput' in code
    
    def test_on_prev_log_page_success(self, job_monitoring_tab, mock_job_service):
        """Test loading previous page of logs."""
        mock_job_service.get_job_logs.return_value = {
            'logs': "Previous page logs",
            'start_line': 0,
            'end_line': 1000,
            'total_lines': 2000,
            'has_more': True
        }
        
        job_logs, log_page_info, prev_btn, next_btn, new_start_line = \
            job_monitoring_tab.on_prev_log_page("job-123", 1000, 1000)
        
        assert job_logs == "Previous page logs"
        assert "Lines 1-1000 of 2000" in log_page_info
        assert new_start_line == 0
        mock_job_service.get_job_logs.assert_called_once_with("job-123", 0, 1000)
    
    def test_on_prev_log_page_empty_job_id(self, job_monitoring_tab):
        """Test loading previous page with empty job ID."""
        job_logs, log_page_info, prev_btn, next_btn, new_start_line = \
            job_monitoring_tab.on_prev_log_page("", 1000, 1000)
        
        assert "No job selected" in job_logs
        assert new_start_line == 0
    
    def test_on_prev_log_page_error_handling(self, job_monitoring_tab, mock_job_service):
        """Test error handling when loading previous page fails."""
        mock_job_service.get_job_logs.side_effect = Exception("Database error")
        
        job_logs, log_page_info, prev_btn, next_btn, new_start_line = \
            job_monitoring_tab.on_prev_log_page("job-123", 1000, 1000)
        
        assert "Error loading logs" in job_logs
    
    def test_on_next_log_page_success(self, job_monitoring_tab, mock_job_service):
        """Test loading next page of logs."""
        mock_job_service.get_job_logs.return_value = {
            'logs': "Next page logs",
            'start_line': 1000,
            'end_line': 2000,
            'total_lines': 2000,
            'has_more': False
        }
        
        job_logs, log_page_info, prev_btn, next_btn, new_start_line = \
            job_monitoring_tab.on_next_log_page("job-123", 0, 1000)
        
        assert job_logs == "Next page logs"
        assert "Lines 1001-2000 of 2000" in log_page_info
        assert new_start_line == 1000
        mock_job_service.get_job_logs.assert_called_once_with("job-123", 1000, 1000)
    
    def test_on_next_log_page_empty_job_id(self, job_monitoring_tab):
        """Test loading next page with empty job ID."""
        job_logs, log_page_info, prev_btn, next_btn, new_start_line = \
            job_monitoring_tab.on_next_log_page("", 0, 1000)
        
        assert "No job selected" in job_logs
        assert new_start_line == 0
    
    def test_on_next_log_page_error_handling(self, job_monitoring_tab, mock_job_service):
        """Test error handling when loading next page fails."""
        mock_job_service.get_job_logs.side_effect = Exception("Database error")
        
        job_logs, log_page_info, prev_btn, next_btn, new_start_line = \
            job_monitoring_tab.on_next_log_page("job-123", 0, 1000)
        
        assert "Error loading logs" in job_logs
