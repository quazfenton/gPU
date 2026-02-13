"""
Job service for GUI interface.

This module provides business logic for job submission, monitoring, and management
through the GUI interface.
"""

from typing import Any, Dict, List, Optional
from datetime import datetime

from notebook_ml_orchestrator.core.interfaces import Job, JobQueueInterface, BackendRouterInterface
from notebook_ml_orchestrator.core.models import JobStatus, JobResult
from notebook_ml_orchestrator.core.logging_config import LoggerMixin
from gui.error_handling import (
    ErrorResponse,
    format_validation_error,
    format_backend_error,
    format_system_error,
    format_generic_error,
    sanitize_error_message
)


class JobService(LoggerMixin):
    """Service for job submission and monitoring."""
    
    def __init__(self, job_queue: JobQueueInterface, backend_router: BackendRouterInterface):
        """
        Initialize job service.
        
        Args:
            job_queue: Job queue manager instance
            backend_router: Backend router instance
        """
        self.job_queue = job_queue
        self.backend_router = backend_router
        self.logger.info("JobService initialized")
    
    def submit_job(
        self,
        template_name: str,
        inputs: Dict[str, Any],
        backend: Optional[str] = None,
        user_id: str = "default_user",
        priority: int = 0,
        routing_strategy: str = "cost-optimized"
    ) -> str:
        """
        Submit a job and return job ID.
        
        Args:
            template_name: Name of the template to execute
            inputs: Dictionary of input parameters
            backend: Optional backend ID for explicit backend selection
            user_id: User ID submitting the job
            priority: Job priority (higher = more important)
            routing_strategy: Routing strategy for automatic backend selection
                            ("cost-optimized", "round-robin", "least-loaded")
            
        Returns:
            Job ID
            
        Raises:
            ValueError: If template_name or inputs are invalid
            Exception: If job submission fails
        """
        try:
            if not template_name:
                error = format_validation_error(
                    "template_name",
                    "Template name is required"
                )
                self.logger.error(f"Validation error: {error.message}")
                raise ValueError(error.message)
            
            if inputs is None:
                error = format_validation_error(
                    "inputs",
                    "Inputs dictionary is required"
                )
                self.logger.error(f"Validation error: {error.message}")
                raise ValueError(error.message)
            
            # Create job instance
            job = Job(
                user_id=user_id,
                template_name=template_name,
                inputs=inputs,
                priority=priority,
                backend_id=backend,  # Set backend_id if explicitly specified
                status=JobStatus.QUEUED,
                created_at=datetime.now(),
                metadata={"routing_strategy": routing_strategy}  # Store routing strategy in metadata
            )
            
            # Submit to job queue
            job_id = self.job_queue.submit_job(job)
            
            self.logger.info(
                f"Job submitted: job_id={job_id}, template={template_name}, "
                f"backend={backend or 'auto'}, routing_strategy={routing_strategy}, user={user_id}"
            )
            
            return job_id
            
        except ValueError:
            # Re-raise validation errors
            raise
        except Exception as e:
            # Handle unexpected errors
            error = format_system_error(
                "job_queue",
                sanitize_error_message(str(e)),
                is_recoverable=True
            )
            self.logger.error(f"Job submission failed: {error.message}", exc_info=True)
            raise Exception(error.message) from e
    
    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """
        Retrieve job status and details.
        
        Args:
            job_id: Job ID to retrieve
            
        Returns:
            Dictionary with job information including:
            - job_id: Job identifier
            - template: Template name
            - status: Current job status
            - backend: Backend ID (if assigned)
            - created_at: Job creation timestamp
            - started_at: Job start timestamp (if started)
            - completed_at: Job completion timestamp (if completed)
            - inputs: Job input parameters
            - result: Job result (if completed)
            - error: Error message (if failed)
            - retry_count: Number of retries
            - duration: Execution duration in seconds (if completed)
            
        Raises:
            ValueError: If job not found
            Exception: If retrieval fails
        """
        try:
            job = self.job_queue.get_job(job_id)
            
            if not job:
                error = format_validation_error(
                    "job_id",
                    f"Job {job_id} not found"
                )
                self.logger.error(f"Job not found: {job_id}")
                raise ValueError(error.message)
            
            # Calculate duration if job has completed
            duration = None
            if job.started_at and job.completed_at:
                duration = (job.completed_at - job.started_at).total_seconds()
            
            return {
                'job_id': job.id,
                'template': job.template_name,
                'status': job.status.value,
                'backend': job.backend_id,
                'created_at': job.created_at.isoformat() if job.created_at else None,
                'started_at': job.started_at.isoformat() if job.started_at else None,
                'completed_at': job.completed_at.isoformat() if job.completed_at else None,
                'inputs': job.inputs,
                'result': self._serialize_result(job.result) if job.result else None,
                'error': sanitize_error_message(job.error) if job.error else None,
                'retry_count': job.retry_count,
                'duration': duration,
                'priority': job.priority,
                'metadata': job.metadata
            }
            
        except ValueError:
            # Re-raise validation errors
            raise
        except Exception as e:
            # Handle unexpected errors
            error = format_system_error(
                "job_queue",
                sanitize_error_message(str(e)),
                is_recoverable=True
            )
            self.logger.error(f"Failed to get job status: {error.message}", exc_info=True)
            raise Exception(error.message) from e
    
    def get_jobs(self, filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Retrieve filtered list of jobs with pagination support.
        
        Args:
            filters: Optional dictionary with filter criteria:
                - status: Filter by job status (e.g., "completed", "running")
                - template: Filter by template name
                - backend: Filter by backend ID
                - user_id: Filter by user ID
                - date_from: Filter jobs created after this date (ISO format)
                - date_to: Filter jobs created before this date (ISO format)
                - page: Page number (1-indexed, default: 1)
                - page_size: Number of jobs per page (default: 50)
                - sort_by: Sort field ("created_at", "completed_at", "duration")
                - sort_order: Sort order ("asc" or "desc", default: "desc")
                
        Returns:
            Dictionary containing:
                - jobs: List of job dictionaries with summary information
                - total_count: Total number of jobs matching filters
                - page: Current page number
                - page_size: Jobs per page
                - total_pages: Total number of pages
                - has_next: Whether there is a next page
                - has_prev: Whether there is a previous page
        """
        filters = filters or {}
        
        # Get pagination parameters
        page = filters.get('page', 1)
        page_size = filters.get('page_size', 50)
        
        # Validate pagination parameters
        if page < 1:
            page = 1
        if page_size < 1:
            page_size = 50
        if page_size > 1000:
            page_size = 1000  # Cap at 1000 to prevent excessive data transfer
        
        # Get user_id filter or default to all users
        user_id = filters.get('user_id', 'default_user')
        
        # Get jobs from queue (fetch more than needed to allow filtering)
        jobs = self.job_queue.get_job_history(user_id, limit=10000)  # Get all jobs for filtering
        
        # Apply filters
        filtered_jobs = []
        for job in jobs:
            # Status filter
            if 'status' in filters and job.status.value != filters['status']:
                continue
            
            # Template filter
            if 'template' in filters and job.template_name != filters['template']:
                continue
            
            # Backend filter
            if 'backend' in filters and job.backend_id != filters['backend']:
                continue
            
            # Date range filters
            if 'date_from' in filters:
                date_from = datetime.fromisoformat(filters['date_from'])
                if job.created_at < date_from:
                    continue
            
            if 'date_to' in filters:
                date_to = datetime.fromisoformat(filters['date_to'])
                if job.created_at > date_to:
                    continue
            
            # Calculate duration
            duration = None
            if job.started_at and job.completed_at:
                duration = (job.completed_at - job.started_at).total_seconds()
            
            filtered_jobs.append({
                'job_id': job.id,
                'template': job.template_name,
                'status': job.status.value,
                'backend': job.backend_id,
                'created_at': job.created_at.isoformat() if job.created_at else None,
                'started_at': job.started_at.isoformat() if job.started_at else None,
                'completed_at': job.completed_at.isoformat() if job.completed_at else None,
                'duration': duration,
                'retry_count': job.retry_count,
                'priority': job.priority
            })
        
        # Sort jobs
        sort_by = filters.get('sort_by', 'created_at')
        sort_order = filters.get('sort_order', 'desc')
        reverse = (sort_order == 'desc')
        
        if sort_by == 'created_at':
            filtered_jobs.sort(key=lambda x: x['created_at'] or '', reverse=reverse)
        elif sort_by == 'completed_at':
            filtered_jobs.sort(key=lambda x: x['completed_at'] or '', reverse=reverse)
        elif sort_by == 'duration':
            filtered_jobs.sort(key=lambda x: x['duration'] or 0, reverse=reverse)
        
        # Calculate pagination metadata
        total_count = len(filtered_jobs)
        total_pages = (total_count + page_size - 1) // page_size if total_count > 0 else 1
        
        # Ensure page is within valid range
        if page > total_pages:
            page = total_pages
        
        # Calculate slice indices
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        
        # Get paginated slice
        paginated_jobs = filtered_jobs[start_idx:end_idx]
        
        # Return pagination result
        return {
            'jobs': paginated_jobs,
            'total_count': total_count,
            'page': page,
            'page_size': page_size,
            'total_pages': total_pages,
            'has_next': page < total_pages,
            'has_prev': page > 1
        }
    
    def get_job_results(self, job_id: str) -> Dict[str, Any]:
        """
        Retrieve job results.
        
        Args:
            job_id: Job ID to retrieve results for
            
        Returns:
            Dictionary with job results including:
            - success: Whether job completed successfully
            - outputs: Job output data
            - error_message: Error message if failed
            - execution_time_seconds: Execution time
            - backend_used: Backend that executed the job
            - metadata: Additional result metadata
            
        Raises:
            ValueError: If job not found or not completed
        """
        job = self.job_queue.get_job(job_id)
        
        if not job:
            raise ValueError(f"Job {job_id} not found")
        
        if job.status not in [JobStatus.COMPLETED, JobStatus.FAILED]:
            raise ValueError(f"Job {job_id} has not completed yet (status: {job.status.value})")
        
        if not job.result:
            return {
                'success': False,
                'outputs': {},
                'error_message': job.error or "No result available",
                'execution_time_seconds': 0.0,
                'backend_used': job.backend_id,
                'metadata': {}
            }
        
        return self._serialize_result(job.result)
    
    def get_job_logs(self, job_id: str, start_line: int = 0, max_lines: int = 1000) -> Dict[str, Any]:
        """
        Retrieve job execution logs with pagination support.
        
        Args:
            job_id: Job ID to retrieve logs for
            start_line: Starting line number (0-indexed)
            max_lines: Maximum number of lines to return
            
        Returns:
            Dictionary containing:
                - logs: Log content as string
                - start_line: Starting line number
                - end_line: Ending line number
                - total_lines: Total number of lines available
                - has_more: Whether more lines are available
            
        Raises:
            ValueError: If job not found
        """
        job = self.job_queue.get_job(job_id)
        
        if not job:
            raise ValueError(f"Job {job_id} not found")
        
        # Check if logs are stored in metadata
        logs = job.metadata.get('logs', '')
        
        if not logs:
            # Generate basic log information from job data
            log_lines = [
                f"Job ID: {job.id}",
                f"Template: {job.template_name}",
                f"Status: {job.status.value}",
                f"Created: {job.created_at.isoformat() if job.created_at else 'N/A'}",
                f"Started: {job.started_at.isoformat() if job.started_at else 'N/A'}",
                f"Completed: {job.completed_at.isoformat() if job.completed_at else 'N/A'}",
                f"Backend: {job.backend_id or 'Not assigned'}",
                f"Retry Count: {job.retry_count}",
            ]
            
            if job.error:
                log_lines.append(f"\nError: {job.error}")
            
            if job.result:
                log_lines.append(f"\nResult: {job.result.success}")
                if job.result.error_message:
                    log_lines.append(f"Error Message: {job.result.error_message}")
            
            logs = "\n".join(log_lines)
        
        # Split logs into lines for pagination
        log_lines_list = logs.split('\n')
        total_lines = len(log_lines_list)
        
        # Validate start_line
        if start_line < 0:
            start_line = 0
        if start_line >= total_lines:
            start_line = max(0, total_lines - max_lines)
        
        # Calculate end line
        end_line = min(start_line + max_lines, total_lines)
        
        # Extract the requested slice
        paginated_logs = '\n'.join(log_lines_list[start_line:end_line])
        
        return {
            'logs': paginated_logs,
            'start_line': start_line,
            'end_line': end_line,
            'total_lines': total_lines,
            'has_more': end_line < total_lines
        }
    
    def _serialize_result(self, result: JobResult) -> Dict[str, Any]:
        """
        Serialize JobResult to dictionary.
        
        Args:
            result: JobResult instance
            
        Returns:
            Dictionary representation of result
        """
        return {
            'success': result.success,
            'outputs': result.outputs,
            'error_message': result.error_message,
            'execution_time_seconds': result.execution_time_seconds,
            'backend_used': result.backend_used,
            'metadata': result.metadata
        }
