"""
SQLite database management for the Notebook ML Orchestrator.

This module provides database connection management, schema creation,
and data persistence for jobs, workflows, and system state.
"""

import sqlite3
import json
import logging
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import threading

from .models import JobStatus, WorkflowStatus, BatchStatus, BackendType, HealthStatus
from .interfaces import Job, Workflow, WorkflowExecution, BatchJob


logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manages SQLite database connections and operations."""
    
    def __init__(self, db_path: str = "orchestrator.db"):
        """
        Initialize database manager.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self._local = threading.local()
        self._ensure_database_exists()
        self._create_tables()
    
    def _ensure_database_exists(self):
        """Ensure database file and directory exist."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.db_path.exists():
            self.db_path.touch()
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
        if not hasattr(self._local, 'connection'):
            self._local.connection = sqlite3.connect(
                str(self.db_path),
                check_same_thread=False,
                timeout=30.0
            )
            self._local.connection.row_factory = sqlite3.Row
            # Enable WAL mode for better concurrency
            self._local.connection.execute("PRAGMA journal_mode=WAL")
            self._local.connection.execute("PRAGMA synchronous=NORMAL")
            self._local.connection.execute("PRAGMA cache_size=10000")
            self._local.connection.execute("PRAGMA temp_store=MEMORY")
        return self._local.connection
    
    @contextmanager
    def get_cursor(self):
        """Context manager for database operations."""
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            yield cursor
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Database operation failed: {e}")
            raise
        finally:
            cursor.close()
    
    def _create_tables(self):
        """Create database tables if they don't exist."""
        with self.get_cursor() as cursor:
            # Jobs table for persistent job queue
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    template_name TEXT NOT NULL,
                    inputs TEXT NOT NULL,
                    status TEXT NOT NULL,
                    backend_id TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    started_at TIMESTAMP,
                    completed_at TIMESTAMP,
                    result TEXT,
                    error TEXT,
                    retry_count INTEGER DEFAULT 0,
                    priority INTEGER DEFAULT 0,
                    metadata TEXT DEFAULT '{}'
                )
            """)
            
            # Workflows table for workflow definitions
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS workflows (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    user_id TEXT NOT NULL,
                    definition TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    version INTEGER DEFAULT 1,
                    metadata TEXT DEFAULT '{}'
                )
            """)
            
            # Workflow executions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS workflow_executions (
                    id TEXT PRIMARY KEY,
                    workflow_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    inputs TEXT,
                    outputs TEXT,
                    current_step TEXT,
                    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    completed_at TIMESTAMP,
                    error TEXT,
                    metadata TEXT DEFAULT '{}',
                    FOREIGN KEY (workflow_id) REFERENCES workflows (id)
                )
            """)
            
            # Backends table for compute resource tracking
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS backends (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    type TEXT NOT NULL,
                    capabilities TEXT NOT NULL,
                    max_concurrent_jobs INTEGER,
                    cost_per_hour REAL,
                    free_tier_limits TEXT,
                    health_status TEXT,
                    last_health_check TIMESTAMP,
                    metadata TEXT DEFAULT '{}'
                )
            """)
            
            # Batch jobs table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS batch_jobs (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    template_name TEXT NOT NULL,
                    items TEXT NOT NULL,
                    status TEXT NOT NULL,
                    progress TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    started_at TIMESTAMP,
                    completed_at TIMESTAMP,
                    metadata TEXT DEFAULT '{}'
                )
            """)
            
            # Create indexes for better performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_jobs_user_id ON jobs(user_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_jobs_priority ON jobs(priority DESC)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_jobs_created_at ON jobs(created_at)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_workflows_user_id ON workflows(user_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_workflow_executions_workflow_id ON workflow_executions(workflow_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_batch_jobs_user_id ON batch_jobs(user_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_batch_jobs_status ON batch_jobs(status)")
    
    def insert_job(self, job: Job) -> bool:
        """
        Insert a new job into the database.
        
        Args:
            job: Job instance to insert
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with self.get_cursor() as cursor:
                cursor.execute("""
                    INSERT INTO jobs (
                        id, user_id, template_name, inputs, status, backend_id,
                        created_at, started_at, completed_at, result, error,
                        retry_count, priority, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    job.id, job.user_id, job.template_name, json.dumps(job.inputs),
                    job.status.value, job.backend_id, 
                    job.created_at.isoformat() if job.created_at else None,
                    job.started_at.isoformat() if job.started_at else None,
                    job.completed_at.isoformat() if job.completed_at else None,
                    json.dumps(job.result.__dict__) if job.result else None,
                    job.error, job.retry_count, job.priority, json.dumps(job.metadata)
                ))
            return True
        except Exception as e:
            logger.error(f"Failed to insert job {job.id}: {e}")
            return False
    
    def update_job(self, job: Job) -> bool:
        """
        Update an existing job in the database.
        
        Args:
            job: Job instance to update
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with self.get_cursor() as cursor:
                cursor.execute("""
                    UPDATE jobs SET
                        status = ?, backend_id = ?, 
                        started_at = ?, completed_at = ?,
                        result = ?, error = ?, retry_count = ?, metadata = ?
                    WHERE id = ?
                """, (
                    job.status.value, job.backend_id, 
                    job.started_at.isoformat() if job.started_at else None,
                    job.completed_at.isoformat() if job.completed_at else None,
                    json.dumps(job.result.__dict__) if job.result else None,
                    job.error, job.retry_count, json.dumps(job.metadata), job.id
                ))
            return True
        except Exception as e:
            logger.error(f"Failed to update job {job.id}: {e}")
            return False
    
    def get_job(self, job_id: str) -> Optional[Job]:
        """
        Retrieve a job by ID.
        
        Args:
            job_id: Job ID to retrieve
            
        Returns:
            Job instance if found, None otherwise
        """
        try:
            with self.get_cursor() as cursor:
                cursor.execute("SELECT * FROM jobs WHERE id = ?", (job_id,))
                row = cursor.fetchone()
                if row:
                    return self._row_to_job(row)
            return None
        except Exception as e:
            logger.error(f"Failed to get job {job_id}: {e}")
            return None
    
    def get_jobs_by_status(self, status: JobStatus, limit: int = 100) -> List[Job]:
        """
        Retrieve jobs by status.
        
        Args:
            status: Job status to filter by
            limit: Maximum number of jobs to return
            
        Returns:
            List of Job instances
        """
        try:
            with self.get_cursor() as cursor:
                cursor.execute("""
                    SELECT * FROM jobs 
                    WHERE status = ? 
                    ORDER BY priority DESC, created_at ASC 
                    LIMIT ?
                """, (status.value, limit))
                return [self._row_to_job(row) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Failed to get jobs by status {status}: {e}")
            return []
    
    def get_user_jobs(self, user_id: str, limit: int = 100) -> List[Job]:
        """
        Retrieve jobs for a specific user.
        
        Args:
            user_id: User ID to filter by
            limit: Maximum number of jobs to return
            
        Returns:
            List of Job instances
        """
        try:
            with self.get_cursor() as cursor:
                cursor.execute("""
                    SELECT * FROM jobs
                    WHERE user_id = ?
                    ORDER BY created_at DESC
                    LIMIT ?
                """, (user_id, limit))
                return [self._row_to_job(row) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Failed to get jobs for user {user_id}: {e}")
            return []

    def _row_to_job(self, row: sqlite3.Row) -> Job:
        """Convert database row to Job instance."""
        from .models import JobResult  # Import here to avoid circular imports

        job = Job(
                id=row['id'],
                user_id=row['user_id'],
                template_name=row['template_name'],
                inputs=json.loads(row['inputs']),
                status=JobStatus(row['status']),
                backend_id=row['backend_id'],
                created_at=datetime.fromisoformat(row['created_at']) if row['created_at'] else datetime.now(),
                started_at=datetime.fromisoformat(row['started_at']) if row['started_at'] else None,
                completed_at=datetime.fromisoformat(row['completed_at']) if row['completed_at'] else None,
                error=row['error'],
                retry_count=row['retry_count'],
                priority=row['priority'],
                metadata=json.loads(row['metadata']) if row['metadata'] else {}
            )
            if row['result']:
                result_data = json.loads(row['result'])
                job.result = JobResult(**result_data)
            return job

        def cleanup_old_jobs(self, days_old: int = 30) -> int:
            """
            Clean up old completed jobs.

            Args:
                days_old: Number of days after which to delete completed jobs

            Returns:
                Number of jobs deleted
            """
            try:
                with self.get_cursor() as cursor:
                    cutoff = (datetime.now() - timedelta(days=days_old)).isoformat()
                    cursor.execute("""
                        DELETE FROM jobs
                        WHERE status IN ('completed', 'failed', 'cancelled')
                        AND created_at < ?
                    """, (cutoff,))
                    return cursor.rowcount
                                return cursor.rowcount
                        except Exception as e:
                            logger.error(f"Failed to cleanup old jobs: {e}")
                            return 0
    def get_job_statistics(self) -> Dict[str, int]:
        """
        Get job statistics by status.
        
        Returns:
            Dictionary with status counts
        """
        try:
            with self.get_cursor() as cursor:
                cursor.execute("""
                    SELECT status, COUNT(*) as count 
                    FROM jobs 
                    GROUP BY status
                """)
                return {row['status']: row['count'] for row in cursor.fetchall()}
        except Exception as e:
            logger.error(f"Failed to get job statistics: {e}")
            return {}
    
    def close(self):
        """Close database connections."""
        if hasattr(self._local, 'connection'):
            self._local.connection.close()
            delattr(self._local, 'connection')