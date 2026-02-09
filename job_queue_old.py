"""
Persistent job queue using SQLite.

Zero-infrastructure job queue that survives process restarts.
Supports batch processing, job chaining, and multi-backend routing.
"""

import sqlite3
import json
import time
import uuid
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class JobStatus(Enum):
    """Job execution status."""
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    RETRY = "retry"
    CANCELLED = "cancelled"


@dataclass
class Job:
    """Job data container."""
    id: str
    template: str
    payload: Dict[str, Any]
    route: str
    status: JobStatus
    result: Optional[Dict[str, Any]]
    error: Optional[str]
    retries: int
    created: float
    updated: float
    workflow_id: Optional[str] = None
    parent_job_id: Optional[str] = None


# Default database path - can be overridden
DEFAULT_DB_PATH = Path(__file__).parent / "jobs.db"


def get_db_path() -> Path:
    """Get database path, checking environment variable first."""
    env_path = os.environ.get("ZAP_JOBS_DB")
    if env_path:
        return Path(env_path)
    return DEFAULT_DB_PATH


def init_db(db_path: Optional[Path] = None) -> None:
    """Initialize the job queue database."""
    db_path = db_path or get_db_path()
    
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS jobs (
            id TEXT PRIMARY KEY,
            template TEXT NOT NULL,
            payload TEXT NOT NULL,
            route TEXT NOT NULL,
            status TEXT NOT NULL,
            result TEXT,
            error TEXT,
            retries INTEGER DEFAULT 0,
            created REAL NOT NULL,
            updated REAL NOT NULL,
            workflow_id TEXT,
            parent_job_id TEXT
        )
    """)
    
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status)
    """)
    
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_jobs_workflow ON jobs(workflow_id)
    """)
    
    conn.commit()
    conn.close()


def enqueue(
    template: str,
    payload: Dict[str, Any],
    route: str = "local",
    workflow_id: Optional[str] = None,
    parent_job_id: Optional[str] = None,
    db_path: Optional[Path] = None
) -> str:
    """
    Add a job to the queue.
    
    Args:
        template: Template name to execute
        payload: Input data for the template
        route: Execution backend (local, modal, hf)
        workflow_id: Optional workflow this job belongs to
        parent_job_id: Optional parent job for chaining
        db_path: Optional database path
        
    Returns:
        Job ID
    """
    db_path = db_path or get_db_path()
    init_db(db_path)
    
    job_id = str(uuid.uuid4())
    now = time.time()
    
    with sqlite3.connect(str(db_path)) as conn:
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO jobs (
                id, template, payload, route, status, result, error,
                retries, created, updated, workflow_id, parent_job_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            job_id,
            template,
            json.dumps(payload),
            route,
            JobStatus.QUEUED.value,
            None,
            None,
            0,
            now,
            now,
            workflow_id,
            parent_job_id
        ))
        
        conn.commit()
    
    return job_id


def enqueue_batch(
    template: str,
    payloads: List[Dict[str, Any]],
    route: str = "local",
    workflow_id: Optional[str] = None,
    db_path: Optional[Path] = None
) -> List[str]:
    """
    Add multiple jobs to the queue.
    
    Args:
        template: Template name to execute
        payloads: List of input data dicts
        route: Execution backend
        workflow_id: Optional workflow ID
        db_path: Optional database path
        
    Returns:
        List of job IDs
    """
    return [
        enqueue(template, p, route, workflow_id, db_path=db_path)
        for p in payloads
    ]


def next_job(
    route: Optional[str] = None,
    db_path: Optional[Path] = None
) -> Optional[Job]:
    """
    Fetch and claim the next queued job.
    
    Args:
        route: Optional route filter
        db_path: Optional database path
        
    Returns:
        Job object or None if no jobs available
    """
    db_path = db_path or get_db_path()
    init_db(db_path)

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    # Use a single atomic UPDATE with RETURNING to claim the next job
    # Use a single atomic UPDATE with RETURNING to claim the next job
    query = """
        UPDATE jobs
        SET status = ?, updated = ?
        WHERE id = (
            SELECT id FROM jobs
            WHERE status IN (?, ?)
    """
    params = [JobStatus.RUNNING.value, time.time(), JobStatus.QUEUED.value, JobStatus.RETRY.value]

    if route:
        query += " AND route = ?"
        params.append(route)

    query += """
        ORDER BY created
        LIMIT 1
    )
    RETURNING id, template, payload, route, status, result, error,
              retries, created, updated, workflow_id, parent_job_id
    """

    cursor.execute(query, params)
    row = cursor.fetchone()

    if not row:
        conn.close()
        return None
    
    conn.commit()
    conn.close()
    
    return Job(
        id=row[0],
        template=row[1],
        payload=json.loads(row[2]),
        route=row[3],
        status=JobStatus(row[4]),
        result=json.loads(row[5]) if row[5] else None,
        error=row[6],
        retries=row[7],
        created=row[8],
        updated=row[9],
        workflow_id=row[10],
        parent_job_id=row[11]
    )

def complete(
    job_id: str,
    result: Dict[str, Any],
    db_path: Optional[Path] = None
) -> None:
    """
    Mark a job as completed with result.
    
    Args:
        job_id: Job ID
        result: Output data from template execution
        db_path: Optional database path
    """
    db_path = db_path or get_db_path()
    
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    cursor.execute("""
        UPDATE jobs
        SET status = ?, result = ?, updated = ?
        WHERE id = ?
    """, (JobStatus.DONE.value, json.dumps(result), time.time(), job_id))
    
    conn.commit()
    conn.close()


def fail(
    job_id: str,
    error: str,
    retry: bool = False,
    max_retries: int = 3,
    db_path: Optional[Path] = None
) -> None:
    """
    Mark a job as failed.
    
    Args:
        job_id: Job ID
        error: Error message
        retry: Whether to retry the job
        max_retries: Maximum retry attempts
        db_path: Optional database path
    """
    db_path = db_path or get_db_path()
    
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    # Get current retry count
    cursor.execute("SELECT retries FROM jobs WHERE id = ?", (job_id,))
    row = cursor.fetchone()
    current_retries = row[0] if row else 0
    
    if retry and current_retries < max_retries:
        status = JobStatus.RETRY.value
        new_retries = current_retries + 1
    else:
        status = JobStatus.FAILED.value
        new_retries = current_retries
    
    cursor.execute("""
        UPDATE jobs
        SET status = ?, error = ?, retries = ?, updated = ?
        WHERE id = ?
    """, (status, error, new_retries, time.time(), job_id))
    
    conn.commit()
    conn.close()


def cancel(job_id: str, db_path: Optional[Path] = None) -> bool:
    """
    Cancel a queued job.
    
    Args:
        job_id: Job ID
        db_path: Optional database path
        
    Returns:
        True if cancelled, False if job was already running/done
    """
    db_path = db_path or get_db_path()
    
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    cursor.execute("""
        UPDATE jobs
        SET status = ?, updated = ?
        WHERE id = ? AND status = ?
    """, (JobStatus.CANCELLED.value, time.time(), job_id, JobStatus.QUEUED.value))
    
    affected = cursor.rowcount
    conn.commit()
    conn.close()
    
    return affected > 0


def get_job(job_id: str, db_path: Optional[Path] = None) -> Optional[Job]:
    """
    Get a job by ID.
    
    Args:
        job_id: Job ID
        db_path: Optional database path
        
    Returns:
        Job object or None
    """
    db_path = db_path or get_db_path()
    init_db(db_path)
    
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT id, template, payload, route, status, result, error,
               retries, created, updated, workflow_id, parent_job_id
        FROM jobs WHERE id = ?
    """, (job_id,))
    
    row = cursor.fetchone()
    conn.close()
    
    if not row:
        return None
    
    return Job(
        id=row[0],
        template=row[1],
        payload=json.loads(row[2]),
        route=row[3],
        status=JobStatus(row[4]),
        result=json.loads(row[5]) if row[5] else None,
        error=row[6],
        retries=row[7],
        created=row[8],
        updated=row[9],
        workflow_id=row[10],
        parent_job_id=row[11]
    )


def list_jobs(
    status: Optional[JobStatus] = None,
    template: Optional[str] = None,
    workflow_id: Optional[str] = None,
    limit: int = 100,
    db_path: Optional[Path] = None
) -> List[Job]:
    """
    List jobs with optional filters.
    
    Args:
        status: Filter by status
        template: Filter by template name
        workflow_id: Filter by workflow
        limit: Maximum jobs to return
        db_path: Optional database path
        
    Returns:
        List of Job objects
    """
    db_path = db_path or get_db_path()
    init_db(db_path)
    
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    query = """
        SELECT id, template, payload, route, status, result, error,
               retries, created, updated, workflow_id, parent_job_id
        FROM jobs WHERE 1=1
    """
    params = []
    
    if status:
        query += " AND status = ?"
        params.append(status.value)
    
    if template:
        query += " AND template = ?"
        params.append(template)
    
    if workflow_id:
        query += " AND workflow_id = ?"
        params.append(workflow_id)
    
    query += " ORDER BY created DESC LIMIT ?"
    params.append(limit)
    
    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()
    
    return [
        Job(
            id=row[0],
            template=row[1],
            payload=json.loads(row[2]),
            route=row[3],
            status=JobStatus(row[4]),
            result=json.loads(row[5]) if row[5] else None,
            error=row[6],
            retries=row[7],
            created=row[8],
            updated=row[9],
            workflow_id=row[10],
            parent_job_id=row[11]
        )
        for row in rows
    ]


def get_queue_stats(db_path: Optional[Path] = None) -> Dict[str, int]:
    """
    Get queue statistics.
    
    Args:
        db_path: Optional database path
        
    Returns:
        Dict with counts by status
    """
    db_path = db_path or get_db_path()
    init_db(db_path)
    
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT status, COUNT(*) FROM jobs GROUP BY status
    """)
    
    stats = {s.value: 0 for s in JobStatus}
    for row in cursor.fetchall():
        stats[row[0]] = row[1]
    
    conn.close()
    return stats


def clear_completed(
    older_than_hours: int = 24,
    db_path: Optional[Path] = None
) -> int:
    """
    Delete completed jobs older than specified hours.
    
    Args:
        older_than_hours: Delete jobs older than this
        db_path: Optional database path
        
    Returns:
        Number of jobs deleted
    """
    db_path = db_path or get_db_path()
    
    cutoff = time.time() - (older_than_hours * 3600)
    
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    cursor.execute("""
        DELETE FROM jobs
        WHERE status IN (?, ?) AND updated < ?
    """, (JobStatus.DONE.value, JobStatus.CANCELLED.value, cutoff))
    
    deleted = cursor.rowcount
    conn.commit()
    conn.close()
    
    return deleted


# Export public API
__all__ = [
    "JobStatus",
    "Job",
    "init_db",
    "enqueue",
    "enqueue_batch",
    "next_job",
    "complete",
    "fail",
    "cancel",
    "get_job",
    "list_jobs",
    "get_queue_stats",
    "clear_completed",
]
