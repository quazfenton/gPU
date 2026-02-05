"""
Command-line interface for the Notebook ML Orchestrator.

This module provides a simple CLI for interacting with the orchestration system.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any

from .core.job_queue import JobQueueManager
from .core.backend_router import MultiBackendRouter
from .core.workflow_engine import WorkflowEngine
from .core.batch_processor import BatchProcessor
from .core.interfaces import Job
from .core.models import JobStatus
from .core.logging_config import configure_default_logging
from .config import get_config


def setup_logging():
    """Set up logging for CLI."""
    configure_default_logging()


def create_job_command(args):
    """Create and submit a new job."""
    config = get_config()
    job_queue = JobQueueManager(config.database.path)
    
    job = Job(
        id=args.job_id,
        user_id=args.user_id,
        template_name=args.template,
        inputs=json.loads(args.inputs) if args.inputs else {},
        priority=args.priority
    )
    
    try:
        job_id = job_queue.submit_job(job)
        print(f"Job {job_id} submitted successfully")
        return 0
    except Exception as e:
        print(f"Error submitting job: {e}")
        return 1


def list_jobs_command(args):
    """List jobs in the queue."""
    config = get_config()
    job_queue = JobQueueManager(config.database.path)
    
    try:
        if args.user_id:
            jobs = job_queue.get_job_history(args.user_id, args.limit)
        else:
            # Get all jobs by status
            all_jobs = []
            for status in JobStatus:
                jobs_by_status = job_queue.db.get_jobs_by_status(status, args.limit)
                all_jobs.extend(jobs_by_status)
            jobs = all_jobs[:args.limit]
        
        if not jobs:
            print("No jobs found")
            return 0
        
        print(f"{'Job ID':<20} {'User ID':<15} {'Template':<20} {'Status':<12} {'Created':<20}")
        print("-" * 87)
        
        for job in jobs:
            created_str = job.created_at.strftime("%Y-%m-%d %H:%M:%S") if job.created_at else "N/A"
            print(f"{job.id:<20} {job.user_id:<15} {job.template_name:<20} {job.status.value:<12} {created_str:<20}")
        
        return 0
    except Exception as e:
        print(f"Error listing jobs: {e}")
        return 1


def job_status_command(args):
    """Get status of a specific job."""
    config = get_config()
    job_queue = JobQueueManager(config.database.path)
    
    try:
        job = job_queue.get_job(args.job_id)
        if not job:
            print(f"Job {args.job_id} not found")
            return 1
        
        print(f"Job ID: {job.id}")
        print(f"User ID: {job.user_id}")
        print(f"Template: {job.template_name}")
        print(f"Status: {job.status.value}")
        print(f"Priority: {job.priority}")
        print(f"Retry Count: {job.retry_count}")
        print(f"Created: {job.created_at}")
        print(f"Started: {job.started_at}")
        print(f"Completed: {job.completed_at}")
        
        if job.inputs:
            print(f"Inputs: {json.dumps(job.inputs, indent=2)}")
        
        if job.result:
            print(f"Result: Success={job.result.success}")
            if job.result.outputs:
                print(f"Outputs: {json.dumps(job.result.outputs, indent=2)}")
        
        if job.error:
            print(f"Error: {job.error}")
        
        return 0
    except Exception as e:
        print(f"Error getting job status: {e}")
        return 1


def queue_stats_command(args):
    """Show queue statistics."""
    config = get_config()
    job_queue = JobQueueManager(config.database.path)
    backend_router = MultiBackendRouter()
    
    try:
        queue_stats = job_queue.get_queue_statistics()
        router_stats = backend_router.get_routing_statistics()
        
        print("=== Queue Statistics ===")
        print(f"Total Jobs: {queue_stats['total_jobs']}")
        print(f"Queue Length: {queue_stats['queue_length']}")
        print(f"Running Jobs: {queue_stats['running_jobs']}")
        
        print("\nJobs by Status:")
        for status, count in queue_stats['by_status'].items():
            print(f"  {status}: {count}")
        
        print("\n=== Backend Statistics ===")
        print(f"Total Backends: {router_stats['total_backends']}")
        print(f"Healthy Backends: {router_stats['healthy_backends']}")
        
        if router_stats['backends_by_type']:
            print("\nBackends by Type:")
            for backend_type, count in router_stats['backends_by_type'].items():
                print(f"  {backend_type}: {count}")
        
        return 0
    except Exception as e:
        print(f"Error getting statistics: {e}")
        return 1


def cleanup_command(args):
    """Clean up old jobs."""
    config = get_config()
    job_queue = JobQueueManager(config.database.path)
    
    try:
        deleted_count = job_queue.cleanup_old_jobs(args.days)
        print(f"Cleaned up {deleted_count} old jobs (older than {args.days} days)")
        return 0
    except Exception as e:
        print(f"Error cleaning up jobs: {e}")
        return 1


def main():
    """Main CLI entry point."""
    setup_logging()
    
    parser = argparse.ArgumentParser(
        description="Notebook ML Orchestrator CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Create job command
    create_parser = subparsers.add_parser('create', help='Create and submit a new job')
    create_parser.add_argument('--job-id', required=True, help='Unique job ID')
    create_parser.add_argument('--user-id', required=True, help='User ID')
    create_parser.add_argument('--template', required=True, help='Template name')
    create_parser.add_argument('--inputs', help='Job inputs as JSON string')
    create_parser.add_argument('--priority', type=int, default=0, help='Job priority (default: 0)')
    create_parser.set_defaults(func=create_job_command)
    
    # List jobs command
    list_parser = subparsers.add_parser('list', help='List jobs')
    list_parser.add_argument('--user-id', help='Filter by user ID')
    list_parser.add_argument('--limit', type=int, default=20, help='Maximum number of jobs to show')
    list_parser.set_defaults(func=list_jobs_command)
    
    # Job status command
    status_parser = subparsers.add_parser('status', help='Get job status')
    status_parser.add_argument('job_id', help='Job ID to check')
    status_parser.set_defaults(func=job_status_command)
    
    # Queue statistics command
    stats_parser = subparsers.add_parser('stats', help='Show queue and backend statistics')
    stats_parser.set_defaults(func=queue_stats_command)
    
    # Cleanup command
    cleanup_parser = subparsers.add_parser('cleanup', help='Clean up old jobs')
    cleanup_parser.add_argument('--days', type=int, default=30, help='Delete jobs older than N days')
    cleanup_parser.set_defaults(func=cleanup_command)
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Execute command
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())