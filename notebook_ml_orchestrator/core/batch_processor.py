"""
Batch processing system for the Notebook ML Orchestrator.

This module implements efficient batch job processing with parallel execution,
progress tracking, and error handling for multiple items.
"""

from typing import Any, Dict, List, Optional
import threading
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from .interfaces import BatchJob, BatchProcessorInterface, MLTemplate
from .models import BatchStatus, BatchProgress, BatchItem, JobStatus, JobResult
from .exceptions import BatchValidationError, BatchError
from .logging_config import LoggerMixin


class BatchOptimizer:
    """Optimizes batch execution across backends."""
    
    def __init__(self):
        self.optimization_strategies = {
            'round_robin': self._round_robin_strategy,
            'load_balanced': self._load_balanced_strategy,
            'cost_optimized': self._cost_optimized_strategy
        }
    
    def optimize_batch_distribution(self, batch: BatchJob, available_backends: List) -> Dict:
        """Optimize distribution of batch items across backends."""
        # Implementation will be added in task 7.1
        pass
    
    def _round_robin_strategy(self, items: List[BatchItem], backends: List) -> Dict:
        """Distribute items using round-robin strategy."""
        # Implementation will be added in task 7.1
        pass
    
    def _load_balanced_strategy(self, items: List[BatchItem], backends: List) -> Dict:
        """Distribute items based on backend load."""
        # Implementation will be added in task 7.1
        pass
    
    def _cost_optimized_strategy(self, items: List[BatchItem], backends: List) -> Dict:
        """Distribute items to minimize cost."""
        # Implementation will be added in task 7.1
        pass


class ParallelExecutor:
    """Handles parallel execution of batch items."""
    
    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
    
    def execute_batch_items(self, items: List[BatchItem], template: MLTemplate, backend) -> List[JobResult]:
        """Execute batch items in parallel."""
        # Implementation will be added in task 7.1
        pass
    
    def execute_single_item(self, item: BatchItem, template: MLTemplate, backend) -> JobResult:
        """Execute a single batch item."""
        # Implementation will be added in task 7.1
        pass


class ProgressTracker:
    """Tracks progress of batch execution."""
    
    def __init__(self):
        self.progress_callbacks = {}
        self._lock = threading.RLock()
    
    def update_item_progress(self, batch_id: str, item_id: str, status: JobStatus, result: JobResult = None):
        """Update progress for a specific batch item."""
        with self._lock:
            # Implementation will be added in task 7.1
            pass
    
    def get_batch_progress(self, batch_id: str) -> BatchProgress:
        """Get current progress for a batch."""
        # Implementation will be added in task 7.1
        pass
    
    def register_progress_callback(self, batch_id: str, callback):
        """Register a callback for progress updates."""
        with self._lock:
            self.progress_callbacks[batch_id] = callback


class BatchProcessor(BatchProcessorInterface, LoggerMixin):
    """Efficient batch processing system."""
    
    def __init__(self, max_parallel_items: int = 4):
        """
        Initialize batch processor.
        
        Args:
            max_parallel_items: Maximum number of items to process in parallel
        """
        self.batches: Dict[str, BatchJob] = {}
        self.batch_optimizer = BatchOptimizer()
        self.parallel_executor = ParallelExecutor(max_workers=max_parallel_items)
        self.progress_tracker = ProgressTracker()
        self._lock = threading.RLock()
        
        self.logger.info("Batch processor initialized")
    
    def submit_batch(self, template: MLTemplate, inputs: List[Dict[str, Any]]) -> BatchJob:
        """
        Submit a batch of jobs for processing.
        
        Args:
            template: ML template to use for processing
            inputs: List of input dictionaries for each item
            
        Returns:
            BatchJob instance
            
        Raises:
            BatchValidationError: If batch validation fails
        """
        if not inputs:
            raise BatchValidationError("Batch must contain at least one item")
        
        # Create batch items
        items = []
        for i, input_data in enumerate(inputs):
            # Validate each input
            if not template.validate_inputs(input_data):
                raise BatchValidationError(f"Invalid input for item {i}")
            
            item = BatchItem(
                inputs=input_data,
                status=JobStatus.QUEUED,
                created_at=datetime.now()
            )
            items.append(item)
        
        # Create batch job
        batch = BatchJob(
            template_name=template.name,
            items=items,
            status=BatchStatus.QUEUED,
            progress=BatchProgress(total_items=len(items)),
            created_at=datetime.now()
        )
        
        with self._lock:
            self.batches[batch.id] = batch
        
        self.logger.info(f"Batch {batch.id} submitted with {len(items)} items")
        return batch
    
    def execute_batch(self, batch_id: str, backend=None) -> BatchJob:
        """
        Execute a batch job.
        
        Args:
            batch_id: Batch ID to execute
            backend: Optional specific backend to use
            
        Returns:
            Updated batch job
            
        Raises:
            BatchValidationError: If batch not found
        """
        # This is a placeholder implementation
        # Full implementation will be added in task 7.1
        
        with self._lock:
            batch = self.batches.get(batch_id)
            if not batch:
                raise BatchValidationError(f"Batch {batch_id} not found")
            
            if batch.status != BatchStatus.QUEUED:
                    # Mark batch as running
                    batch.status = BatchStatus.RUNNING
                    batch.started_at = datetime.now()

                    # For now, just mark all items as completed
                    # Full execution logic will be implemented in task 7.1
                    all_items_successful = True # This flag will track if ALL items ultimately succeeded
                    for item in batch.items:
                        item.status = JobStatus.COMPLETED
                        item.started_at = datetime.now()
                        item.completed_at = datetime.now()
                        item.result = JobResult(
                            success=True, # Placeholder: always True for now
                            outputs={"placeholder": "item completed"},
                            execution_time_seconds=1.0
                        )
                        batch.progress.completed_items += 1
                        # In future implementation (task 7.1), this flag would be set to False
                        # if any item.result.success is False.

                    # Determine final batch status and completion timestamp based on item outcomes
                    if all_items_successful:
                        batch.status = BatchStatus.COMPLETED
                        batch.completed_at = datetime.now() # Batch truly completed successfully
                    else:
                        # If not all items were successful (e.g., some failed), the batch itself is considered FAILED.
                        # In this interpretation, 'completed_at' specifically signifies successful completion.
                        batch.status = BatchStatus.FAILED
                        batch.completed_at = None # Explicitly set to None for a batch that did not complete successfully

                 else:
                    # This 'else' block likely corresponds to an initial failure condition
                    # (e.g., batch not in QUEUED status as per the earlier validation).
                    # In such a case, the batch didn't even start processing items successfully.
                    batch.status = BatchStatus.FAILED
                    # batch.completed_at should remain None here, as it was never successfully executed
                    # beyond the initial validation stages.

                self.logger.info(f"Batch {batch_id} execution completed")
                return batch
        """
        Track progress of batch execution.
        
        Args:
            batch_id: Batch ID to track
            
        Returns:
            Current batch progress
        """
        batch = self.batches.get(batch_id)
        if not batch:
            return BatchProgress(total_items=0)
        
        # Update progress from items
        progress = BatchProgress(total_items=len(batch.items))
        for item in batch.items:
            if item.status == JobStatus.COMPLETED:
                progress.completed_items += 1
            elif item.status == JobStatus.FAILED:
                progress.failed_items += 1
            elif item.status == JobStatus.RUNNING:
                progress.running_items += 1
            elif item.status == JobStatus.QUEUED:
                progress.queued_items += 1
        
        batch.progress = progress
        return progress
    
    def cancel_batch(self, batch_id: str) -> bool:
        """
        Cancel a batch job.
        
        Args:
            batch_id: Batch ID to cancel
            
        Returns:
            True if successful, False otherwise
        """
        with self._lock:
            batch = self.batches.get(batch_id)
            if not batch:
                return False
            
            if batch.status in [BatchStatus.COMPLETED, BatchStatus.FAILED, BatchStatus.CANCELLED]:
                return False  # Cannot cancel finished batches
            
            batch.status = BatchStatus.CANCELLED
            batch.completed_at = datetime.now()
            
            # Cancel individual items
            for item in batch.items:
                if item.status in [JobStatus.QUEUED, JobStatus.RUNNING]:
                    item.status = JobStatus.CANCELLED
                    item.completed_at = datetime.now()
            
            self.logger.info(f"Batch {batch_id} cancelled")
            return True
    
    def get_batch(self, batch_id: str) -> Optional[BatchJob]:
        """
        Get batch by ID.
        
        Args:
            batch_id: Batch ID
            
        Returns:
            Batch job or None if not found
        """
        return self.batches.get(batch_id)
    
    def list_batches(self, user_id: str = None) -> List[BatchJob]:
        """
        List batch jobs.
        
        Args:
            user_id: Optional user ID to filter by
            
        Returns:
            List of batch jobs
        """
        batches = list(self.batches.values())
        if user_id:
            batches = [b for b in batches if b.user_id == user_id]
        return batches
    
    def handle_batch_failures(self, batch_id: str, failed_items: List[str]):
        """
        Handle individual item failures in batch.
        
        Args:
            batch_id: Batch ID
            failed_items: List of failed item IDs
        """
        batch = self.batches.get(batch_id)
        if not batch:
            return
        
        failed_count = 0
        for item in batch.items:
            if item.id in failed_items:
                item.status = JobStatus.FAILED
                item.completed_at = datetime.now()
                failed_count += 1
        
        # Update batch status based on failures
        total_items = len(batch.items)
        completed_items = sum(1 for item in batch.items if item.status == JobStatus.COMPLETED)
        
        if failed_count == total_items:
            batch.status = BatchStatus.FAILED
        elif failed_count > 0:
            batch.status = BatchStatus.PARTIALLY_FAILED
        elif completed_items == total_items:
            batch.status = BatchStatus.COMPLETED
        
        batch.completed_at = datetime.now()
        
        self.logger.warning(f"Batch {batch_id} had {failed_count} failed items")
    
    def get_batch_statistics(self) -> Dict:
        """
        Get batch processing statistics.
        
        Returns:
            Dictionary with batch statistics
        """
        with self._lock:
            stats = {
                'total_batches': len(self.batches),
                'batches_by_status': {},
                'total_items': 0,
                'items_by_status': {},
                'average_batch_size': 0.0
            }
            
            batch_sizes = []
            for batch in self.batches.values():
                status = batch.status.value
                if status not in stats['batches_by_status']:
                    stats['batches_by_status'][status] = 0
                stats['batches_by_status'][status] += 1
                
                batch_size = len(batch.items)
                batch_sizes.append(batch_size)
                stats['total_items'] += batch_size
                
                for item in batch.items:
                    item_status = item.status.value
                    if item_status not in stats['items_by_status']:
                        stats['items_by_status'][item_status] = 0
                    stats['items_by_status'][item_status] += 1
            
            if batch_sizes:
                stats['average_batch_size'] = sum(batch_sizes) / len(batch_sizes)
            
            return stats