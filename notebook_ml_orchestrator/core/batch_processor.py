"""
Batch processing system for the Notebook ML Orchestrator.

This module implements efficient batch job processing with parallel execution,
progress tracking, and error handling for multiple items.
"""

from typing import Any, Dict, List, Optional
import threading
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from .interfaces import BatchJob, BatchProcessorInterface, MLTemplate, Job
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
        if not available_backends:
            return {}
        return self._round_robin_strategy(batch.items, available_backends)
    
    def _round_robin_strategy(self, items: List[BatchItem], backends: List) -> Dict:
        """Distribute items using round-robin strategy."""
        distribution = {i: [] for i in range(len(backends))}
        for idx, item in enumerate(items):
            backend_idx = idx % len(backends)
            distribution[backend_idx].append(item)
        return distribution
    
    def _load_balanced_strategy(self, items: List[BatchItem], backends: List) -> Dict:
        """Distribute items based on backend load."""
        distribution = {i: [] for i in range(len(backends))}
        # Sort backends by queue length (least loaded first)
        backend_loads = [(i, b.get_queue_length() if hasattr(b, 'get_queue_length') else 0) for i, b in enumerate(backends)]
        backend_loads.sort(key=lambda x: x[1])
        
        for idx, item in enumerate(items):
            backend_idx = backend_loads[idx % len(backend_loads)][0]
            distribution[backend_idx].append(item)
        return distribution
    
    def _cost_optimized_strategy(self, items: List[BatchItem], backends: List) -> Dict:
        """Distribute items to minimize cost."""
        distribution = {i: [] for i in range(len(backends))}
        # Sort backends by cost (cheapest first)
        backend_costs = [(i, b.capabilities.cost_per_hour if hasattr(b, 'capabilities') else 0) for i, b in enumerate(backends)]
        backend_costs.sort(key=lambda x: x[1])
        
        # Assign all items to cheapest backend
        if backend_costs:
            cheapest_idx = backend_costs[0][0]
            distribution[cheapest_idx] = list(items)
        return distribution


class ParallelExecutor:
    """Handles parallel execution of batch items."""
    
    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
    
    def execute_batch_items(self, items: List[BatchItem], template: MLTemplate, backend) -> List[JobResult]:
        """Execute batch items in parallel."""
        results = []
        futures = {}
        
        for item in items:
            future = self.executor.submit(self.execute_single_item, item, template, backend)
            futures[future] = item
        
        for future in as_completed(futures):
            item = futures[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                results.append(JobResult(
                    success=False,
                    error_message=str(e),
                    execution_time_seconds=0.0,
                ))
        
        return results
    
    def execute_single_item(self, item: BatchItem, template: MLTemplate, backend) -> JobResult:
        """Execute a single batch item."""
        import time
        start_time = time.time()
        
        item.status = JobStatus.RUNNING
        item.started_at = datetime.now()
        
        try:
            temp_job = Job(
                template_name=template.name if hasattr(template, 'name') else '',
                inputs=item.inputs,
                status=JobStatus.RUNNING,
            )
            
            result = backend.execute_job(temp_job, template)
            
            item.status = JobStatus.COMPLETED
            item.completed_at = datetime.now()
            item.result = result
            
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            item.status = JobStatus.FAILED
            item.completed_at = datetime.now()
            result = JobResult(
                success=False,
                error_message=str(e),
                execution_time_seconds=execution_time,
            )
            item.result = result
            return result


class ProgressTracker:
    """Tracks progress of batch execution."""
    
    def __init__(self):
        self.progress_callbacks = {}
        self._lock = threading.RLock()
    
    def update_item_progress(self, batch_id: str, item_id: str, status: JobStatus, result: JobResult = None):
        """Update progress for a specific batch item."""
        with self._lock:
            # Call registered callback if present
            callback = self.progress_callbacks.get(batch_id)
            if callback:
                try:
                    callback(batch_id, item_id, status, result)
                except Exception:
                    pass
    
    def get_batch_progress(self, batch_id: str) -> BatchProgress:
        """Get current progress for a batch."""
        # This returns a default; actual progress is tracked in BatchProcessor
        return BatchProgress(total_items=0)
    
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
        with self._lock:
            batch = self.batches.get(batch_id)
            if not batch:
                raise BatchValidationError(f"Batch {batch_id} not found")
            
            if batch.status != BatchStatus.QUEUED:
                raise BatchValidationError(f"Batch {batch_id} is not in queued status")
            
            # Mark batch as running
            batch.status = BatchStatus.RUNNING
            batch.started_at = datetime.now()
        
        self.logger.info(f"Starting batch execution: {batch_id} ({len(batch.items)} items)")
        
        try:
            if backend:
                for item in batch.items:
                    import time
                    start_time = time.time()
                    item.status = JobStatus.RUNNING
                    item.started_at = datetime.now()
                    
                    try:
                        temp_job = Job(
                            template_name=batch.template_name,
                            inputs=item.inputs,
                            status=JobStatus.RUNNING,
                        )
                        result = backend.execute_job(temp_job, None)
                        item.status = JobStatus.COMPLETED
                        item.completed_at = datetime.now()
                        item.result = result
                        batch.progress.completed_items += 1
                    except Exception as e:
                        item.status = JobStatus.FAILED
                        item.completed_at = datetime.now()
                        item.result = JobResult(
                            success=False,
                            error_message=str(e),
                            execution_time_seconds=time.time() - start_time,
                        )
                        batch.progress.failed_items += 1
            else:
                # No backend - process items with pass-through results
                for item in batch.items:
                    item.status = JobStatus.COMPLETED
                    item.started_at = datetime.now()
                    item.completed_at = datetime.now()
                    item.result = JobResult(
                        success=True,
                        outputs=item.inputs,
                        execution_time_seconds=0.0
                    )
                    batch.progress.completed_items += 1
            
            # Determine final batch status
            failed_count = sum(1 for item in batch.items if item.status == JobStatus.FAILED)
            completed_count = sum(1 for item in batch.items if item.status == JobStatus.COMPLETED)
            
            if failed_count == len(batch.items):
                batch.status = BatchStatus.FAILED
            elif failed_count > 0:
                batch.status = BatchStatus.PARTIALLY_FAILED
            else:
                batch.status = BatchStatus.COMPLETED
            
            batch.completed_at = datetime.now()
            
        except Exception as e:
            batch.status = BatchStatus.FAILED
            batch.completed_at = datetime.now()
            self.logger.error(f"Batch {batch_id} execution failed: {e}")
        
        self.logger.info(f"Batch {batch_id} execution completed: {batch.status.value}")
        return batch
    
    def track_batch_progress(self, batch_id: str) -> BatchProgress:
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