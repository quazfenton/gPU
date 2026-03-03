"""
MLflow Integration for experiment tracking.

This module provides integration with MLflow for:
- Experiment tracking
- Metric logging
- Parameter logging
- Artifact storage
- Model registry integration

Supports both local MLflow server and remote tracking server.
"""

import json
import logging
import os
import tempfile
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class MLflowConfig:
    """MLflow configuration."""
    tracking_uri: Optional[str] = None  # If None, uses local ./mlruns
    experiment_name: str = "notebook-ml-orchestrator"
    registry_uri: Optional[str] = None
    artifact_location: Optional[str] = None
    create_experiment_if_not_exists: bool = True


class MLflowTracker:
    """
    MLflow experiment tracker.
    
    Integrates with the job queue to automatically track:
    - Job parameters
    - Execution metrics
    - Output artifacts
    - Model versions
    
    Example:
        tracker = MLflowTracker()
        
        # Start tracking a job
        with tracker.start_run(job_id="job-123", template="image-classification"):
            tracker.log_params({"model": "resnet50", "epochs": 10})
            tracker.log_metrics({"accuracy": 0.95, "loss": 0.05})
            tracker.log_artifact("model.pth", "models")
    """
    
    def __init__(self, config: Optional[MLflowConfig] = None):
        """
        Initialize MLflow tracker.
        
        Args:
            config: MLflow configuration
        """
        self.config = config or MLflowConfig()
        self._mlflow = None
        self._experiment_id = None
        self._active_run = None
        
        logger.info(f"MLflowTracker initialized (tracking_uri: {self.config.tracking_uri or 'local'})")
    
    def _import_mlflow(self) -> None:
        """Import MLflow library."""
        try:
            import mlflow
            self._mlflow = mlflow
            
            # Set tracking URI
            if self.config.tracking_uri:
                self._mlflow.set_tracking_uri(self.config.tracking_uri)
                logger.info(f"MLflow tracking URI set to: {self.config.tracking_uri}")
            
            # Set registry URI
            if self.config.registry_uri:
                self._mlflow.set_registry_uri(self.config.registry_uri)
                logger.info(f"MLflow registry URI set to: {self.config.registry_uri}")
            
            # Get or create experiment
            self._setup_experiment()
            
        except ImportError:
            logger.warning("MLflow not installed. Install with: pip install mlflow")
            raise
    
    def _setup_experiment(self) -> None:
        """Set up MLflow experiment."""
        if not self._mlflow:
            return
        
        experiment = self._mlflow.get_experiment_by_name(self.config.experiment_name)
        
        if experiment is None:
            if self.config.create_experiment_if_not_exists:
                experiment_id = self._mlflow.create_experiment(
                    self.config.experiment_name,
                    artifact_location=self.config.artifact_location
                )
                logger.info(f"Created MLflow experiment: {self.config.experiment_name}")
                self._experiment_id = experiment_id
            else:
                logger.warning(f"Experiment {self.config.experiment_name} does not exist")
        else:
            self._experiment_id = experiment.experiment_id
            logger.info(f"Using existing MLflow experiment: {self.config.experiment_name}")
    
    def start_run(
        self,
        job_id: str,
        template: str,
        user_id: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None
    ) -> 'MLflowRun':
        """
        Start a new MLflow run for a job.
        
        Args:
            job_id: Job ID
            template: Template name
            user_id: Optional user ID
            tags: Optional additional tags
            
        Returns:
            MLflowRun context manager
        """
        if self._mlflow is None:
            self._import_mlflow()
        
        return MLflowRun(
            tracker=self,
            job_id=job_id,
            template=template,
            user_id=user_id,
            tags=tags
        )
    
    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        """
        Get run information.
        
        Args:
            run_id: Run ID
            
        Returns:
            Run information dictionary
        """
        if self._mlflow is None:
            return None
        
        try:
            run = self._mlflow.get_run(run_id)
            return {
                'run_id': run.info.run_id,
                'experiment_id': run.info.experiment_id,
                'status': run.info.status,
                'start_time': datetime.fromtimestamp(run.info.start_time / 1000),
                'end_time': datetime.fromtimestamp(run.info.end_time / 1000) if run.info.end_time else None,
                'artifact_uri': run.info.artifact_uri,
                'tags': dict(run.data.tags),
                'params': dict(run.data.params),
                'metrics': dict(run.data.metrics)
            }
        except Exception as e:
            logger.error(f"Failed to get run {run_id}: {e}")
            return None
    
    def search_runs(
        self,
        filter_string: Optional[str] = None,
        max_results: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Search runs.
        
        Args:
            filter_string: MLflow filter string
            max_results: Maximum results to return
            
        Returns:
            List of run information dictionaries
        """
        if self._mlflow is None:
            return []
        
        try:
            runs = self._mlflow.search_runs(
                experiment_names=[self.config.experiment_name],
                filter_string=filter_string,
                max_results=max_results
            )
            
            return runs.to_dict('records') if len(runs) > 0 else []
            
        except Exception as e:
            logger.error(f"Failed to search runs: {e}")
            return []
    
    def log_job_completion(
        self,
        job_id: str,
        run_id: str,
        result: Dict[str, Any],
        execution_time: float
    ) -> None:
        """
        Log job completion.
        
        Args:
            job_id: Job ID
            run_id: Run ID
            result: Job result
            execution_time: Execution time in seconds
        """
        if self._mlflow is None:
            return
        
        try:
            with self._mlflow.start_run(run_id=run_id):
                # Log metrics
                if 'metrics' in result:
                    for key, value in result['metrics'].items():
                        if isinstance(value, (int, float)):
                            self._mlflow.log_metric(key, value)
                
                # Log execution time
                self._mlflow.log_metric('execution_time_seconds', execution_time)
                
                # Log status
                self._mlflow.log_param('status', 'completed')
                
                # Log artifacts
                if 'artifacts' in result:
                    for artifact_path, artifact_data in result['artifacts'].items():
                        self._log_artifact_data(artifact_path, artifact_data)
                
        except Exception as e:
            logger.error(f"Failed to log job completion: {e}")
    
    def _log_artifact_data(self, path: str, data: Any) -> None:
        """Log artifact data."""
        if not self._mlflow:
            return
        
        try:
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
                json.dump(data, f)
                temp_path = f.name
            
            self._mlflow.log_artifact(temp_path, os.path.dirname(path))
            os.unlink(temp_path)
            
        except Exception as e:
            logger.error(f"Failed to log artifact {path}: {e}")


class MLflowRun:
    """
    MLflow run context manager.
    
    Manages the lifecycle of an MLflow run for a job.
    """
    
    def __init__(
        self,
        tracker: MLflowTracker,
        job_id: str,
        template: str,
        user_id: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None
    ):
        """
        Initialize MLflow run.
        
        Args:
            tracker: Parent tracker
            job_id: Job ID
            template: Template name
            user_id: Optional user ID
            tags: Optional additional tags
        """
        self.tracker = tracker
        self.job_id = job_id
        self.template = template
        self.user_id = user_id
        self.tags = tags or {}
        self.run_id = None
        self._run = None
    
    def __enter__(self):
        """Enter run context."""
        if self.tracker._mlflow is None:
            return self
        
        # Prepare tags
        tags = {
            'job_id': self.job_id,
            'template': self.template,
            'user_id': self.user_id or 'anonymous',
            'orchestrator': 'notebook-ml-orchestrator',
            **self.tags
        }
        
        # Start run
        self._run = self.tracker._mlflow.start_run(
            experiment_id=self.tracker._experiment_id,
            tags=tags
        )
        self.run_id = self._run.info.run_id
        
        logger.info(f"Started MLflow run {self.run_id} for job {self.job_id}")
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit run context."""
        if self.tracker._mlflow is None or self._run is None:
            return
        
        try:
            if exc_type is not None:
                # Run failed
                self.tracker._mlflow.end_run(status='FAILED')
                logger.warning(f"MLflow run {self.run_id} failed: {exc_val}")
            else:
                # Run completed
                self.tracker._mlflow.end_run(status='FINISHED')
                logger.info(f"MLflow run {self.run_id} completed successfully")
        except Exception as e:
            logger.error(f"Failed to end MLflow run: {e}")
    
    def log_params(self, params: Dict[str, Any]) -> None:
        """
        Log parameters.
        
        Args:
            params: Parameter dictionary
        """
        if self.tracker._mlflow is None or self._run is None:
            return
        
        for key, value in params.items():
            try:
                self.tracker._mlflow.log_param(self.run_id, key, str(value))
            except Exception as e:
                logger.warning(f"Failed to log param {key}: {e}")
    
    def log_metrics(self, metrics: Dict[str, Union[int, float]]) -> None:
        """
        Log metrics.
        
        Args:
            metrics: Metric dictionary
        """
        if self.tracker._mlflow is None or self._run is None:
            return
        
        for key, value in metrics.items():
            try:
                self.tracker._mlflow.log_metric(self.run_id, key, value)
            except Exception as e:
                logger.warning(f"Failed to log metric {key}: {e}")
    
    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None) -> None:
        """
        Log artifact file.
        
        Args:
            local_path: Local file path
            artifact_path: Optional artifact path in MLflow
        """
        if self.tracker._mlflow is None or self._run is None:
            return
        
        try:
            self.tracker._mlflow.log_artifact(local_path, artifact_path)
            logger.info(f"Logged artifact {local_path}")
        except Exception as e:
            logger.error(f"Failed to log artifact {local_path}: {e}")
    
    def log_artifacts(self, local_dir: str, artifact_path: Optional[str] = None) -> None:
        """
        Log artifact directory.
        
        Args:
            local_dir: Local directory path
            artifact_path: Optional artifact path in MLflow
        """
        if self.tracker._mlflow is None or self._run is None:
            return
        
        try:
            self.tracker._mlflow.log_artifacts(local_dir, artifact_path)
            logger.info(f"Logged artifacts from {local_dir}")
        except Exception as e:
            logger.error(f"Failed to log artifacts from {local_dir}: {e}")


# Module-level instance for easy integration
_tracker: Optional[MLflowTracker] = None


def get_mlflow_tracker(config: Optional[MLflowConfig] = None) -> MLflowTracker:
    """Get or create module-level MLflow tracker."""
    global _tracker
    if _tracker is None:
        _tracker = MLflowTracker(config)
    return _tracker


def track_job(job_id: str, template: str, user_id: Optional[str] = None):
    """
    Context manager for tracking a job with MLflow.
    
    Usage:
        with track_job("job-123", "image-classification"):
            # Run job
            pass
    """
    tracker = get_mlflow_tracker()
    return tracker.start_run(job_id, template, user_id)
