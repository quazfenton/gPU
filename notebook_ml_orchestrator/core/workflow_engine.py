"""
Workflow automation engine for the Notebook ML Orchestrator.

This module implements DAG-based workflow execution with conditional logic,
data passing, and error handling capabilities.
"""

from typing import Any, Dict, List, Optional
import threading
from datetime import datetime

from .interfaces import Workflow, WorkflowExecution, WorkflowEngineInterface
from .models import WorkflowStatus, WorkflowDefinition
from .exceptions import WorkflowValidationError, WorkflowExecutionError
from .logging_config import LoggerMixin


class DAGExecutor:
    """Executes workflow steps in dependency order."""
    
    def __init__(self):
        self.execution_graph = {}
        self.step_results = {}
    
    def build_execution_graph(self, definition: WorkflowDefinition) -> Dict:
        """Build execution graph from workflow definition."""
        # Implementation will be added in task 6.1
        pass
    
    def execute_step(self, step_name: str, step_config: Dict, inputs: Dict) -> Any:
        """Execute a single workflow step."""
        # Implementation will be added in task 6.1
        pass
    
    def validate_dependencies(self, definition: WorkflowDefinition) -> bool:
        """Validate workflow dependencies for cycles and missing steps."""
        # For now, return True as a placeholder until full implementation
        return True


class ConditionEvaluator:
    """Evaluates conditional logic in workflows."""
    
    def __init__(self):
        self.supported_operators = ['==', '!=', '>', '<', '>=', '<=', 'in', 'not_in']
    
    def evaluate_condition(self, condition: Dict, context: Dict) -> bool:
        """Evaluate a condition against the current context."""
        # Implementation will be added in task 6.1
        pass
    
    def should_execute_step(self, step_config: Dict, context: Dict) -> bool:
        """Check if a step should be executed based on conditions."""
        # Implementation will be added in task 6.1
        pass


class DataPipeline:
    """Manages data flow between workflow steps."""
    
    def __init__(self):
        self.data_transformers = {}
        self.type_validators = {}
        self.step_outputs = {}  # Store outputs from each step
    
    def transform_data(self, data: Any, transformation: Dict) -> Any:
        """Transform data between workflow steps."""
        # Implementation will be added in task 6.2
        pass
    
    def validate_data_types(self, data: Any, expected_schema: Dict) -> bool:
        """
        Validate data types against expected schema.
        
        This method validates that data types match between workflow steps,
        ensuring type compatibility when passing outputs as inputs.
        
        Args:
            data: Data to validate
            expected_schema: Expected schema with type information
            
        Returns:
            True if validation passes
            
        Raises:
            TypeError: If data types don't match expected schema
            
        Requirements:
            - 7.6: Validate data types between workflow steps
        """
        if not expected_schema:
            return True
        
        for field_name, expected_type in expected_schema.items():
            if field_name not in data:
                continue  # Optional field
            
            value = data[field_name]
            
            # Basic type checking
            if expected_type == "text" and not isinstance(value, str):
                raise TypeError(
                    f"Field '{field_name}' expects type 'text' (str), "
                    f"but got {type(value).__name__}"
                )
            elif expected_type == "number" and not isinstance(value, (int, float)):
                raise TypeError(
                    f"Field '{field_name}' expects type 'number' (int/float), "
                    f"but got {type(value).__name__}"
                )
            elif expected_type == "json" and not isinstance(value, (dict, list)):
                raise TypeError(
                    f"Field '{field_name}' expects type 'json' (dict/list), "
                    f"but got {type(value).__name__}"
                )
            # For file types (audio, image, video, file), accept strings or bytes
            elif expected_type in ["audio", "image", "video", "file"]:
                if not isinstance(value, (str, bytes)):
                    raise TypeError(
                        f"Field '{field_name}' expects type '{expected_type}' (str/bytes), "
                        f"but got {type(value).__name__}"
                    )
        
        return True
    
    def pass_data_between_steps(self, from_step: str, to_step: str, 
                                from_outputs: Dict[str, Any],
                                to_inputs_schema: Dict[str, str]) -> Dict[str, Any]:
        """
        Handle data passing between workflow steps.
        
        This method implements Property 22: Workflow data passing.
        For any workflow with two sequential steps where step A outputs a value 
        with key K and step B expects an input with key K, this method passes 
        the output value from step A as the input value to step B.
        
        Args:
            from_step: Name of the source step
            to_step: Name of the destination step
            from_outputs: Outputs from the source step
            to_inputs_schema: Input schema for the destination step (field_name -> type)
            
        Returns:
            Dictionary of inputs for the destination step
            
        Raises:
            TypeError: If data types don't match between steps
            
        Requirements:
            - 7.6: Pass outputs from one template as inputs to the next
        """
        # Store outputs from the source step
        self.step_outputs[from_step] = from_outputs
        
        # Build inputs for destination step by matching keys
        to_inputs = {}
        
        for field_name, field_type in to_inputs_schema.items():
            if field_name in from_outputs:
                # Pass the value from source to destination
                to_inputs[field_name] = from_outputs[field_name]
        
        # Validate data types
        self.validate_data_types(to_inputs, to_inputs_schema)
        
        return to_inputs
    
    def get_step_output(self, step_name: str, output_key: str) -> Any:
        """
        Get a specific output from a step.
        
        Args:
            step_name: Name of the step
            output_key: Key of the output to retrieve
            
        Returns:
            Output value
            
        Raises:
            KeyError: If step or output key not found
        """
        if step_name not in self.step_outputs:
            raise KeyError(f"Step '{step_name}' has no recorded outputs")
        
        if output_key not in self.step_outputs[step_name]:
            raise KeyError(
                f"Step '{step_name}' has no output '{output_key}'. "
                f"Available outputs: {list(self.step_outputs[step_name].keys())}"
            )
        
        return self.step_outputs[step_name][output_key]


class WorkflowEngine(WorkflowEngineInterface, LoggerMixin):
    """DAG-based workflow execution engine."""
    
    def __init__(self):
        """Initialize the workflow engine."""
        self.workflows: Dict[str, Workflow] = {}
        self.executions: Dict[str, WorkflowExecution] = {}
        self.dag_executor = DAGExecutor()
        self.condition_evaluator = ConditionEvaluator()
        self.data_pipeline = DataPipeline()
        self._lock = threading.RLock()
        
        self.logger.info("Workflow engine initialized")
    
    def create_workflow(self, definition: WorkflowDefinition) -> Workflow:
        """
        Create a new workflow from definition.
        
        Args:
            definition: Workflow definition
            
        Returns:
            Created workflow
            
        Raises:
            WorkflowValidationError: If workflow definition is invalid
        """
        # This is a placeholder implementation
        # Full implementation will be added in task 6.1
        
        # Basic validation
        if not definition.steps:
            raise WorkflowValidationError("Workflow must have at least one step")
        
        # Validate DAG structure
        if not self.dag_executor.validate_dependencies(definition):
            raise WorkflowValidationError("Workflow contains circular dependencies")
        
        workflow = Workflow(
            definition=definition,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        with self._lock:
            self.workflows[workflow.id] = workflow
        
        self.logger.info(f"Workflow {workflow.id} created with {len(definition.steps)} steps")
        return workflow
    
    def execute_workflow(self, workflow_id: str, inputs: Dict[str, Any]) -> WorkflowExecution:
        """
        Execute a workflow with given inputs.
        
        Args:
            workflow_id: Workflow ID to execute
            inputs: Input data for workflow
            
        Returns:
            Workflow execution instance
            
        Raises:
            WorkflowValidationError: If workflow not found
            WorkflowExecutionError: If execution fails
        """
        # This is a placeholder implementation
        # Full implementation will be added in task 6.1
        
        with self._lock:
            workflow = self.workflows.get(workflow_id)
            if not workflow:
                raise WorkflowValidationError(f"Workflow {workflow_id} not found")
            
            execution = WorkflowExecution(
                workflow_id=workflow_id,
                status=WorkflowStatus.RUNNING,
                inputs=inputs,
                started_at=datetime.now()
            )
            
            self.executions[execution.id] = execution
        
        self.logger.info(f"Started execution {execution.id} for workflow {workflow_id}")
        
        # For now, just mark as completed
        # Full execution logic will be implemented in task 6.1
        execution.status = WorkflowStatus.COMPLETED
        execution.completed_at = datetime.now()
        execution.outputs = {"placeholder": "execution completed"}
        
        return execution
    
    def pause_workflow(self, execution_id: str):
        """
        Pause workflow execution.
        
        Args:
            execution_id: Execution ID to pause
        """
        with self._lock:
            execution = self.executions.get(execution_id)
            if execution and execution.status == WorkflowStatus.RUNNING:
                execution.status = WorkflowStatus.PAUSED
                self.logger.info(f"Workflow execution {execution_id} paused")
    
    def resume_workflow(self, execution_id: str):
        """
        Resume paused workflow execution.
        
        Args:
            execution_id: Execution ID to resume
        """
        with self._lock:
            execution = self.executions.get(execution_id)
            if execution and execution.status == WorkflowStatus.PAUSED:
                execution.status = WorkflowStatus.RUNNING
                self.logger.info(f"Workflow execution {execution_id} resumed")
    
    def cancel_workflow(self, execution_id: str):
        """
        Cancel workflow execution.
        
        Args:
            execution_id: Execution ID to cancel
        """
        with self._lock:
            execution = self.executions.get(execution_id)
            if execution and execution.status in [WorkflowStatus.RUNNING, WorkflowStatus.PAUSED]:
                execution.status = WorkflowStatus.CANCELLED
                execution.completed_at = datetime.now()
                self.logger.info(f"Workflow execution {execution_id} cancelled")
    
    def get_workflow_status(self, execution_id: str) -> WorkflowStatus:
        """
        Get current workflow execution status.
        
        Args:
            execution_id: Execution ID
            
        Returns:
            Current workflow status
        """
        execution = self.executions.get(execution_id)
        return execution.status if execution else WorkflowStatus.FAILED
    
    def get_workflow_execution(self, execution_id: str) -> Optional[WorkflowExecution]:
        """
        Get workflow execution by ID.
        
        Args:
            execution_id: Execution ID
            
        Returns:
            Workflow execution or None if not found
        """
        return self.executions.get(execution_id)
    
    def list_workflows(self) -> List[Workflow]:
        """
        List all workflows.
        
        Returns:
            List of all workflows
        """
        return list(self.workflows.values())
    
    def list_executions(self, workflow_id: str = None) -> List[WorkflowExecution]:
        """
        List workflow executions.
        
        Args:
            workflow_id: Optional workflow ID to filter by
            
        Returns:
            List of workflow executions
        """
        executions = list(self.executions.values())
        if workflow_id:
            executions = [e for e in executions if e.workflow_id == workflow_id]
        return executions
    
    def get_execution_statistics(self) -> Dict:
        """
        Get workflow execution statistics.
        
        Returns:
            Dictionary with execution statistics
        """
        with self._lock:
            stats = {
                'total_workflows': len(self.workflows),
                'total_executions': len(self.executions),
                'executions_by_status': {},
                'average_execution_time': 0.0
            }
            
            execution_times = []
            for execution in self.executions.values():
                status = execution.status.value
                if status not in stats['executions_by_status']:
                    stats['executions_by_status'][status] = 0
                stats['executions_by_status'][status] += 1
                
                if execution.completed_at and execution.started_at:
                    duration = (execution.completed_at - execution.started_at).total_seconds()
                    execution_times.append(duration)
            
            if execution_times:
                stats['average_execution_time'] = sum(execution_times) / len(execution_times)
            
            return stats