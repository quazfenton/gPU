"""
Workflow service for GUI interface.

This module provides business logic for workflow validation, execution, and monitoring
through the GUI interface.
"""

import json
from typing import Any, Dict, Optional, Tuple
from datetime import datetime

from notebook_ml_orchestrator.core.interfaces import WorkflowEngineInterface
from notebook_ml_orchestrator.core.models import WorkflowDefinition, WorkflowStatus
from notebook_ml_orchestrator.core.logging_config import LoggerMixin
from notebook_ml_orchestrator.core.exceptions import WorkflowValidationError, WorkflowExecutionError


class WorkflowService(LoggerMixin):
    """Service for workflow management and execution."""
    
    def __init__(self, workflow_engine: WorkflowEngineInterface):
        """
        Initialize workflow service.
        
        Args:
            workflow_engine: Workflow engine instance
        """
        self.workflow_engine = workflow_engine
        self.logger.info("WorkflowService initialized")
    
    def validate_workflow(self, workflow_json: str) -> Tuple[bool, str]:
        """
        Validate workflow structure and type compatibility.
        
        This method performs comprehensive validation including:
        - JSON parsing validation
        - Required fields validation (name, steps)
        - Step structure validation
        - Type compatibility validation for connections
        - Circular dependency detection
        - Missing input validation
        
        Args:
            workflow_json: JSON string representing the workflow
            
        Returns:
            Tuple of (is_valid, error_message)
            - is_valid: True if workflow is valid, False otherwise
            - error_message: Empty string if valid, error description if invalid
        """
        try:
            # Parse JSON
            try:
                workflow_data = json.loads(workflow_json)
            except json.JSONDecodeError as e:
                return False, f"Invalid JSON format: {str(e)}"
            
            # Validate required fields
            if not isinstance(workflow_data, dict):
                return False, "Workflow must be a JSON object"
            
            if 'name' not in workflow_data:
                return False, "Workflow must have a 'name' field"
            
            if 'steps' not in workflow_data:
                return False, "Workflow must have a 'steps' field"
            
            steps = workflow_data.get('steps', [])
            if not isinstance(steps, list):
                return False, "'steps' must be a list"
            
            if len(steps) == 0:
                return False, "Workflow must have at least one step"
            
            # Validate each step
            step_ids = set()
            for i, step in enumerate(steps):
                if not isinstance(step, dict):
                    return False, f"Step {i} must be a JSON object"
                
                # Check required step fields
                if 'id' not in step:
                    return False, f"Step {i} is missing 'id' field"
                
                if 'template' not in step:
                    return False, f"Step {i} (id: {step.get('id')}) is missing 'template' field"
                
                step_id = step['id']
                
                # Check for duplicate step IDs
                if step_id in step_ids:
                    return False, f"Duplicate step ID: {step_id}"
                step_ids.add(step_id)
                
                # Validate inputs field
                if 'inputs' in step and not isinstance(step['inputs'], dict):
                    return False, f"Step {step_id}: 'inputs' must be a dictionary"
                
                # Validate outputs field
                if 'outputs' in step and not isinstance(step['outputs'], list):
                    return False, f"Step {step_id}: 'outputs' must be a list"
            
            # Validate connections if present
            connections = workflow_data.get('connections', [])
            if not isinstance(connections, list):
                return False, "'connections' must be a list"
            
            for i, conn in enumerate(connections):
                if not isinstance(conn, dict):
                    return False, f"Connection {i} must be a JSON object"
                
                # Check required connection fields
                required_fields = ['from', 'to', 'output', 'input']
                for field in required_fields:
                    if field not in conn:
                        return False, f"Connection {i} is missing '{field}' field"
                
                # Validate that referenced steps exist
                from_step = conn['from']
                to_step = conn['to']
                
                if from_step not in step_ids:
                    return False, f"Connection {i} references non-existent step: {from_step}"
                
                if to_step not in step_ids:
                    return False, f"Connection {i} references non-existent step: {to_step}"
                
                # Check for self-connections
                if from_step == to_step:
                    return False, f"Connection {i}: Step cannot connect to itself ({from_step})"
            
            # Create WorkflowDefinition for deeper validation
            definition = WorkflowDefinition(
                steps=steps,
                connections=connections,
                conditions=workflow_data.get('conditions', []),
                metadata=workflow_data.get('metadata', {})
            )
            
            # Use workflow engine to validate DAG structure (circular dependencies)
            try:
                # Create a temporary workflow to validate
                workflow = self.workflow_engine.create_workflow(definition)
                self.logger.info(f"Workflow validation successful: {workflow_data.get('name')}")
                return True, ""
            except WorkflowValidationError as e:
                return False, f"Workflow validation failed: {str(e)}"
            
        except Exception as e:
            self.logger.error(f"Unexpected error during workflow validation: {str(e)}")
            return False, f"Validation error: {str(e)}"
    
    def execute_workflow(self, workflow_json: str) -> str:
        """
        Execute workflow and return workflow ID.
        
        This method:
        1. Validates the workflow structure
        2. Creates a workflow definition
        3. Submits it to the workflow engine for execution
        4. Returns the execution ID for tracking
        
        Args:
            workflow_json: JSON string representing the workflow
            
        Returns:
            Workflow execution ID
            
        Raises:
            ValueError: If workflow JSON is invalid
            WorkflowValidationError: If workflow validation fails
            WorkflowExecutionError: If workflow execution fails
        """
        # First validate the workflow
        is_valid, error_message = self.validate_workflow(workflow_json)
        if not is_valid:
            raise ValueError(f"Invalid workflow: {error_message}")
        
        try:
            # Parse workflow JSON
            workflow_data = json.loads(workflow_json)
            
            # Create WorkflowDefinition
            definition = WorkflowDefinition(
                steps=workflow_data.get('steps', []),
                connections=workflow_data.get('connections', []),
                conditions=workflow_data.get('conditions', []),
                metadata={
                    'name': workflow_data.get('name', 'Unnamed Workflow'),
                    'description': workflow_data.get('description', ''),
                    **workflow_data.get('metadata', {})
                }
            )
            
            # Create workflow in engine
            workflow = self.workflow_engine.create_workflow(definition)
            
            # Execute workflow with inputs
            inputs = workflow_data.get('inputs', {})
            execution = self.workflow_engine.execute_workflow(workflow.id, inputs)
            
            self.logger.info(
                f"Workflow executed: workflow_id={workflow.id}, "
                f"execution_id={execution.id}, name={workflow_data.get('name')}"
            )
            
            return execution.id
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format: {str(e)}")
        except WorkflowValidationError as e:
            raise WorkflowValidationError(f"Workflow validation failed: {str(e)}")
        except Exception as e:
            self.logger.error(f"Workflow execution failed: {str(e)}")
            raise WorkflowExecutionError(f"Workflow execution failed: {str(e)}")
    
    def get_workflow_status(self, workflow_id: str) -> Dict[str, Any]:
        """
        Retrieve workflow execution status.
        
        Args:
            workflow_id: Workflow execution ID
            
        Returns:
            Dictionary with workflow execution information including:
            - execution_id: Workflow execution identifier
            - workflow_id: Original workflow identifier
            - status: Current execution status
            - started_at: Execution start timestamp
            - completed_at: Execution completion timestamp (if completed)
            - current_step: Currently executing step (if running)
            - steps: List of step statuses
            - inputs: Workflow input parameters
            - outputs: Workflow outputs (if completed)
            - error: Error message (if failed)
            - duration: Execution duration in seconds (if completed)
            
        Raises:
            ValueError: If workflow execution not found
        """
        execution = self.workflow_engine.get_workflow_execution(workflow_id)
        
        if not execution:
            raise ValueError(f"Workflow execution {workflow_id} not found")
        
        # Calculate duration if workflow has completed
        duration = None
        if execution.started_at and execution.completed_at:
            duration = (execution.completed_at - execution.started_at).total_seconds()
        
        # Get step statuses
        step_statuses = []
        if hasattr(execution, 'step_results') and execution.step_results:
            for step_id, result in execution.step_results.items():
                step_statuses.append({
                    'step_id': step_id,
                    'status': result.get('status', 'unknown'),
                    'outputs': result.get('outputs', {}),
                    'error': result.get('error')
                })
        
        return {
            'execution_id': execution.id,
            'workflow_id': execution.workflow_id,
            'status': execution.status.value if isinstance(execution.status, WorkflowStatus) else str(execution.status),
            'started_at': execution.started_at.isoformat() if execution.started_at else None,
            'completed_at': execution.completed_at.isoformat() if execution.completed_at else None,
            'current_step': getattr(execution, 'current_step', None),
            'steps': step_statuses,
            'inputs': execution.inputs,
            'outputs': execution.outputs,
            'error': getattr(execution, 'error', None),
            'duration': duration
        }
