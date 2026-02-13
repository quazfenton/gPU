"""Tests for WorkflowService."""

import pytest
import json
from datetime import datetime
from unittest.mock import Mock, MagicMock

from gui.services.workflow_service import WorkflowService
from notebook_ml_orchestrator.core.models import WorkflowDefinition, WorkflowStatus
from notebook_ml_orchestrator.core.exceptions import WorkflowValidationError, WorkflowExecutionError


class MockWorkflow:
    """Mock workflow object."""
    def __init__(self, workflow_id="workflow-123"):
        self.id = workflow_id
        self.definition = None
        self.created_at = datetime.now()
        self.updated_at = datetime.now()


class MockWorkflowExecution:
    """Mock workflow execution object."""
    def __init__(self, execution_id="exec-123", workflow_id="workflow-123"):
        self.id = execution_id
        self.workflow_id = workflow_id
        self.status = WorkflowStatus.RUNNING
        self.inputs = {}
        self.outputs = None
        self.started_at = datetime.now()
        self.completed_at = None
        self.step_results = {}


class TestWorkflowService:
    """Test suite for WorkflowService."""
    
    @pytest.fixture
    def mock_workflow_engine(self):
        """Create mock workflow engine."""
        return Mock()
    
    @pytest.fixture
    def workflow_service(self, mock_workflow_engine):
        """Create WorkflowService instance with mock."""
        return WorkflowService(mock_workflow_engine)
    
    def test_validate_workflow_success(self, workflow_service, mock_workflow_engine):
        """Test successful workflow validation."""
        # Setup
        workflow_json = json.dumps({
            "name": "Test Workflow",
            "steps": [
                {
                    "id": "step1",
                    "name": "Load Data",
                    "template": "data_loader",
                    "inputs": {"path": "/data/input.csv"},
                    "outputs": ["data"]
                },
                {
                    "id": "step2",
                    "name": "Process Data",
                    "template": "data_processor",
                    "inputs": {"data": "${step1.data}"},
                    "outputs": ["result"]
                }
            ],
            "connections": [
                {"from": "step1", "to": "step2", "output": "data", "input": "data"}
            ]
        })
        
        mock_workflow_engine.create_workflow.return_value = MockWorkflow()
        
        # Execute
        is_valid, error_message = workflow_service.validate_workflow(workflow_json)
        
        # Verify
        assert is_valid is True
        assert error_message == ""
        mock_workflow_engine.create_workflow.assert_called_once()
    
    def test_validate_workflow_invalid_json(self, workflow_service):
        """Test validation with invalid JSON."""
        # Setup
        workflow_json = "{ invalid json }"
        
        # Execute
        is_valid, error_message = workflow_service.validate_workflow(workflow_json)
        
        # Verify
        assert is_valid is False
        assert "Invalid JSON format" in error_message
    
    def test_validate_workflow_missing_name(self, workflow_service):
        """Test validation with missing name field."""
        # Setup
        workflow_json = json.dumps({
            "steps": [
                {"id": "step1", "template": "test_template"}
            ]
        })
        
        # Execute
        is_valid, error_message = workflow_service.validate_workflow(workflow_json)
        
        # Verify
        assert is_valid is False
        assert "must have a 'name' field" in error_message
    
    def test_validate_workflow_missing_steps(self, workflow_service):
        """Test validation with missing steps field."""
        # Setup
        workflow_json = json.dumps({
            "name": "Test Workflow"
        })
        
        # Execute
        is_valid, error_message = workflow_service.validate_workflow(workflow_json)
        
        # Verify
        assert is_valid is False
        assert "must have a 'steps' field" in error_message
    
    def test_validate_workflow_empty_steps(self, workflow_service):
        """Test validation with empty steps list."""
        # Setup
        workflow_json = json.dumps({
            "name": "Test Workflow",
            "steps": []
        })
        
        # Execute
        is_valid, error_message = workflow_service.validate_workflow(workflow_json)
        
        # Verify
        assert is_valid is False
        assert "must have at least one step" in error_message
    
    def test_validate_workflow_step_missing_id(self, workflow_service):
        """Test validation with step missing id field."""
        # Setup
        workflow_json = json.dumps({
            "name": "Test Workflow",
            "steps": [
                {"template": "test_template"}
            ]
        })
        
        # Execute
        is_valid, error_message = workflow_service.validate_workflow(workflow_json)
        
        # Verify
        assert is_valid is False
        assert "missing 'id' field" in error_message
    
    def test_validate_workflow_step_missing_template(self, workflow_service):
        """Test validation with step missing template field."""
        # Setup
        workflow_json = json.dumps({
            "name": "Test Workflow",
            "steps": [
                {"id": "step1"}
            ]
        })
        
        # Execute
        is_valid, error_message = workflow_service.validate_workflow(workflow_json)
        
        # Verify
        assert is_valid is False
        assert "missing 'template' field" in error_message
    
    def test_validate_workflow_duplicate_step_ids(self, workflow_service):
        """Test validation with duplicate step IDs."""
        # Setup
        workflow_json = json.dumps({
            "name": "Test Workflow",
            "steps": [
                {"id": "step1", "template": "template1"},
                {"id": "step1", "template": "template2"}
            ]
        })
        
        # Execute
        is_valid, error_message = workflow_service.validate_workflow(workflow_json)
        
        # Verify
        assert is_valid is False
        assert "Duplicate step ID" in error_message
    
    def test_validate_workflow_connection_missing_fields(self, workflow_service):
        """Test validation with connection missing required fields."""
        # Setup
        workflow_json = json.dumps({
            "name": "Test Workflow",
            "steps": [
                {"id": "step1", "template": "template1"},
                {"id": "step2", "template": "template2"}
            ],
            "connections": [
                {"from": "step1", "to": "step2"}  # Missing output and input
            ]
        })
        
        # Execute
        is_valid, error_message = workflow_service.validate_workflow(workflow_json)
        
        # Verify
        assert is_valid is False
        assert "missing" in error_message.lower()
    
    def test_validate_workflow_connection_nonexistent_step(self, workflow_service):
        """Test validation with connection referencing non-existent step."""
        # Setup
        workflow_json = json.dumps({
            "name": "Test Workflow",
            "steps": [
                {"id": "step1", "template": "template1"}
            ],
            "connections": [
                {"from": "step1", "to": "step2", "output": "data", "input": "data"}
            ]
        })
        
        # Execute
        is_valid, error_message = workflow_service.validate_workflow(workflow_json)
        
        # Verify
        assert is_valid is False
        assert "non-existent step" in error_message
    
    def test_validate_workflow_self_connection(self, workflow_service):
        """Test validation with step connecting to itself."""
        # Setup
        workflow_json = json.dumps({
            "name": "Test Workflow",
            "steps": [
                {"id": "step1", "template": "template1"}
            ],
            "connections": [
                {"from": "step1", "to": "step1", "output": "data", "input": "data"}
            ]
        })
        
        # Execute
        is_valid, error_message = workflow_service.validate_workflow(workflow_json)
        
        # Verify
        assert is_valid is False
        assert "cannot connect to itself" in error_message
    
    def test_validate_workflow_circular_dependency(self, workflow_service, mock_workflow_engine):
        """Test validation with circular dependencies."""
        # Setup
        workflow_json = json.dumps({
            "name": "Test Workflow",
            "steps": [
                {"id": "step1", "template": "template1"},
                {"id": "step2", "template": "template2"}
            ],
            "connections": [
                {"from": "step1", "to": "step2", "output": "data", "input": "data"}
            ]
        })
        
        # Mock workflow engine to raise validation error for circular dependency
        mock_workflow_engine.create_workflow.side_effect = WorkflowValidationError(
            "Workflow contains circular dependencies"
        )
        
        # Execute
        is_valid, error_message = workflow_service.validate_workflow(workflow_json)
        
        # Verify
        assert is_valid is False
        assert "circular dependencies" in error_message.lower()
    
    def test_execute_workflow_success(self, workflow_service, mock_workflow_engine):
        """Test successful workflow execution."""
        # Setup
        workflow_json = json.dumps({
            "name": "Test Workflow",
            "description": "A test workflow",
            "steps": [
                {"id": "step1", "template": "template1", "inputs": {}, "outputs": ["data"]}
            ],
            "connections": [],
            "inputs": {"param1": "value1"}
        })
        
        mock_workflow = MockWorkflow("workflow-123")
        mock_execution = MockWorkflowExecution("exec-456", "workflow-123")
        
        mock_workflow_engine.create_workflow.return_value = mock_workflow
        mock_workflow_engine.execute_workflow.return_value = mock_execution
        
        # Execute
        execution_id = workflow_service.execute_workflow(workflow_json)
        
        # Verify
        assert execution_id == "exec-456"
        # create_workflow is called twice: once in validate_workflow, once in execute_workflow
        assert mock_workflow_engine.create_workflow.call_count == 2
        mock_workflow_engine.execute_workflow.assert_called_once_with(
            "workflow-123",
            {"param1": "value1"}
        )
    
    def test_execute_workflow_invalid_json(self, workflow_service):
        """Test workflow execution with invalid JSON."""
        # Setup
        workflow_json = "{ invalid json }"
        
        # Execute & Verify
        with pytest.raises(ValueError, match="Invalid workflow"):
            workflow_service.execute_workflow(workflow_json)
    
    def test_execute_workflow_validation_fails(self, workflow_service, mock_workflow_engine):
        """Test workflow execution when validation fails."""
        # Setup
        workflow_json = json.dumps({
            "name": "Test Workflow",
            "steps": []  # Empty steps will fail validation
        })
        
        # Execute & Verify
        with pytest.raises(ValueError, match="Invalid workflow"):
            workflow_service.execute_workflow(workflow_json)
    
    def test_execute_workflow_engine_error(self, workflow_service, mock_workflow_engine):
        """Test workflow execution when engine raises error."""
        # Setup
        workflow_json = json.dumps({
            "name": "Test Workflow",
            "steps": [
                {"id": "step1", "template": "template1"}
            ]
        })
        
        mock_workflow = MockWorkflow()
        mock_workflow_engine.create_workflow.return_value = mock_workflow
        mock_workflow_engine.execute_workflow.side_effect = Exception("Engine error")
        
        # Execute & Verify
        with pytest.raises(WorkflowExecutionError, match="Workflow execution failed"):
            workflow_service.execute_workflow(workflow_json)
    
    def test_get_workflow_status_success(self, workflow_service, mock_workflow_engine):
        """Test retrieving workflow execution status."""
        # Setup
        mock_execution = MockWorkflowExecution("exec-123", "workflow-456")
        mock_execution.status = WorkflowStatus.COMPLETED
        mock_execution.started_at = datetime(2024, 1, 1, 10, 0, 0)
        mock_execution.completed_at = datetime(2024, 1, 1, 10, 5, 0)
        mock_execution.inputs = {"param1": "value1"}
        mock_execution.outputs = {"result": "output_value"}
        mock_execution.step_results = {
            "step1": {
                "status": "completed",
                "outputs": {"data": "processed_data"}
            }
        }
        
        mock_workflow_engine.get_workflow_execution.return_value = mock_execution
        
        # Execute
        status = workflow_service.get_workflow_status("exec-123")
        
        # Verify
        assert status['execution_id'] == "exec-123"
        assert status['workflow_id'] == "workflow-456"
        assert status['status'] == "completed"
        assert status['duration'] == 300.0  # 5 minutes
        assert status['inputs'] == {"param1": "value1"}
        assert status['outputs'] == {"result": "output_value"}
        assert len(status['steps']) == 1
        assert status['steps'][0]['step_id'] == "step1"
        assert status['steps'][0]['status'] == "completed"
    
    def test_get_workflow_status_running(self, workflow_service, mock_workflow_engine):
        """Test retrieving status of a running workflow."""
        # Setup
        mock_execution = MockWorkflowExecution("exec-789", "workflow-101")
        mock_execution.status = WorkflowStatus.RUNNING
        mock_execution.started_at = datetime(2024, 1, 1, 10, 0, 0)
        mock_execution.completed_at = None
        mock_execution.current_step = "step2"
        
        mock_workflow_engine.get_workflow_execution.return_value = mock_execution
        
        # Execute
        status = workflow_service.get_workflow_status("exec-789")
        
        # Verify
        assert status['execution_id'] == "exec-789"
        assert status['status'] == "running"
        assert status['duration'] is None  # Not completed yet
        assert status['current_step'] == "step2"
    
    def test_get_workflow_status_not_found(self, workflow_service, mock_workflow_engine):
        """Test retrieving status of non-existent workflow."""
        # Setup
        mock_workflow_engine.get_workflow_execution.return_value = None
        
        # Execute & Verify
        with pytest.raises(ValueError, match="Workflow execution exec-999 not found"):
            workflow_service.get_workflow_status("exec-999")
    
    def test_validate_workflow_invalid_inputs_type(self, workflow_service):
        """Test validation with invalid inputs type."""
        # Setup
        workflow_json = json.dumps({
            "name": "Test Workflow",
            "steps": [
                {"id": "step1", "template": "template1", "inputs": "invalid"}  # Should be dict
            ]
        })
        
        # Execute
        is_valid, error_message = workflow_service.validate_workflow(workflow_json)
        
        # Verify
        assert is_valid is False
        assert "'inputs' must be a dictionary" in error_message
    
    def test_validate_workflow_invalid_outputs_type(self, workflow_service):
        """Test validation with invalid outputs type."""
        # Setup
        workflow_json = json.dumps({
            "name": "Test Workflow",
            "steps": [
                {"id": "step1", "template": "template1", "outputs": "invalid"}  # Should be list
            ]
        })
        
        # Execute
        is_valid, error_message = workflow_service.validate_workflow(workflow_json)
        
        # Verify
        assert is_valid is False
        assert "'outputs' must be a list" in error_message
    
    def test_execute_workflow_with_metadata(self, workflow_service, mock_workflow_engine):
        """Test workflow execution with metadata."""
        # Setup
        workflow_json = json.dumps({
            "name": "Test Workflow",
            "description": "Test description",
            "steps": [
                {"id": "step1", "template": "template1"}
            ],
            "metadata": {
                "author": "test_user",
                "version": "1.0"
            }
        })
        
        mock_workflow = MockWorkflow()
        mock_execution = MockWorkflowExecution()
        
        mock_workflow_engine.create_workflow.return_value = mock_workflow
        mock_workflow_engine.execute_workflow.return_value = mock_execution
        
        # Execute
        execution_id = workflow_service.execute_workflow(workflow_json)
        
        # Verify
        assert execution_id is not None
        
        # Check that metadata was passed to create_workflow
        call_args = mock_workflow_engine.create_workflow.call_args[0][0]
        assert call_args.metadata['name'] == "Test Workflow"
        assert call_args.metadata['description'] == "Test description"
        assert call_args.metadata['author'] == "test_user"
        assert call_args.metadata['version'] == "1.0"
