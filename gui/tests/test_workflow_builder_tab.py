"""
Unit tests for WorkflowBuilderTab component.
"""

import pytest
import json
from unittest.mock import Mock, MagicMock
from gui.components.workflow_builder_tab import WorkflowBuilderTab
from gui.services.workflow_service import WorkflowService
from gui.services.template_service import TemplateService


@pytest.fixture
def mock_workflow_service():
    """Create mock workflow service."""
    service = Mock(spec=WorkflowService)
    service.validate_workflow = Mock(return_value=(True, ""))
    service.execute_workflow = Mock(return_value="workflow-exec-123")
    service.get_workflow_status = Mock(return_value={
        'execution_id': 'workflow-exec-123',
        'workflow_id': 'workflow-123',
        'status': 'running',
        'started_at': '2024-01-01T00:00:00',
        'completed_at': None,
        'current_step': 'step1',
        'steps': [],
        'inputs': {},
        'outputs': None,
        'error': None,
        'duration': None
    })
    return service


@pytest.fixture
def mock_template_service():
    """Create mock template service."""
    service = Mock(spec=TemplateService)
    service.get_templates = Mock(return_value=[
        {'name': 'template1', 'category': 'test', 'description': 'Test template 1'},
        {'name': 'template2', 'category': 'test', 'description': 'Test template 2'}
    ])
    service.get_template_metadata = Mock(return_value={
        'name': 'template1',
        'category': 'test',
        'description': 'Test template',
        'version': '1.0.0',
        'inputs': [
            {'name': 'input1', 'type': 'text', 'description': 'Test input', 'required': True}
        ],
        'outputs': [
            {'name': 'output1', 'type': 'text', 'description': 'Test output'}
        ],
        'gpu_required': False,
        'memory_mb': 512,
        'timeout_sec': 300
    })
    return service


@pytest.fixture
def workflow_builder_tab(mock_workflow_service, mock_template_service):
    """Create WorkflowBuilderTab instance."""
    return WorkflowBuilderTab(mock_workflow_service, mock_template_service)


@pytest.fixture
def sample_workflow_json():
    """Create sample workflow JSON."""
    workflow = {
        "name": "Test Workflow",
        "description": "Test workflow description",
        "steps": [
            {
                "id": "step1",
                "name": "Step 1",
                "template": "template1",
                "inputs": {},
                "outputs": ["output1"]
            }
        ],
        "connections": [],
        "inputs": {},
        "metadata": {}
    }
    return json.dumps(workflow, indent=2)


class TestWorkflowBuilderTab:
    """Test suite for WorkflowBuilderTab."""
    
    def test_initialization(self, workflow_builder_tab):
        """Test that WorkflowBuilderTab initializes correctly."""
        assert workflow_builder_tab.workflow_service is not None
        assert workflow_builder_tab.template_service is not None
    
    def test_get_template_choices(self, workflow_builder_tab, mock_template_service):
        """Test retrieving template choices for dropdown."""
        choices = workflow_builder_tab._get_template_choices()
        
        assert len(choices) == 2
        assert 'template1' in choices
        assert 'template2' in choices
        mock_template_service.get_templates.assert_called_once()
    
    def test_get_initial_workflow_json(self, workflow_builder_tab):
        """Test getting initial workflow JSON structure."""
        initial_json = workflow_builder_tab._get_initial_workflow_json()
        
        # Parse to verify it's valid JSON
        workflow_data = json.loads(initial_json)
        
        assert 'name' in workflow_data
        assert 'steps' in workflow_data
        assert 'connections' in workflow_data
        assert isinstance(workflow_data['steps'], list)
        assert isinstance(workflow_data['connections'], list)
    
    def test_get_initial_canvas_html(self, workflow_builder_tab):
        """Test getting initial canvas HTML."""
        canvas_html = workflow_builder_tab._get_initial_canvas_html()
        
        assert 'workflow-canvas' in canvas_html
        assert 'renderWorkflow' in canvas_html
        assert '<script>' in canvas_html
    
    def test_on_add_step_success(self, workflow_builder_tab):
        """Test successfully adding a step to workflow."""
        initial_json = workflow_builder_tab._get_initial_workflow_json()
        
        updated_json, canvas_html, status = workflow_builder_tab.on_add_step(
            initial_json,
            'template1',
            'My Step'
        )
        
        # Parse updated workflow
        workflow_data = json.loads(updated_json)
        
        assert len(workflow_data['steps']) == 1
        assert workflow_data['steps'][0]['id'] == 'my_step'
        assert workflow_data['steps'][0]['name'] == 'My Step'
        assert workflow_data['steps'][0]['template'] == 'template1'
        assert 'Success' in status.value
    
    def test_on_add_step_no_template(self, workflow_builder_tab):
        """Test adding step without template selection."""
        initial_json = workflow_builder_tab._get_initial_workflow_json()
        
        updated_json, canvas_html, status = workflow_builder_tab.on_add_step(
            initial_json,
            None,
            'My Step'
        )
        
        # Workflow should be unchanged
        assert updated_json == initial_json
        assert 'Error' in status.value
        assert 'template' in status.value.lower()
    
    def test_on_add_step_no_name(self, workflow_builder_tab):
        """Test adding step without step name."""
        initial_json = workflow_builder_tab._get_initial_workflow_json()
        
        updated_json, canvas_html, status = workflow_builder_tab.on_add_step(
            initial_json,
            'template1',
            ''
        )
        
        # Workflow should be unchanged
        assert updated_json == initial_json
        assert 'Error' in status.value
        assert 'name' in status.value.lower()
    
    def test_on_add_step_duplicate_id(self, workflow_builder_tab):
        """Test adding step with duplicate ID."""
        # Create workflow with a step that has ID 'my_step'
        workflow = {
            "name": "Test Workflow",
            "steps": [
                {"id": "my_step", "name": "My Step", "template": "template1", "inputs": {}, "outputs": []}
            ],
            "connections": []
        }
        workflow_json = json.dumps(workflow, indent=2)
        
        # Try to add a step with same name (which generates same ID)
        updated_json, canvas_html, status = workflow_builder_tab.on_add_step(
            workflow_json,
            'template1',
            'My Step'  # This will generate 'my_step' which already exists
        )
        
        # Workflow should be unchanged
        workflow_data = json.loads(updated_json)
        assert len(workflow_data['steps']) == 1  # Still only one step
        assert 'Error' in status.value
        assert 'already exists' in status.value.lower()
    
    def test_on_connect_steps_success(self, workflow_builder_tab):
        """Test successfully connecting two steps."""
        # Create workflow with two steps
        workflow = {
            "name": "Test Workflow",
            "steps": [
                {"id": "step1", "name": "Step 1", "template": "template1", "inputs": {}, "outputs": ["output1"]},
                {"id": "step2", "name": "Step 2", "template": "template2", "inputs": {}, "outputs": []}
            ],
            "connections": []
        }
        workflow_json = json.dumps(workflow, indent=2)
        
        updated_json, canvas_html, status = workflow_builder_tab.on_connect_steps(
            workflow_json,
            'step1',
            'step2',
            'output1',
            'input1'
        )
        
        # Parse updated workflow
        workflow_data = json.loads(updated_json)
        
        assert len(workflow_data['connections']) == 1
        assert workflow_data['connections'][0]['from'] == 'step1'
        assert workflow_data['connections'][0]['to'] == 'step2'
        assert workflow_data['connections'][0]['output'] == 'output1'
        assert workflow_data['connections'][0]['input'] == 'input1'
        
        # Check that target step input was updated
        step2 = next(s for s in workflow_data['steps'] if s['id'] == 'step2')
        assert 'input1' in step2['inputs']
        assert step2['inputs']['input1'] == '${step1.output1}'
        
        assert 'Success' in status.value
    
    def test_on_connect_steps_missing_fields(self, workflow_builder_tab, sample_workflow_json):
        """Test connecting steps with missing fields."""
        updated_json, canvas_html, status = workflow_builder_tab.on_connect_steps(
            sample_workflow_json,
            'step1',
            '',  # Missing to_step
            'output1',
            'input1'
        )
        
        assert 'Error' in status.value
        assert 'required' in status.value.lower()
    
    def test_on_connect_steps_nonexistent_source(self, workflow_builder_tab, sample_workflow_json):
        """Test connecting steps with nonexistent source step."""
        updated_json, canvas_html, status = workflow_builder_tab.on_connect_steps(
            sample_workflow_json,
            'nonexistent_step',
            'step1',
            'output1',
            'input1'
        )
        
        assert 'Error' in status.value
        assert 'not found' in status.value.lower()
    
    def test_on_connect_steps_self_connection(self, workflow_builder_tab, sample_workflow_json):
        """Test connecting step to itself."""
        updated_json, canvas_html, status = workflow_builder_tab.on_connect_steps(
            sample_workflow_json,
            'step1',
            'step1',  # Same as source
            'output1',
            'input1'
        )
        
        assert 'Error' in status.value
        assert 'itself' in status.value.lower()
    
    def test_on_validate_workflow_success(self, workflow_builder_tab, mock_workflow_service, sample_workflow_json):
        """Test validating a valid workflow."""
        mock_workflow_service.validate_workflow.return_value = (True, "")
        
        status = workflow_builder_tab.on_validate_workflow(sample_workflow_json)
        
        assert 'Success' in status.value or '✅' in status.value
        mock_workflow_service.validate_workflow.assert_called_once_with(sample_workflow_json)
    
    def test_on_validate_workflow_failure(self, workflow_builder_tab, mock_workflow_service, sample_workflow_json):
        """Test validating an invalid workflow."""
        mock_workflow_service.validate_workflow.return_value = (False, "Missing required field")
        
        status = workflow_builder_tab.on_validate_workflow(sample_workflow_json)
        
        assert 'Failed' in status.value or '❌' in status.value
        assert 'Missing required field' in status.value
    
    def test_on_execute_workflow_success(self, workflow_builder_tab, mock_workflow_service, sample_workflow_json):
        """Test executing a valid workflow."""
        execution_id, status, loading_indicator = workflow_builder_tab.on_execute_workflow(sample_workflow_json)
        
        assert execution_id.value == 'workflow-exec-123'
        assert 'Success' in status.value
        mock_workflow_service.execute_workflow.assert_called_once_with(sample_workflow_json)
    
    def test_on_execute_workflow_validation_error(self, workflow_builder_tab, mock_workflow_service, sample_workflow_json):
        """Test executing workflow with validation error."""
        mock_workflow_service.execute_workflow.side_effect = ValueError("Invalid workflow")
        
        execution_id, status, loading_indicator = workflow_builder_tab.on_execute_workflow(sample_workflow_json)
        
        assert execution_id.visible is False
        assert 'Error' in status.value or 'Validation' in status.value
    
    def test_on_save_workflow(self, workflow_builder_tab, sample_workflow_json):
        """Test saving workflow."""
        status = workflow_builder_tab.on_save_workflow(sample_workflow_json, "My Workflow")
        
        # Should provide feedback about save
        assert status.value is not None
        assert len(status.value) > 0
    
    def test_on_json_changed_valid(self, workflow_builder_tab, sample_workflow_json):
        """Test updating canvas when JSON is changed with valid JSON."""
        canvas_html = workflow_builder_tab.on_json_changed(sample_workflow_json)
        
        assert 'workflow-canvas' in canvas_html
        assert 'renderWorkflow' in canvas_html
    
    def test_on_json_changed_invalid(self, workflow_builder_tab):
        """Test updating canvas when JSON is changed with invalid JSON."""
        invalid_json = "{ invalid json"
        
        canvas_html = workflow_builder_tab.on_json_changed(invalid_json)
        
        # Should return initial canvas on error
        assert 'workflow-canvas' in canvas_html
    
    def test_render_canvas(self, workflow_builder_tab):
        """Test rendering canvas with workflow data."""
        workflow_data = {
            "name": "Test Workflow",
            "steps": [
                {"id": "step1", "name": "Step 1", "template": "template1"}
            ],
            "connections": []
        }
        
        canvas_html = workflow_builder_tab._render_canvas(workflow_data)
        
        assert 'workflow-canvas' in canvas_html
        assert 'renderWorkflow' in canvas_html
        assert 'step1' in canvas_html or 'Step 1' in canvas_html
    
    def test_websocket_client_code_present(self, workflow_builder_tab):
        """Test that WebSocket client code is generated."""
        websocket_code = workflow_builder_tab._get_websocket_client_code()
        
        # Verify WebSocket connection code is present
        assert 'WebSocket' in websocket_code
        assert 'connectWebSocket' in websocket_code
        assert 'workflow.step_completed' in websocket_code
        
        # Verify event handlers are present
        assert 'handleWorkflowStepCompleted' in websocket_code
        assert 'updateWorkflowProgress' in websocket_code
        
        # Verify reconnection logic is present
        assert 'reconnectAttempts' in websocket_code
        assert 'maxReconnectAttempts' in websocket_code
        
        # Verify notification function is present
        assert 'showNotification' in websocket_code
    
    def test_render_includes_websocket_code(self, workflow_builder_tab):
        """Test that render method includes WebSocket client code."""
        # The render method should include WebSocket HTML component
        # We can't directly test Gradio rendering, but we can verify the method exists
        assert hasattr(workflow_builder_tab, '_get_websocket_client_code')
        
        # Verify the WebSocket code is valid HTML/JavaScript
        websocket_code = workflow_builder_tab._get_websocket_client_code()
        assert '<script>' in websocket_code
        assert '</script>' in websocket_code
