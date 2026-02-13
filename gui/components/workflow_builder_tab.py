"""
Workflow Builder Tab component for GUI interface.

This module provides the UI component for building and executing multi-step ML workflows
using a visual DAG editor.
"""

from typing import Any, Dict, List, Optional, Tuple
import gradio as gr
import json

from gui.services.workflow_service import WorkflowService
from gui.services.template_service import TemplateService
from gui.error_handling import (
    format_workflow_error,
    format_validation_error,
    format_generic_error,
    create_success_message,
    create_loading_message,
    create_warning_message,
    sanitize_error_message
)
from notebook_ml_orchestrator.core.logging_config import LoggerMixin


class WorkflowBuilderTab(LoggerMixin):
    """UI component for building and executing workflows."""
    
    def __init__(self, workflow_service: WorkflowService, template_service: TemplateService):
        """
        Initialize workflow builder tab.
        
        Args:
            workflow_service: Workflow service instance for workflow validation and execution
            template_service: Template service instance for template discovery
        """
        self.workflow_service = workflow_service
        self.template_service = template_service
        self.logger.info("WorkflowBuilderTab initialized")
    
    def render(self) -> gr.Blocks:
        """
        Render the workflow builder tab within a Gradio Blocks context.
        
        Returns:
            Gradio Blocks component containing the workflow builder interface
        """
        with gr.Blocks() as tab:
            gr.Markdown("## Workflow Builder")
            gr.Markdown("Create and execute multi-step ML workflows using a visual DAG editor.")
            
            # WebSocket client code for real-time workflow updates
            gr.HTML(self._get_websocket_client_code())
            
            # Main layout: Canvas on left, controls on right
            with gr.Row():
                # Left column: Workflow canvas and visualization
                with gr.Column(scale=2):
                    gr.Markdown("### Workflow Canvas")
                    
                    # Workflow canvas with JavaScript for DAG visualization
                    workflow_canvas = gr.HTML(
                        value=self._get_initial_canvas_html(),
                        label="Workflow DAG"
                    )
                    
                    # Workflow JSON editor
                    gr.Markdown("### Workflow JSON")
                    workflow_json_editor = gr.Code(
                        label="Workflow Definition",
                        language="json",
                        value=self._get_initial_workflow_json(),
                        interactive=True,
                        lines=15
                    )
                
                # Right column: Controls and configuration
                with gr.Column(scale=1):
                    gr.Markdown("### Workflow Controls")
                    
                    # Workflow name
                    workflow_name = gr.Textbox(
                        label="Workflow Name",
                        placeholder="Enter workflow name...",
                        value="My Workflow",
                        interactive=True
                    )
                    
                    # Template selector for adding steps
                    gr.Markdown("#### Add Step")
                    template_selector = gr.Dropdown(
                        label="Select Template",
                        choices=self._get_template_choices(),
                        value=None,
                        interactive=True,
                        info="Choose a template to add as a workflow step"
                    )
                    
                    step_name = gr.Textbox(
                        label="Step Name",
                        placeholder="Enter step name...",
                        value="",
                        interactive=True
                    )
                    
                    add_step_button = gr.Button(
                        "Add Step",
                        variant="primary",
                        size="sm"
                    )
                    
                    # Step configuration panel
                    gr.Markdown("#### Step Configuration")
                    step_config_panel = gr.JSON(
                        label="Step Configuration",
                        value={},
                        interactive=True
                    )
                    
                    # Connection controls
                    gr.Markdown("#### Connect Steps")
                    
                    from_step = gr.Textbox(
                        label="From Step ID",
                        placeholder="source_step_id",
                        interactive=True
                    )
                    
                    output_field = gr.Textbox(
                        label="Output Field",
                        placeholder="output_field_name",
                        interactive=True
                    )
                    
                    to_step = gr.Textbox(
                        label="To Step ID",
                        placeholder="target_step_id",
                        interactive=True
                    )
                    
                    input_field = gr.Textbox(
                        label="Input Field",
                        placeholder="input_field_name",
                        interactive=True
                    )
                    
                    connect_button = gr.Button(
                        "Connect Steps",
                        variant="secondary",
                        size="sm"
                    )
            
            # Action buttons row
            with gr.Row():
                validate_button = gr.Button(
                    "Validate Workflow",
                    variant="secondary"
                )
                
                save_button = gr.Button(
                    "Save Workflow",
                    variant="secondary"
                )
                
                load_button = gr.Button(
                    "Load Workflow",
                    variant="secondary"
                )
                
                execute_button = gr.Button(
                    "Execute Workflow",
                    variant="primary"
                )
            
            # File upload for loading workflows
            with gr.Row():
                workflow_file_upload = gr.File(
                    label="Upload Workflow JSON",
                    file_types=[".json"],
                    visible=False
                )
            
            # Status and results section
            with gr.Row():
                with gr.Column():
                    # Loading indicator
                    loading_indicator = gr.HTML(
                        value="",
                        visible=False
                    )
                    
                    # Validation status
                    validation_status = gr.Markdown(
                        value="",
                        visible=False
                    )
                    
                    # Execution status
                    execution_status = gr.Textbox(
                        label="Workflow Execution ID",
                        value="",
                        interactive=False,
                        visible=False
                    )
                    
                    # Status message
                    status_message = gr.Markdown(
                        value="",
                        visible=False
                    )
            
            # Event handlers
            
            # Add step button
            add_step_button.click(
                fn=self.on_add_step,
                inputs=[
                    workflow_json_editor,
                    template_selector,
                    step_name
                ],
                outputs=[
                    workflow_json_editor,
                    workflow_canvas,
                    status_message
                ]
            )
            
            # Connect steps button
            connect_button.click(
                fn=self.on_connect_steps,
                inputs=[
                    workflow_json_editor,
                    from_step,
                    to_step,
                    output_field,
                    input_field
                ],
                outputs=[
                    workflow_json_editor,
                    workflow_canvas,
                    status_message
                ]
            )
            
            # Validate button
            validate_button.click(
                fn=self.on_validate_workflow,
                inputs=[workflow_json_editor],
                outputs=[validation_status]
            )
            
            # Save button
            save_button.click(
                fn=self.on_save_workflow,
                inputs=[workflow_json_editor, workflow_name],
                outputs=[status_message]
            )
            
            # Load button (toggle file upload visibility)
            load_button.click(
                fn=lambda: gr.File(visible=True),
                inputs=[],
                outputs=[workflow_file_upload]
            )
            
            # File upload handler
            workflow_file_upload.change(
                fn=self.on_load_workflow,
                inputs=[workflow_file_upload],
                outputs=[
                    workflow_json_editor,
                    workflow_canvas,
                    workflow_name,
                    status_message,
                    workflow_file_upload
                ]
            )
            
            # Execute button
            execute_button.click(
                fn=self.on_execute_workflow,
                inputs=[workflow_json_editor],
                outputs=[
                    execution_status,
                    status_message,
                    loading_indicator
                ]
            )
            
            # Update canvas when JSON is edited manually
            workflow_json_editor.change(
                fn=self.on_json_changed,
                inputs=[workflow_json_editor],
                outputs=[workflow_canvas]
            )
        
        return tab

    
    def _get_template_choices(self) -> List[str]:
        """
        Get list of available template names for dropdown.
        
        Returns:
            List of template names
        """
        try:
            templates = self.template_service.get_templates()
            return [t['name'] for t in templates]
        except Exception as e:
            self.logger.error(f"Failed to retrieve templates: {e}")
            return []
    
    def _get_initial_workflow_json(self) -> str:
        """
        Get initial workflow JSON template.
        
        Returns:
            JSON string with empty workflow structure
        """
        initial_workflow = {
            "name": "My Workflow",
            "description": "",
            "steps": [],
            "connections": [],
            "inputs": {},
            "metadata": {}
        }
        return json.dumps(initial_workflow, indent=2)
    
    def _get_initial_canvas_html(self) -> str:
        """
        Get initial HTML for workflow canvas with DAG visualization.
        
        Returns:
            HTML string with canvas and JavaScript for visualization
        """
        return """
        <div id="workflow-canvas" style="width: 100%; height: 400px; border: 1px solid #ccc; border-radius: 4px; padding: 20px; background-color: #f9f9f9; position: relative; overflow: auto;">
            <div style="text-align: center; color: #666; padding-top: 180px;">
                <p style="font-size: 16px; margin: 0;">Workflow Canvas</p>
                <p style="font-size: 14px; margin: 10px 0 0 0;">Add steps to visualize your workflow</p>
            </div>
        </div>
        <script>
            // Simple workflow visualization
            function renderWorkflow(workflowData) {
                const canvas = document.getElementById('workflow-canvas');
                if (!canvas || !workflowData || !workflowData.steps) return;
                
                // Clear canvas
                canvas.innerHTML = '';
                
                const steps = workflowData.steps;
                if (steps.length === 0) {
                    canvas.innerHTML = '<div style="text-align: center; color: #666; padding-top: 180px;"><p style="font-size: 16px; margin: 0;">Workflow Canvas</p><p style="font-size: 14px; margin: 10px 0 0 0;">Add steps to visualize your workflow</p></div>';
                    return;
                }
                
                // Simple vertical layout
                const stepHeight = 80;
                const stepWidth = 200;
                const verticalSpacing = 40;
                
                steps.forEach((step, index) => {
                    const stepDiv = document.createElement('div');
                    stepDiv.setAttribute('data-step-id', step.id);
                    stepDiv.style.cssText = `
                        position: absolute;
                        left: 50%;
                        transform: translateX(-50%);
                        top: ${index * (stepHeight + verticalSpacing)}px;
                        width: ${stepWidth}px;
                        height: ${stepHeight}px;
                        background-color: #4CAF50;
                        color: white;
                        border-radius: 8px;
                        padding: 10px;
                        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
                        font-size: 12px;
                    `;
                    
                    stepDiv.innerHTML = `
                        <div style="font-weight: bold; margin-bottom: 5px;">${step.name || step.id}</div>
                        <div style="font-size: 11px; opacity: 0.9;">Template: ${step.template}</div>
                        <div style="font-size: 10px; opacity: 0.8; margin-top: 5px;">ID: ${step.id}</div>
                    `;
                    
                    canvas.appendChild(stepDiv);
                });
                
                // Draw connections
                const connections = workflowData.connections || [];
                connections.forEach(conn => {
                    const fromIndex = steps.findIndex(s => s.id === conn.from);
                    const toIndex = steps.findIndex(s => s.id === conn.to);
                    
                    if (fromIndex >= 0 && toIndex >= 0) {
                        const line = document.createElement('div');
                        const fromY = fromIndex * (stepHeight + verticalSpacing) + stepHeight;
                        const toY = toIndex * (stepHeight + verticalSpacing);
                        const lineHeight = toY - fromY - 10;
                        
                        line.style.cssText = `
                            position: absolute;
                            left: 50%;
                            transform: translateX(-50%);
                            top: ${fromY + 5}px;
                            width: 2px;
                            height: ${lineHeight}px;
                            background-color: #2196F3;
                        `;
                        
                        canvas.appendChild(line);
                        
                        // Arrow head
                        const arrow = document.createElement('div');
                        arrow.style.cssText = `
                            position: absolute;
                            left: 50%;
                            transform: translateX(-50%) translateY(-5px);
                            top: ${toY - 5}px;
                            width: 0;
                            height: 0;
                            border-left: 5px solid transparent;
                            border-right: 5px solid transparent;
                            border-top: 8px solid #2196F3;
                        `;
                        canvas.appendChild(arrow);
                    }
                });
                
                // Adjust canvas height
                const totalHeight = steps.length * (stepHeight + verticalSpacing) + 40;
                canvas.style.minHeight = `${Math.max(400, totalHeight)}px`;
            }
        </script>
        """
    
    def on_add_step(
        self,
        workflow_json: str,
        template_name: Optional[str],
        step_name: str
    ) -> Tuple[str, str, gr.Markdown]:
        """
        Add a step to the workflow.
        
        Args:
            workflow_json: Current workflow JSON string
            template_name: Template to use for the step
            step_name: Name for the new step
            
        Returns:
            Tuple of (
                updated_workflow_json,
                updated_canvas_html,
                status_message
            )
        """
        if not template_name:
            return (
                workflow_json,
                self._get_initial_canvas_html(),
                gr.Markdown(
                    value="❌ **Error:** Please select a template",
                    visible=True
                )
            )
        
        if not step_name:
            return (
                workflow_json,
                self._get_initial_canvas_html(),
                gr.Markdown(
                    value="❌ **Error:** Please enter a step name",
                    visible=True
                )
            )
        
        try:
            # Parse current workflow
            workflow_data = json.loads(workflow_json)
            
            # Generate step ID from step name
            step_id = step_name.lower().replace(' ', '_')
            
            # Check for duplicate step IDs
            existing_ids = [s['id'] for s in workflow_data.get('steps', [])]
            if step_id in existing_ids:
                return (
                    workflow_json,
                    self._render_canvas(workflow_data),
                    gr.Markdown(
                        value=f"❌ **Error:** Step ID '{step_id}' already exists. Please use a different step name.",
                        visible=True
                    )
                )
            
            # Create new step
            new_step = {
                "id": step_id,
                "name": step_name,
                "template": template_name,
                "inputs": {},
                "outputs": []
            }
            
            # Add step to workflow
            if 'steps' not in workflow_data:
                workflow_data['steps'] = []
            workflow_data['steps'].append(new_step)
            
            # Update JSON
            updated_json = json.dumps(workflow_data, indent=2)
            
            self.logger.info(f"Added step '{step_name}' (ID: {step_id}) with template '{template_name}'")
            
            return (
                updated_json,
                self._render_canvas(workflow_data),
                gr.Markdown(
                    value=f"✅ **Success:** Added step '{step_name}' to workflow",
                    visible=True
                )
            )
            
        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid workflow JSON: {e}")
            return (
                workflow_json,
                self._get_initial_canvas_html(),
                gr.Markdown(
                    value=f"❌ **Error:** Invalid workflow JSON\n\n**Details:** {str(e)}",
                    visible=True
                )
            )
        except Exception as e:
            self.logger.error(f"Failed to add step: {e}")
            return (
                workflow_json,
                self._get_initial_canvas_html(),
                gr.Markdown(
                    value=f"❌ **Error:** Failed to add step\n\n**Details:** {str(e)}",
                    visible=True
                )
            )
    
    def on_connect_steps(
        self,
        workflow_json: str,
        from_step: str,
        to_step: str,
        output_field: str,
        input_field: str
    ) -> Tuple[str, str, gr.Markdown]:
        """
        Connect two workflow steps.
        
        Args:
            workflow_json: Current workflow JSON string
            from_step: Source step ID
            to_step: Target step ID
            output_field: Output field name from source step
            input_field: Input field name for target step
            
        Returns:
            Tuple of (
                updated_workflow_json,
                updated_canvas_html,
                status_message
            )
        """
        if not all([from_step, to_step, output_field, input_field]):
            return (
                workflow_json,
                self._get_initial_canvas_html(),
                gr.Markdown(
                    value="❌ **Error:** All connection fields are required",
                    visible=True
                )
            )
        
        try:
            # Parse current workflow
            workflow_data = json.loads(workflow_json)
            
            # Validate that steps exist
            step_ids = [s['id'] for s in workflow_data.get('steps', [])]
            if from_step not in step_ids:
                return (
                    workflow_json,
                    self._render_canvas(workflow_data),
                    gr.Markdown(
                        value=f"❌ **Error:** Source step '{from_step}' not found",
                        visible=True
                    )
                )
            
            if to_step not in step_ids:
                return (
                    workflow_json,
                    self._render_canvas(workflow_data),
                    gr.Markdown(
                        value=f"❌ **Error:** Target step '{to_step}' not found",
                        visible=True
                    )
                )
            
            # Check for self-connection
            if from_step == to_step:
                return (
                    workflow_json,
                    self._render_canvas(workflow_data),
                    gr.Markdown(
                        value="❌ **Error:** Cannot connect a step to itself",
                        visible=True
                    )
                )
            
            # Create connection
            new_connection = {
                "from": from_step,
                "to": to_step,
                "output": output_field,
                "input": input_field
            }
            
            # Add connection to workflow
            if 'connections' not in workflow_data:
                workflow_data['connections'] = []
            
            # Check for duplicate connection
            for conn in workflow_data['connections']:
                if (conn['from'] == from_step and conn['to'] == to_step and
                    conn['output'] == output_field and conn['input'] == input_field):
                    return (
                        workflow_json,
                        self._render_canvas(workflow_data),
                        gr.Markdown(
                            value="⚠️ **Warning:** This connection already exists",
                            visible=True
                        )
                    )
            
            workflow_data['connections'].append(new_connection)
            
            # Update target step inputs to reference source step output
            for step in workflow_data['steps']:
                if step['id'] == to_step:
                    if 'inputs' not in step:
                        step['inputs'] = {}
                    step['inputs'][input_field] = f"${{{from_step}.{output_field}}}"
                    break
            
            # Update JSON
            updated_json = json.dumps(workflow_data, indent=2)
            
            self.logger.info(
                f"Connected steps: {from_step}.{output_field} -> {to_step}.{input_field}"
            )
            
            return (
                updated_json,
                self._render_canvas(workflow_data),
                gr.Markdown(
                    value=f"✅ **Success:** Connected {from_step} → {to_step}",
                    visible=True
                )
            )
            
        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid workflow JSON: {e}")
            return (
                workflow_json,
                self._get_initial_canvas_html(),
                gr.Markdown(
                    value=f"❌ **Error:** Invalid workflow JSON\n\n**Details:** {str(e)}",
                    visible=True
                )
            )
        except Exception as e:
            self.logger.error(f"Failed to connect steps: {e}")
            return (
                workflow_json,
                self._get_initial_canvas_html(),
                gr.Markdown(
                    value=f"❌ **Error:** Failed to connect steps\n\n**Details:** {str(e)}",
                    visible=True
                )
            )
    
    def on_validate_workflow(self, workflow_json: str) -> gr.Markdown:
        """
        Validate workflow structure and type compatibility.
        
        Args:
            workflow_json: Workflow JSON string to validate
            
        Returns:
            Markdown component with validation results
        """
        try:
            # Use workflow service to validate
            is_valid, error_message = self.workflow_service.validate_workflow(workflow_json)
            
            if is_valid:
                self.logger.info("Workflow validation successful")
                success_msg = create_success_message(
                    "Workflow is valid and ready to execute",
                    {"Status": "All checks passed"}
                )
                return gr.Markdown(value=success_msg, visible=True)
            else:
                self.logger.warning(f"Workflow validation failed: {error_message}")
                # Parse workflow to get name if possible
                try:
                    workflow_data = json.loads(workflow_json)
                    workflow_name = workflow_data.get('name', 'Unnamed Workflow')
                except:
                    workflow_name = 'Unnamed Workflow'
                
                error = format_workflow_error(
                    workflow_name,
                    sanitize_error_message(error_message)
                )
                return gr.Markdown(value=error.to_markdown(), visible=True)
                
        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid workflow JSON: {e}")
            error = format_validation_error(
                "workflow_json",
                f"Invalid JSON format: {sanitize_error_message(str(e))}"
            )
            return gr.Markdown(value=error.to_markdown(), visible=True)
        except Exception as e:
            self.logger.error(f"Validation error: {e}", exc_info=True)
            error = format_generic_error(e, "Workflow validation failed")
            return gr.Markdown(value=error.to_markdown(), visible=True)
    
    def on_save_workflow(
        self,
        workflow_json: str,
        workflow_name: str
    ) -> gr.Markdown:
        """
        Save workflow to file.
        
        Args:
            workflow_json: Workflow JSON string
            workflow_name: Name for the workflow file
            
        Returns:
            Markdown component with save status
        """
        try:
            # Parse and update workflow name
            workflow_data = json.loads(workflow_json)
            workflow_data['name'] = workflow_name
            
            # Generate filename
            filename = workflow_name.lower().replace(' ', '_') + '.json'
            
            # In a real implementation, this would save to a file or database
            # For now, we'll just provide feedback
            self.logger.info(f"Workflow '{workflow_name}' ready to save as '{filename}'")
            
            return gr.Markdown(
                value=f"ℹ️ **Info:** Workflow '{workflow_name}' is ready.\n\nIn a full implementation, this would save to `{filename}`.",
                visible=True
            )
            
        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid workflow JSON: {e}")
            return gr.Markdown(
                value=f"❌ **Error:** Invalid workflow JSON\n\n**Details:** {str(e)}",
                visible=True
            )
        except Exception as e:
            self.logger.error(f"Failed to save workflow: {e}")
            return gr.Markdown(
                value=f"❌ **Error:** Failed to save workflow\n\n**Details:** {str(e)}",
                visible=True
            )
    
    def on_load_workflow(
        self,
        file
    ) -> Tuple[str, str, str, gr.Markdown, gr.File]:
        """
        Load workflow from uploaded file.
        
        Args:
            file: Uploaded file object
            
        Returns:
            Tuple of (
                workflow_json,
                canvas_html,
                workflow_name,
                status_message,
                file_upload_component
            )
        """
        if file is None:
            return (
                self._get_initial_workflow_json(),
                self._get_initial_canvas_html(),
                "My Workflow",
                gr.Markdown(value="", visible=False),
                gr.File(visible=False)
            )
        
        try:
            # Read file content
            with open(file.name, 'r') as f:
                workflow_json = f.read()
            
            # Parse to validate
            workflow_data = json.loads(workflow_json)
            
            # Extract workflow name
            workflow_name = workflow_data.get('name', 'Loaded Workflow')
            
            self.logger.info(f"Loaded workflow '{workflow_name}' from file")
            
            return (
                workflow_json,
                self._render_canvas(workflow_data),
                workflow_name,
                gr.Markdown(
                    value=f"✅ **Success:** Loaded workflow '{workflow_name}'",
                    visible=True
                ),
                gr.File(visible=False)
            )
            
        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid workflow file: {e}")
            return (
                self._get_initial_workflow_json(),
                self._get_initial_canvas_html(),
                "My Workflow",
                gr.Markdown(
                    value=f"❌ **Error:** Invalid workflow file\n\n**Details:** {str(e)}",
                    visible=True
                ),
                gr.File(visible=False)
            )
        except Exception as e:
            self.logger.error(f"Failed to load workflow: {e}")
            return (
                self._get_initial_workflow_json(),
                self._get_initial_canvas_html(),
                "My Workflow",
                gr.Markdown(
                    value=f"❌ **Error:** Failed to load workflow\n\n**Details:** {str(e)}",
                    visible=True
                ),
                gr.File(visible=False)
            )
    
    def on_execute_workflow(
        self,
        workflow_json: str
    ) -> Tuple[gr.Textbox, gr.Markdown, gr.HTML]:
        """
        Execute workflow.
        
        Args:
            workflow_json: Workflow JSON string to execute
            
        Returns:
            Tuple of (
                execution_id_textbox,
                status_message,
                loading_indicator
            )
        """
        try:
            # Parse workflow to get name
            workflow_data = json.loads(workflow_json)
            workflow_name = workflow_data.get('name', 'Unnamed Workflow')
            
            # Execute workflow using workflow service
            execution_id = self.workflow_service.execute_workflow(workflow_json)
            
            self.logger.info(f"Workflow execution started: {execution_id}")
            
            success_msg = create_success_message(
                "Workflow execution started",
                {
                    "Execution ID": f"`{execution_id}`",
                    "Workflow": workflow_name,
                    "Next Step": "Monitor the workflow in the **Job Monitoring** tab"
                }
            )
            
            return (
                gr.Textbox(value=execution_id, visible=True),
                gr.Markdown(value=success_msg, visible=True),
                gr.HTML(value="", visible=False)
            )
            
        except ValueError as e:
            # Validation error
            self.logger.warning(f"Workflow execution failed - validation error: {e}")
            error = format_validation_error(
                "workflow",
                sanitize_error_message(str(e))
            )
            return (
                gr.Textbox(value="", visible=False),
                gr.Markdown(value=error.to_markdown(), visible=True),
                gr.HTML(value="", visible=False)
            )
        except json.JSONDecodeError as e:
            # JSON parsing error
            self.logger.error(f"Invalid workflow JSON: {e}")
            error = format_validation_error(
                "workflow_json",
                f"Invalid JSON format: {sanitize_error_message(str(e))}"
            )
            return (
                gr.Textbox(value="", visible=False),
                gr.Markdown(value=error.to_markdown(), visible=True),
                gr.HTML(value="", visible=False)
            )
        except Exception as e:
            # Generic error
            self.logger.error(f"Workflow execution failed: {e}", exc_info=True)
            error = format_generic_error(e, "Workflow execution failed")
            return (
                gr.Textbox(value="", visible=False),
                gr.Markdown(value=error.to_markdown(), visible=True),
                gr.HTML(value="", visible=False)
            )
    
    def on_json_changed(self, workflow_json: str) -> str:
        """
        Update canvas when JSON is manually edited.
        
        Args:
            workflow_json: Updated workflow JSON string
            
        Returns:
            Updated canvas HTML
        """
        try:
            workflow_data = json.loads(workflow_json)
            return self._render_canvas(workflow_data)
        except json.JSONDecodeError:
            # Invalid JSON, return current canvas
            return self._get_initial_canvas_html()
        except Exception as e:
            self.logger.error(f"Failed to update canvas: {e}")
            return self._get_initial_canvas_html()
    
    def _render_canvas(self, workflow_data: Dict[str, Any]) -> str:
        """
        Render workflow canvas with current workflow data.
        
        Args:
            workflow_data: Workflow data dictionary
            
        Returns:
            HTML string with rendered canvas
        """
        # Get base canvas HTML
        canvas_html = self._get_initial_canvas_html()
        
        # Add script to render workflow
        workflow_json_escaped = json.dumps(workflow_data).replace("'", "\\'")
        render_script = f"""
        <script>
            setTimeout(function() {{
                const workflowData = {workflow_json_escaped};
                renderWorkflow(workflowData);
            }}, 100);
        </script>
        """
        
        return canvas_html + render_script
    
    def _get_websocket_client_code(self) -> str:
        """
        Generate JavaScript code for WebSocket client connection.
        
        This method creates the JavaScript code that:
        - Connects to the WebSocket server
        - Handles workflow step completion events
        - Updates the workflow progress display when events are received
        
        Returns:
            HTML string containing JavaScript code for WebSocket client
            
        Requirements validated: 3.10
        """
        return """
        <script>
        (function() {
            // WebSocket connection for real-time workflow updates
            let ws = null;
            let reconnectAttempts = 0;
            const maxReconnectAttempts = 10;
            const reconnectDelay = 3000; // 3 seconds
            
            function connectWebSocket() {
                // Determine WebSocket URL based on current location
                const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                const host = window.location.hostname;
                const port = 7861; // WebSocket server port from config
                const wsUrl = `${protocol}//${host}:${port}/ws`;
                
                console.log('Connecting to WebSocket:', wsUrl);
                
                try {
                    ws = new WebSocket(wsUrl);
                    
                    ws.onopen = function(event) {
                        console.log('WebSocket connected');
                        reconnectAttempts = 0;
                    };
                    
                    ws.onmessage = function(event) {
                        try {
                            const message = JSON.parse(event.data);
                            console.log('WebSocket message received:', message);
                            
                            // Handle workflow step completion events
                            if (message.event_type === 'workflow.step_completed') {
                                handleWorkflowStepCompleted(message.data);
                            }
                        } catch (error) {
                            console.error('Error parsing WebSocket message:', error);
                        }
                    };
                    
                    ws.onerror = function(error) {
                        console.error('WebSocket error:', error);
                    };
                    
                    ws.onclose = function(event) {
                        console.log('WebSocket disconnected');
                        ws = null;
                        
                        // Attempt to reconnect with exponential backoff
                        if (reconnectAttempts < maxReconnectAttempts) {
                            reconnectAttempts++;
                            const delay = reconnectDelay * Math.pow(1.5, reconnectAttempts - 1);
                            console.log(`Reconnecting in ${delay}ms (attempt ${reconnectAttempts}/${maxReconnectAttempts})`);
                            setTimeout(connectWebSocket, delay);
                        } else {
                            console.error('Max reconnection attempts reached');
                        }
                    };
                } catch (error) {
                    console.error('Error creating WebSocket connection:', error);
                }
            }
            
            function handleWorkflowStepCompleted(data) {
                console.log('Workflow step completed:', data);
                
                // Show notification for step completion
                const stepName = data.step_name || data.step_id || 'Unknown step';
                const workflowId = data.workflow_id || data.execution_id || 'Unknown workflow';
                showNotification(`Workflow ${workflowId}: Step "${stepName}" completed`, 'success');
                
                // Update workflow progress display
                updateWorkflowProgress(data);
            }
            
            function updateWorkflowProgress(data) {
                // Find the workflow canvas and update it with step status
                const canvas = document.getElementById('workflow-canvas');
                if (!canvas) return;
                
                // Find the step element by ID
                const stepId = data.step_id;
                if (!stepId) return;
                
                // Look for step elements in the canvas
                const stepElements = canvas.querySelectorAll('div[data-step-id]');
                stepElements.forEach(function(stepElement) {
                    if (stepElement.getAttribute('data-step-id') === stepId) {
                        // Update step visual to show completion
                        stepElement.style.backgroundColor = '#4CAF50'; // Green for completed
                        stepElement.style.border = '2px solid #2E7D32';
                        
                        // Add completion indicator
                        const completionBadge = document.createElement('div');
                        completionBadge.innerHTML = '✓';
                        completionBadge.style.cssText = `
                            position: absolute;
                            top: 5px;
                            right: 5px;
                            background-color: #2E7D32;
                            color: white;
                            border-radius: 50%;
                            width: 20px;
                            height: 20px;
                            display: flex;
                            align-items: center;
                            justify-content: center;
                            font-size: 14px;
                            font-weight: bold;
                        `;
                        
                        // Remove existing badge if present
                        const existingBadge = stepElement.querySelector('div[style*="position: absolute"]');
                        if (existingBadge) {
                            existingBadge.remove();
                        }
                        
                        stepElement.appendChild(completionBadge);
                    }
                });
                
                // If the workflow execution ID matches the current workflow, refresh the status
                const executionIdInput = document.querySelector('input[label="Workflow Execution ID"]');
                if (executionIdInput && executionIdInput.value === data.execution_id) {
                    // Optionally trigger a refresh of workflow status
                    console.log('Current workflow updated, execution ID:', data.execution_id);
                }
            }
            
            function showNotification(message, type) {
                // Create a simple notification element
                const notification = document.createElement('div');
                notification.textContent = message;
                notification.style.position = 'fixed';
                notification.style.top = '20px';
                notification.style.right = '20px';
                notification.style.padding = '15px 20px';
                notification.style.borderRadius = '5px';
                notification.style.zIndex = '10000';
                notification.style.boxShadow = '0 2px 10px rgba(0,0,0,0.2)';
                notification.style.maxWidth = '400px';
                notification.style.fontSize = '14px';
                
                if (type === 'success') {
                    notification.style.backgroundColor = '#4CAF50';
                    notification.style.color = 'white';
                } else if (type === 'error') {
                    notification.style.backgroundColor = '#f44336';
                    notification.style.color = 'white';
                } else {
                    notification.style.backgroundColor = '#2196F3';
                    notification.style.color = 'white';
                }
                
                document.body.appendChild(notification);
                
                // Remove notification after 5 seconds
                setTimeout(function() {
                    notification.style.transition = 'opacity 0.5s';
                    notification.style.opacity = '0';
                    setTimeout(function() {
                        document.body.removeChild(notification);
                    }, 500);
                }, 5000);
            }
            
            // Connect when the page loads
            if (document.readyState === 'loading') {
                document.addEventListener('DOMContentLoaded', connectWebSocket);
            } else {
                connectWebSocket();
            }
            
            // Cleanup on page unload
            window.addEventListener('beforeunload', function() {
                if (ws) {
                    ws.close();
                }
            });
        })();
        </script>
        """
    
    def _get_loading_indicator_html(self) -> str:
        """
        Generate HTML for loading indicator.
        
        Returns:
            HTML string with animated loading spinner
        """
        return """
        <div style="display: flex; align-items: center; justify-content: center; padding: 20px;">
            <div style="border: 4px solid #f3f3f3; border-top: 4px solid #3498db; border-radius: 50%; width: 40px; height: 40px; animation: spin 1s linear infinite;"></div>
            <style>
                @keyframes spin {
                    0% { transform: rotate(0deg); }
                    100% { transform: rotate(360deg); }
                }
            </style>
            <span style="margin-left: 15px; font-size: 16px; color: #666;">Executing workflow...</span>
        </div>
        """
