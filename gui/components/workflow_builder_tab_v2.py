"""
Workflow Builder Tab with Visual DAG Editor - Enhanced Version

This module provides a visual workflow builder with:
- Mermaid.js DAG visualization
- Drag-and-drop step addition
- Connection validation
- Workflow export/import
- Real-time validation
"""

from typing import Any, Dict, List, Optional, Tuple
import gradio as gr
import json

from gui.services.workflow_service import WorkflowService
from gui.services.template_service import TemplateService
from notebook_ml_orchestrator.core.logging_config import LoggerMixin


class WorkflowBuilderTabV2(LoggerMixin):
    """Enhanced UI component for building and executing workflows with visual DAG editor."""

    def __init__(self, workflow_service: WorkflowService, template_service: TemplateService):
        """Initialize workflow builder tab."""
        self.workflow_service = workflow_service
        self.template_service = template_service
        self.logger.info("WorkflowBuilderTabV2 initialized")

    def render(self) -> gr.Blocks:
        """Render the workflow builder tab with visual DAG editor."""
        with gr.Blocks() as tab:
            gr.Markdown("## Workflow Builder")
            gr.Markdown("Create multi-step ML workflows visually with drag-and-drop interface")
            
            # Connection status indicator
            with gr.Row():
                gr.HTML("""
                    <div id="websocket-status" class="status-indicator status-disconnected" 
                         title="WebSocket: Disconnected"></div>
                """)
            
            with gr.Row():
                # Left panel: Template palette and step configuration
                with gr.Column(scale=1):
                    gr.Markdown("### Template Palette")
                    
                    # Template search
                    template_search = gr.Textbox(
                        label="Search Templates",
                        placeholder="Type to search templates...",
                        interactive=True
                    )
                    
                    # Template list
                    template_list = gr.Dataframe(
                        label="Available Templates",
                        headers=["Name", "Category", "GPU Required"],
                        datatype=["str", "str", "str"],
                        interactive=False,
                        wrap=True,
                        max_rows=20
                    )
                    
                    gr.Markdown("---")
                    gr.Markdown("### Step Configuration")
                    
                    # Step name
                    step_name = gr.Textbox(
                        label="Step Name",
                        placeholder="e.g., load-data, preprocess, train-model",
                        interactive=True
                    )
                    
                    # Selected template
                    selected_template = gr.Dropdown(
                        label="Selected Template",
                        choices=[],
                        interactive=True
                    )
                    
                    # Step inputs (JSON)
                    step_inputs = gr.Code(
                        label="Step Inputs (JSON)",
                        language="json",
                        value="{}",
                        lines=10,
                        interactive=True
                    )
                    
                    # Add/Update step buttons
                    with gr.Row():
                        add_step_btn = gr.Button("➕ Add Step", variant="primary")
                        update_step_btn = gr.Button("🔄 Update Step", variant="secondary")
                        delete_step_btn = gr.Button("🗑️ Delete Step", variant="stop")
                    
                    gr.Markdown("---")
                    gr.Markdown("### Workflow Actions")
                    
                    # Workflow name
                    workflow_name = gr.Textbox(
                        label="Workflow Name",
                        placeholder="My ML Workflow",
                        interactive=True
                    )
                    
                    # Action buttons
                    with gr.Row():
                        validate_workflow_btn = gr.Button("✅ Validate", variant="secondary")
                        save_workflow_btn = gr.Button("💾 Save", variant="primary")
                        load_workflow_btn = gr.Button("📂 Load", variant="secondary")
                    
                    with gr.Row():
                        execute_workflow_btn = gr.Button("▶️ Execute", variant="primary")
                        export_workflow_btn = gr.Button("📤 Export", variant="secondary")
                        import_workflow_btn = gr.Button("📥 Import", variant="secondary")
                    
                    # Workflow file for import/export
                    workflow_file = gr.File(
                        label="Workflow File",
                        file_types=[".json"],
                        visible=False
                    )
                
                # Right panel: Visual DAG editor and workflow JSON
                with gr.Column(scale=2):
                    gr.Markdown("### Workflow Visualization")
                    
                    # Mermaid DAG visualization
                    dag_visualization = gr.HTML(
                        value=self._generate_mermaid_diagram([]),
                        label="Workflow DAG",
                        elem_classes=["mermaid-diagram"]
                    )
                    
                    gr.Markdown("---")
                    gr.Markdown("### Workflow Steps")
                    
                    # Current steps list
                    steps_list = gr.Dataframe(
                        label="Workflow Steps",
                        headers=["Step Name", "Template", "Status"],
                        datatype=["str", "str", "str"],
                        interactive=True,
                        wrap=True,
                        max_rows=15
                    )
                    
                    gr.Markdown("---")
                    gr.Markdown("### Workflow JSON")
                    
                    # Workflow JSON editor
                    workflow_json = gr.Code(
                        label="Workflow Definition (JSON)",
                        language="json",
                        value=self._get_empty_workflow_json(),
                        lines=15,
                        interactive=True
                    )
                    
                    # Status and results
                    gr.Markdown("---")
                    gr.Markdown("### Execution Status")
                    
                    execution_status = gr.Textbox(
                        label="Status",
                        value="Ready",
                        interactive=False
                    )
                    
                    execution_results = gr.JSON(
                        label="Results",
                        value={}
                    )
            
            # Hidden state for workflow data
            workflow_state = gr.State(value={
                'steps': [],
                'connections': [],
                'current_step': None
            })
            
            # Event handlers
            
            # Load templates on tab open
            tab.load(
                fn=self._load_templates,
                outputs=[template_list, selected_template]
            )
            
            # Template search
            template_search.change(
                fn=self._filter_templates,
                inputs=[template_search, template_list],
                outputs=[template_list]
            )
            
            # Template selection
            template_list.select(
                fn=self._on_template_selected,
                inputs=[template_list],
                outputs=[selected_template]
            )
            
            # Add step
            add_step_btn.click(
                fn=self._add_step,
                inputs=[step_name, selected_template, step_inputs, workflow_state],
                outputs=[workflow_state, steps_list, dag_visualization, workflow_json]
            )
            
            # Update step
            update_step_btn.click(
                fn=self._update_step,
                inputs=[step_name, selected_template, step_inputs, workflow_state],
                outputs=[workflow_state, steps_list, dag_visualization, workflow_json]
            )
            
            # Delete step
            delete_step_btn.click(
                fn=self._delete_step,
                inputs=[step_name, workflow_state],
                outputs=[workflow_state, steps_list, dag_visualization, workflow_json]
            )
            
            # Validate workflow
            validate_workflow_btn.click(
                fn=self._validate_workflow,
                inputs=[workflow_state],
                outputs=[execution_status, execution_results]
            )
            
            # Save workflow
            save_workflow_btn.click(
                fn=self._save_workflow,
                inputs=[workflow_name, workflow_state],
                outputs=[execution_status, workflow_file]
            )
            
            # Execute workflow
            execute_workflow_btn.click(
                fn=self._execute_workflow,
                inputs=[workflow_name, workflow_state],
                outputs=[execution_status, execution_results]
            )
            ...
            def _execute_workflow(self, workflow_name: str, workflow_state: Dict) -> Tuple[str, Dict]:
                """Execute the workflow."""
                try:
                    # Build full workflow JSON expected by WorkflowService
                    workflow_data = {
                        "name": workflow_name or "Untitled Workflow",
                        "steps": workflow_state.get("steps", []),
                        "connections": workflow_state.get("connections", []),
                        "conditions": workflow_state.get("conditions", []),
                        "metadata": workflow_state.get("metadata", {}),
                    }

                    # Call workflow service to execute
                    workflow_id = self.workflow_service.execute_workflow(json.dumps(workflow_data))

                    workflow_json.change(
                        fn=self._sync_json_to_state,
                        inputs=[workflow_json, workflow_state],
                        outputs=[workflow_state, steps_list, dag_visualization]
                    )
                except Exception as e:
                    return f"❌ Execution failed: {str(e)}", {"error": str(e)}
            # Workflow JSON changes (sync with state)
            workflow_json.change(
                fn=self._sync_json_to_state,
                inputs=[workflow_json],
                outputs=[workflow_state, steps_list, dag_visualization]
            )
            
            # Add CSS for Mermaid diagram
            gr.HTML("""
                <style>
                .mermaid-diagram {
                    border: 1px solid #e0e0e0;
                    border-radius: 8px;
                    padding: 20px;
                    background: #fafafa;
                    min-height: 400px;
                }
                
                .status-indicator {
                    width: 12px;
                    height: 12px;
                    border-radius: 50%;
                    display: inline-block;
                    margin-right: 8px;
                }
                
                .status-connected {
                    background: #4CAF50;
                    box-shadow: 0 0 8px #4CAF50;
                }
                
                .status-disconnected {
                    background: #9E9E9E;
                }
                
                .status-error {
                    background: #f44336;
                    box-shadow: 0 0 8px #f44336;
                }
                </style>
            """)
            
            # Add Mermaid.js library
            gr.HTML("""
                <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
                <script>
                mermaid.initialize({ 
                    startOnLoad: true,
                    theme: 'default',
                    flowchart: {
                        useMaxWidth: true,
                        htmlLabels: true,
                        curve: 'basis'
                    }
                });
                
                // Re-render Mermaid diagrams when content changes
                function renderMermaid() {
                    mermaid.init(undefined, document.querySelectorAll('.mermaid-diagram'));
                }
                
                // Listen for DOM changes
                const observer = new MutationObserver(renderMermaid);
                observer.observe(document.querySelector('.mermaid-diagram'), {
                    childList: true,
                    subtree: true,
                    characterData: true
                });
                </script>
            """)
        
        return tab
    
    def _load_templates(self) -> Tuple[List[List[str]], List[str]]:
        """Load available templates."""
        try:
            templates = self.template_service.get_templates()
            
            # Create dataframe rows
            rows = []
            choices = []
            for t in templates:
                rows.append([
                    t['name'],
                    t['category'],
                    'Yes' if t.get('gpu_required') else 'No'
                ])
                choices.append(t['name'])
            
            return rows, choices
            
        except Exception as e:
            self.logger.error(f"Failed to load templates: {e}")
            return [], []
    
    def _filter_templates(self, search_query: str, template_list: List[List[str]]) -> List[List[str]]:
        """Filter templates based on search query."""
        if not search_query:
            return template_list
        
        search_lower = search_query.lower()
        filtered = []
        
        for row in template_list:
            if (search_lower in row[0].lower() or  # Name
                search_lower in row[1].lower()):    # Category
                filtered.append(row)
        
        return filtered
    
    def _on_template_selected(self, template_list: List[List[str]], 
                             evt: gr.SelectData) -> str:
        """Handle template selection."""
        if evt.index[0] < len(template_list):
            return template_list[evt.index[0]][0]
        return ""
    
    def _add_step(self, step_name: str, template_name: str, 
                  step_inputs: str, workflow_state: Dict) -> Tuple[Dict, List[List[str]], str, str]:
        """Add a step to the workflow."""
        if not step_name or not template_name:
            return workflow_state, self._get_steps_list(workflow_state), \
                   self._generate_mermaid_diagram(workflow_state.get('steps', [])), \
                   json.dumps(workflow_state, indent=2)
        
        # Initialize workflow state
        if 'steps' not in workflow_state:
            workflow_state['steps'] = []
        if 'connections' not in workflow_state:
            workflow_state['connections'] = []
        
        # Parse inputs
        try:
            inputs = json.loads(step_inputs) if step_inputs else {}
        except json.JSONDecodeError:
            inputs = {}
        
        # Add step
        step = {
            'id': step_name,
            'name': step_name,
            'template': template_name,
            'inputs': inputs,
            'status': 'pending'
        }
        
        workflow_state['steps'].append(step)
        
        return workflow_state, \
               self._get_steps_list(workflow_state), \
               self._generate_mermaid_diagram(workflow_state['steps']), \
               json.dumps(workflow_state, indent=2)
    
    def _update_step(self, step_name: str, template_name: str,
                    step_inputs: str, workflow_state: Dict) -> Tuple[Dict, List[List[str]], str, str]:
        """Update an existing step."""
        if 'steps' not in workflow_state:
            return workflow_state, self._get_steps_list(workflow_state), \
                   self._generate_mermaid_diagram([]), \
                   json.dumps(workflow_state, indent=2)
        
        # Find and update step
        for step in workflow_state['steps']:
            if step['id'] == step_name:
                step['template'] = template_name
                try:
                    step['inputs'] = json.loads(step_inputs) if step_inputs else {}
                except json.JSONDecodeError:
                    pass
                break
        
        return workflow_state, \
               self._get_steps_list(workflow_state), \
               self._generate_mermaid_diagram(workflow_state['steps']), \
               json.dumps(workflow_state, indent=2)
    
    def _delete_step(self, step_name: str, workflow_state: Dict) -> Tuple[Dict, List[List[str]], str, str]:
        """Delete a step from the workflow."""
        if 'steps' not in workflow_state:
            return workflow_state, [], self._generate_mermaid_diagram([]), json.dumps(workflow_state, indent=2)
        
        # Remove step
        workflow_state['steps'] = [s for s in workflow_state['steps'] if s['id'] != step_name]
        
        # Remove connections involving this step
        workflow_state['connections'] = [
            c for c in workflow_state.get('connections', [])
            if c.get('from') != step_name and c.get('to') != step_name
        ]
        
        return workflow_state, \
               self._get_steps_list(workflow_state), \
               self._generate_mermaid_diagram(workflow_state['steps']), \
               json.dumps(workflow_state, indent=2)
    
    def _validate_workflow(self, workflow_state: Dict) -> Tuple[str, Dict]:
        """Validate workflow structure."""
        try:
            # Call workflow service validation
            result = self.workflow_service.validate_workflow(json.dumps(workflow_state))
            
            if result[0]:  # Valid
                return "✅ Workflow validation passed", {"valid": True, "message": "Workflow is valid"}
            else:
                return f"❌ Validation failed: {result[1]}", {"valid": False, "error": result[1]}
                
        except Exception as e:
            return f"❌ Validation error: {str(e)}", {"valid": False, "error": str(e)}
    
    def _save_workflow(self, workflow_name: str, workflow_state: Dict) -> Tuple[str, str]:
        """Save workflow to file."""
        try:
            workflow_data = {
                'name': workflow_name or 'Untitled Workflow',
                'workflow': workflow_state,
                'created_at': str(gradio.utils.get_time())
            }
            
            # In a real implementation, this would save to disk
            # For now, we'll return the JSON as a file download
            import tempfile
            import os
            
            temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
            json.dump(workflow_data, temp_file, indent=2)
            temp_file.close()
            
            return f"✅ Workflow saved: {workflow_name}", temp_file.name
            
        except Exception as e:
            return f"❌ Save failed: {str(e)}", ""
    
    def _execute_workflow(self, workflow_name: str, workflow_state: Dict) -> Tuple[str, Dict]:
        """Execute the workflow."""
        try:
            # Call workflow service to execute
            workflow_id = self.workflow_service.execute_workflow(json.dumps(workflow_state))
            
            return f"▶️ Workflow executing (ID: {workflow_id})", {
                "workflow_id": workflow_id,
                "status": "running",
                "message": "Workflow execution started"
            }
            
        except Exception as e:
            return f"❌ Execution failed: {str(e)}", {"error": str(e)}
    
    def _export_workflow(self, workflow_name: str, workflow_state: Dict) -> str:
        """Export workflow to file."""
        try:
            import tempfile
            
            workflow_data = {
                'name': workflow_name or 'Untitled Workflow',
                'workflow': workflow_state
            }
            
            temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
            json.dump(workflow_data, temp_file, indent=2)
            temp_file.close()
            
            return temp_file.name
            
        except Exception as e:
            return ""
    
    def _import_workflow(self, workflow_file) -> Tuple[str, Dict, List[List[str]], str, str]:
        """Import workflow from file."""
        try:
            if not workflow_file:
                return "", {}, [], self._generate_mermaid_diagram([]), "{}"
            
            with open(workflow_file, 'r') as f:
                workflow_data = json.load(f)
            
            workflow_name = workflow_data.get('name', 'Imported Workflow')
            workflow_state = workflow_data.get('workflow', {})
            
            return workflow_name, \
                   workflow_state, \
                   self._get_steps_list(workflow_state), \
                   self._generate_mermaid_diagram(workflow_state.get('steps', [])), \
                   json.dumps(workflow_state, indent=2)
                   
        except Exception as e:
            return f"Import Error: {str(e)}", {}, [], self._generate_mermaid_diagram([]), "{}"
    
    def _sync_json_to_state(self, workflow_json: str) -> Tuple[Dict, List[List[str]], str]:
        """Sync workflow JSON to state."""
        try:
            workflow_state = json.loads(workflow_json)
            return workflow_state, \
                   self._get_steps_list(workflow_state), \
                   self._generate_mermaid_diagram(workflow_state.get('steps', []))
        except:
            return {}, [], self._generate_mermaid_diagram([])
    
    def _get_steps_list(self, workflow_state: Dict) -> List[List[str]]:
        """Get steps as dataframe rows."""
        steps = workflow_state.get('steps', [])
        return [[s['name'], s['template'], s.get('status', 'pending')] for s in steps]
    
    def _generate_mermaid_diagram(self, steps: List[Dict]) -> str:
        """Generate Mermaid.js diagram from workflow steps."""
        if not steps:
            return """
            <div class="mermaid">
            graph TD
                A[No Steps] --> B[Add steps to see workflow]
                style A fill:#f9f9f9,stroke:#333,stroke-width:2px
                style B fill:#f9f9f9,stroke:#333,stroke-width:2px
            </div>
            """
        
        # Build Mermaid diagram
        mermaid_lines = ['graph TD']
        
        # Add nodes
        for i, step in enumerate(steps):
            node_id = f"step_{i}"
            node_label = f"{step['name']}<br/>({step['template']})"
            mermaid_lines.append(f"    {node_id}[{node_label}]")
            
            # Add connections
            if i > 0:
                prev_id = f"step_{i-1}"
                mermaid_lines.append(f"    {prev_id} --> {node_id}")
        
        # Add styling
        mermaid_lines.append("    style step_0 fill:#4CAF50,color:#fff,stroke:#333,stroke-width:2px")
        if steps:
            mermaid_lines.append(f"    style step_{len(steps)-1} fill:#2196F3,color:#fff,stroke:#333,stroke-width:2px")
        
        return f"<div class=\"mermaid\">{''.join(mermaid_lines)}</div>"
    
    def _get_empty_workflow_json(self) -> str:
        """Get empty workflow JSON template."""
        return json.dumps({
            "steps": [],
            "connections": []
        }, indent=2)
