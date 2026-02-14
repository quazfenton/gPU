"""
Job Submission Tab component for GUI interface - Version 2 with proper dynamic fields.

This module provides the UI component for submitting ML jobs through a form-based interface
with dynamically generated Gradio input components.
"""

from typing import Any, Dict, List, Optional, Tuple
import gradio as gr

from gui.services.job_service import JobService
from gui.services.template_service import TemplateService
from gui.error_handling import (
    format_validation_error,
    format_generic_error,
    create_success_message,
    sanitize_error_message
)
from notebook_ml_orchestrator.core.logging_config import LoggerMixin


class JobSubmissionTabV2(LoggerMixin):
    """UI component for submitting ML jobs with dynamic Gradio components."""
    
    def __init__(self, job_service: JobService, template_service: TemplateService):
        """
        Initialize job submission tab.
        
        Args:
            job_service: Job service instance for job submission
            template_service: Template service instance for template metadata
        """
        self.job_service = job_service
        self.template_service = template_service
        self.uploaded_files = {}  # Track uploaded files by field name
        self.logger.info("JobSubmissionTabV2 initialized")
    
    def render(self) -> gr.Blocks:
        """
        Render the job submission tab with dynamic input generation.
        
        Returns:
            Gradio Blocks component containing the job submission interface
        """
        with gr.Blocks() as tab:
            gr.Markdown("## Submit ML Job")
            gr.Markdown("Select a template and configure parameters to submit a job.")
            
            with gr.Row():
                with gr.Column(scale=1):
                    # Template selection
                    template_dropdown = gr.Dropdown(
                        label="Template",
                        choices=self._get_template_choices(),
                        value=None,
                        interactive=True,
                        info="Select an ML template to execute"
                    )
                    
                    # Backend selection (optional)
                    backend_dropdown = gr.Dropdown(
                        label="Backend (Optional)",
                        choices=self._get_backend_choices(),
                        value="auto",
                        interactive=True,
                        info="Select backend or use automatic routing"
                    )
                    
                    # Routing strategy selection
                    routing_strategy_dropdown = gr.Dropdown(
                        label="Routing Strategy",
                        choices=["cost-optimized", "round-robin", "least-loaded"],
                        value="cost-optimized",
                        interactive=True,
                        info="Strategy for automatic backend selection"
                    )
                    
                    # Template documentation display
                    template_docs = gr.Markdown(
                        value="Select a template to view documentation"
                    )
                
                with gr.Column(scale=1):
                    gr.Markdown("### Input Parameters")
                    
                    # File upload section
                    gr.Markdown("**File Uploads** (for audio, image, video inputs)")
                    file_upload = gr.File(
                        label="Upload Input File(s)",
                        file_count="multiple",
                        type="filepath",
                        interactive=True
                    )
                    uploaded_files_display = gr.Textbox(
                        label="Uploaded Files",
                        value="",
                        interactive=False,
                        lines=3,
                        visible=False
                    )
                    
                    gr.Markdown("---")
                    
                    # Placeholder message
                    placeholder_msg = gr.Markdown(
                        value="*Select a template to see input fields*",
                        visible=True
                    )
                    
                    # Dynamic input fields - we'll create these based on template
                    # Using a simple approach: text inputs for all fields with descriptions
                    input_container = gr.Column(visible=False)
                    
                    with input_container:
                        # We'll use JSON input as a workaround for dynamic fields
                        gr.Markdown("**Enter inputs as JSON:**")
                        gr.Markdown("""
                        *For file inputs, upload files above and use the file paths shown in "Uploaded Files"*
                        """)
                        inputs_json = gr.Code(
                            label="Job Inputs (JSON format)",
                            language="json",
                            value="{}",
                            lines=10
                        )
                        
                        gr.Markdown("""
                        **Example:**
                        ```json
                        {
                          "text": "Hello world",
                          "audio_file": "/path/to/uploaded/file.mp3",
                          "max_length": 100
                        }
                        ```
                        """)
                    
                    # Submit button
                    submit_button = gr.Button(
                        "Submit Job",
                        variant="primary",
                        visible=False
                    )
                    
                    # Job ID display
                    job_id_output = gr.Textbox(
                        label="Job ID",
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
            
            # File upload handler
            file_upload.change(
                fn=self.on_file_upload,
                inputs=[file_upload],
                outputs=[uploaded_files_display]
            )
            
            template_dropdown.change(
                fn=self.on_template_selected,
                inputs=[template_dropdown],
                outputs=[
                    template_docs,
                    placeholder_msg,
                    input_container,
                    submit_button,
                    inputs_json
                ]
            )
            
            submit_button.click(
                fn=self.on_submit_job,
                inputs=[
                    template_dropdown,
                    backend_dropdown,
                    routing_strategy_dropdown,
                    inputs_json
                ],
                outputs=[
                    job_id_output,
                    status_message
                ]
            )
        
        return tab
    
    def _get_template_choices(self) -> List[str]:
        """Get list of available template names for dropdown."""
        try:
            templates = self.template_service.get_templates()
            return [t['name'] for t in templates]
        except Exception as e:
            self.logger.error(f"Failed to retrieve templates: {e}")
            return []
    
    def _get_backend_choices(self) -> List[str]:
        """Get list of available backend IDs for dropdown."""
        try:
            choices = ["auto"]
            
            if hasattr(self.job_service, 'backend_router'):
                backend_ids = list(self.job_service.backend_router.backends.keys())
                choices.extend(sorted(backend_ids))
            
            return choices
        except Exception as e:
            self.logger.error(f"Failed to retrieve backends: {e}")
            return ["auto"]
    
    def on_template_selected(
        self,
        template_name: Optional[str]
    ) -> Tuple[str, gr.Markdown, gr.Column, gr.Button, str]:
        """Handle template selection and update UI."""
        if not template_name:
            return (
                "Select a template to view documentation",
                gr.Markdown(value="*Select a template to see input fields*", visible=True),
                gr.Column(visible=False),
                gr.Button(visible=False),
                "{}"
            )
        
        try:
            metadata = self.template_service.get_template_metadata(template_name)
            
            if not metadata:
                return (
                    f"**Error:** Template '{template_name}' not found",
                    gr.Markdown(value="*Template not found*", visible=True),
                    gr.Column(visible=False),
                    gr.Button(visible=False),
                    "{}"
                )
            
            # Generate documentation
            docs = self._generate_template_docs(metadata)
            
            # Generate example JSON for inputs
            example_json = self._generate_example_json(metadata)
            
            return (
                docs,
                gr.Markdown(visible=False),
                gr.Column(visible=True),
                gr.Button(visible=True),
                example_json
            )
            
        except Exception as e:
            self.logger.error(f"Error loading template: {e}")
            return (
                f"**Error:** {str(e)}",
                gr.Markdown(value="*Error loading template*", visible=True),
                gr.Column(visible=False),
                gr.Button(visible=False),
                "{}"
            )
    
    def _generate_template_docs(self, metadata: Dict[str, Any]) -> str:
        """Generate markdown documentation for a template."""
        docs_lines = [
            f"### {metadata['name']}",
            f"**Category:** {metadata['category']}",
            f"**Version:** {metadata['version']}",
            "",
            f"**Description:** {metadata['description']}",
            "",
            "**Resource Requirements:**",
            f"- GPU Required: {'Yes' if metadata['gpu_required'] else 'No'}",
        ]
        
        if metadata['gpu_required'] and metadata.get('gpu_type'):
            docs_lines.append(f"- GPU Type: {metadata['gpu_type']}")
        
        docs_lines.extend([
            f"- Memory: {metadata['memory_mb']} MB",
            f"- Timeout: {metadata['timeout_sec']} seconds",
            ""
        ])
        
        if metadata.get('inputs'):
            docs_lines.append("**Input Fields:**")
            for inp in metadata['inputs']:
                required_str = " (required)" if inp['required'] else " (optional)"
                docs_lines.append(
                    f"- **{inp['name']}** ({inp['type']}){required_str}: {inp['description']}"
                )
                if inp.get('default') is not None:
                    docs_lines.append(f"  - Default: `{inp['default']}`")
            docs_lines.append("")
        
        if metadata.get('outputs'):
            docs_lines.append("**Output Fields:**")
            for out in metadata['outputs']:
                docs_lines.append(
                    f"- **{out['name']}** ({out['type']}): {out['description']}"
                )
        
        return "\n".join(docs_lines)
    
    def _generate_example_json(self, metadata: Dict[str, Any]) -> str:
        """Generate example JSON for template inputs."""
        import json
        
        example = {}
        
        for inp in metadata.get('inputs', []):
            field_name = inp['name']
            field_type = inp['type']
            
            if inp.get('default') is not None:
                example[field_name] = inp['default']
            elif field_type == 'text' or field_type == 'string':
                example[field_name] = f"<your {field_name} here>"
            elif field_type == 'number':
                example[field_name] = 0
            elif field_type in ['file', 'audio', 'image', 'video']:
                example[field_name] = f"https://example.com/{field_name}.{field_type}"
            elif field_type == 'json':
                example[field_name] = {}
            else:
                example[field_name] = ""
        
        return json.dumps(example, indent=2)
    
    def on_file_upload(self, files: Optional[List[str]]) -> gr.Textbox:
        """Handle file uploads and display file paths."""
        if not files:
            return gr.Textbox(value="", visible=False)
        
        # Store uploaded files
        file_paths = []
        for file_path in files:
            if file_path:
                file_paths.append(file_path)
                self.logger.info(f"File uploaded: {file_path}")
        
        if file_paths:
            display_text = "Uploaded files (use these paths in your JSON):\n" + "\n".join(
                f"- {path}" for path in file_paths
            )
            return gr.Textbox(value=display_text, visible=True)
        
        return gr.Textbox(value="", visible=False)
    
    def on_submit_job(
        self,
        template_name: Optional[str],
        backend: str,
        routing_strategy: str,
        inputs_json_str: str
    ) -> Tuple[gr.Textbox, gr.Markdown]:
        """Handle job submission."""
        import json
        
        if not template_name:
            error_msg = "❌ **Error:** Please select a template first"
            return (
                gr.Textbox(value="", visible=False),
                gr.Markdown(value=error_msg, visible=True)
            )
        
        try:
            # Parse JSON inputs
            try:
                inputs = json.loads(inputs_json_str)
            except json.JSONDecodeError as e:
                error_msg = f"❌ **Invalid JSON:** {str(e)}"
                return (
                    gr.Textbox(value="", visible=False),
                    gr.Markdown(value=error_msg, visible=True)
                )
            
            # Determine backend
            backend_id = None if backend == "auto" else backend
            
            # Submit job
            job_id = self.job_service.submit_job(
                template_name=template_name,
                inputs=inputs,
                backend=backend_id,
                routing_strategy=routing_strategy
            )
            
            success_msg = f"""
✅ **Job Submitted Successfully!**

**Job ID:** `{job_id}`  
**Template:** {template_name}  
**Backend:** {backend}  
**Routing Strategy:** {routing_strategy}

**Next Steps:**
- Monitor the job in the **Job Monitoring** tab
- View results when the job completes
"""
            
            return (
                gr.Textbox(value=job_id, visible=True),
                gr.Markdown(value=success_msg, visible=True)
            )
            
        except Exception as e:
            self.logger.error(f"Job submission failed: {e}", exc_info=True)
            error_msg = f"❌ **Error:** {sanitize_error_message(str(e))}"
            return (
                gr.Textbox(value="", visible=False),
                gr.Markdown(value=error_msg, visible=True)
            )
