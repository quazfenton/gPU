"""
Job Submission Tab component for GUI interface.

This module provides the UI component for submitting ML jobs through a form-based interface.
"""

from typing import Any, Dict, List, Optional, Tuple
import gradio as gr

from gui.services.job_service import JobService
from gui.services.template_service import TemplateService
from gui.validation import validate_inputs, format_validation_errors
from gui.error_handling import (
    format_validation_error,
    format_generic_error,
    create_success_message,
    create_loading_message,
    sanitize_error_message
)
from notebook_ml_orchestrator.core.logging_config import LoggerMixin
from templates.base import Template, InputField


class JobSubmissionTab(LoggerMixin):
    """UI component for submitting ML jobs."""
    
    def __init__(self, job_service: JobService, template_service: TemplateService):
        """
        Initialize job submission tab.
        
        Args:
            job_service: Job service instance for job submission
            template_service: Template service instance for template metadata
        """
        self.job_service = job_service
        self.template_service = template_service
        self.logger.info("JobSubmissionTab initialized")
    
    def render(self) -> gr.Blocks:
        """
        Render the job submission tab within a Gradio Blocks context.
        
        Returns:
            Gradio Blocks component containing the job submission interface
        """
        with gr.Blocks() as tab:
            gr.Markdown("## Submit ML Job")
            gr.Markdown("Select a template and configure parameters to submit a job.")
            
            # Store current template metadata
            current_template_state = gr.State(value=None)
            
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
                    with gr.Row():
                        backend_dropdown = gr.Dropdown(
                            label="Backend (Optional)",
                            choices=self._get_backend_choices(),
                            value="auto",
                            interactive=True,
                            info="Select backend or use automatic routing",
                            scale=4
                        )
                        refresh_backends_btn = gr.Button(
                            "🔄",
                            variant="secondary",
                            scale=1,
                            min_width=10,
                            size="sm"
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
                        value="Select a template to view documentation",
                        label="Template Documentation"
                    )
                
                with gr.Column(scale=1):
                    # Dynamic input fields container
                    gr.Markdown("### Input Parameters")
                    input_fields_html = gr.HTML(
                        value="<p style='color: #666; font-style: italic;'>Select a template to see input fields</p>"
                    )
                    
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
            refresh_backends_btn.click(
                fn=self.on_refresh_backends,
                outputs=[backend_dropdown]
            )

            template_dropdown.change(
                fn=self.on_template_selected,
                inputs=[template_dropdown],
                outputs=[
                    template_docs,
                    input_fields_html,
                    submit_button,
                    current_template_state
                ]
            )
            
            submit_button.click(
                fn=self.on_submit_job_with_form,
                inputs=[
                    template_dropdown,
                    backend_dropdown,
                    routing_strategy_dropdown,
                    current_template_state
                ],
                outputs=[
                    job_id_output,
                    status_message
                ]
            )

        return tab

    def on_refresh_backends(self) -> gr.Dropdown:
        """
        Refresh the list of available backends.
        
        Returns:
            Updated dropdown component
        """
        choices = self._get_backend_choices()
        return gr.Dropdown(choices=choices)
    
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
    
    def _get_backend_choices(self) -> List[str]:
        """
        Get list of available backend IDs for dropdown.
        
        Returns:
            List of backend IDs with "auto" as the first option
        """
        try:
            # Always include "auto" as the first option for automatic routing
            choices = ["auto"]
            
            # Get registered backends from job service's backend router
            if hasattr(self.job_service, 'backend_router'):
                backends = self.job_service.backend_router.list_backends()
                backend_ids = [backend.id for backend in backends]
                choices.extend(sorted(backend_ids))
            else:
                # Fallback to common backend types if router not available
                choices.extend(["colab", "kaggle", "huggingface", "modal"])
            
            return choices
        except Exception as e:
            self.logger.error(f"Failed to retrieve backends: {e}")
            # Return default backend options
            return ["auto", "colab", "kaggle", "huggingface", "modal"]
    
    def on_template_selected(
        self,
        template_name: Optional[str]
    ) -> Tuple[str, str, gr.Button, Optional[Dict[str, Any]]]:
        """
        Handle template selection and return dynamic UI updates.
        
        This method:
        1. Fetches template metadata
        2. Generates template documentation
        3. Creates dynamic input fields HTML based on template schema
        4. Shows/hides UI elements appropriately
        
        Args:
            template_name: Selected template name
            
        Returns:
            Tuple of (
                template_docs: Markdown documentation string,
                input_fields_html: HTML string with input form fields,
                submit_button: Updated button visibility,
                current_template_state: Template metadata for submission
            )
        """
        if not template_name:
            return (
                "Select a template to view documentation",
                "<p style='color: #666; font-style: italic;'>Select a template to see input fields</p>",
                gr.Button(visible=False),
                None
            )
        
        try:
            # Fetch template metadata
            metadata = self.template_service.get_template_metadata(template_name)
            
            if not metadata:
                self.logger.error(f"Template '{template_name}' not found")
                return (
                    f"**Error:** Template '{template_name}' not found",
                    "<p style='color: red;'>Template not found</p>",
                    gr.Button(visible=False),
                    None
                )
            
            # Generate documentation
            docs = self._generate_template_docs(metadata)
            
            # Generate input fields HTML
            input_html = self._generate_input_fields_html(metadata)
            
            # Show submit button
            return (
                docs,
                input_html,
                gr.Button(visible=True),
                metadata
            )
            
        except Exception as e:
            self.logger.error(f"Error loading template '{template_name}': {e}")
            return (
                f"**Error:** Failed to load template: {str(e)}",
                f"<p style='color: red;'>Error: {str(e)}</p>",
                gr.Button(visible=False),
                None
            )
    
    def _generate_input_fields_html(self, metadata: Dict[str, Any]) -> str:
        """
        Generate HTML form fields for template inputs.
        
        Args:
            metadata: Template metadata dictionary
            
        Returns:
            HTML string with form input fields
        """
        if not metadata.get('inputs'):
            return "<p style='color: #666; font-style: italic;'>This template has no input parameters</p>"
        
        html_parts = []
        html_parts.append("<div style='padding: 10px;'>")
        
        for inp in metadata['inputs']:
            field_name = inp['name']
            field_type = inp['type']
            required = inp['required']
            description = inp.get('description', '')
            default = inp.get('default')
            
            # Field label
            required_marker = "<span style='color: red;'>*</span>" if required else ""
            html_parts.append(f"<div style='margin-bottom: 15px;'>")
            html_parts.append(
                f"<label style='display: block; font-weight: bold; margin-bottom: 5px;'>"
                f"{field_name} {required_marker}</label>"
            )
            html_parts.append(
                f"<p style='margin: 0 0 5px 0; font-size: 0.9em; color: #666;'>{description}</p>"
            )
            
            # Generate appropriate input field based on type
            if field_type in ['text', 'string']:
                default_val = default if default else ''
                html_parts.append(
                    f"<input type='text' id='input_{field_name}' name='{field_name}' "
                    f"value='{default_val}' "
                    f"style='width: 100%; padding: 8px; border: 1px solid #ccc; border-radius: 4px;' "
                    f"{'required' if required else ''}/>"
                )
            elif field_type == 'number':
                default_val = default if default is not None else ''
                html_parts.append(
                    f"<input type='number' id='input_{field_name}' name='{field_name}' "
                    f"value='{default_val}' "
                    f"style='width: 100%; padding: 8px; border: 1px solid #ccc; border-radius: 4px;' "
                    f"{'required' if required else ''}/>"
                )
            elif field_type in ['file', 'audio', 'image', 'video']:
                html_parts.append(
                    f"<input type='file' id='input_{field_name}' name='{field_name}' "
                    f"accept='{self._get_file_accept_types(field_type)}' "
                    f"style='width: 100%; padding: 8px; border: 1px solid #ccc; border-radius: 4px;' "
                    f"{'required' if required else ''}/>"
                )
                html_parts.append(
                    f"<p style='margin: 5px 0 0 0; font-size: 0.85em; color: #888;'>"
                    f"Or enter a URL:</p>"
                )
                html_parts.append(
                    f"<input type='url' id='input_{field_name}_url' name='{field_name}_url' "
                    f"placeholder='https://example.com/file.{field_type}' "
                    f"style='width: 100%; padding: 8px; border: 1px solid #ccc; border-radius: 4px;'/>"
                )
            elif field_type == 'json':
                default_val = default if default else '{}'
                html_parts.append(
                    f"<textarea id='input_{field_name}' name='{field_name}' "
                    f"rows='4' "
                    f"style='width: 100%; padding: 8px; border: 1px solid #ccc; border-radius: 4px; font-family: monospace;' "
                    f"{'required' if required else ''}>{default_val}</textarea>"
                )
            elif inp.get('options'):
                # Dropdown for fields with options
                html_parts.append(f"<select id='input_{field_name}' name='{field_name}' "
                               f"style='width: 100%; padding: 8px; border: 1px solid #ccc; border-radius: 4px;' "
                               f"{'required' if required else ''}>")
                if not required:
                    html_parts.append("<option value=''>-- Select --</option>")
                for option in inp['options']:
                    selected = 'selected' if option == default else ''
                    html_parts.append(f"<option value='{option}' {selected}>{option}</option>")
                html_parts.append("</select>")
            else:
                # Default to text input
                default_val = default if default else ''
                html_parts.append(
                    f"<input type='text' id='input_{field_name}' name='{field_name}' "
                    f"value='{default_val}' "
                    f"style='width: 100%; padding: 8px; border: 1px solid #ccc; border-radius: 4px;' "
                    f"{'required' if required else ''}/>"
                )
            
            html_parts.append("</div>")
        
        html_parts.append("</div>")
        html_parts.append(
            "<p style='margin-top: 10px; font-size: 0.9em; color: #666;'>"
            "<span style='color: red;'>*</span> Required fields</p>"
        )
        
        return "\n".join(html_parts)
    
    def _get_file_accept_types(self, field_type: str) -> str:
        """
        Get HTML accept attribute value for file input based on field type.
        
        Args:
            field_type: Field type (audio, image, video, file)
            
        Returns:
            Accept attribute value
        """
        accept_map = {
            'audio': 'audio/*,.mp3,.wav,.ogg,.flac',
            'image': 'image/*,.jpg,.jpeg,.png,.gif,.bmp,.webp',
            'video': 'video/*,.mp4,.avi,.mov,.wmv,.flv,.webm',
            'file': '*/*'
        }
        return accept_map.get(field_type, '*/*')
    
    def _generate_template_docs(self, metadata: Dict[str, Any]) -> str:
        """
        Generate markdown documentation for a template.
        
        Args:
            metadata: Template metadata dictionary
            
        Returns:
            Markdown formatted documentation string
        """
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
        
        # Input fields documentation
        if metadata.get('inputs'):
            docs_lines.append("**Input Fields:**")
            for inp in metadata['inputs']:
                required_str = " (required)" if inp['required'] else " (optional)"
                docs_lines.append(
                    f"- **{inp['name']}** ({inp['type']}){required_str}: {inp['description']}"
                )
                if inp.get('default') is not None:
                    docs_lines.append(f"  - Default: {inp['default']}")
                if inp.get('options'):
                    docs_lines.append(f"  - Options: {inp['options']}")
            docs_lines.append("")
        
        # Output fields documentation
        if metadata.get('outputs'):
            docs_lines.append("**Output Fields:**")
            for out in metadata['outputs']:
                docs_lines.append(
                    f"- **{out['name']}** ({out['type']}): {out['description']}"
                )
            docs_lines.append("")
        
        # Supported backends
        if metadata.get('supported_backends'):
            backends = ", ".join(metadata['supported_backends'])
            docs_lines.append(f"**Supported Backends:** {backends}")
        
        return "\n".join(docs_lines)
    
    def on_submit_job_with_form(
        self,
        template_name: Optional[str],
        backend: str,
        routing_strategy: str,
        template_metadata: Optional[Dict[str, Any]]
    ) -> Tuple[gr.Textbox, gr.Markdown]:
        """
        Handle job submission with form data extraction via JavaScript.
        
        Note: This is a simplified version. In a real implementation, you would
        need to use Gradio's JavaScript integration to extract form values.
        For now, we'll provide instructions to the user.
        
        Args:
            template_name: Selected template name
            backend: Selected backend (or "auto")
            routing_strategy: Routing strategy for automatic backend selection
            template_metadata: Template metadata from state
            
        Returns:
            Tuple of (job_id_output, status_message)
        """
        if not template_name or not template_metadata:
            error_msg = "❌ **Error:** Please select a template first"
            return (
                gr.Textbox(value="", visible=False),
                gr.Markdown(value=error_msg, visible=True)
            )
        
        # Note: In Gradio, we can't easily extract HTML form values directly
        # This is a limitation of the current approach
        # A better solution would be to use Gradio's native components
        
        info_msg = """
        ⚠️ **Note:** Due to Gradio limitations, form submission via HTML inputs requires a different approach.
        
        **Alternative Options:**
        1. Use the CLI to submit jobs: `python -m notebook_ml_orchestrator.cli submit --template {template} --input key=value`
        2. Use the API directly
        3. Wait for a future update that uses Gradio's native dynamic components
        
        **Template Selected:** `{template}`  
        **Backend:** `{backend}`  
        **Routing Strategy:** `{routing_strategy}`
        """.format(
            template=template_name,
            backend=backend,
            routing_strategy=routing_strategy
        )
        
        return (
            gr.Textbox(value="", visible=False),
            gr.Markdown(value=info_msg, visible=True)
        )
    
    def _validate_inputs_from_metadata(
        self,
        metadata: Dict[str, Any],
        inputs: Dict[str, Any]
    ) -> List[str]:
        """
        Validate inputs against template metadata.
        
        Args:
            metadata: Template metadata dictionary
            inputs: Input values to validate
            
        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        
        # Check required fields
        for inp in metadata.get('inputs', []):
            field_name = inp['name']
            
            if inp['required'] and field_name not in inputs:
                errors.append(f"Required field '{field_name}' is missing")
            
            # Type validation (basic)
            if field_name in inputs:
                value = inputs[field_name]
                field_type = inp['type']
                
                if value is None and inp['required']:
                    errors.append(f"Required field '{field_name}' cannot be empty")
                elif value is not None:
                    # Basic type checking
                    if field_type == "number" and not isinstance(value, (int, float)):
                        try:
                            float(value)
                        except (ValueError, TypeError):
                            errors.append(
                                f"Field '{field_name}' must be a number"
                            )
                    elif field_type in ["text", "string"] and not isinstance(value, str):
                        errors.append(
                            f"Field '{field_name}' must be text"
                        )
        
        return errors
    
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
            <span style="margin-left: 15px; font-size: 16px; color: #666;">Submitting job...</span>
        </div>
        """
