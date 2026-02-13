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
                        value="Select a template to view documentation",
                        label="Template Documentation"
                    )
                
                with gr.Column(scale=1):
                    # Dynamic input fields container
                    input_fields_container = gr.Column(visible=False)
                    
                    with input_fields_container:
                        gr.Markdown("### Input Parameters")
                        
                        # Placeholder for dynamic input fields
                        # These will be created dynamically based on template selection
                        dynamic_inputs = gr.State(value={})
                        input_components = gr.State(value={})
                    
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
                    
                    # Loading indicator
                    loading_indicator = gr.HTML(
                        value="",
                        visible=False
                    )
            
            # Event handlers
            template_dropdown.change(
                fn=self.on_template_selected,
                inputs=[template_dropdown],
                outputs=[
                    template_docs,
                    input_fields_container,
                    submit_button,
                    dynamic_inputs,
                    input_components
                ]
            )
            
            submit_button.click(
                fn=self.on_submit_job,
                inputs=[
                    template_dropdown,
                    backend_dropdown,
                    routing_strategy_dropdown,
                    dynamic_inputs
                ],
                outputs=[
                    job_id_output,
                    status_message,
                    loading_indicator
                ]
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
    ) -> Tuple[str, gr.Column, gr.Button, Dict[str, Any], Dict[str, Any]]:
        """
        Handle template selection and return dynamic UI updates.
        
        This method:
        1. Fetches template metadata
        2. Generates template documentation
        3. Creates dynamic input fields based on template schema
        4. Shows/hides UI elements appropriately
        
        Args:
            template_name: Selected template name
            
        Returns:
            Tuple of (
                template_docs: Markdown documentation string,
                input_fields_container: Updated container visibility,
                submit_button: Updated button visibility,
                dynamic_inputs: State dict for input values,
                input_components: State dict for input component references
            )
        """
        if not template_name:
            return (
                "Select a template to view documentation",
                gr.Column(visible=False),
                gr.Button(visible=False),
                {},
                {}
            )
        
        try:
            # Fetch template metadata
            metadata = self.template_service.get_template_metadata(template_name)
            
            if not metadata:
                self.logger.error(f"Template '{template_name}' not found")
                return (
                    f"**Error:** Template '{template_name}' not found",
                    gr.Column(visible=False),
                    gr.Button(visible=False),
                    {},
                    {}
                )
            
            # Generate documentation
            docs = self._generate_template_docs(metadata)
            
            # Initialize input state
            dynamic_inputs = {}
            input_components = {}
            
            # Show input container and submit button
            return (
                docs,
                gr.Column(visible=True),
                gr.Button(visible=True),
                dynamic_inputs,
                input_components
            )
            
        except Exception as e:
            self.logger.error(f"Error loading template '{template_name}': {e}")
            return (
                f"**Error:** Failed to load template: {str(e)}",
                gr.Column(visible=False),
                gr.Button(visible=False),
                {},
                {}
            )
    
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
    
    def on_submit_job(
        self,
        template_name: Optional[str],
        backend: str,
        routing_strategy: str,
        dynamic_inputs: Dict[str, Any]
    ) -> Tuple[gr.Textbox, gr.Markdown, gr.HTML]:
        """
        Handle job submission.
        
        This method:
        1. Validates inputs against template schema
        2. Submits job to job queue
        3. Returns job ID and status message
        
        Args:
            template_name: Selected template name
            backend: Selected backend (or "auto")
            routing_strategy: Routing strategy for automatic backend selection
            dynamic_inputs: Dictionary of input values from dynamic fields
            
        Returns:
            Tuple of (
                job_id_output: Updated textbox with job ID,
                status_message: Updated markdown with status/error message,
                loading_indicator: Loading indicator HTML
            )
        """
        if not template_name:
            error = format_validation_error(
                "template",
                "Please select a template"
            )
            return (
                gr.Textbox(value="", visible=False),
                gr.Markdown(value=error.to_markdown(), visible=True),
                gr.HTML(value="", visible=False)
            )
        
        try:
            # Get template metadata for validation
            metadata = self.template_service.get_template_metadata(template_name)
            
            if not metadata:
                error = format_validation_error(
                    "template",
                    f"Template '{template_name}' not found"
                )
                return (
                    gr.Textbox(value="", visible=False),
                    gr.Markdown(value=error.to_markdown(), visible=True),
                    gr.HTML(value="", visible=False)
                )
            
            # Validate inputs
            validation_errors = self._validate_inputs_from_metadata(
                metadata,
                dynamic_inputs
            )
            
            if validation_errors:
                error_msg = "❌ **Validation Errors:**\n\n" + "\n".join(
                    f"- {err}" for err in validation_errors
                )
                return (
                    gr.Textbox(value="", visible=False),
                    gr.Markdown(value=error_msg, visible=True),
                    gr.HTML(value="", visible=False)
                )
            
            # Determine backend
            backend_id = None if backend == "auto" else backend
            
            # Submit job with routing strategy
            job_id = self.job_service.submit_job(
                template_name=template_name,
                inputs=dynamic_inputs,
                backend=backend_id,
                routing_strategy=routing_strategy
            )
            
            success_msg = create_success_message(
                "Job submitted successfully",
                {
                    "Job ID": f"`{job_id}`",
                    "Template": template_name,
                    "Backend": backend,
                    "Routing Strategy": routing_strategy,
                    "Next Step": "Monitor the job in the **Job Monitoring** tab"
                }
            )
            
            return (
                gr.Textbox(value=job_id, visible=True),
                gr.Markdown(value=success_msg, visible=True),
                gr.HTML(value="", visible=False)
            )
            
        except ValueError as e:
            # Validation errors
            self.logger.error(f"Validation error during job submission: {e}")
            error = format_validation_error(
                "job_submission",
                sanitize_error_message(str(e))
            )
            return (
                gr.Textbox(value="", visible=False),
                gr.Markdown(value=error.to_markdown(), visible=True),
                gr.HTML(value="", visible=False)
            )
        except Exception as e:
            # Generic errors
            self.logger.error(f"Job submission failed: {e}", exc_info=True)
            error = format_generic_error(e, "Job submission failed")
            return (
                gr.Textbox(value="", visible=False),
                gr.Markdown(value=error.to_markdown(), visible=True),
                gr.HTML(value="", visible=False)
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
