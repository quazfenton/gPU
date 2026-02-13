"""
Template Management Tab component for GUI interface.

This module provides the UI component for browsing and exploring available templates.
"""

from typing import Any, Dict, List, Optional, Tuple
import gradio as gr
import pandas as pd

from gui.services.template_service import TemplateService
from gui.error_handling import (
    format_generic_error,
    format_validation_error,
    create_success_message,
    sanitize_error_message
)
from notebook_ml_orchestrator.core.logging_config import LoggerMixin


class TemplateManagementTab(LoggerMixin):
    """UI component for browsing and managing templates."""
    
    def __init__(self, template_service: TemplateService):
        """
        Initialize template management tab.
        
        Args:
            template_service: Template service instance for template discovery
        """
        self.template_service = template_service
        self.logger.info("TemplateManagementTab initialized")
    
    def render(self) -> gr.Blocks:
        """
        Render the template management tab within a Gradio Blocks context.
        
        Returns:
            Gradio Blocks component containing the template management interface
        """
        with gr.Blocks() as tab:
            gr.Markdown("## Template Management")
            gr.Markdown("Browse and explore available ML templates.")
            
            with gr.Row():
                # Filter and search controls
                with gr.Column(scale=1):
                    gr.Markdown("### Filters")
                    
                    category_filter = gr.Radio(
                        label="Category",
                        choices=["All", "Audio", "Vision", "Language", "Multimodal"],
                        value="All",
                        interactive=True
                    )
                    
                    search_box = gr.Textbox(
                        label="Search",
                        placeholder="Search by name, category, or capability...",
                        interactive=True
                    )
                    
                    search_button = gr.Button("Search", variant="primary")
                    clear_button = gr.Button("Clear", variant="secondary")
                
                # Template list table
                with gr.Column(scale=2):
                    gr.Markdown("### Templates")
                    
                    template_list_table = gr.Dataframe(
                        headers=["Name", "Category", "Description", "GPU Required"],
                        datatype=["str", "str", "str", "str"],
                        interactive=False,
                        wrap=True,
                        value=self._get_initial_templates()
                    )
            
            # Template details section
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Template Details")
                    
                    selected_template_name = gr.Textbox(
                        label="Selected Template",
                        value="",
                        interactive=True,
                        placeholder="Enter template name or select from table"
                    )
                    
                    load_details_button = gr.Button("Load Details", variant="primary")
                    
                    # Template metadata panel
                    template_details_panel = gr.JSON(
                        label="Template Metadata",
                        value={}
                    )
            
            # Input/Output schema section
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Input Schema")
                    
                    input_schema_display = gr.JSON(
                        label="Input Fields",
                        value={}
                    )
                
                with gr.Column():
                    gr.Markdown("### Output Schema")
                    
                    output_schema_display = gr.JSON(
                        label="Output Fields",
                        value={}
                    )
            
            # Resource requirements and supported backends
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Resource Requirements")
                    
                    resource_requirements_display = gr.Markdown(
                        value="Select a template to view resource requirements."
                    )
                
                with gr.Column():
                    gr.Markdown("### Supported Backends")
                    
                    supported_backends_display = gr.Markdown(
                        value="Select a template to view supported backends."
                    )
            
            # Example usage code
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Example Usage")
                    
                    example_usage_code = gr.Code(
                        label="Python Example",
                        language="python",
                        value="# Select a template to view example usage",
                        interactive=False
                    )
            
            # Create Job button
            with gr.Row():
                create_job_button = gr.Button(
                    "Create Job with This Template",
                    variant="primary",
                    size="lg",
                    visible=False
                )
                
                create_job_message = gr.Textbox(
                    label="",
                    value="",
                    visible=False,
                    interactive=False
                )
            
            # Event handlers
            category_filter.change(
                fn=self.on_category_changed,
                inputs=[category_filter],
                outputs=[template_list_table]
            )
            
            search_button.click(
                fn=self.on_search,
                inputs=[search_box],
                outputs=[template_list_table]
            )
            
            clear_button.click(
                fn=self.on_clear_search,
                inputs=[],
                outputs=[search_box, template_list_table]
            )
            
            load_details_button.click(
                fn=self.on_load_template_details,
                inputs=[selected_template_name],
                outputs=[
                    template_details_panel,
                    input_schema_display,
                    output_schema_display,
                    resource_requirements_display,
                    supported_backends_display,
                    example_usage_code,
                    create_job_button,
                    create_job_message
                ]
            )
            
            create_job_button.click(
                fn=self.on_create_job,
                inputs=[selected_template_name],
                outputs=[create_job_message]
            )
            
            # Table row selection (if supported by Gradio)
            template_list_table.select(
                fn=self.on_template_selected_from_table,
                inputs=[template_list_table],
                outputs=[selected_template_name]
            )
        
        return tab
    
    def _get_initial_templates(self) -> pd.DataFrame:
        """
        Get initial template list for display.
        
        Returns:
            DataFrame with all templates
        """
        try:
            templates = self.template_service.get_templates()
            return self._templates_to_dataframe(templates)
        except Exception as e:
            self.logger.error(f"Failed to load initial templates: {e}")
            return self._get_empty_dataframe()
    
    def _get_empty_dataframe(self) -> pd.DataFrame:
        """
        Get empty dataframe with correct columns.
        
        Returns:
            Empty pandas DataFrame with template columns
        """
        return pd.DataFrame(columns=["Name", "Category", "Description", "GPU Required"])
    
    def _templates_to_dataframe(self, templates: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Convert template list to DataFrame format.
        
        Args:
            templates: List of template dictionaries
            
        Returns:
            DataFrame with template data
        """
        if not templates:
            return self._get_empty_dataframe()
        
        rows = []
        for template in templates:
            rows.append({
                "Name": template['name'],
                "Category": template['category'],
                "Description": template['description'],
                "GPU Required": "Yes" if template['gpu_required'] else "No"
            })
        
        return pd.DataFrame(rows)
    
    def on_category_changed(self, category: str) -> pd.DataFrame:
        """
        Handle category filter change.
        
        Args:
            category: Selected category ("All" or specific category)
            
        Returns:
            Updated DataFrame with filtered templates
        """
        try:
            # Convert "All" to None for service call
            category_filter = None if category == "All" else category.lower()
            
            templates = self.template_service.get_templates(category=category_filter)
            
            self.logger.debug(f"Category filter changed to '{category}', found {len(templates)} templates")
            
            return self._templates_to_dataframe(templates)
            
        except Exception as e:
            self.logger.error(f"Failed to filter templates by category '{category}': {e}")
            return self._get_empty_dataframe()
    
    def on_search(self, query: str) -> pd.DataFrame:
        """
        Handle search query.
        
        Args:
            query: Search query string
            
        Returns:
            Updated DataFrame with search results
        """
        try:
            templates = self.template_service.search_templates(query)
            
            self.logger.debug(f"Search query '{query}' returned {len(templates)} templates")
            
            return self._templates_to_dataframe(templates)
            
        except Exception as e:
            self.logger.error(f"Failed to search templates with query '{query}': {e}")
            return self._get_empty_dataframe()
    
    def on_clear_search(self) -> Tuple[str, pd.DataFrame]:
        """
        Clear search and reset to all templates.
        
        Returns:
            Tuple of (empty search box, all templates DataFrame)
        """
        try:
            templates = self.template_service.get_templates()
            return "", self._templates_to_dataframe(templates)
        except Exception as e:
            self.logger.error(f"Failed to clear search: {e}")
            return "", self._get_empty_dataframe()
    
    def on_load_template_details(
        self,
        template_name: str
    ) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], str, str, str, gr.Button, str]:
        """
        Load template details, schemas, and metadata.
        
        Args:
            template_name: Template name to load
            
        Returns:
            Tuple of (
                template_details,
                input_schema,
                output_schema,
                resource_requirements_markdown,
                supported_backends_markdown,
                example_usage_code,
                create_job_button,
                create_job_message
            )
        """
        if not template_name:
            return (
                {},
                {},
                {},
                "Select a template to view resource requirements.",
                "Select a template to view supported backends.",
                "# Select a template to view example usage",
                gr.Button(visible=False),
                ""
            )
        
        try:
            # Get template metadata
            metadata = self.template_service.get_template_metadata(template_name)
            
            if metadata is None:
                return (
                    {},
                    {},
                    {},
                    f"Template '{template_name}' not found.",
                    "",
                    "# Template not found",
                    gr.Button(visible=False),
                    ""
                )
            
            # Extract input schema
            input_schema = {}
            if 'inputs' in metadata:
                for input_field in metadata['inputs']:
                    input_schema[input_field['name']] = {
                        'type': input_field['type'],
                        'description': input_field.get('description', ''),
                        'required': input_field.get('required', False)
                    }
            
            # Extract output schema
            output_schema = {}
            if 'outputs' in metadata:
                for output_field in metadata['outputs']:
                    output_schema[output_field['name']] = {
                        'type': output_field['type'],
                        'description': output_field.get('description', '')
                    }
            
            # Format resource requirements
            resource_md = self._format_resource_requirements(metadata)
            
            # Format supported backends
            backends_md = self._format_supported_backends(metadata)
            
            # Generate example usage code
            example_code = self._generate_example_usage(template_name, metadata)
            
            # Prepare template details (summary info)
            template_details = {
                'name': metadata['name'],
                'category': metadata['category'],
                'description': metadata['description'],
                'version': metadata['version'],
                'gpu_required': metadata['gpu_required'],
                'memory_mb': metadata['memory_mb'],
                'timeout_sec': metadata['timeout_sec']
            }
            
            return (
                template_details,
                input_schema,
                output_schema,
                resource_md,
                backends_md,
                example_code,
                gr.Button(visible=True),
                ""
            )
            
        except Exception as e:
            self.logger.error(f"Failed to load template details for '{template_name}': {e}")
            return (
                {},
                {},
                {},
                f"Error loading template: {str(e)}",
                "",
                "# Error loading template",
                gr.Button(visible=False),
                ""
            )
    
    def on_template_selected_from_table(
        self,
        table_data: pd.DataFrame
    ) -> str:
        """
        Handle template selection from table.
        
        Args:
            table_data: Selected row data from table
            
        Returns:
            Template name from selected row
        """
        try:
            if table_data is not None and len(table_data) > 0:
                # First column is template name
                return str(table_data.iloc[0, 0])
        except Exception as e:
            self.logger.error(f"Failed to extract template name from table selection: {e}")
        
        return ""
    
    def on_create_job(self, template_name: str) -> str:
        """
        Handle "Create Job" button click.
        
        This provides feedback to the user. In a full implementation,
        this would navigate to the Job Submission tab with the template pre-selected.
        
        Args:
            template_name: Template name to create job with
            
        Returns:
            Message to display to user
        """
        if not template_name:
            return "⚠️ Please select a template first."
        
        # In a full implementation, this would trigger navigation to Job Submission tab
        # For now, provide a message
        return (
            f"ℹ️ To create a job with template **'{template_name}'**, "
            f"please navigate to the **Job Submission** tab and select this template."
        )
    
    def _format_resource_requirements(self, metadata: Dict[str, Any]) -> str:
        """
        Format resource requirements as markdown.
        
        Args:
            metadata: Template metadata dictionary
            
        Returns:
            Markdown formatted resource requirements
        """
        lines = []
        
        # GPU requirements
        if metadata.get('gpu_required', False):
            gpu_type = metadata.get('gpu_type', 'Any')
            lines.append(f"**GPU Required:** Yes ({gpu_type})")
        else:
            lines.append("**GPU Required:** No")
        
        # Memory requirements
        memory_mb = metadata.get('memory_mb', 0)
        if memory_mb > 0:
            memory_gb = memory_mb / 1024
            lines.append(f"**Memory:** {memory_gb:.1f} GB")
        
        # Timeout
        timeout_sec = metadata.get('timeout_sec', 0)
        if timeout_sec > 0:
            timeout_min = timeout_sec / 60
            lines.append(f"**Timeout:** {timeout_min:.0f} minutes")
        
        # Pip packages
        pip_packages = metadata.get('pip_packages', [])
        if pip_packages:
            lines.append(f"**Required Packages:** {', '.join(pip_packages)}")
        
        return "\n\n".join(lines) if lines else "No specific requirements."
    
    def _format_supported_backends(self, metadata: Dict[str, Any]) -> str:
        """
        Format supported backends as markdown.
        
        Args:
            metadata: Template metadata dictionary
            
        Returns:
            Markdown formatted supported backends
        """
        backends = metadata.get('supported_backends', [])
        
        if not backends:
            return "No backends currently support this template."
        
        lines = ["**Supported Backends:**"]
        for backend in backends:
            lines.append(f"- {backend}")
        
        return "\n".join(lines)
    
    def _generate_example_usage(self, template_name: str, metadata: Dict[str, Any]) -> str:
        """
        Generate example usage code.
        
        Args:
            template_name: Template name
            metadata: Template metadata dictionary
            
        Returns:
            Python code example
        """
        lines = [
            "# Example: Submit a job using this template",
            "",
            "from notebook_ml_orchestrator.core.job_queue import JobQueue",
            "",
            "# Initialize job queue",
            "job_queue = JobQueue()",
            "",
            "# Define job inputs"
        ]
        
        # Generate example inputs based on schema
        inputs = metadata.get('inputs', [])
        if inputs:
            lines.append("inputs = {")
            for input_field in inputs:
                field_name = input_field['name']
                field_type = input_field['type']
                
                # Generate example value based on type
                if field_type == 'string':
                    example_value = f'"example_{field_name}"'
                elif field_type == 'integer':
                    example_value = "42"
                elif field_type == 'float':
                    example_value = "3.14"
                elif field_type == 'boolean':
                    example_value = "True"
                elif field_type == 'file':
                    example_value = '"/path/to/file"'
                else:
                    example_value = '"value"'
                
                lines.append(f'    "{field_name}": {example_value},')
            lines.append("}")
        else:
            lines.append("inputs = {}")
        
        lines.extend([
            "",
            "# Submit job",
            f'job_id = job_queue.submit_job(',
            f'    template="{template_name}",',
            '    inputs=inputs',
            ')',
            "",
            'print(f"Job submitted: {job_id}")'
        ])
        
        return "\n".join(lines)
