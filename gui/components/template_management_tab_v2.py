"""
Template Management Tab component - V2 with real implementation.

This module provides the UI component for browsing and managing templates.
"""

from typing import Any, Dict, List, Optional
import gradio as gr
import pandas as pd

from gui.services.template_service import TemplateService
from notebook_ml_orchestrator.core.logging_config import LoggerMixin


class TemplateManagementTabV2(LoggerMixin):
    """UI component for browsing and managing templates."""
    
    def __init__(self, template_service: TemplateService):
        """Initialize template management tab."""
        self.template_service = template_service
        self.logger.info("TemplateManagementTabV2 initialized")
    
    def render(self) -> gr.Blocks:
        """Render the template management tab."""
        with gr.Blocks() as tab:
            gr.Markdown("## Template Management")
            gr.Markdown("Browse and explore available ML templates.")
            
            with gr.Row():
                # Category filter
                category_filter = gr.Radio(
                    label="Category",
                    choices=["All", "Audio", "Vision", "Language", "Multimodal", "Test"],
                    value="All",
                    interactive=True
                )
                
                # Search box
                search_box = gr.Textbox(
                    label="Search Templates",
                    placeholder="Search by name or description...",
                    interactive=True
                )
            
            # Template list
            template_list_df = gr.Dataframe(
                label="Templates",
                headers=["Name", "Category", "Description", "GPU Required"],
                datatype=["str", "str", "str", "str"],
                interactive=False,
                wrap=True
            )
            
            # Template details
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Template Details")
                    template_details = gr.Markdown(
                        value="Select a template to view details"
                    )
                
                with gr.Column():
                    gr.Markdown("### Input/Output Schema")
                    template_schema = gr.Markdown(
                        value="Select a template to view schema"
                    )
            
            # Event handlers
            def load_templates(category, search_query):
                return self.get_templates_table(category, search_query)
            
            # Load templates on tab open
            tab.load(
                fn=lambda: self.get_templates_table("All", ""),
                outputs=[template_list_df]
            )
            
            # Category filter change
            category_filter.change(
                fn=load_templates,
                inputs=[category_filter, search_box],
                outputs=[template_list_df]
            )
            
            # Search box change
            search_box.change(
                fn=load_templates,
                inputs=[category_filter, search_box],
                outputs=[template_list_df]
            )
        
        return tab
    
    def get_templates_table(
        self,
        category: str = "All",
        search_query: str = ""
    ) -> pd.DataFrame:
        """Get templates as DataFrame."""
        try:
            # Get templates from service
            category_filter = None if category == "All" else category
            templates = self.template_service.get_templates(category=category_filter)
            
            # Apply search filter
            if search_query:
                search_lower = search_query.lower()
                templates = [
                    t for t in templates
                    if search_lower in t['name'].lower() or
                       search_lower in t.get('description', '').lower()
                ]
            
            if not templates:
                return pd.DataFrame(columns=[
                    "Name", "Category", "Description", "GPU Required"
                ])
            
            # Convert to DataFrame
            rows = []
            for t in templates:
                rows.append([
                    t['name'],
                    t['category'],
                    t.get('description', 'No description')[:100] + "...",
                    "Yes" if t.get('gpu_required', False) else "No"
                ])
            
            return pd.DataFrame(rows, columns=[
                "Name", "Category", "Description", "GPU Required"
            ])
            
        except Exception as e:
            self.logger.error(f"Failed to load templates: {e}")
            return pd.DataFrame(columns=[
                "Name", "Category", "Description", "GPU Required"
            ])
