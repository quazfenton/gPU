"""
Test template for validating the Template Registry.
This is a minimal template used for testing purposes.
"""

from typing import Any, Dict
from templates.base import Template, InputField, OutputField, RouteType


class TestTemplate(Template):
    """A simple test template for registry validation."""
    
    name = "test-template"
    category = "Test"
    description = "A test template for validating the registry"
    version = "1.0.0"
    
    inputs = [
        InputField(
            name="text",
            type="text",
            description="Input text",
            required=True
        )
    ]
    
    outputs = [
        OutputField(
            name="result",
            type="text",
            description="Output result"
        )
    ]
    
    routing = [RouteType.LOCAL]
    gpu_required = False
    memory_mb = 512
    timeout_sec = 60
    pip_packages = []
    
    def run(self, **kwargs) -> Dict[str, Any]:
        """Execute the test template."""
        text = kwargs.get("text", "")
        return {"result": f"Processed: {text}"}
