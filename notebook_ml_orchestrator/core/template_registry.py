"""
Template Registry for discovering, validating, and managing ML templates.

The Template Registry is responsible for:
- Discovering all templates in the templates directory
- Validating that templates inherit from the Template base class
- Registering templates with their metadata
- Providing thread-safe access to registered templates
- Handling errors gracefully during discovery
"""

import importlib
import importlib.util
import inspect
import logging
import os
import sys
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

# Import Template base class
from templates.base import Template

logger = logging.getLogger(__name__)


class TemplateRegistry:
    """
    Central registry for ML templates.
    Discovers, validates, and manages template instances.
    """
    
    def __init__(self, templates_dir: str = "templates"):
        """
        Initialize the Template Registry.
        
        Args:
            templates_dir: Path to the directory containing template modules
        """
        self.templates_dir = templates_dir
        self.templates: Dict[str, Template] = {}
        self.templates_by_category: Dict[str, List[str]] = {}
        self._lock = threading.RLock()
        self.failed_templates: List[str] = []
        
        logger.info(f"Initialized TemplateRegistry with templates_dir: {templates_dir}")
    
    def discover_templates(self) -> int:
        """
        Discover all templates in the templates directory.
        
        Scans the templates directory for Python files, imports them,
        and registers any valid Template subclasses found.
        
        Returns:
            The number of templates successfully discovered and registered
        """
        discovered_count = 0
        
        with self._lock:
            templates_path = Path(self.templates_dir)
            
            if not templates_path.exists():
                logger.error(f"Templates directory does not exist: {self.templates_dir}")
                return 0
            
            if not templates_path.is_dir():
                logger.error(f"Templates path is not a directory: {self.templates_dir}")
                return 0
            
            logger.info(f"Discovering templates in: {templates_path.absolute()}")
            
            # Find all Python files in the templates directory
            python_files = list(templates_path.glob("*.py"))
            logger.info(f"Found {len(python_files)} Python files to scan")
            
            for py_file in python_files:
                # Skip __init__.py and base.py
                if py_file.name in ["__init__.py", "base.py"]:
                    continue
                
                try:
                    # Import the module
                    module_name = py_file.stem
                    spec = importlib.util.spec_from_file_location(
                        f"templates.{module_name}",
                        py_file
                    )
                    
                    if spec is None or spec.loader is None:
                        logger.warning(f"Could not load spec for {py_file.name}")
                        self.failed_templates.append(py_file.name)
                        continue
                    
                    module = importlib.util.module_from_spec(spec)
                    sys.modules[f"templates.{module_name}"] = module
                    spec.loader.exec_module(module)
                    
                    # Find all Template subclasses in the module
                    for name, obj in inspect.getmembers(module, inspect.isclass):
                        # Check if it's a Template subclass (but not Template itself)
                        if (issubclass(obj, Template) and 
                            obj is not Template and
                            obj.__module__ == module.__name__):
                            
                            try:
                                # Instantiate and register the template
                                template_instance = obj()
                                
                                if self.register_template(template_instance):
                                    discovered_count += 1
                                    logger.info(f"Discovered and registered template: {template_instance.name}")
                                else:
                                    logger.warning(f"Failed to register template from {py_file.name}: {name}")
                                    self.failed_templates.append(f"{py_file.name}::{name}")
                            
                            except Exception as e:
                                logger.error(f"Error instantiating template {name} from {py_file.name}: {e}")
                                self.failed_templates.append(f"{py_file.name}::{name}")
                                continue
                
                except Exception as e:
                    logger.error(f"Error loading template file {py_file.name}: {e}")
                    self.failed_templates.append(py_file.name)
                    continue
            
            logger.info(f"Template discovery complete. Registered: {discovered_count}, Failed: {len(self.failed_templates)}")
            
            if self.failed_templates:
                logger.warning(f"Failed templates: {', '.join(self.failed_templates)}")
        
        return discovered_count
    
    def register_template(self, template: Template) -> bool:
        """
        Register a template instance.
        
        Validates the template and adds it to the registry if valid.
        Handles duplicate names by keeping the first registered template.
        
        Args:
            template: Template instance to register
            
        Returns:
            True if successful, False otherwise
        """
        with self._lock:
            try:
                # Validate the template
                if not self.validate_template(template):
                    logger.error(f"Template validation failed for: {template.name}")
                    return False
                
                # Check for duplicate names
                if template.name in self.templates:
                    logger.warning(
                        f"Template with name '{template.name}' already registered. "
                        f"Skipping duplicate."
                    )
                    return False
                
                # Register the template
                self.templates[template.name] = template
                
                # Add to category index
                category = template.category
                if category not in self.templates_by_category:
                    self.templates_by_category[category] = []
                self.templates_by_category[category].append(template.name)
                
                logger.info(f"Successfully registered template: {template.name} (category: {category})")
                return True
            
            except Exception as e:
                logger.error(f"Error registering template: {e}")
                return False
    
    def validate_template(self, template: Template) -> bool:
        """
        Validate that a template meets requirements.
        
        Checks:
        - Template inherits from Template base class
        - Required metadata fields are present and non-empty
        - Input and output fields are properly defined
        - Resource requirements are valid
        
        Args:
            template: Template instance to validate
            
        Returns:
            True if valid, False otherwise
        """
        try:
            # Check inheritance
            if not isinstance(template, Template):
                logger.error(f"Template does not inherit from Template base class")
                return False
            
            # Check required metadata fields
            required_fields = {
                "name": str,
                "category": str,
                "description": str,
                "version": str,
            }
            
            for field_name, field_type in required_fields.items():
                if not hasattr(template, field_name):
                    logger.error(f"Template missing required field: {field_name}")
                    return False
                
                value = getattr(template, field_name)
                if not isinstance(value, field_type):
                    logger.error(f"Template field {field_name} has wrong type: {type(value)} (expected {field_type})")
                    return False
                
                if isinstance(value, str) and not value.strip():
                    logger.error(f"Template field {field_name} is empty")
                    return False
            
            # Check resource requirements
            if not hasattr(template, "memory_mb") or template.memory_mb <= 0:
                logger.error(f"Template has invalid memory_mb: {getattr(template, 'memory_mb', None)}")
                return False
            
            if not hasattr(template, "timeout_sec") or template.timeout_sec <= 0:
                logger.error(f"Template has invalid timeout_sec: {getattr(template, 'timeout_sec', None)}")
                return False
            
            # Check pip_packages is a list
            if not hasattr(template, "pip_packages") or not isinstance(template.pip_packages, list):
                logger.error(f"Template pip_packages must be a list")
                return False
            
            # Validate input fields
            if hasattr(template, "inputs") and template.inputs:
                for inp in template.inputs:
                    if not hasattr(inp, "name") or not inp.name:
                        logger.error(f"Input field missing name")
                        return False
                    if not hasattr(inp, "type") or not inp.type:
                        logger.error(f"Input field '{inp.name}' missing type")
                        return False
                    if not hasattr(inp, "description"):
                        logger.error(f"Input field '{inp.name}' missing description")
                        return False
                    if not hasattr(inp, "required"):
                        logger.error(f"Input field '{inp.name}' missing required flag")
                        return False
            
            # Validate output fields
            if hasattr(template, "outputs") and template.outputs:
                for out in template.outputs:
                    if not hasattr(out, "name") or not out.name:
                        logger.error(f"Output field missing name")
                        return False
                    if not hasattr(out, "type") or not out.type:
                        logger.error(f"Output field '{out.name}' missing type")
                        return False
                    if not hasattr(out, "description"):
                        logger.error(f"Output field '{out.name}' missing description")
                        return False
            
            # Validate GPU requirements
            if hasattr(template, "gpu_required") and template.gpu_required:
                if not hasattr(template, "gpu_type") or not template.gpu_type:
                    logger.error(f"Template requires GPU but gpu_type is not specified")
                    return False
                
                valid_gpu_types = ["T4", "A10G", "A100"]
                if template.gpu_type not in valid_gpu_types:
                    logger.error(f"Template has invalid gpu_type: {template.gpu_type} (must be one of {valid_gpu_types})")
                    return False
            
            return True
        
        except Exception as e:
            logger.error(f"Error validating template: {e}")
            return False
    
    def get_template(self, name: str) -> Optional[Template]:
        """
        Get a template by name.
        
        Args:
            name: Template name
            
        Returns:
            Template instance if found, None otherwise
        """
        with self._lock:
            return self.templates.get(name)
    
    def list_templates(self, category: Optional[str] = None) -> List[Template]:
        """
        List all templates, optionally filtered by category.
        
        Args:
            category: Optional category to filter by
            
        Returns:
            List of template instances
        """
        with self._lock:
            if category is None:
                return list(self.templates.values())
            
            # Get template names for this category
            template_names = self.templates_by_category.get(category, [])
            return [self.templates[name] for name in template_names if name in self.templates]
    
    def get_template_metadata(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata for a specific template.
        
        Args:
            name: Template name
            
        Returns:
            Dictionary containing template metadata, or None if not found
        """
        with self._lock:
            template = self.templates.get(name)
            if template is None:
                return None
            
            return template.to_dict()
    
    def get_all_categories(self) -> List[str]:
        """
        Get list of all template categories.
        
        Returns:
            List of category names
        """
        with self._lock:
            return list(self.templates_by_category.keys())
    
    def get_registry_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the registry.
        
        Returns:
            Dictionary with registry statistics
        """
        with self._lock:
            return {
                "total_templates": len(self.templates),
                "categories": len(self.templates_by_category),
                "templates_by_category": {
                    cat: len(names) for cat, names in self.templates_by_category.items()
                },
                "failed_templates": len(self.failed_templates),
                "failed_template_list": self.failed_templates.copy(),
            }
