"""
Template discovery and registration.

Automatically discovers all Template subclasses in the templates directory
and provides a registry for easy access.
"""

import os
import sys
import importlib
import importlib.util
import inspect
from pathlib import Path
from typing import Dict, List, Optional, Type

from .base import Template, RouteType, InputField, OutputField


# Template registry
_templates: Optional[Dict[str, Template]] = None


def discover_templates(templates_dir: Optional[Path] = None) -> Dict[str, Template]:
    """
    Discover and instantiate all Template subclasses in the templates directory.

    Args:
        templates_dir: Optional path to templates directory. Defaults to this directory.

    Returns:
        Dict mapping template names to Template instances.
    """
    global _templates

    if templates_dir is None:
        templates_dir = Path(__file__).parent

    discovered_templates = {}

    # Scan all Python files in templates directory
    for py_file in templates_dir.glob("*.py"):
        if py_file.name.startswith("_"):
            continue
        if py_file.name == "base.py":
            continue

        module_name = py_file.stem

        try:
            # Load module
            spec = importlib.util.spec_from_file_location(
                f"templates.{module_name}",
                py_file
            )
            if spec is None or spec.loader is None:
                continue

            module = importlib.util.module_from_spec(spec)
            sys.modules[f"templates.{module_name}"] = module
            spec.loader.exec_module(module)

            # Find Template subclasses
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if (issubclass(obj, Template) and
                    obj is not Template and
                    not inspect.isabstract(obj)):
                    try:
                        instance = obj()
                        discovered_templates[instance.name] = instance
                    except Exception as e:
                        print(f"Warning: Could not instantiate {name}: {e}")

        except Exception as e:
            print(f"Warning: Could not load {py_file.name}: {e}")

    # Cache the discovered templates
    _templates = discovered_templates
    return _templates


def get_templates() -> Dict[str, Template]:
    """Get all discovered templates."""
    global _templates
    if _templates is None:
        discover_templates()
    return _templates


def get_template(name: str) -> Optional[Template]:
    """Get a template by name."""
    templates = get_templates()
    return templates.get(name)


def list_templates() -> List[str]:
    """List all template names."""
    return list(get_templates().keys())


def get_templates_by_category(category: str) -> Dict[str, Template]:
    """Get all templates in a category."""
    return {
        name: tpl 
        for name, tpl in get_templates().items() 
        if tpl.category.lower() == category.lower()
    }


def get_categories() -> List[str]:
    """Get list of all unique categories."""
    return list(set(tpl.category for tpl in get_templates().values()))


def register_template(template: Template) -> None:
    """Manually register a template."""
    global _templates
    _templates[template.name] = template


# Export public API
__all__ = [
    "Template",
    "RouteType", 
    "InputField",
    "OutputField",
    "discover_templates",
    "get_templates",
    "get_template",
    "list_templates",
    "get_templates_by_category",
    "get_categories",
    "register_template",
]
