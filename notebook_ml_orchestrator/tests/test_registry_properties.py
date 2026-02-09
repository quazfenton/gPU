"""
Property-based tests for Template Registry.

This module implements property-based tests using Hypothesis to verify
that the Template Registry correctly discovers, validates, and manages
ML templates according to the design specifications.

**Validates: Requirements 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 6.6, 6.7**
"""

import pytest
import tempfile
import os
from pathlib import Path
from typing import Dict, Any, List, Optional

from hypothesis import given, strategies as st, settings, assume, example
from hypothesis.stateful import RuleBasedStateMachine, Bundle, rule, initialize, invariant

from notebook_ml_orchestrator.core.template_registry import TemplateRegistry
from templates.base import Template, InputField, OutputField, RouteType


# ============================================================================
# Test Template Classes
# ============================================================================

class ValidAudioTemplate(Template):
    """Valid audio template for testing."""
    name = "audio-test"
    category = "Audio"
    description = "Audio test template"
    version = "1.0.0"
    inputs = [InputField(name="audio", type="audio", description="Audio input", required=True)]
    outputs = [OutputField(name="text", type="text", description="Text output")]
    routing = [RouteType.LOCAL]
    gpu_required = True
    gpu_type = "T4"
    memory_mb = 2048
    timeout_sec = 300
    pip_packages = ["torch"]
    
    def run(self, **kwargs):
        return {"text": "transcribed"}


class ValidVisionTemplate(Template):
    """Valid vision template for testing."""
    name = "vision-test"
    category = "Vision"
    description = "Vision test template"
    version = "1.0.0"
    inputs = [InputField(name="image", type="image", description="Image input", required=True)]
    outputs = [OutputField(name="detections", type="json", description="Detection results")]
    routing = [RouteType.LOCAL]
    gpu_required = True
    gpu_type = "T4"
    memory_mb = 4096
    timeout_sec = 300
    pip_packages = ["opencv-python"]
    
    def run(self, **kwargs):
        return {"detections": []}


class ValidLanguageTemplate(Template):
    """Valid language template for testing."""
    name = "language-test"
    category = "Language"
    description = "Language test template"
    version = "1.0.0"
    inputs = [InputField(name="text", type="text", description="Text input", required=True)]
    outputs = [OutputField(name="entities", type="json", description="Extracted entities")]
    routing = [RouteType.LOCAL]
    gpu_required = False
    memory_mb = 1024
    timeout_sec = 60
    pip_packages = ["spacy"]
    
    def run(self, **kwargs):
        return {"entities": []}


class NotATemplate:
    """Class that does NOT inherit from Template."""
    name = "not-a-template"
    
    def run(self):
        return {}


class InvalidTemplateNoName(Template):
    """Invalid template missing name."""
    name = ""  # Empty name
    category = "Test"
    description = "Invalid"
    version = "1.0.0"
    memory_mb = 512
    timeout_sec = 60
    pip_packages = []
    
    def run(self, **kwargs):
        return {}


class InvalidTemplateNoCategory(Template):
    """Invalid template missing category."""
    name = "invalid-no-category"
    category = ""  # Empty category
    description = "Invalid"
    version = "1.0.0"
    memory_mb = 512
    timeout_sec = 60
    pip_packages = []
    
    def run(self, **kwargs):
        return {}


class InvalidTemplateInvalidMemory(Template):
    """Invalid template with invalid memory."""
    name = "invalid-memory"
    category = "Test"
    description = "Invalid"
    version = "1.0.0"
    memory_mb = 0  # Invalid
    timeout_sec = 60
    pip_packages = []
    
    def run(self, **kwargs):
        return {}


class InvalidTemplateGPUNoType(Template):
    """Invalid template requiring GPU but missing gpu_type."""
    name = "invalid-gpu"
    category = "Test"
    description = "Invalid GPU template"
    version = "1.0.0"
    gpu_required = True
    gpu_type = None  # Missing
    memory_mb = 2048
    timeout_sec = 300
    pip_packages = []
    
    def run(self, **kwargs):
        return {}


# ============================================================================
# Hypothesis Strategies
# ============================================================================

def valid_template_strategy():
    """Strategy for generating valid template instances."""
    templates = [
        ValidAudioTemplate,
        ValidVisionTemplate,
        ValidLanguageTemplate,
    ]
    return st.sampled_from(templates).map(lambda cls: cls())


def invalid_template_strategy():
    """Strategy for generating invalid template instances."""
    templates = [
        InvalidTemplateNoName,
        InvalidTemplateNoCategory,
        InvalidTemplateInvalidMemory,
        InvalidTemplateGPUNoType,
    ]
    return st.sampled_from(templates).map(lambda cls: cls())


def non_template_class_strategy():
    """Strategy for generating non-Template classes."""
    return st.just(NotATemplate)


def category_strategy():
    """Strategy for generating category names."""
    return st.sampled_from(["Audio", "Vision", "Language", "Multimodal", "Test"])


def template_name_strategy():
    """Strategy for generating template names."""
    return st.text(min_size=1, max_size=50, alphabet=st.characters(
        whitelist_categories=('Lu', 'Ll', 'Nd'), 
        whitelist_characters='-_'
    ))


# ============================================================================
# Property 9: Template inheritance validation
# ============================================================================

class TestProperty9TemplateInheritanceValidation:
    """
    Property 9: Template inheritance validation
    
    For any Python class discovered in the templates directory, if it does not 
    inherit from the Template base class, it SHALL NOT be registered in the 
    Template_Registry.
    
    **Validates: Requirements 5.2**
    """
    
    @settings(max_examples=100)
    @given(non_template_class_strategy())
    def test_non_template_class_not_registered(self, non_template_class):
        """Non-Template classes should not be registered."""
        registry = TemplateRegistry()
        
        # Try to instantiate and register
        try:
            instance = non_template_class()
            # This should fail validation because it's not a Template
            result = registry.register_template(instance)
            assert result is False, "Non-Template class should not be registered"
        except (TypeError, AttributeError):
            # Expected - non-Template classes can't be registered
            pass
        
        # Verify nothing was registered
        assert len(registry.templates) == 0
    
    def test_template_inheritance_validation_with_temp_directory(self):
        """Test that non-Template classes in files are not registered during discovery."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a file with a non-Template class
            non_template_file = Path(tmpdir) / "not_a_template.py"
            non_template_file.write_text("""
class NotATemplate:
    name = "not-a-template"
    
    def run(self):
        return {}
""")
            
            # Create a file with a valid Template class
            template_file = Path(tmpdir) / "valid_template.py"
            template_file.write_text("""
from templates.base import Template, InputField, OutputField

class ValidTemplate(Template):
    name = "valid-template"
    category = "Test"
    description = "Valid"
    version = "1.0.0"
    inputs = []
    outputs = []
    memory_mb = 512
    timeout_sec = 60
    pip_packages = []
    
    def run(self, **kwargs):
        return {}
""")
            
            registry = TemplateRegistry(templates_dir=tmpdir)
            count = registry.discover_templates()
            
            # Only the valid Template should be registered
            assert count == 1
            assert "valid-template" in registry.templates
            assert "not-a-template" not in registry.templates


# ============================================================================
# Property 10: Registration metadata preservation
# ============================================================================

class TestProperty10RegistrationMetadataPreservation:
    """
    Property 10: Registration metadata preservation
    
    For any template that is successfully registered, calling 
    get_template_metadata(template.name) SHALL return a dictionary containing 
    all metadata fields from the template.
    
    **Validates: Requirements 5.3, 6.6, 6.7**
    """
    
    @settings(max_examples=100)
    @given(valid_template_strategy())
    def test_metadata_preservation_after_registration(self, template):
        """Registered templates should preserve all metadata."""
        registry = TemplateRegistry()
        
        # Register the template
        result = registry.register_template(template)
        assume(result is True)  # Only test successfully registered templates
        
        # Retrieve metadata
        metadata = registry.get_template_metadata(template.name)
        
        # Verify metadata is not None
        assert metadata is not None, f"Metadata should exist for registered template {template.name}"
        
        # Verify all required metadata fields are present
        required_fields = [
            "name", "category", "description", "version",
            "inputs", "outputs", "routing",
            "gpu_required", "gpu_type", "memory_mb", "timeout_sec",
            "pip_packages"
        ]
        
        for field in required_fields:
            assert field in metadata, f"Metadata missing required field: {field}"
        
        # Verify metadata values match template attributes
        assert metadata["name"] == template.name
        assert metadata["category"] == template.category
        assert metadata["description"] == template.description
        assert metadata["version"] == template.version
        assert metadata["gpu_required"] == template.gpu_required
        assert metadata["gpu_type"] == template.gpu_type
        assert metadata["memory_mb"] == template.memory_mb
        assert metadata["timeout_sec"] == template.timeout_sec
        assert metadata["pip_packages"] == template.pip_packages
        
        # Verify inputs are serialized correctly
        assert len(metadata["inputs"]) == len(template.inputs)
        for i, inp in enumerate(template.inputs):
            assert metadata["inputs"][i]["name"] == inp.name
            assert metadata["inputs"][i]["type"] == inp.type
            assert metadata["inputs"][i]["description"] == inp.description
            assert metadata["inputs"][i]["required"] == inp.required
        
        # Verify outputs are serialized correctly
        assert len(metadata["outputs"]) == len(template.outputs)
        for i, out in enumerate(template.outputs):
            assert metadata["outputs"][i]["name"] == out.name
            assert metadata["outputs"][i]["type"] == out.type
            assert metadata["outputs"][i]["description"] == out.description


# ============================================================================
# Property 11: Failed registration isolation
# ============================================================================

class TestProperty11FailedRegistrationIsolation:
    """
    Property 11: Failed registration isolation
    
    For any set of templates where at least one fails validation, all valid 
    templates SHALL still be successfully registered in the Template_Registry.
    
    **Validates: Requirements 5.4**
    """
    
    @settings(max_examples=100)
    @given(
        st.lists(valid_template_strategy(), min_size=1, max_size=5),
        st.lists(invalid_template_strategy(), min_size=1, max_size=3)
    )
    def test_valid_templates_registered_despite_invalid_ones(self, valid_templates, invalid_templates):
        """Valid templates should be registered even when invalid ones fail."""
        registry = TemplateRegistry()
        
        # Mix valid and invalid templates
        all_templates = valid_templates + invalid_templates
        
        # Track which templates should succeed
        expected_valid_names = set()
        for template in valid_templates:
            # Only expect registration if name is unique
            if template.name not in expected_valid_names:
                expected_valid_names.add(template.name)
        
        # Register all templates
        for template in all_templates:
            registry.register_template(template)
        
        # Verify all valid templates with unique names were registered
        for name in expected_valid_names:
            assert name in registry.templates, f"Valid template {name} should be registered"
        
        # Verify the count matches
        assert len(registry.templates) >= len(expected_valid_names)
    
    def test_failed_registration_isolation_with_discovery(self):
        """Test that discovery continues after encountering invalid templates."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create multiple template files, some valid, some invalid
            
            # Valid template 1
            valid1 = Path(tmpdir) / "valid1.py"
            valid1.write_text("""
from templates.base import Template

class Valid1(Template):
    name = "valid-1"
    category = "Test"
    description = "Valid 1"
    version = "1.0.0"
    inputs = []
    outputs = []
    memory_mb = 512
    timeout_sec = 60
    pip_packages = []
    
    def run(self, **kwargs):
        return {}
""")
            
            # Invalid template (syntax error)
            invalid1 = Path(tmpdir) / "invalid1.py"
            invalid1.write_text("""
from templates.base import Template

class Invalid1(Template):
    syntax error here
""")
            
            # Valid template 2
            valid2 = Path(tmpdir) / "valid2.py"
            valid2.write_text("""
from templates.base import Template

class Valid2(Template):
    name = "valid-2"
    category = "Test"
    description = "Valid 2"
    version = "1.0.0"
    inputs = []
    outputs = []
    memory_mb = 512
    timeout_sec = 60
    pip_packages = []
    
    def run(self, **kwargs):
        return {}
""")
            
            # Invalid template (missing required fields)
            invalid2 = Path(tmpdir) / "invalid2.py"
            invalid2.write_text("""
from templates.base import Template

class Invalid2(Template):
    name = ""  # Invalid empty name
    category = "Test"
    description = "Invalid"
    version = "1.0.0"
    inputs = []
    outputs = []
    memory_mb = 512
    timeout_sec = 60
    pip_packages = []
    
    def run(self, **kwargs):
        return {}
""")
            
            registry = TemplateRegistry(templates_dir=tmpdir)
            count = registry.discover_templates()
            
            # Both valid templates should be registered
            assert count == 2
            assert "valid-1" in registry.templates
            assert "valid-2" in registry.templates
            
            # Failed templates should be tracked
            assert len(registry.failed_templates) >= 2


# ============================================================================
# Property 12: Template discovery completeness
# ============================================================================

class TestProperty12TemplateDiscoveryCompleteness:
    """
    Property 12: Template discovery completeness
    
    For any valid template file in the templates directory at startup, the 
    template SHALL be present in the Template_Registry after discovery completes.
    
    **Validates: Requirements 5.1**
    """
    
    @settings(max_examples=100)
    @given(st.lists(valid_template_strategy(), min_size=1, max_size=10, unique_by=lambda t: t.name))
    def test_all_valid_templates_discovered(self, templates):
        """All valid templates in directory should be discovered."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create template files
            template_names = set()
            for i, template in enumerate(templates):
                template_file = Path(tmpdir) / f"template_{i}.py"
                
                # Generate Python code for the template
                code = f"""
from templates.base import Template, InputField, OutputField, RouteType

class Template{i}(Template):
    name = "{template.name}"
    category = "{template.category}"
    description = "{template.description}"
    version = "{template.version}"
    inputs = []
    outputs = []
    routing = [RouteType.LOCAL]
    gpu_required = {template.gpu_required}
    gpu_type = {repr(template.gpu_type)}
    memory_mb = {template.memory_mb}
    timeout_sec = {template.timeout_sec}
    pip_packages = {repr(template.pip_packages)}
    
    def run(self, **kwargs):
        return {{}}
"""
                template_file.write_text(code)
                template_names.add(template.name)
            
            # Discover templates
            registry = TemplateRegistry(templates_dir=tmpdir)
            count = registry.discover_templates()
            
            # All templates should be discovered
            assert count == len(template_names)
            
            # Each template should be in the registry
            for name in template_names:
                assert name in registry.templates, f"Template {name} should be discovered"
    
    def test_discovery_completeness_with_real_templates(self):
        """Test that real templates in the templates directory are discovered."""
        registry = TemplateRegistry(templates_dir="templates")
        count = registry.discover_templates()
        
        # Should discover at least the test template
        assert count >= 1
        
        # test-template should be discovered
        assert "test-template" in registry.templates


# ============================================================================
# Property 13: Category filtering
# ============================================================================

class TestProperty13CategoryFiltering:
    """
    Property 13: Category filtering
    
    For any category string, calling list_templates(category) SHALL return only 
    templates where template.category equals that category string.
    
    **Validates: Requirements 5.5**
    """
    
    @settings(max_examples=100)
    @given(category_strategy())
    def test_category_filtering_returns_only_matching_templates(self, category):
        """list_templates(category) should return only templates in that category."""
        registry = TemplateRegistry()
        
        # Register templates with different categories
        audio_template = ValidAudioTemplate()
        vision_template = ValidVisionTemplate()
        language_template = ValidLanguageTemplate()
        
        registry.register_template(audio_template)
        registry.register_template(vision_template)
        registry.register_template(language_template)
        
        # Get templates for the specified category
        filtered_templates = registry.list_templates(category=category)
        
        # Verify all returned templates have the correct category
        for template in filtered_templates:
            assert template.category == category, \
                f"Template {template.name} has category {template.category}, expected {category}"
        
        # Verify we didn't miss any templates in this category
        all_templates = registry.list_templates()
        expected_count = sum(1 for t in all_templates if t.category == category)
        assert len(filtered_templates) == expected_count
    
    @settings(max_examples=100)
    @given(st.lists(valid_template_strategy(), min_size=1, max_size=10))
    def test_category_filtering_completeness(self, templates):
        """Category filtering should return all templates in that category."""
        registry = TemplateRegistry()
        
        # Register all templates
        registered_names = set()
        for template in templates:
            if template.name not in registered_names:
                registry.register_template(template)
                registered_names.add(template.name)
        
        # Get all categories
        categories = registry.get_all_categories()
        
        # For each category, verify filtering works correctly
        for category in categories:
            filtered = registry.list_templates(category=category)
            
            # All returned templates should have this category
            for template in filtered:
                assert template.category == category
            
            # Count should match expected
            all_templates = registry.list_templates()
            expected_in_category = [t for t in all_templates if t.category == category]
            assert len(filtered) == len(expected_in_category)
    
    def test_empty_category_returns_empty_list(self):
        """Filtering by non-existent category should return empty list."""
        registry = TemplateRegistry()
        
        # Register some templates
        registry.register_template(ValidAudioTemplate())
        registry.register_template(ValidVisionTemplate())
        
        # Filter by non-existent category
        filtered = registry.list_templates(category="NonExistentCategory")
        
        assert len(filtered) == 0
        assert isinstance(filtered, list)


# ============================================================================
# Property 14: Template retrieval by name
# ============================================================================

class TestProperty14TemplateRetrievalByName:
    """
    Property 14: Template retrieval by name
    
    For any registered template with name N, calling get_template(N) SHALL 
    return the same template instance that was registered.
    
    **Validates: Requirements 5.6**
    """
    
    @settings(max_examples=100)
    @given(valid_template_strategy())
    def test_get_template_returns_same_instance(self, template):
        """get_template should return the exact instance that was registered."""
        registry = TemplateRegistry()
        
        # Register the template
        result = registry.register_template(template)
        assume(result is True)
        
        # Retrieve the template
        retrieved = registry.get_template(template.name)
        
        # Should return the same instance
        assert retrieved is not None
        assert retrieved is template, "Should return the same instance"
        assert id(retrieved) == id(template), "Should be the exact same object"
    
    @settings(max_examples=100)
    @given(
        st.lists(valid_template_strategy(), min_size=1, max_size=10, unique_by=lambda t: t.name)
    )
    def test_get_template_retrieves_correct_template(self, templates):
        """get_template should retrieve the correct template by name."""
        registry = TemplateRegistry()
        
        # Register all templates
        template_map = {}
        for template in templates:
            result = registry.register_template(template)
            if result:
                template_map[template.name] = template
        
        # Retrieve each template and verify it's correct
        for name, original_template in template_map.items():
            retrieved = registry.get_template(name)
            
            assert retrieved is not None, f"Template {name} should be retrievable"
            assert retrieved is original_template, f"Should retrieve the same instance for {name}"
            assert retrieved.name == name
            assert retrieved.category == original_template.category
            assert retrieved.description == original_template.description
    
    @settings(max_examples=100)
    @given(template_name_strategy())
    def test_get_nonexistent_template_returns_none(self, nonexistent_name):
        """get_template should return None for non-existent templates."""
        registry = TemplateRegistry()
        
        # Register some templates
        registry.register_template(ValidAudioTemplate())
        registry.register_template(ValidVisionTemplate())
        
        # Assume the name doesn't exist
        assume(nonexistent_name not in registry.templates)
        
        # Try to retrieve non-existent template
        retrieved = registry.get_template(nonexistent_name)
        
        assert retrieved is None, f"Non-existent template {nonexistent_name} should return None"


# ============================================================================
# Stateful Property Testing
# ============================================================================

class TemplateRegistryStateMachine(RuleBasedStateMachine):
    """
    Stateful property testing for Template Registry.
    
    This tests that the registry maintains consistency across a sequence of
    operations (register, retrieve, list, etc.).
    """
    
    templates = Bundle("templates")
    
    def __init__(self):
        super().__init__()
        self.registry = TemplateRegistry()
        self.registered_templates = {}
        self.next_template_id = 0
    
    @initialize()
    def init_registry(self):
        """Initialize the registry."""
        self.registry = TemplateRegistry()
        self.registered_templates = {}
        self.next_template_id = 0
    
    @rule(target=templates)
    def register_valid_template(self):
        """Register a valid template."""
        # Create a unique template
        template_id = self.next_template_id
        self.next_template_id += 1
        
        class DynamicTemplate(Template):
            name = f"template-{template_id}"
            category = "Test"
            description = f"Template {template_id}"
            version = "1.0.0"
            memory_mb = 512
            timeout_sec = 60
            pip_packages = []
            
            def run(self, **kwargs):
                return {}
        
        template = DynamicTemplate()
        result = self.registry.register_template(template)
        
        if result:
            self.registered_templates[template.name] = template
        
        return template
    
    @rule(template=templates)
    def retrieve_template(self, template):
        """Retrieve a registered template."""
        retrieved = self.registry.get_template(template.name)
        
        if template.name in self.registered_templates:
            assert retrieved is not None
            assert retrieved is self.registered_templates[template.name]
        else:
            assert retrieved is None
    
    @rule()
    def list_all_templates(self):
        """List all templates."""
        all_templates = self.registry.list_templates()
        
        # Should match registered templates
        assert len(all_templates) == len(self.registered_templates)
        
        for template in all_templates:
            assert template.name in self.registered_templates
    
    @rule(category=category_strategy())
    def list_by_category(self, category):
        """List templates by category."""
        filtered = self.registry.list_templates(category=category)
        
        # All returned templates should have the correct category
        for template in filtered:
            assert template.category == category
        
        # Count should match expected
        expected_count = sum(
            1 for t in self.registered_templates.values() 
            if t.category == category
        )
        assert len(filtered) == expected_count
    
    @invariant()
    def registry_consistency(self):
        """Registry should maintain consistency."""
        # All registered templates should be retrievable
        for name, template in self.registered_templates.items():
            retrieved = self.registry.get_template(name)
            assert retrieved is template
        
        # Template count should match
        all_templates = self.registry.list_templates()
        assert len(all_templates) == len(self.registered_templates)


# Run the stateful test
TestRegistryStateMachine = TemplateRegistryStateMachine.TestCase
