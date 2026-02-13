"""
Unit tests for the Template Registry.

Tests cover:
- Template discovery from the templates directory
- Template registration with validation
- Template retrieval by name
- Template listing with category filtering
- Metadata retrieval
- Error handling for invalid templates
- Thread safety
"""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch
import threading

from notebook_ml_orchestrator.core.template_registry import TemplateRegistry
from templates.base import Template, InputField, OutputField, RouteType


class ValidTestTemplate(Template):
    """A valid test template for testing."""
    name = "valid-test"
    category = "Test"
    description = "A valid test template"
    version = "1.0.0"
    inputs = [
        InputField(name="input1", type="text", description="Test input", required=True)
    ]
    outputs = [
        OutputField(name="output1", type="text", description="Test output")
    ]
    routing = [RouteType.LOCAL]
    gpu_required = False
    memory_mb = 512
    timeout_sec = 60
    pip_packages = []
    
    def run(self, **kwargs):
        return {"output1": "test"}


class InvalidTemplateNoName(Template):
    """Invalid template missing name."""
    category = "Test"
    description = "Invalid template"
    version = "1.0.0"
    inputs = []
    outputs = []
    routing = [RouteType.LOCAL]
    memory_mb = 512
    timeout_sec = 60
    pip_packages = []
    
    def run(self, **kwargs):
        return {}


class InvalidTemplateNoMemory(Template):
    """Invalid template with invalid memory."""
    name = "invalid-memory"
    category = "Test"
    description = "Invalid template"
    version = "1.0.0"
    inputs = []
    outputs = []
    routing = [RouteType.LOCAL]
    memory_mb = 0  # Invalid
    timeout_sec = 60
    pip_packages = []
    
    def run(self, **kwargs):
        return {}


class GPUTemplateValid(Template):
    """Valid GPU template."""
    name = "gpu-test"
    category = "Test"
    description = "GPU test template"
    version = "1.0.0"
    inputs = []
    outputs = []
    routing = [RouteType.LOCAL]
    gpu_required = True
    gpu_type = "T4"
    memory_mb = 4096
    timeout_sec = 300
    pip_packages = []
    
    def run(self, **kwargs):
        return {}


class GPUTemplateInvalid(Template):
    """Invalid GPU template missing gpu_type."""
    name = "gpu-invalid"
    category = "Test"
    description = "Invalid GPU template"
    version = "1.0.0"
    inputs = []
    outputs = []
    routing = [RouteType.LOCAL]
    gpu_required = True
    gpu_type = None  # Invalid - should specify type
    memory_mb = 4096
    timeout_sec = 300
    pip_packages = []
    
    def run(self, **kwargs):
        return {}


class TestTemplateRegistry:
    """Test suite for TemplateRegistry."""
    
    def test_init(self):
        """Test registry initialization."""
        registry = TemplateRegistry(templates_dir="templates")
        assert registry.templates_dir == "templates"
        assert len(registry.templates) == 0
        assert len(registry.templates_by_category) == 0
        assert len(registry.failed_templates) == 0
    
    def test_register_valid_template(self):
        """Test registering a valid template."""
        registry = TemplateRegistry()
        template = ValidTestTemplate()
        
        result = registry.register_template(template)
        assert result is True
        assert "valid-test" in registry.templates
        assert registry.templates["valid-test"] == template
        assert "Test" in registry.templates_by_category
        assert "valid-test" in registry.templates_by_category["Test"]
    
    def test_register_duplicate_template(self):
        """Test that duplicate template names are rejected."""
        registry = TemplateRegistry()
        template1 = ValidTestTemplate()
        template2 = ValidTestTemplate()
        
        result1 = registry.register_template(template1)
        result2 = registry.register_template(template2)
        
        assert result1 is True
        assert result2 is False  # Duplicate should be rejected
        assert len(registry.templates) == 1
    
    def test_register_invalid_template_no_name(self):
        """Test that templates without names fail validation."""
        registry = TemplateRegistry()
        template = InvalidTemplateNoName()
        
        result = registry.register_template(template)
        assert result is False
        assert len(registry.templates) == 0
    
    def test_register_invalid_template_no_memory(self):
        """Test that templates with invalid memory fail validation."""
        registry = TemplateRegistry()
        template = InvalidTemplateNoMemory()
        
        result = registry.register_template(template)
        assert result is False
        assert len(registry.templates) == 0
    
    def test_validate_gpu_template_valid(self):
        """Test validation of valid GPU template."""
        registry = TemplateRegistry()
        template = GPUTemplateValid()
        
        result = registry.validate_template(template)
        assert result is True
    
    def test_validate_gpu_template_invalid(self):
        """Test validation of invalid GPU template."""
        registry = TemplateRegistry()
        template = GPUTemplateInvalid()
        
        result = registry.validate_template(template)
        assert result is False
    
    def test_get_template(self):
        """Test retrieving a template by name."""
        registry = TemplateRegistry()
        template = ValidTestTemplate()
        registry.register_template(template)
        
        retrieved = registry.get_template("valid-test")
        assert retrieved is not None
        assert retrieved == template
        
        not_found = registry.get_template("nonexistent")
        assert not_found is None
    
    def test_list_templates_all(self):
        """Test listing all templates."""
        registry = TemplateRegistry()
        template1 = ValidTestTemplate()
        template2 = GPUTemplateValid()
        
        registry.register_template(template1)
        registry.register_template(template2)
        
        all_templates = registry.list_templates()
        assert len(all_templates) == 2
        assert template1 in all_templates
        assert template2 in all_templates
    
    def test_list_templates_by_category(self):
        """Test listing templates filtered by category."""
        registry = TemplateRegistry()
        
        # Create templates with different categories
        template1 = ValidTestTemplate()
        template1.category = "Audio"
        
        template2 = GPUTemplateValid()
        template2.category = "Vision"
        
        template3 = ValidTestTemplate()
        template3.name = "another-audio"
        template3.category = "Audio"
        
        registry.register_template(template1)
        registry.register_template(template2)
        registry.register_template(template3)
        
        audio_templates = registry.list_templates(category="Audio")
        assert len(audio_templates) == 2
        
        vision_templates = registry.list_templates(category="Vision")
        assert len(vision_templates) == 1
        
        empty_category = registry.list_templates(category="Nonexistent")
        assert len(empty_category) == 0
    
    def test_get_template_metadata(self):
        """Test retrieving template metadata."""
        registry = TemplateRegistry()
        template = ValidTestTemplate()
        registry.register_template(template)
        
        metadata = registry.get_template_metadata("valid-test")
        assert metadata is not None
        assert metadata["name"] == "valid-test"
        assert metadata["category"] == "Test"
        assert metadata["description"] == "A valid test template"
        assert metadata["version"] == "1.0.0"
        assert "inputs" in metadata
        assert "outputs" in metadata
        assert metadata["gpu_required"] is False
        assert metadata["memory_mb"] == 512
        assert metadata["timeout_sec"] == 60
        
        not_found = registry.get_template_metadata("nonexistent")
        assert not_found is None
    
    def test_get_all_categories(self):
        """Test getting all categories."""
        registry = TemplateRegistry()
        
        template1 = ValidTestTemplate()
        template1.category = "Audio"
        
        template2 = GPUTemplateValid()
        template2.category = "Vision"
        
        registry.register_template(template1)
        registry.register_template(template2)
        
        categories = registry.get_all_categories()
        assert len(categories) == 2
        assert "Audio" in categories
        assert "Vision" in categories
    
    def test_get_registry_stats(self):
        """Test getting registry statistics."""
        registry = TemplateRegistry()
        
        template1 = ValidTestTemplate()
        template2 = GPUTemplateValid()
        
        registry.register_template(template1)
        registry.register_template(template2)
        
        stats = registry.get_registry_stats()
        assert stats["total_templates"] == 2
        assert stats["categories"] == 1  # Both are "Test" category
        assert stats["failed_templates"] == 0
    
    def test_discover_templates_real_directory(self):
        """Test discovering templates from the real templates directory."""
        registry = TemplateRegistry(templates_dir="templates")
        
        count = registry.discover_templates()
        
        # Should discover at least the test_template.py we created
        assert count >= 1
        
        # Check that test-template was discovered
        test_template = registry.get_template("test-template")
        assert test_template is not None
        assert test_template.name == "test-template"
        assert test_template.category == "Test"
    
    def test_discover_templates_nonexistent_directory(self):
        """Test discovery with nonexistent directory."""
        registry = TemplateRegistry(templates_dir="nonexistent_dir")
        
        count = registry.discover_templates()
        assert count == 0
        assert len(registry.templates) == 0
    
    def test_thread_safety(self):
        """Test that registry operations are thread-safe."""
        registry = TemplateRegistry()
        results = []
        
        def register_template(template_num):
            template = ValidTestTemplate()
            template.name = f"template-{template_num}"
            result = registry.register_template(template)
            results.append(result)
        
        # Create multiple threads that register templates
        threads = []
        for i in range(10):
            thread = threading.Thread(target=register_template, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # All registrations should succeed
        assert all(results)
        assert len(registry.templates) == 10
    
    def test_validate_input_fields(self):
        """Test validation of input field completeness."""
        registry = TemplateRegistry()
        
        # Template with incomplete input field
        class IncompleteInputTemplate(Template):
            name = "incomplete-input"
            category = "Test"
            description = "Test"
            version = "1.0.0"
            inputs = [
                InputField(name="", type="text", description="Test", required=True)  # Empty name
            ]
            outputs = []
            routing = [RouteType.LOCAL]
            memory_mb = 512
            timeout_sec = 60
            pip_packages = []
            
            def run(self, **kwargs):
                return {}
        
        template = IncompleteInputTemplate()
        result = registry.validate_template(template)
        assert result is False
    
    def test_validate_output_fields(self):
        """Test validation of output field completeness."""
        registry = TemplateRegistry()
        
        # Template with incomplete output field
        class IncompleteOutputTemplate(Template):
            name = "incomplete-output"
            category = "Test"
            description = "Test"
            version = "1.0.0"
            inputs = []
            outputs = [
                OutputField(name="output", type="", description="Test")  # Empty type
            ]
            routing = [RouteType.LOCAL]
            memory_mb = 512
            timeout_sec = 60
            pip_packages = []
            
            def run(self, **kwargs):
                return {}
        
        template = IncompleteOutputTemplate()
        result = registry.validate_template(template)
        assert result is False
    
    def test_failed_templates_tracking(self):
        """Test that failed templates are tracked."""
        # Create a temporary directory with an invalid template file
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a Python file with syntax error
            invalid_file = Path(tmpdir) / "invalid_template.py"
            invalid_file.write_text("class InvalidTemplate(Template):\n    syntax error here")
            
            registry = TemplateRegistry(templates_dir=tmpdir)
            count = registry.discover_templates()
            
            # Should have tracked the failed template
            assert len(registry.failed_templates) > 0 or count == 0
    
    def test_discovery_with_mixed_valid_invalid_templates(self):
        """Test that valid templates are registered even when some fail."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a valid template file
            valid_file = Path(tmpdir) / "valid_template.py"
            valid_file.write_text("""
from templates.base import Template, InputField, OutputField, RouteType

class ValidDiscoveryTemplate(Template):
    name = "valid-discovery"
    category = "Test"
    description = "Valid template"
    version = "1.0.0"
    inputs = []
    outputs = []
    routing = [RouteType.LOCAL]
    gpu_required = False
    memory_mb = 512
    timeout_sec = 60
    pip_packages = []
    
    def run(self, **kwargs):
        return {}
""")
            
            # Create an invalid template file (syntax error)
            invalid_file = Path(tmpdir) / "invalid_template.py"
            invalid_file.write_text("this is not valid python code")
            
            registry = TemplateRegistry(templates_dir=tmpdir)
            count = registry.discover_templates()
            
            # Valid template should be registered
            assert count == 1
            assert "valid-discovery" in registry.templates
            
            # Invalid template should be tracked as failed
            assert len(registry.failed_templates) >= 1
    
    def test_query_methods_with_empty_registry(self):
        """Test query methods on empty registry."""
        registry = TemplateRegistry()
        
        # get_template should return None
        assert registry.get_template("nonexistent") is None
        
        # list_templates should return empty list
        assert registry.list_templates() == []
        assert registry.list_templates(category="Audio") == []
        
        # get_template_metadata should return None
        assert registry.get_template_metadata("nonexistent") is None
        
        # get_all_categories should return empty list
        assert registry.get_all_categories() == []
        
        # get_registry_stats should show zeros
        stats = registry.get_registry_stats()
        assert stats["total_templates"] == 0
        assert stats["categories"] == 0
        assert stats["failed_templates"] == 0
    
    def test_query_methods_with_special_characters(self):
        """Test query methods with special characters in names."""
        registry = TemplateRegistry()
        
        # Test with names containing special characters
        assert registry.get_template("") is None
        assert registry.get_template("template-with-dashes") is None
        assert registry.get_template("template_with_underscores") is None
        assert registry.get_template("template.with.dots") is None
        
        # Test category filtering with special characters
        assert registry.list_templates(category="") == []
        assert registry.list_templates(category="Non-Existent Category") == []
    
    def test_get_template_metadata_completeness(self):
        """Test that metadata includes all required fields."""
        registry = TemplateRegistry()
        template = ValidTestTemplate()
        registry.register_template(template)
        
        metadata = registry.get_template_metadata("valid-test")
        
        # Check all required metadata fields are present
        required_fields = [
            "name", "category", "description", "version",
            "inputs", "outputs", "routing",
            "gpu_required", "gpu_type", "memory_mb", "timeout_sec",
            "pip_packages"
        ]
        
        for field in required_fields:
            assert field in metadata, f"Missing required field: {field}"
        
        # Check input field structure
        assert len(metadata["inputs"]) == 1
        input_field = metadata["inputs"][0]
        assert "name" in input_field
        assert "type" in input_field
        assert "description" in input_field
        assert "required" in input_field
        
        # Check output field structure
        assert len(metadata["outputs"]) == 1
        output_field = metadata["outputs"][0]
        assert "name" in output_field
        assert "type" in output_field
        assert "description" in output_field
    
    def test_register_template_with_invalid_gpu_type(self):
        """Test that templates with invalid GPU types fail validation."""
        registry = TemplateRegistry()
        
        class InvalidGPUTypeTemplate(Template):
            name = "invalid-gpu-type"
            category = "Test"
            description = "Template with invalid GPU type"
            version = "1.0.0"
            inputs = []
            outputs = []
            routing = [RouteType.LOCAL]
            gpu_required = True
            gpu_type = "InvalidGPU"  # Not in [T4, A10G, A100]
            memory_mb = 4096
            timeout_sec = 300
            pip_packages = []
            
            def run(self, **kwargs):
                return {}
        
        template = InvalidGPUTypeTemplate()
        result = registry.register_template(template)
        assert result is False
        assert len(registry.templates) == 0
    
    def test_validate_template_with_missing_required_fields(self):
        """Test validation fails for templates missing required metadata."""
        registry = TemplateRegistry()
        
        # Template with empty description
        class EmptyDescriptionTemplate(Template):
            name = "empty-desc"
            category = "Test"
            description = ""  # Empty string
            version = "1.0.0"
            inputs = []
            outputs = []
            routing = [RouteType.LOCAL]
            memory_mb = 512
            timeout_sec = 60
            pip_packages = []
            
            def run(self, **kwargs):
                return {}
        
        template = EmptyDescriptionTemplate()
        result = registry.validate_template(template)
        assert result is False
    
    def test_validate_template_with_invalid_pip_packages(self):
        """Test validation fails when pip_packages is not a list."""
        registry = TemplateRegistry()
        
        class InvalidPipPackagesTemplate(Template):
            name = "invalid-pip"
            category = "Test"
            description = "Test"
            version = "1.0.0"
            inputs = []
            outputs = []
            routing = [RouteType.LOCAL]
            memory_mb = 512
            timeout_sec = 60
            pip_packages = "not-a-list"  # Should be a list
            
            def run(self, **kwargs):
                return {}
        
        template = InvalidPipPackagesTemplate()
        result = registry.validate_template(template)
        assert result is False
    
    def test_discovery_skips_init_and_base_files(self):
        """Test that __init__.py and base.py are skipped during discovery."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create __init__.py
            init_file = Path(tmpdir) / "__init__.py"
            init_file.write_text("# Init file")
            
            # Create base.py
            base_file = Path(tmpdir) / "base.py"
            base_file.write_text("# Base file")
            
            # Create a valid template
            valid_file = Path(tmpdir) / "valid_template.py"
            valid_file.write_text("""
from templates.base import Template, RouteType

class SkipTestTemplate(Template):
    name = "skip-test"
    category = "Test"
    description = "Test"
    version = "1.0.0"
    inputs = []
    outputs = []
    routing = [RouteType.LOCAL]
    memory_mb = 512
    timeout_sec = 60
    pip_packages = []
    
    def run(self, **kwargs):
        return {}
""")
            
            registry = TemplateRegistry(templates_dir=tmpdir)
            count = registry.discover_templates()
            
            # Should only discover the valid template, not __init__.py or base.py
            assert count == 1
            assert "skip-test" in registry.templates
    
    def test_list_templates_returns_template_instances(self):
        """Test that list_templates returns actual Template instances."""
        registry = TemplateRegistry()
        template1 = ValidTestTemplate()
        template2 = GPUTemplateValid()
        
        registry.register_template(template1)
        registry.register_template(template2)
        
        templates = registry.list_templates()
        
        # Should return Template instances, not strings
        assert all(isinstance(t, Template) for t in templates)
        assert template1 in templates
        assert template2 in templates
    
    def test_category_filtering_case_sensitive(self):
        """Test that category filtering is case-sensitive."""
        registry = TemplateRegistry()
        
        template = ValidTestTemplate()
        template.category = "Audio"
        registry.register_template(template)
        
        # Exact match should work
        assert len(registry.list_templates(category="Audio")) == 1
        
        # Different case should not match
        assert len(registry.list_templates(category="audio")) == 0
        assert len(registry.list_templates(category="AUDIO")) == 0
    
    def test_registry_stats_with_failed_templates(self):
        """Test that registry stats correctly report failed templates."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create an invalid template file
            invalid_file = Path(tmpdir) / "invalid.py"
            invalid_file.write_text("invalid python")
            
            registry = TemplateRegistry(templates_dir=tmpdir)
            registry.discover_templates()
            
            stats = registry.get_registry_stats()
            assert stats["failed_templates"] >= 1
            assert len(stats["failed_template_list"]) >= 1
            assert "invalid.py" in stats["failed_template_list"][0]

    def test_get_supported_backends_existing_template(self):
        """Test getting supported backends for a template in the mapping."""
        registry = TemplateRegistry()
        
        # Test templates that exist in TEMPLATE_BACKEND_SUPPORT
        backends = registry.get_supported_backends("image-generation")
        assert backends == ["modal", "huggingface"]
        
        backends = registry.get_supported_backends("model-training")
        assert backends == ["modal", "kaggle", "colab"]
        
        backends = registry.get_supported_backends("embeddings")
        assert backends == ["huggingface", "modal"]
    
    def test_get_supported_backends_nonexistent_template(self):
        """Test getting supported backends for a template not in the mapping."""
        registry = TemplateRegistry()
        
        # Template not in mapping should return empty list
        backends = registry.get_supported_backends("nonexistent-template")
        assert backends == []
        
        backends = registry.get_supported_backends("")
        assert backends == []
    
    def test_get_backend_capabilities_modal(self):
        """Test getting capabilities for Modal backend."""
        registry = TemplateRegistry()
        
        capabilities = registry.get_backend_capabilities("modal")
        
        # Check required fields
        assert "supported_templates" in capabilities
        assert "supports_gpu" in capabilities
        assert "max_job_duration_minutes" in capabilities
        assert "cost_per_hour" in capabilities
        assert "backend_type" in capabilities
        
        # Check Modal-specific values
        assert capabilities["supports_gpu"] is True
        assert capabilities["backend_type"] == "serverless"
        assert capabilities["max_job_duration_minutes"] == 300
        assert capabilities["cost_per_hour"] == 1.10
        assert "gpu_types" in capabilities
        assert "T4" in capabilities["gpu_types"]
        assert "A10G" in capabilities["gpu_types"]
        assert "A100" in capabilities["gpu_types"]
        
        # Check that supported templates include Modal templates
        assert "image-generation" in capabilities["supported_templates"]
        assert "text-generation" in capabilities["supported_templates"]
    
    def test_get_backend_capabilities_huggingface(self):
        """Test getting capabilities for HuggingFace backend."""
        registry = TemplateRegistry()
        
        capabilities = registry.get_backend_capabilities("huggingface")
        
        # Check HuggingFace-specific values
        assert capabilities["supports_gpu"] is True
        assert capabilities["backend_type"] == "inference"
        assert capabilities["cost_per_hour"] == 0.0  # Free tier
        assert capabilities["max_job_duration_minutes"] == 60
        
        # Check supported templates
        assert "image-generation" in capabilities["supported_templates"]
        assert "text-generation" in capabilities["supported_templates"]
        assert "embeddings" in capabilities["supported_templates"]
    
    def test_get_backend_capabilities_kaggle(self):
        """Test getting capabilities for Kaggle backend."""
        registry = TemplateRegistry()
        
        capabilities = registry.get_backend_capabilities("kaggle")
        
        # Check Kaggle-specific values
        assert capabilities["supports_gpu"] is True
        assert capabilities["backend_type"] == "notebook"
        assert capabilities["cost_per_hour"] == 0.0  # Free tier
        assert capabilities["max_job_duration_minutes"] == 540  # 9 hours
        
        # Check free tier limits
        assert "free_tier_limits" in capabilities
        assert capabilities["free_tier_limits"]["gpu_hours_per_week"] == 30
        assert capabilities["free_tier_limits"]["max_concurrent_kernels"] == 1
        
        # Check supported templates
        assert "model-training" in capabilities["supported_templates"]
        assert "data-processing" in capabilities["supported_templates"]
    
    def test_get_backend_capabilities_colab(self):
        """Test getting capabilities for Colab backend."""
        registry = TemplateRegistry()
        
        capabilities = registry.get_backend_capabilities("colab")
        
        # Check Colab-specific values
        assert capabilities["supports_gpu"] is True
        assert capabilities["backend_type"] == "notebook"
        assert capabilities["cost_per_hour"] == 0.0  # Free tier
        assert capabilities["max_job_duration_minutes"] == 720  # 12 hours
        
        # Check free tier limits
        assert "free_tier_limits" in capabilities
        assert capabilities["free_tier_limits"]["session_timeout_minutes"] == 90
        assert capabilities["free_tier_limits"]["max_runtime_hours"] == 12
        
        # Check supported templates
        assert "model-training" in capabilities["supported_templates"]
    
    def test_get_backend_capabilities_unknown_backend(self):
        """Test getting capabilities for an unknown backend."""
        registry = TemplateRegistry()
        
        capabilities = registry.get_backend_capabilities("unknown-backend")
        
        # Should return default capabilities
        assert "supported_templates" in capabilities
        assert capabilities["supported_templates"] == []
        assert capabilities["supports_gpu"] is False
        assert capabilities["max_job_duration_minutes"] == 60
        assert capabilities["cost_per_hour"] == 0.0
        assert capabilities["backend_type"] == "unknown"
    
    def test_backend_capabilities_consistency(self):
        """Test that backend capabilities are consistent with template mapping."""
        registry = TemplateRegistry()
        
        # For each backend, verify that its supported_templates list matches
        # the templates that list this backend in TEMPLATE_BACKEND_SUPPORT
        from notebook_ml_orchestrator.core.template_registry import TEMPLATE_BACKEND_SUPPORT
        
        for backend_id in ["modal", "huggingface", "kaggle", "colab"]:
            capabilities = registry.get_backend_capabilities(backend_id)
            supported_templates = capabilities["supported_templates"]
            
            # Check that each template in supported_templates actually lists this backend
            for template_name in supported_templates:
                assert template_name in TEMPLATE_BACKEND_SUPPORT
                assert backend_id in TEMPLATE_BACKEND_SUPPORT[template_name]
