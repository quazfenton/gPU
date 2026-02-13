"""
Property-based tests for Language templates.

This module implements property-based tests using Hypothesis to verify
that Language templates conform to the design specifications for input/output types,
and metadata completeness.

**Validates: Requirements 3.5, 3.6, 6.1, 6.2, 6.3, 6.4, 6.5**
"""

import pytest
from typing import Dict, Any, List, Optional

from hypothesis import given, strategies as st, settings, assume

from templates.base import Template, InputField, OutputField, RouteType
from templates.ner_template import NERTemplate
from templates.sentiment_analysis_template import SentimentAnalysisTemplate
from templates.translation_template import TranslationTemplate
from templates.summarization_template import SummarizationTemplate


# ============================================================================
# Hypothesis Strategies
# ============================================================================

def language_template_strategy():
    """Strategy for generating Language template instances."""
    return st.sampled_from([
        NERTemplate(),
        SentimentAnalysisTemplate(),
        TranslationTemplate(),
        SummarizationTemplate()
    ])


def valid_language_input_types():
    """Valid input types for Language templates."""
    return st.sampled_from(["text"])


def valid_language_output_types():
    """Valid output types for Language templates."""
    return st.sampled_from(["text", "json"])


# ============================================================================
# Property 3: Language template I/O types
# ============================================================================

class TestLanguageTemplateIOTypes:
    """
    **Property 3: Language template I/O types**
    
    For any template in the Language category, all input types SHALL be "text",
    and all output types SHALL be either "text" or "json". Configuration parameters
    (number, etc.) are allowed as additional inputs.
    
    **Validates: Requirements 3.5, 3.6**
    """
    
    @settings(max_examples=100)
    @given(language_template_strategy())
    def test_language_template_has_text_input(self, template):
        """Language templates must have at least one 'text' input."""
        # Verify template is in Language category
        assert template.category == "Language", f"Template {template.name} is not in Language category"
        
        # Check that at least one input is text
        has_text_input = any(
            input_field.type == "text" 
            for input_field in template.inputs
        )
        
        assert has_text_input, (
            f"Language template '{template.name}' must have at least one input of type 'text'. "
            f"Found input types: {[f.type for f in template.inputs]}"
        )
    
    @settings(max_examples=100)
    @given(language_template_strategy())
    def test_language_template_primary_inputs_are_text(self, template):
        """Language templates' primary data inputs should be 'text' type."""
        # Verify template is in Language category
        assert template.category == "Language", f"Template {template.name} is not in Language category"
        
        # Check that primary inputs (not configuration) are text
        # Configuration inputs are typically: number, boolean, etc.
        config_types = {"number", "boolean"}
        
        for input_field in template.inputs:
            if input_field.type not in config_types:
                # This is a data input, should be text
                assert input_field.type == "text", (
                    f"Language template '{template.name}' has non-text data input '{input_field.name}' "
                    f"of type '{input_field.type}'. Expected 'text'."
                )
    
    @settings(max_examples=100)
    @given(language_template_strategy())
    def test_language_template_output_types_are_valid(self, template):
        """All Language template outputs must be 'text' or 'json' types."""
        # Verify template is in Language category
        assert template.category == "Language", f"Template {template.name} is not in Language category"
        
        # Check all output types
        valid_output_types = {"text", "json"}
        for output_field in template.outputs:
            assert output_field.type in valid_output_types, (
                f"Language template '{template.name}' has invalid output type '{output_field.type}'. "
                f"Expected one of {valid_output_types}"
            )
    
    @settings(max_examples=100)
    @given(language_template_strategy())
    def test_language_template_has_at_least_one_input(self, template):
        """Language templates must have at least one input field."""
        assert len(template.inputs) > 0, (
            f"Language template '{template.name}' has no input fields"
        )
    
    @settings(max_examples=100)
    @given(language_template_strategy())
    def test_language_template_has_at_least_one_output(self, template):
        """Language templates must have at least one output field."""
        assert len(template.outputs) > 0, (
            f"Language template '{template.name}' has no output fields"
        )


# ============================================================================
# Property 6: Required metadata fields
# ============================================================================

class TestRequiredMetadataFields:
    """
    **Property 6: Required metadata fields**
    
    For any registered template, the template SHALL provide non-empty values 
    for name, category, description, version, memory_mb (> 0), timeout_sec (> 0), 
    and pip_packages (list, possibly empty).
    
    **Validates: Requirements 6.1, 6.4, 6.5, 8.3, 8.4**
    """
    
    @settings(max_examples=100)
    @given(language_template_strategy())
    def test_template_has_non_empty_name(self, template):
        """Templates must have a non-empty name."""
        assert template.name, f"Template has empty name"
        assert isinstance(template.name, str), f"Template name must be a string"
        assert len(template.name.strip()) > 0, f"Template name must not be whitespace only"
    
    @settings(max_examples=100)
    @given(language_template_strategy())
    def test_template_has_non_empty_category(self, template):
        """Templates must have a non-empty category."""
        assert template.category, f"Template '{template.name}' has empty category"
        assert isinstance(template.category, str), f"Template category must be a string"
        assert len(template.category.strip()) > 0, f"Template category must not be whitespace only"
    
    @settings(max_examples=100)
    @given(language_template_strategy())
    def test_template_has_non_empty_description(self, template):
        """Templates must have a non-empty description."""
        assert template.description, f"Template '{template.name}' has empty description"
        assert isinstance(template.description, str), f"Template description must be a string"
        assert len(template.description.strip()) > 0, f"Template description must not be whitespace only"
    
    @settings(max_examples=100)
    @given(language_template_strategy())
    def test_template_has_non_empty_version(self, template):
        """Templates must have a non-empty version."""
        assert template.version, f"Template '{template.name}' has empty version"
        assert isinstance(template.version, str), f"Template version must be a string"
        assert len(template.version.strip()) > 0, f"Template version must not be whitespace only"
    
    @settings(max_examples=100)
    @given(language_template_strategy())
    def test_template_has_positive_memory_mb(self, template):
        """Templates must have memory_mb > 0."""
        assert template.memory_mb > 0, (
            f"Template '{template.name}' has memory_mb={template.memory_mb} (must be > 0)"
        )
        assert isinstance(template.memory_mb, int), (
            f"Template '{template.name}' memory_mb must be an integer"
        )
    
    @settings(max_examples=100)
    @given(language_template_strategy())
    def test_template_has_positive_timeout_sec(self, template):
        """Templates must have timeout_sec > 0."""
        assert template.timeout_sec > 0, (
            f"Template '{template.name}' has timeout_sec={template.timeout_sec} (must be > 0)"
        )
        assert isinstance(template.timeout_sec, int), (
            f"Template '{template.name}' timeout_sec must be an integer"
        )
    
    @settings(max_examples=100)
    @given(language_template_strategy())
    def test_template_has_pip_packages_list(self, template):
        """Templates must have pip_packages as a list (possibly empty)."""
        assert isinstance(template.pip_packages, list), (
            f"Template '{template.name}' pip_packages must be a list"
        )
        # Verify all items in the list are strings
        for pkg in template.pip_packages:
            assert isinstance(pkg, str), (
                f"Template '{template.name}' has non-string package in pip_packages: {pkg}"
            )


# ============================================================================
# Property 7: Input field completeness
# ============================================================================

class TestInputFieldCompleteness:
    """
    **Property 7: Input field completeness**
    
    For any registered template, each input field SHALL have a name, type, 
    description, and required flag defined.
    
    **Validates: Requirements 6.2**
    """
    
    @settings(max_examples=100)
    @given(language_template_strategy())
    def test_input_fields_have_name(self, template):
        """All input fields must have a non-empty name."""
        for input_field in template.inputs:
            assert input_field.name, (
                f"Template '{template.name}' has input field with empty name"
            )
            assert isinstance(input_field.name, str), (
                f"Template '{template.name}' input field name must be a string"
            )
            assert len(input_field.name.strip()) > 0, (
                f"Template '{template.name}' has input field with whitespace-only name"
            )
    
    @settings(max_examples=100)
    @given(language_template_strategy())
    def test_input_fields_have_type(self, template):
        """All input fields must have a non-empty type."""
        for input_field in template.inputs:
            assert input_field.type, (
                f"Template '{template.name}' has input field '{input_field.name}' with empty type"
            )
            assert isinstance(input_field.type, str), (
                f"Template '{template.name}' input field '{input_field.name}' type must be a string"
            )
            assert len(input_field.type.strip()) > 0, (
                f"Template '{template.name}' input field '{input_field.name}' has whitespace-only type"
            )
    
    @settings(max_examples=100)
    @given(language_template_strategy())
    def test_input_fields_have_description(self, template):
        """All input fields must have a non-empty description."""
        for input_field in template.inputs:
            assert input_field.description, (
                f"Template '{template.name}' has input field '{input_field.name}' with empty description"
            )
            assert isinstance(input_field.description, str), (
                f"Template '{template.name}' input field '{input_field.name}' description must be a string"
            )
            assert len(input_field.description.strip()) > 0, (
                f"Template '{template.name}' input field '{input_field.name}' has whitespace-only description"
            )
    
    @settings(max_examples=100)
    @given(language_template_strategy())
    def test_input_fields_have_required_flag(self, template):
        """All input fields must have a required flag defined."""
        for input_field in template.inputs:
            assert hasattr(input_field, 'required'), (
                f"Template '{template.name}' input field '{input_field.name}' missing 'required' attribute"
            )
            assert isinstance(input_field.required, bool), (
                f"Template '{template.name}' input field '{input_field.name}' required flag must be a boolean"
            )


# ============================================================================
# Property 8: Output field completeness
# ============================================================================

class TestOutputFieldCompleteness:
    """
    **Property 8: Output field completeness**
    
    For any registered template, each output field SHALL have a name, type, 
    and description defined.
    
    **Validates: Requirements 6.3**
    """
    
    @settings(max_examples=100)
    @given(language_template_strategy())
    def test_output_fields_have_name(self, template):
        """All output fields must have a non-empty name."""
        for output_field in template.outputs:
            assert output_field.name, (
                f"Template '{template.name}' has output field with empty name"
            )
            assert isinstance(output_field.name, str), (
                f"Template '{template.name}' output field name must be a string"
            )
            assert len(output_field.name.strip()) > 0, (
                f"Template '{template.name}' has output field with whitespace-only name"
            )
    
    @settings(max_examples=100)
    @given(language_template_strategy())
    def test_output_fields_have_type(self, template):
        """All output fields must have a non-empty type."""
        for output_field in template.outputs:
            assert output_field.type, (
                f"Template '{template.name}' has output field '{output_field.name}' with empty type"
            )
            assert isinstance(output_field.type, str), (
                f"Template '{template.name}' output field '{output_field.name}' type must be a string"
            )
            assert len(output_field.type.strip()) > 0, (
                f"Template '{template.name}' output field '{output_field.name}' has whitespace-only type"
            )
    
    @settings(max_examples=100)
    @given(language_template_strategy())
    def test_output_fields_have_description(self, template):
        """All output fields must have a non-empty description."""
        for output_field in template.outputs:
            assert output_field.description, (
                f"Template '{template.name}' has output field '{output_field.name}' with empty description"
            )
            assert isinstance(output_field.description, str), (
                f"Template '{template.name}' output field '{output_field.name}' description must be a string"
            )
            assert len(output_field.description.strip()) > 0, (
                f"Template '{template.name}' output field '{output_field.name}' has whitespace-only description"
            )
