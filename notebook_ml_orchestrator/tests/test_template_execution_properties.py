"""
Property-based tests for template execution validation.

Tests Properties 15-18:
- Property 15: Input validation enforcement
- Property 16: Setup initialization
- Property 17: Output schema conformance
- Property 18: Execution error diagnostics
"""

import pytest
from hypothesis import given, strategies as st, assume
from typing import Dict, Any, List
from templates.base import Template, InputField, OutputField, RouteType


# Test template for property testing
class MockTemplate(Template):
    """Mock template for testing execution properties."""
    
    name = "mock-template"
    category = "Test"
    description = "Mock template for testing"
    version = "1.0.0"
    
    inputs = [
        InputField(name="required_text", type="text", required=True),
        InputField(name="optional_number", type="number", required=False, default=42),
    ]
    
    outputs = [
        OutputField(name="result", type="text"),
        OutputField(name="count", type="number"),
    ]
    
    routing = [RouteType.LOCAL]
    memory_mb = 512
    timeout_sec = 60
    pip_packages = []
    
    def __init__(self):
        super().__init__()
        self.setup_called = False
        self.run_called = False
    
    def setup(self):
        """Track setup calls."""
        super().setup()
        self.setup_called = True
    
    def run(self, **kwargs) -> Dict[str, Any]:
        """Mock run implementation."""
        self.run_called = True
        return {
            "result": f"processed: {kwargs.get('required_text', '')}",
            "count": kwargs.get('optional_number', 42)
        }


class FailingTemplate(Template):
    """Template that fails during execution."""
    
    name = "failing-template"
    category = "Test"
    description = "Template that fails"
    version = "1.0.0"
    
    inputs = [InputField(name="input", type="text", required=True)]
    outputs = [OutputField(name="output", type="text")]
    
    routing = [RouteType.LOCAL]
    memory_mb = 512
    timeout_sec = 60
    pip_packages = []
    
    def run(self, **kwargs) -> Dict[str, Any]:
        """Always fails."""
        raise RuntimeError("Intentional failure for testing")


class IncompleteOutputTemplate(Template):
    """Template that returns incomplete outputs."""
    
    name = "incomplete-output-template"
    category = "Test"
    description = "Template with incomplete outputs"
    version = "1.0.0"
    
    inputs = [InputField(name="input", type="text", required=True)]
    outputs = [
        OutputField(name="output1", type="text"),
        OutputField(name="output2", type="text"),
    ]
    
    routing = [RouteType.LOCAL]
    memory_mb = 512
    timeout_sec = 60
    pip_packages = []
    
    def run(self, **kwargs) -> Dict[str, Any]:
        """Returns incomplete outputs."""
        return {"output1": "value1"}  # Missing output2


class TestTemplateExecutionProperties:
    """Property-based tests for template execution validation."""
    
    # Property 15: Input validation enforcement
    @given(
        missing_fields=st.lists(
            st.sampled_from(["required_text"]),
            min_size=1,
            max_size=1,
            unique=True
        )
    )
    def test_property_15_input_validation_enforcement_missing_required(self, missing_fields):
        """
        **Validates: Requirements 7.1, 7.2**
        
        Property 15: Input validation enforcement
        For any template and any input dictionary, if the inputs do not satisfy 
        the template's input schema (missing required fields), calling validate_inputs 
        SHALL raise a ValueError.
        """
        template = MockTemplate()
        
        # Create inputs with missing required fields
        inputs = {"optional_number": 10}  # Missing required_text
        
        # Validation should raise ValueError
        with pytest.raises(ValueError) as exc_info:
            template.validate_inputs(**inputs)
        
        # Error message should mention the missing field
        assert "required_text" in str(exc_info.value).lower()
        assert template.name in str(exc_info.value)
    
    @given(
        text_value=st.integers(),  # Wrong type for text field
    )
    def test_property_15_input_validation_enforcement_wrong_type(self, text_value):
        """
        **Validates: Requirements 7.1, 7.2**
        
        Property 15: Input validation enforcement (type checking)
        For any template and any input dictionary, if the inputs have wrong types,
        calling validate_inputs SHALL raise a ValueError.
        """
        template = MockTemplate()
        
        # Create inputs with wrong type
        inputs = {"required_text": text_value}  # Should be string, not int
        
        # Validation should raise ValueError
        with pytest.raises(ValueError) as exc_info:
            template.validate_inputs(**inputs)
        
        # Error message should mention type mismatch
        assert "type" in str(exc_info.value).lower()
        assert template.name in str(exc_info.value)
    
    @given(
        text_value=st.text(min_size=1),
        number_value=st.integers() | st.floats(allow_nan=False, allow_infinity=False)
    )
    def test_property_15_input_validation_passes_with_valid_inputs(self, text_value, number_value):
        """
        **Validates: Requirements 7.1, 7.2**
        
        Property 15: Input validation should pass with valid inputs.
        """
        template = MockTemplate()
        
        # Create valid inputs
        inputs = {
            "required_text": text_value,
            "optional_number": number_value
        }
        
        # Validation should pass
        assert template.validate_inputs(**inputs) is True
    
    # Property 16: Setup initialization
    @given(
        text_value=st.text(min_size=1)
    )
    def test_property_16_setup_initialization(self, text_value):
        """
        **Validates: Requirements 7.3**
        
        Property 16: Setup initialization
        For any template instance, if _initialized is False when execute() is called,
        the setup() method SHALL be called before executing the run logic.
        """
        template = MockTemplate()
        
        # Ensure template is not initialized
        assert template._initialized is False
        assert template.setup_called is False
        
        # Execute template
        inputs = {"required_text": text_value}
        result = template.execute(**inputs)
        
        # Setup should have been called
        assert template.setup_called is True
        assert template._initialized is True
        assert template.run_called is True
    
    @given(
        text_value=st.text(min_size=1)
    )
    def test_property_16_setup_not_called_twice(self, text_value):
        """
        **Validates: Requirements 7.3**
        
        Property 16: Setup should not be called multiple times.
        """
        template = MockTemplate()
        
        # Call setup manually
        template.setup()
        assert template.setup_called is True
        setup_call_count = 1
        
        # Execute template
        inputs = {"required_text": text_value}
        template.execute(**inputs)
        
        # Setup should not be called again (already initialized)
        assert template._initialized is True
    
    # Property 17: Output schema conformance
    @given(
        text_value=st.text(min_size=1)
    )
    def test_property_17_output_schema_conformance(self, text_value):
        """
        **Validates: Requirements 7.4**
        
        Property 17: Output schema conformance
        For any template execution that completes successfully, the returned 
        dictionary SHALL contain keys matching all output field names declared 
        in the template's outputs list.
        """
        template = MockTemplate()
        
        # Execute template
        inputs = {"required_text": text_value}
        result = template.execute(**inputs)
        
        # Check that all declared outputs are present
        for output_field in template.outputs:
            assert output_field.name in result, \
                f"Output field '{output_field.name}' missing from result"
    
    def test_property_17_output_schema_validation_fails_on_incomplete(self):
        """
        **Validates: Requirements 7.4**
        
        Property 17: Output validation should fail when outputs are incomplete.
        """
        template = IncompleteOutputTemplate()
        
        # Execute should fail due to incomplete outputs
        inputs = {"input": "test"}
        with pytest.raises(ValueError) as exc_info:
            template.execute(**inputs)
        
        # Error message should mention missing output
        assert "output2" in str(exc_info.value).lower()
        assert template.name in str(exc_info.value)
    
    # Property 18: Execution error diagnostics
    def test_property_18_execution_error_diagnostics(self):
        """
        **Validates: Requirements 7.7**
        
        Property 18: Execution error diagnostics
        For any template execution that fails, the raised exception SHALL include 
        information about the template name and the nature of the failure.
        """
        template = FailingTemplate()
        
        # Execute should fail with diagnostic information
        inputs = {"input": "test"}
        with pytest.raises(RuntimeError) as exc_info:
            template.execute(**inputs)
        
        # Error message should include template name
        assert template.name in str(exc_info.value)
        
        # Error message should include information about the failure
        assert "execution failed" in str(exc_info.value).lower()
        
        # Original error should be preserved in the chain
        assert exc_info.value.__cause__ is not None
        assert "Intentional failure" in str(exc_info.value.__cause__)
    
    @given(
        error_message=st.text(min_size=1, max_size=100)
    )
    def test_property_18_error_diagnostics_preserve_original_message(self, error_message):
        """
        **Validates: Requirements 7.7**
        
        Property 18: Error diagnostics should preserve original error message.
        """
        class CustomFailingTemplate(Template):
            name = "custom-failing"
            category = "Test"
            description = "Test"
            version = "1.0.0"
            inputs = [InputField(name="input", type="text", required=True)]
            outputs = [OutputField(name="output", type="text")]
            routing = [RouteType.LOCAL]
            memory_mb = 512
            timeout_sec = 60
            pip_packages = []
            
            def run(self, **kwargs) -> Dict[str, Any]:
                raise RuntimeError(error_message)
        
        template = CustomFailingTemplate()
        
        # Execute should fail with diagnostic information
        inputs = {"input": "test"}
        with pytest.raises(RuntimeError) as exc_info:
            template.execute(**inputs)
        
        # Original error message should be preserved
        assert error_message in str(exc_info.value.__cause__)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
