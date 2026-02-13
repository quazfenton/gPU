"""
Unit tests for template execution validation.

Tests specific examples and edge cases for:
- Input validation with various invalid inputs
- Setup initialization behavior
- Output schema validation
- Error message quality
"""

import pytest
from typing import Dict, Any
from templates.base import Template, InputField, OutputField, RouteType


class SimpleTemplate(Template):
    """Simple template for testing."""
    
    name = "simple-template"
    category = "Test"
    description = "Simple test template"
    version = "1.0.0"
    
    inputs = [
        InputField(name="text_input", type="text", required=True, description="Text input field"),
        InputField(name="number_input", type="number", required=False, default=10, description="Number input field"),
    ]
    
    outputs = [
        OutputField(name="output_text", type="text", description="Output text"),
        OutputField(name="output_number", type="number", description="Output number"),
    ]
    
    routing = [RouteType.LOCAL]
    memory_mb = 512
    timeout_sec = 60
    pip_packages = []
    
    def run(self, **kwargs) -> Dict[str, Any]:
        return {
            "output_text": f"Processed: {kwargs['text_input']}",
            "output_number": kwargs.get('number_input', 10) * 2
        }


class TestInputValidation:
    """Test input validation with various invalid inputs."""
    
    def test_missing_required_field(self):
        """Test validation fails when required field is missing."""
        template = SimpleTemplate()
        
        with pytest.raises(ValueError) as exc_info:
            template.validate_inputs(number_input=5)  # Missing text_input
        
        error_msg = str(exc_info.value)
        assert "text_input" in error_msg
        assert "required" in error_msg.lower()
        assert "simple-template" in error_msg
    
    def test_wrong_type_for_text_field(self):
        """Test validation fails when text field receives non-string."""
        template = SimpleTemplate()
        
        with pytest.raises(ValueError) as exc_info:
            template.validate_inputs(text_input=123)  # Should be string
        
        error_msg = str(exc_info.value)
        assert "text_input" in error_msg
        assert "type" in error_msg.lower()
        assert "simple-template" in error_msg
    
    def test_wrong_type_for_number_field(self):
        """Test validation fails when number field receives non-numeric."""
        template = SimpleTemplate()
        
        with pytest.raises(ValueError) as exc_info:
            template.validate_inputs(text_input="hello", number_input="not a number")
        
        error_msg = str(exc_info.value)
        assert "number_input" in error_msg
        assert "type" in error_msg.lower()
    
    def test_valid_inputs_with_all_fields(self):
        """Test validation passes with all valid fields."""
        template = SimpleTemplate()
        
        result = template.validate_inputs(text_input="hello", number_input=42)
        assert result is True
    
    def test_valid_inputs_with_only_required_fields(self):
        """Test validation passes with only required fields."""
        template = SimpleTemplate()
        
        result = template.validate_inputs(text_input="hello")
        assert result is True
    
    def test_empty_string_is_valid_for_text(self):
        """Test that empty string is valid for text fields."""
        template = SimpleTemplate()
        
        result = template.validate_inputs(text_input="")
        assert result is True
    
    def test_zero_is_valid_for_number(self):
        """Test that zero is valid for number fields."""
        template = SimpleTemplate()
        
        result = template.validate_inputs(text_input="test", number_input=0)
        assert result is True
    
    def test_negative_number_is_valid(self):
        """Test that negative numbers are valid."""
        template = SimpleTemplate()
        
        result = template.validate_inputs(text_input="test", number_input=-5)
        assert result is True
    
    def test_float_is_valid_for_number(self):
        """Test that floats are valid for number fields."""
        template = SimpleTemplate()
        
        result = template.validate_inputs(text_input="test", number_input=3.14)
        assert result is True


class TestSetupInitialization:
    """Test setup initialization behavior."""
    
    def test_setup_called_on_first_execute(self):
        """Test that setup is called on first execute."""
        class TrackingTemplate(SimpleTemplate):
            def __init__(self):
                super().__init__()
                self.setup_count = 0
            
            def setup(self):
                super().setup()
                self.setup_count += 1
        
        template = TrackingTemplate()
        assert template.setup_count == 0
        assert template._initialized is False
        
        template.execute(text_input="test")
        
        assert template.setup_count == 1
        assert template._initialized is True
    
    def test_setup_not_called_on_subsequent_executes(self):
        """Test that setup is not called on subsequent executes."""
        class TrackingTemplate(SimpleTemplate):
            def __init__(self):
                super().__init__()
                self.setup_count = 0
            
            def setup(self):
                super().setup()
                self.setup_count += 1
        
        template = TrackingTemplate()
        
        template.execute(text_input="test1")
        template.execute(text_input="test2")
        template.execute(text_input="test3")
        
        assert template.setup_count == 1
    
    def test_manual_setup_prevents_auto_setup(self):
        """Test that manually calling setup prevents auto-setup."""
        class TrackingTemplate(SimpleTemplate):
            def __init__(self):
                super().__init__()
                self.setup_count = 0
            
            def setup(self):
                super().setup()
                self.setup_count += 1
        
        template = TrackingTemplate()
        template.setup()
        
        assert template.setup_count == 1
        
        template.execute(text_input="test")
        
        # Should still be 1, not 2
        assert template.setup_count == 1


class TestOutputValidation:
    """Test output schema validation."""
    
    def test_missing_output_field(self):
        """Test validation fails when output field is missing."""
        class IncompleteOutputTemplate(Template):
            name = "incomplete"
            category = "Test"
            description = "Test"
            version = "1.0.0"
            
            inputs = [InputField(name="input", type="text", required=True)]
            outputs = [
                OutputField(name="output1", type="text", description="First output"),
                OutputField(name="output2", type="text", description="Second output"),
            ]
            
            routing = [RouteType.LOCAL]
            memory_mb = 512
            timeout_sec = 60
            pip_packages = []
            
            def run(self, **kwargs) -> Dict[str, Any]:
                return {"output1": "value1"}  # Missing output2
        
        template = IncompleteOutputTemplate()
        
        with pytest.raises(ValueError) as exc_info:
            template.execute(input="test")
        
        error_msg = str(exc_info.value)
        assert "output2" in error_msg
        assert "missing" in error_msg.lower()
        assert "incomplete" in error_msg
    
    def test_all_outputs_present(self):
        """Test validation passes when all outputs are present."""
        template = SimpleTemplate()
        
        result = template.execute(text_input="hello")
        
        assert "output_text" in result
        assert "output_number" in result
    
    def test_extra_outputs_allowed(self):
        """Test that extra outputs beyond schema are allowed."""
        class ExtraOutputTemplate(Template):
            name = "extra"
            category = "Test"
            description = "Test"
            version = "1.0.0"
            
            inputs = [InputField(name="input", type="text", required=True)]
            outputs = [OutputField(name="output", type="text", description="Output")]
            
            routing = [RouteType.LOCAL]
            memory_mb = 512
            timeout_sec = 60
            pip_packages = []
            
            def run(self, **kwargs) -> Dict[str, Any]:
                return {
                    "output": "value",
                    "extra_field": "extra_value"
                }
        
        template = ExtraOutputTemplate()
        result = template.execute(input="test")
        
        assert result["output"] == "value"
        assert result["extra_field"] == "extra_value"


class TestErrorMessages:
    """Test error message quality."""
    
    def test_input_validation_error_includes_template_name(self):
        """Test that input validation errors include template name."""
        template = SimpleTemplate()
        
        with pytest.raises(ValueError) as exc_info:
            template.validate_inputs()  # Missing required field
        
        assert "simple-template" in str(exc_info.value)
    
    def test_input_validation_error_includes_field_info(self):
        """Test that input validation errors include field information."""
        template = SimpleTemplate()
        
        with pytest.raises(ValueError) as exc_info:
            template.validate_inputs()  # Missing text_input
        
        error_msg = str(exc_info.value)
        assert "text_input" in error_msg
        assert "text" in error_msg  # type
        assert "Text input field" in error_msg  # description
    
    def test_output_validation_error_includes_template_name(self):
        """Test that output validation errors include template name."""
        class BadOutputTemplate(Template):
            name = "bad-output"
            category = "Test"
            description = "Test"
            version = "1.0.0"
            
            inputs = [InputField(name="input", type="text", required=True)]
            outputs = [OutputField(name="output", type="text", description="Output field")]
            
            routing = [RouteType.LOCAL]
            memory_mb = 512
            timeout_sec = 60
            pip_packages = []
            
            def run(self, **kwargs) -> Dict[str, Any]:
                return {}  # Missing output
        
        template = BadOutputTemplate()
        
        with pytest.raises(ValueError) as exc_info:
            template.execute(input="test")
        
        assert "bad-output" in str(exc_info.value)
    
    def test_execution_error_includes_template_name(self):
        """Test that execution errors include template name."""
        class FailingTemplate(Template):
            name = "failing"
            category = "Test"
            description = "Test"
            version = "1.0.0"
            
            inputs = [InputField(name="input", type="text", required=True)]
            outputs = [OutputField(name="output", type="text", description="Output")]
            
            routing = [RouteType.LOCAL]
            memory_mb = 512
            timeout_sec = 60
            pip_packages = []
            
            def run(self, **kwargs) -> Dict[str, Any]:
                raise RuntimeError("Something went wrong")
        
        template = FailingTemplate()
        
        with pytest.raises(RuntimeError) as exc_info:
            template.execute(input="test")
        
        error_msg = str(exc_info.value)
        assert "failing" in error_msg
        assert "execution failed" in error_msg.lower()
    
    def test_execution_error_preserves_original_error(self):
        """Test that execution errors preserve the original error."""
        class FailingTemplate(Template):
            name = "failing"
            category = "Test"
            description = "Test"
            version = "1.0.0"
            
            inputs = [InputField(name="input", type="text", required=True)]
            outputs = [OutputField(name="output", type="text", description="Output")]
            
            routing = [RouteType.LOCAL]
            memory_mb = 512
            timeout_sec = 60
            pip_packages = []
            
            def run(self, **kwargs) -> Dict[str, Any]:
                raise ValueError("Original error message")
        
        template = FailingTemplate()
        
        with pytest.raises(RuntimeError) as exc_info:
            template.execute(input="test")
        
        # Original error should be in the chain
        assert exc_info.value.__cause__ is not None
        assert "Original error message" in str(exc_info.value.__cause__)


class TestFileTypeValidation:
    """Test validation for file-type inputs (audio, image, video, file)."""
    
    def test_audio_field_accepts_string_path(self):
        """Test that audio fields accept string paths."""
        class AudioTemplate(Template):
            name = "audio"
            category = "Audio"
            description = "Test"
            version = "1.0.0"
            
            inputs = [InputField(name="audio", type="audio", required=True)]
            outputs = [OutputField(name="result", type="text")]
            
            routing = [RouteType.LOCAL]
            memory_mb = 512
            timeout_sec = 60
            pip_packages = []
            
            def run(self, **kwargs) -> Dict[str, Any]:
                return {"result": "processed"}
        
        template = AudioTemplate()
        assert template.validate_inputs(audio="/path/to/audio.wav") is True
    
    def test_audio_field_accepts_bytes(self):
        """Test that audio fields accept bytes."""
        class AudioTemplate(Template):
            name = "audio"
            category = "Audio"
            description = "Test"
            version = "1.0.0"
            
            inputs = [InputField(name="audio", type="audio", required=True)]
            outputs = [OutputField(name="result", type="text")]
            
            routing = [RouteType.LOCAL]
            memory_mb = 512
            timeout_sec = 60
            pip_packages = []
            
            def run(self, **kwargs) -> Dict[str, Any]:
                return {"result": "processed"}
        
        template = AudioTemplate()
        assert template.validate_inputs(audio=b"audio data") is True
    
    def test_audio_field_rejects_invalid_type(self):
        """Test that audio fields reject invalid types."""
        class AudioTemplate(Template):
            name = "audio"
            category = "Audio"
            description = "Test"
            version = "1.0.0"
            
            inputs = [InputField(name="audio", type="audio", required=True)]
            outputs = [OutputField(name="result", type="text")]
            
            routing = [RouteType.LOCAL]
            memory_mb = 512
            timeout_sec = 60
            pip_packages = []
            
            def run(self, **kwargs) -> Dict[str, Any]:
                return {"result": "processed"}
        
        template = AudioTemplate()
        
        with pytest.raises(ValueError) as exc_info:
            template.validate_inputs(audio=123)  # Invalid type
        
        assert "audio" in str(exc_info.value)
        assert "type" in str(exc_info.value).lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
