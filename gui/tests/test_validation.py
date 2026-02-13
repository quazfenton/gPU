"""Tests for input validation module."""

import pytest
from templates.base import Template, InputField
from gui.validation import (
    validate_inputs,
    ValidationError,
    format_validation_errors,
    _validate_type,
    _validate_number_range
)


class MockTemplate(Template):
    """Mock template for testing."""
    
    name = "test_template"
    category = "Test"
    description = "Test template"
    
    def __init__(self, input_fields=None):
        super().__init__()
        if input_fields:
            self.inputs = input_fields
    
    def run(self, **kwargs):
        return {}


class TestValidationError:
    """Test ValidationError class."""
    
    def test_validation_error_creation(self):
        """Test creating a ValidationError."""
        error = ValidationError("field1", "Error message")
        assert error.field == "field1"
        assert error.message == "Error message"
    
    def test_validation_error_repr(self):
        """Test ValidationError string representation."""
        error = ValidationError("field1", "Error message")
        assert "field1" in repr(error)
        assert "Error message" in repr(error)
    
    def test_validation_error_equality(self):
        """Test ValidationError equality comparison."""
        error1 = ValidationError("field1", "Error message")
        error2 = ValidationError("field1", "Error message")
        error3 = ValidationError("field2", "Error message")
        
        assert error1 == error2
        assert error1 != error3
        assert error1 != "not an error"


class TestValidateInputs:
    """Test validate_inputs function."""
    
    def test_empty_inputs_with_no_required_fields(self):
        """Test validation with empty inputs when no fields are required."""
        template = MockTemplate([
            InputField(name="optional_field", type="text", required=False)
        ])
        
        is_valid, errors = validate_inputs(template, {})
        
        assert is_valid
        assert len(errors) == 0
    
    def test_missing_required_field(self):
        """Test validation fails when required field is missing."""
        template = MockTemplate([
            InputField(name="required_field", type="text", required=True)
        ])
        
        is_valid, errors = validate_inputs(template, {})
        
        assert not is_valid
        assert len(errors) == 1
        assert errors[0].field == "required_field"
        assert "missing" in errors[0].message.lower()
    
    def test_multiple_missing_required_fields(self):
        """Test validation fails when multiple required fields are missing."""
        template = MockTemplate([
            InputField(name="field1", type="text", required=True),
            InputField(name="field2", type="number", required=True),
            InputField(name="field3", type="text", required=False)
        ])
        
        is_valid, errors = validate_inputs(template, {})
        
        assert not is_valid
        assert len(errors) == 2
        error_fields = {e.field for e in errors}
        assert "field1" in error_fields
        assert "field2" in error_fields
    
    def test_unknown_field(self):
        """Test validation fails when unknown field is provided."""
        template = MockTemplate([
            InputField(name="known_field", type="text", required=False)
        ])
        
        is_valid, errors = validate_inputs(template, {"unknown_field": "value"})
        
        assert not is_valid
        assert len(errors) == 1
        assert errors[0].field == "unknown_field"
        assert "unknown" in errors[0].message.lower()
    
    def test_valid_string_input(self):
        """Test validation passes for valid string input."""
        template = MockTemplate([
            InputField(name="text_field", type="text", required=True)
        ])
        
        is_valid, errors = validate_inputs(template, {"text_field": "hello"})
        
        assert is_valid
        assert len(errors) == 0
    
    def test_invalid_string_type(self):
        """Test validation fails when string field receives non-string value."""
        template = MockTemplate([
            InputField(name="text_field", type="text", required=True)
        ])
        
        is_valid, errors = validate_inputs(template, {"text_field": 123})
        
        assert not is_valid
        assert len(errors) == 1
        assert errors[0].field == "text_field"
        assert "string" in errors[0].message.lower()
    
    def test_valid_number_input(self):
        """Test validation passes for valid number input."""
        template = MockTemplate([
            InputField(name="num_field", type="number", required=True)
        ])
        
        is_valid, errors = validate_inputs(template, {"num_field": 42})
        
        assert is_valid
        assert len(errors) == 0
    
    def test_valid_float_input(self):
        """Test validation passes for valid float input."""
        template = MockTemplate([
            InputField(name="num_field", type="number", required=True)
        ])
        
        is_valid, errors = validate_inputs(template, {"num_field": 3.14})
        
        assert is_valid
        assert len(errors) == 0
    
    def test_invalid_number_type(self):
        """Test validation fails when number field receives non-number value."""
        template = MockTemplate([
            InputField(name="num_field", type="number", required=True)
        ])
        
        is_valid, errors = validate_inputs(template, {"num_field": "not a number"})
        
        assert not is_valid
        assert len(errors) == 1
        assert errors[0].field == "num_field"
        assert "number" in errors[0].message.lower()
    
    def test_valid_json_dict_input(self):
        """Test validation passes for valid JSON dict input."""
        template = MockTemplate([
            InputField(name="json_field", type="json", required=True)
        ])
        
        is_valid, errors = validate_inputs(template, {"json_field": {"key": "value"}})
        
        assert is_valid
        assert len(errors) == 0
    
    def test_valid_json_list_input(self):
        """Test validation passes for valid JSON list input."""
        template = MockTemplate([
            InputField(name="json_field", type="json", required=True)
        ])
        
        is_valid, errors = validate_inputs(template, {"json_field": [1, 2, 3]})
        
        assert is_valid
        assert len(errors) == 0
    
    def test_invalid_json_type(self):
        """Test validation fails when JSON field receives non-JSON value."""
        template = MockTemplate([
            InputField(name="json_field", type="json", required=True)
        ])
        
        is_valid, errors = validate_inputs(template, {"json_field": "not json"})
        
        assert not is_valid
        assert len(errors) == 1
        assert errors[0].field == "json_field"
        assert "json" in errors[0].message.lower()
    
    def test_valid_file_input(self):
        """Test validation passes for valid file path input."""
        template = MockTemplate([
            InputField(name="file_field", type="file", required=True)
        ])
        
        is_valid, errors = validate_inputs(template, {"file_field": "/path/to/file.txt"})
        
        assert is_valid
        assert len(errors) == 0
    
    def test_invalid_file_type(self):
        """Test validation fails when file field receives non-string value."""
        template = MockTemplate([
            InputField(name="file_field", type="file", required=True)
        ])
        
        is_valid, errors = validate_inputs(template, {"file_field": 123})
        
        assert not is_valid
        assert len(errors) == 1
        assert errors[0].field == "file_field"
    
    def test_valid_audio_file_path(self):
        """Test validation passes for valid audio file path."""
        template = MockTemplate([
            InputField(name="audio_field", type="audio", required=True)
        ])
        
        is_valid, errors = validate_inputs(template, {"audio_field": "/path/to/audio.mp3"})
        
        assert is_valid
        assert len(errors) == 0
    
    def test_valid_audio_binary_data(self):
        """Test validation passes for valid audio binary data."""
        template = MockTemplate([
            InputField(name="audio_field", type="audio", required=True)
        ])
        
        is_valid, errors = validate_inputs(template, {"audio_field": b"binary audio data"})
        
        assert is_valid
        assert len(errors) == 0
    
    def test_valid_image_file_path(self):
        """Test validation passes for valid image file path."""
        template = MockTemplate([
            InputField(name="image_field", type="image", required=True)
        ])
        
        is_valid, errors = validate_inputs(template, {"image_field": "/path/to/image.jpg"})
        
        assert is_valid
        assert len(errors) == 0
    
    def test_valid_video_file_path(self):
        """Test validation passes for valid video file path."""
        template = MockTemplate([
            InputField(name="video_field", type="video", required=True)
        ])
        
        is_valid, errors = validate_inputs(template, {"video_field": "/path/to/video.mp4"})
        
        assert is_valid
        assert len(errors) == 0
    
    def test_none_value_for_optional_field(self):
        """Test validation passes when None is provided for optional field."""
        template = MockTemplate([
            InputField(name="optional_field", type="text", required=False)
        ])
        
        is_valid, errors = validate_inputs(template, {"optional_field": None})
        
        assert is_valid
        assert len(errors) == 0
    
    def test_none_value_for_required_field(self):
        """Test validation fails when None is provided for required field."""
        template = MockTemplate([
            InputField(name="required_field", type="text", required=True)
        ])
        
        is_valid, errors = validate_inputs(template, {"required_field": None})
        
        assert not is_valid
        assert len(errors) == 1
        assert errors[0].field == "required_field"
        assert "none" in errors[0].message.lower()
    
    def test_options_validation_valid(self):
        """Test validation passes when value is in allowed options."""
        template = MockTemplate([
            InputField(name="choice_field", type="text", required=True, options=["option1", "option2", "option3"])
        ])
        
        is_valid, errors = validate_inputs(template, {"choice_field": "option2"})
        
        assert is_valid
        assert len(errors) == 0
    
    def test_options_validation_invalid(self):
        """Test validation fails when value is not in allowed options."""
        template = MockTemplate([
            InputField(name="choice_field", type="text", required=True, options=["option1", "option2", "option3"])
        ])
        
        is_valid, errors = validate_inputs(template, {"choice_field": "invalid_option"})
        
        assert not is_valid
        assert len(errors) == 1
        assert errors[0].field == "choice_field"
        assert "must be one of" in errors[0].message.lower()
    
    def test_number_range_validation_min(self):
        """Test validation fails when number is below minimum."""
        template = MockTemplate([
            InputField(name="num_field", type="number", required=True, options={"min": 0, "max": 100})
        ])
        
        is_valid, errors = validate_inputs(template, {"num_field": -5})
        
        assert not is_valid
        assert len(errors) == 1
        assert errors[0].field == "num_field"
        assert ">=" in errors[0].message
    
    def test_number_range_validation_max(self):
        """Test validation fails when number is above maximum."""
        template = MockTemplate([
            InputField(name="num_field", type="number", required=True, options={"min": 0, "max": 100})
        ])
        
        is_valid, errors = validate_inputs(template, {"num_field": 150})
        
        assert not is_valid
        assert len(errors) == 1
        assert errors[0].field == "num_field"
        assert "<=" in errors[0].message
    
    def test_number_range_validation_within_range(self):
        """Test validation passes when number is within range."""
        template = MockTemplate([
            InputField(name="num_field", type="number", required=True, options={"min": 0, "max": 100})
        ])
        
        is_valid, errors = validate_inputs(template, {"num_field": 50})
        
        assert is_valid
        assert len(errors) == 0
    
    def test_multiple_validation_errors(self):
        """Test validation collects multiple errors."""
        template = MockTemplate([
            InputField(name="field1", type="text", required=True),
            InputField(name="field2", type="number", required=True),
            InputField(name="field3", type="text", required=True, options=["a", "b", "c"])
        ])
        
        is_valid, errors = validate_inputs(template, {
            "field2": "not a number",
            "field3": "invalid"
        })
        
        assert not is_valid
        assert len(errors) == 3  # missing field1, wrong type field2, invalid option field3
        error_fields = {e.field for e in errors}
        assert "field1" in error_fields
        assert "field2" in error_fields
        assert "field3" in error_fields
    
    def test_complex_valid_inputs(self):
        """Test validation with multiple valid inputs of different types."""
        template = MockTemplate([
            InputField(name="text_field", type="text", required=True),
            InputField(name="num_field", type="number", required=True),
            InputField(name="json_field", type="json", required=False),
            InputField(name="file_field", type="file", required=False)
        ])
        
        is_valid, errors = validate_inputs(template, {
            "text_field": "hello",
            "num_field": 42,
            "json_field": {"key": "value"},
            "file_field": "/path/to/file.txt"
        })
        
        assert is_valid
        assert len(errors) == 0


class TestFormatValidationErrors:
    """Test format_validation_errors function."""
    
    def test_format_empty_errors(self):
        """Test formatting empty error list."""
        result = format_validation_errors([])
        assert result == ""
    
    def test_format_single_error(self):
        """Test formatting single error."""
        errors = [ValidationError("field1", "Field is required")]
        result = format_validation_errors(errors)
        
        assert "Validation error:" in result
        assert "Field is required" in result
    
    def test_format_multiple_errors(self):
        """Test formatting multiple errors."""
        errors = [
            ValidationError("field1", "Field is required"),
            ValidationError("field2", "Must be a number"),
            ValidationError("field3", "Invalid option")
        ]
        result = format_validation_errors(errors)
        
        assert "Validation errors:" in result
        assert "Field is required" in result
        assert "Must be a number" in result
        assert "Invalid option" in result
        assert result.count("-") == 3  # Three bullet points


class TestValidateType:
    """Test _validate_type helper function."""
    
    def test_validate_string_type_valid(self):
        """Test string type validation with valid input."""
        field = InputField(name="test", type="string", required=True)
        error = _validate_type(field, "hello")
        assert error is None
    
    def test_validate_text_type_valid(self):
        """Test text type validation with valid input."""
        field = InputField(name="test", type="text", required=True)
        error = _validate_type(field, "hello")
        assert error is None
    
    def test_validate_string_type_invalid(self):
        """Test string type validation with invalid input."""
        field = InputField(name="test", type="string", required=True)
        error = _validate_type(field, 123)
        assert error is not None
        assert "string" in error.lower()
    
    def test_validate_number_type_valid_int(self):
        """Test number type validation with valid integer."""
        field = InputField(name="test", type="number", required=True)
        error = _validate_type(field, 42)
        assert error is None
    
    def test_validate_number_type_valid_float(self):
        """Test number type validation with valid float."""
        field = InputField(name="test", type="number", required=True)
        error = _validate_type(field, 3.14)
        assert error is None
    
    def test_validate_number_type_invalid(self):
        """Test number type validation with invalid input."""
        field = InputField(name="test", type="number", required=True)
        error = _validate_type(field, "not a number")
        assert error is not None
        assert "number" in error.lower()
    
    def test_validate_unknown_type(self):
        """Test validation with unknown type accepts any value."""
        field = InputField(name="test", type="custom_type", required=True)
        error = _validate_type(field, "any value")
        assert error is None


class TestValidateNumberRange:
    """Test _validate_number_range helper function."""
    
    def test_validate_range_no_constraints(self):
        """Test range validation with no constraints."""
        field = InputField(name="test", type="number", required=True)
        error = _validate_number_range(field, 42)
        assert error is None
    
    def test_validate_range_within_bounds(self):
        """Test range validation within bounds."""
        field = InputField(name="test", type="number", required=True, options={"min": 0, "max": 100})
        error = _validate_number_range(field, 50)
        assert error is None
    
    def test_validate_range_below_min(self):
        """Test range validation below minimum."""
        field = InputField(name="test", type="number", required=True, options={"min": 0, "max": 100})
        error = _validate_number_range(field, -5)
        assert error is not None
        assert ">=" in error
    
    def test_validate_range_above_max(self):
        """Test range validation above maximum."""
        field = InputField(name="test", type="number", required=True, options={"min": 0, "max": 100})
        error = _validate_number_range(field, 150)
        assert error is not None
        assert "<=" in error
    
    def test_validate_range_at_min_boundary(self):
        """Test range validation at minimum boundary."""
        field = InputField(name="test", type="number", required=True, options={"min": 0, "max": 100})
        error = _validate_number_range(field, 0)
        assert error is None
    
    def test_validate_range_at_max_boundary(self):
        """Test range validation at maximum boundary."""
        field = InputField(name="test", type="number", required=True, options={"min": 0, "max": 100})
        error = _validate_number_range(field, 100)
        assert error is None
    
    def test_validate_range_only_min(self):
        """Test range validation with only minimum constraint."""
        field = InputField(name="test", type="number", required=True, options={"min": 0})
        error = _validate_number_range(field, 50)
        assert error is None
        
        error = _validate_number_range(field, -5)
        assert error is not None
    
    def test_validate_range_only_max(self):
        """Test range validation with only maximum constraint."""
        field = InputField(name="test", type="number", required=True, options={"max": 100})
        error = _validate_number_range(field, 50)
        assert error is None
        
        error = _validate_number_range(field, 150)
        assert error is not None
