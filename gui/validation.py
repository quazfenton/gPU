"""
Input validation module for GUI interface.

Validates user inputs against template schemas, supporting:
- Type validation (string, number, file, etc.)
- Required field validation
- Value range validation
- Descriptive error messages
"""

from typing import Any, Dict, List, Optional, Tuple
from templates.base import Template, InputField


class ValidationError:
    """Represents a validation error with field and message."""
    
    def __init__(self, field: str, message: str):
        self.field = field
        self.message = message
    
    def __repr__(self) -> str:
        return f"ValidationError(field='{self.field}', message='{self.message}')"
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, ValidationError):
            return False
        return self.field == other.field and self.message == other.message


def validate_inputs(template: Template, inputs: Dict[str, Any]) -> Tuple[bool, List[ValidationError]]:
    """
    Validate inputs against template schema.
    
    Args:
        template: Template instance with input field definitions
        inputs: Dictionary of input values to validate
        
    Returns:
        Tuple of (is_valid, list of ValidationError objects)
        
    Validates:
        - Required fields are present
        - Field types match expected types
        - Values are within valid ranges (for numbers)
        - Values are in allowed options (if options specified)
    """
    errors: List[ValidationError] = []
    
    # Create a map of input fields for easy lookup
    input_fields_map = {field.name: field for field in template.inputs}
    
    # Check required fields
    for field in template.inputs:
        if field.required and field.name not in inputs:
            errors.append(ValidationError(
                field=field.name,
                message=f"Required field '{field.name}' is missing"
            ))
    
    # Validate each provided input
    for input_name, input_value in inputs.items():
        # Check if field exists in template
        if input_name not in input_fields_map:
            errors.append(ValidationError(
                field=input_name,
                message=f"Unknown field '{input_name}' not defined in template"
            ))
            continue
        
        field = input_fields_map[input_name]
        
        # Skip validation for None values if field is not required
        if input_value is None:
            if field.required:
                errors.append(ValidationError(
                    field=input_name,
                    message=f"Required field '{input_name}' cannot be None"
                ))
            continue
        
        # Type validation
        type_error = _validate_type(field, input_value)
        if type_error:
            errors.append(ValidationError(field=input_name, message=type_error))
            continue
        
        # Options validation (if options are specified as a list)
        if field.options is not None and isinstance(field.options, list) and len(field.options) > 0:
            if input_value not in field.options:
                errors.append(ValidationError(
                    field=input_name,
                    message=f"Value '{input_value}' for field '{input_name}' must be one of {field.options}"
                ))
        
        # Range validation for numbers (if options are specified as a dict with min/max)
        if field.type == "number":
            range_error = _validate_number_range(field, input_value)
            if range_error:
                errors.append(ValidationError(field=input_name, message=range_error))
    
    is_valid = len(errors) == 0
    return is_valid, errors


def _validate_type(field: InputField, value: Any) -> Optional[str]:
    """
    Validate that value matches the expected field type.
    
    Args:
        field: Input field definition
        value: Value to validate
        
    Returns:
        Error message if validation fails, None otherwise
    """
    field_type = field.type.lower()
    
    if field_type == "text" or field_type == "string":
        if not isinstance(value, str):
            return f"Field '{field.name}' must be a string, got {type(value).__name__}"
    
    elif field_type == "number":
        if not isinstance(value, (int, float)):
            return f"Field '{field.name}' must be a number, got {type(value).__name__}"
    
    elif field_type == "json":
        if not isinstance(value, (dict, list)):
            return f"Field '{field.name}' must be a JSON object or array, got {type(value).__name__}"
    
    elif field_type == "file":
        if not isinstance(value, str):
            return f"Field '{field.name}' must be a file path (string), got {type(value).__name__}"
    
    elif field_type in ["audio", "image", "video"]:
        # These are typically file paths or binary data
        if not isinstance(value, (str, bytes)):
            return f"Field '{field.name}' must be a file path (string) or binary data, got {type(value).__name__}"
    
    # For other types, accept any value (extensibility)
    return None


def _validate_number_range(field: InputField, value: float) -> Optional[str]:
    """
    Validate that a number is within valid range if specified in field metadata.
    
    Args:
        field: Input field definition
        value: Numeric value to validate
        
    Returns:
        Error message if validation fails, None otherwise
        
    Note:
        Range constraints can be specified in field.options as a dict with 'min' and 'max' keys.
        This is a common pattern for numeric inputs.
    """
    # Check if options contains range constraints
    if field.options and isinstance(field.options, dict):
        min_val = field.options.get('min')
        max_val = field.options.get('max')
        
        if min_val is not None and value < min_val:
            return f"Field '{field.name}' must be >= {min_val}, got {value}"
        
        if max_val is not None and value > max_val:
            return f"Field '{field.name}' must be <= {max_val}, got {value}"
    
    return None


def format_validation_errors(errors: List[ValidationError]) -> str:
    """
    Format validation errors into a user-friendly message.
    
    Args:
        errors: List of ValidationError objects
        
    Returns:
        Formatted error message string
    """
    if not errors:
        return ""
    
    if len(errors) == 1:
        return f"Validation error: {errors[0].message}"
    
    error_lines = ["Validation errors:"]
    for error in errors:
        error_lines.append(f"  - {error.message}")
    
    return "\n".join(error_lines)
