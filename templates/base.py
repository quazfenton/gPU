"""
Template base class for ML services.

All templates inherit from this base class and implement:
- name: Unique template identifier
- category: Audio, Vision, Text, etc.
- description: Human-readable description
- inputs: List of input field definitions
- outputs: List of output field definitions
- routing: Supported backends (local, modal, hf)
- setup(): One-time initialization
- run(**kwargs): Execute the template
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum


class RouteType(Enum):
    """Supported execution backends."""
    LOCAL = "local"
    MODAL = "modal"
    HF = "hf"
    COLAB = "colab"


@dataclass
class InputField:
    """Definition of a template input field."""
    name: str
    type: str  # audio, image, video, text, number, json, file
    description: str = ""
    required: bool = True
    default: Any = None
    options: Optional[List[Any]] = None  # For dropdown/select inputs


@dataclass
class OutputField:
    """Definition of a template output field."""
    name: str
    type: str  # audio, image, video, text, json, file
    description: str = ""


class Template(ABC):
    """Base class for all ML service templates."""
    
    # Template metadata (override in subclasses)
    name: str = "Base Template"
    category: str = "General"
    description: str = "Base template class"
    version: str = "1.0.0"
    
    # Input/output definitions (override in subclasses)
    inputs: List[InputField] = []
    outputs: List[OutputField] = []
    
    # Supported routing backends (override in subclasses)
    routing: List[RouteType] = []
    
    # Resource requirements
    gpu_required: bool = False
    gpu_type: Optional[str] = None  # T4, A10G, A100
    memory_mb: int = 512
    timeout_sec: int = 300
    
    # Dependencies
    pip_packages: List[str] = []
    
    _initialized: bool = False
    
    def __init__(self):
        """Initialize template instance."""
        # Copy class-level lists to prevent shared mutable state across instances
        # Check if the attribute is a Field object (from dataclass) or a regular value
        if hasattr(self, 'inputs'):
            if isinstance(self.inputs, list):
                self.inputs = list(self.inputs)
            else:
                self.inputs = []
        else:
            self.inputs = []
            
        if hasattr(self, 'outputs'):
            if isinstance(self.outputs, list):
                self.outputs = list(self.outputs)
            else:
                self.outputs = []
        else:
            self.outputs = []
            
        if hasattr(self, 'routing'):
            if isinstance(self.routing, list):
                self.routing = list(self.routing)
            else:
                self.routing = [RouteType.LOCAL]
        else:
            self.routing = [RouteType.LOCAL]
            
        if hasattr(self, 'pip_packages'):
            if isinstance(self.pip_packages, list):
                self.pip_packages = list(self.pip_packages)
            else:
                self.pip_packages = []
        else:
            self.pip_packages = []
    
    def setup(self) -> None:
        """
        One-time initialization (download models, etc.).
        Override in subclasses if needed.
        """
        self._initialized = True
    
    @abstractmethod
    def run(self, **kwargs) -> Dict[str, Any]:
        """
        Execute the template with given inputs.
        
        This method should be overridden in subclasses. The base implementation
        ensures setup() is called if not already initialized and validates outputs.
        
        Args:
            **kwargs: Input arguments matching self.inputs
            
        Returns:
            Dict mapping output names to values
            
        Raises:
            ValueError: If inputs or outputs don't match schema
            Exception: If execution fails
        """
        raise NotImplementedError
    
    def execute(self, **kwargs) -> Dict[str, Any]:
        """
        Execute the template with automatic setup and validation.
        
        This is the main entry point for template execution. It:
        1. Validates inputs
        2. Calls setup() if not initialized
        3. Executes run()
        4. Validates outputs
        
        Args:
            **kwargs: Input arguments matching self.inputs
            
        Returns:
            Dict mapping output names to values
            
        Raises:
            ValueError: If validation fails
            Exception: If execution fails with diagnostic information
        """
        # Validate inputs (may raise ValueError)
        self.validate_inputs(**kwargs)
        
        # Ensure setup is called
        if not self._initialized:
            self.setup()
        
        try:
            # Execute the template
            outputs = self.run(**kwargs)
        except Exception as e:
            # Wrap execution errors with diagnostic information
            raise RuntimeError(
                f"Template '{self.name}' execution failed: {str(e)}"
            ) from e
        
        # Validate outputs (may raise ValueError)
        self.validate_outputs(outputs)
        
        return outputs
    
    def validate_inputs(self, **kwargs) -> bool:
        """
        Validate inputs match expected schema.
        
        Args:
            **kwargs: Input arguments to validate
            
        Returns:
            True if validation passes
            
        Raises:
            ValueError: If validation fails with descriptive error message
        """
        # Check for missing required fields
        for inp in self.inputs:
            if inp.required and inp.name not in kwargs:
                raise ValueError(
                    f"Template '{self.name}': Missing required input field '{inp.name}' "
                    f"(type: {inp.type}, description: {inp.description})"
                )
        
        # Check for type compatibility (basic validation)
        for inp in self.inputs:
            if inp.name in kwargs:
                value = kwargs[inp.name]
                # Basic type checking
                if inp.type == "number" and not isinstance(value, (int, float)):
                    raise ValueError(
                        f"Template '{self.name}': Input field '{inp.name}' expects type 'number', "
                        f"but got {type(value).__name__}"
                    )
                elif inp.type == "text" and not isinstance(value, str):
                    raise ValueError(
                        f"Template '{self.name}': Input field '{inp.name}' expects type 'text', "
                        f"but got {type(value).__name__}"
                    )
                # For file types (audio, image, video, file), we accept strings (paths) or bytes
                elif inp.type in ["audio", "image", "video", "file"]:
                    if not isinstance(value, (str, bytes)):
                        raise ValueError(
                            f"Template '{self.name}': Input field '{inp.name}' expects type '{inp.type}' "
                            f"(string path or bytes), but got {type(value).__name__}"
                        )
        
        return True
    
    def validate_outputs(self, outputs: Dict[str, Any]) -> bool:
        """
        Validate outputs match expected schema.
        
        Args:
            outputs: Output dictionary to validate
            
        Returns:
            True if validation passes
            
        Raises:
            ValueError: If validation fails with descriptive error message
        """
        # Check that all declared output fields are present
        for out in self.outputs:
            if out.name not in outputs:
                raise ValueError(
                    f"Template '{self.name}': Missing expected output field '{out.name}' "
                    f"(type: {out.type}, description: {out.description})"
                )
        
        return True
    
    def get_modal_decorator_args(self) -> Dict[str, Any]:
        """Get arguments for Modal @app.function decorator."""
        args = {
            "timeout": self.timeout_sec,
        }
        if self.gpu_required and self.gpu_type:
            args["gpu"] = self.gpu_type
        return args
    
    def get_pip_install_list(self) -> List[str]:
        """Get list of pip packages to install."""
        return self.pip_packages.copy()
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize template metadata to dict."""
        return {
            "name": self.name,
            "category": self.category,
            "description": self.description,
            "version": self.version,
            "inputs": [
                {
                    "name": i.name,
                    "type": i.type,
                    "description": i.description,
                    "required": i.required,
                    "default": i.default,
                    "options": i.options,
                }
                for i in self.inputs
            ],
            "outputs": [
                {
                    "name": o.name,
                    "type": o.type,
                    "description": o.description,
                }
                for o in self.outputs
            ],
            "routing": [r.value for r in self.routing],
            "gpu_required": self.gpu_required,
            "gpu_type": self.gpu_type,
            "memory_mb": self.memory_mb,
            "timeout_sec": self.timeout_sec,
            "pip_packages": self.pip_packages,
        }
    
    def __repr__(self) -> str:
        return f"<Template: {self.name} ({self.category})>"
