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

    # Input/output definitions
    inputs: List[InputField]
    outputs: List[OutputField]

    # Supported routing backends
    routing: List[RouteType]

    def __init__(self):
        self.inputs = []
        self.outputs = []
        self.routing = [RouteType.LOCAL]
    # Resource requirements
    gpu_required: bool = False
    gpu_type: Optional[str] = None  # T4, A10G, A100
    memory_mb: int = 512
    timeout_sec: int = 300
    
    # Dependencies
    pip_packages: List[str] = field(default_factory=list)
    
    _initialized: bool = False
    
    def __init__(self):
        """Initialize template instance."""
        self.inputs = self.inputs if hasattr(self, 'inputs') else []
        self.outputs = self.outputs if hasattr(self, 'outputs') else []
        self.routing = self.routing if hasattr(self, 'routing') else [RouteType.LOCAL]
        self.pip_packages = self.pip_packages if hasattr(self, 'pip_packages') else []
    
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
        
        Args:
            **kwargs: Input arguments matching self.inputs
            
        Returns:
            Dict mapping output names to values
        """
        raise NotImplementedError
    
    def validate_inputs(self, **kwargs) -> bool:
        """Validate inputs match expected schema."""
        for inp in self.inputs:
            if inp.required and inp.name not in kwargs:
                raise ValueError(f"Missing required input: {inp.name}")
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
