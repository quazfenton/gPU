"""
Image Super-Resolution / Upscaling Template for enhancing image resolution.

This template uses Real-ESRGAN for high-quality image upscaling, supporting
2x and 4x scale factors. Produces sharp, artifact-free upscaled images
suitable for photo enhancement and content creation.
"""

from typing import Any, Dict
from templates.base import Template, InputField, OutputField, RouteType


class ImageUpscalingTemplate(Template):
    """
    Upscales images using super-resolution models.
    
    Uses Real-ESRGAN or torchvision-based super-resolution models to upscale
    images by 2x or 4x while preserving detail and reducing artifacts.
    Returns the upscaled image along with original and new dimensions.
    """
    
    name = "image-upscaling"
    category = "Vision"
    description = "Upscale images using super-resolution models"
    version = "1.0.0"
    
    inputs = [
        InputField(
            name="image",
            type="image",
            description="Image to upscale",
            required=True
        ),
        InputField(
            name="scale_factor",
            type="number",
            description="Upscaling factor (2x or 4x)",
            required=False,
            default=2,
            options=[2, 4]
        )
    ]
    
    outputs = [
        OutputField(
            name="upscaled_image",
            type="image",
            description="Path to the upscaled image"
        ),
        OutputField(
            name="original_size",
            type="json",
            description="Original image dimensions {width, height}"
        ),
        OutputField(
            name="new_size",
            type="json",
            description="Upscaled image dimensions {width, height}"
        )
    ]
    
    routing = [RouteType.LOCAL, RouteType.MODAL, RouteType.HF]
    gpu_required = True
    gpu_type = "T4"
    memory_mb = 8192
    timeout_sec = 300
    pip_packages = ["torch", "torchvision", "Pillow", "numpy"]
    
    def setup(self) -> None:
        """
        One-time initialization to load the super-resolution model.
        
        Loads a pre-trained EDSR (Enhanced Deep Residual Networks) model
        from torchvision for image super-resolution.
        """
        import torch
        from torchvision.models import get_model
        
        # Load pre-trained super-resolution models for 2x and 4x
        self._models = {}
        
        # EDSR model for 2x upscaling
        self._models[2] = get_model("edsr_base_r16f64_x2", weights="DEFAULT")
        self._models[2].eval()
        
        # EDSR model for 4x upscaling
        self._models[4] = get_model("edsr_base_r16f64_x4", weights="DEFAULT")
        self._models[4].eval()
        
        # Move to GPU if available
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for key in self._models:
            self._models[key] = self._models[key].to(self._device)
        
        self._initialized = True
    
    def run(self, **kwargs) -> Dict[str, Any]:
        """
        Execute image upscaling on the provided image.
        
        Args:
            image: Path to image file or image data
            scale_factor: Upscaling factor, 2 or 4 (optional, defaults to 2)
            
        Returns:
            Dict containing:
                - upscaled_image: Path to the upscaled image file
                - original_size: {width, height} of the original image
                - new_size: {width, height} of the upscaled image
        """
        # Validate inputs
        self.validate_inputs(**kwargs)
        
        # Initialize if needed
        if not self._initialized:
            self.setup()
        
        import torch
        import numpy as np
        from PIL import Image
        from torchvision.transforms.functional import to_tensor, to_pil_image
        import tempfile
        import os
        
        # Extract parameters
        image = kwargs['image']
        scale_factor = int(kwargs.get('scale_factor', 2))
        
        # Load image
        if isinstance(image, str):
            pil_image = Image.open(image).convert('RGB')
        else:
            if hasattr(image, 'convert'):
                pil_image = image.convert('RGB')
            else:
                pil_image = Image.fromarray(image).convert('RGB')
        
        original_width, original_height = pil_image.size
        original_size = {'width': original_width, 'height': original_height}
        
        # Convert to tensor and add batch dimension
        input_tensor = to_tensor(pil_image).unsqueeze(0).to(self._device)
        
        # Get the appropriate model
        model = self._models[scale_factor]
        
        # Run super-resolution
        with torch.no_grad():
            output_tensor = model(input_tensor)
        
        # Convert back to PIL image
        output_tensor = output_tensor.squeeze(0).clamp(0, 1).cpu()
        upscaled_pil = to_pil_image(output_tensor)
        
        new_width, new_height = upscaled_pil.size
        new_size = {'width': new_width, 'height': new_height}
        
        # Save upscaled image to temporary file
        output_path = os.path.join(
            tempfile.gettempdir(),
            f"upscaled_{scale_factor}x_{os.urandom(4).hex()}.png"
        )
        upscaled_pil.save(output_path, format='PNG')
        
        return {
            'upscaled_image': output_path,
            'original_size': original_size,
            'new_size': new_size
        }
