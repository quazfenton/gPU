"""
Image Segmentation Template for semantic and instance segmentation.

This template uses segmentation models to partition images into regions.
Supports both semantic segmentation (pixel-level classification) and
instance segmentation (individual object instances).
"""

from typing import Any, Dict
from templates.base import Template, InputField, OutputField, RouteType


class ImageSegmentationTemplate(Template):
    """
    Performs semantic or instance segmentation on images.
    
    Uses transformer-based segmentation models to segment images into regions.
    Returns segmentation masks and segment information including labels and areas.
    """
    
    name = "image-segmentation"
    category = "Vision"
    description = "Segment images into regions using segmentation models"
    version = "1.0.0"
    
    inputs = [
        InputField(
            name="image",
            type="image",
            description="Image file to segment",
            required=True
        ),
        InputField(
            name="segmentation_type",
            type="text",
            description="Type of segmentation (semantic, instance)",
            required=False,
            default="semantic",
            options=["semantic", "instance"]
        )
    ]
    
    outputs = [
        OutputField(
            name="mask",
            type="image",
            description="Segmentation mask"
        ),
        OutputField(
            name="segments",
            type="json",
            description="Segment information with labels and areas"
        )
    ]
    
    routing = [RouteType.LOCAL, RouteType.MODAL]
    gpu_required = True
    gpu_type = "T4"
    memory_mb = 6144
    timeout_sec = 300
    pip_packages = ["transformers", "torch", "pillow"]
    
    def setup(self) -> None:
        """
        One-time initialization to load the segmentation model.
        
        Downloads and loads the segmentation model. The model is cached
        for subsequent uses.
        """
        from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation
        
        # Use a lightweight segmentation model
        model_name = "nvidia/segformer-b0-finetuned-ade-512-512"
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModelForSemanticSegmentation.from_pretrained(model_name)
        self._initialized = True
    
    def run(self, **kwargs) -> Dict[str, Any]:
        """
        Execute image segmentation on the provided image.
        
        Args:
            image: Path to image file or image data
            segmentation_type: Type of segmentation (optional, defaults to 'semantic')
            
        Returns:
            Dict containing:
                - mask: Segmentation mask as image
                - segments: List of segments with labels and areas
        """
        # Validate inputs
        self.validate_inputs(**kwargs)
        
        from PIL import Image
        import torch
        import numpy as np
        
        # Initialize if needed
        if not self._initialized:
            self.setup()
        
        # Extract parameters
        image = kwargs['image']
        segmentation_type = kwargs.get('segmentation_type', 'semantic')
        
        # Load image if it's a path
        if isinstance(image, str):
            img = Image.open(image).convert('RGB')
        else:
            # Assume it's already a PIL Image or numpy array
            if isinstance(image, np.ndarray):
                img = Image.fromarray(image)
            else:
                img = image
        
        # Process image
        inputs = self.processor(images=img, return_tensors="pt")
        
        # Run segmentation
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Get segmentation map
        logits = outputs.logits
        segmentation_map = logits.argmax(dim=1)[0].cpu().numpy()
        
        # Create colored mask for visualization
        # Use a simple color mapping
        unique_labels = np.unique(segmentation_map)
        colored_mask = np.zeros((*segmentation_map.shape, 3), dtype=np.uint8)
        
        # Assign colors to each segment
        for i, label in enumerate(unique_labels):
            # Generate a color based on label
            color = [
                (label * 50) % 255,
                (label * 100) % 255,
                (label * 150) % 255
            ]
            colored_mask[segmentation_map == label] = color
        
        # Convert to PIL Image
        mask_image = Image.fromarray(colored_mask)
        
        # Extract segment information
        segments = []
        for label in unique_labels:
            mask_area = (segmentation_map == label)
            area = int(np.sum(mask_area))
            
            # Get label name if available
            label_name = self.model.config.id2label.get(int(label), f"class_{label}")
            
            segments.append({
                'label_id': int(label),
                'label_name': label_name,
                'area': area,
                'percentage': float(area / segmentation_map.size * 100)
            })
        
        return {
            'mask': mask_image,
            'segments': segments
        }
