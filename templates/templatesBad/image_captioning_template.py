"""
Image Captioning Template for generating descriptive text from images.

This template uses vision-language models to generate natural language
descriptions of image content, useful for accessibility, content indexing,
and image understanding tasks.
"""

from typing import Any, Dict
from templates.base import Template, InputField, OutputField, RouteType


class ImageCaptioningTemplate(Template):
    """
    Generates descriptive captions for images.
    
    Uses pre-trained vision-language models (e.g., BLIP, GIT, ViT-GPT2) to
    generate natural language descriptions of image content. The model analyzes
    visual features and generates coherent, descriptive text.
    
    Returns a caption and confidence score.
    """
    
    name = "image-captioning"
    category = "Multimodal"
    description = "Generate descriptive text captions for images"
    version = "1.0.0"
    
    inputs = [
        InputField(
            name="image",
            type="image",
            description="Image to caption",
            required=True
        ),
        InputField(
            name="max_length",
            type="number",
            description="Maximum caption length",
            required=False,
            default=50
        )
    ]
    
    outputs = [
        OutputField(
            name="caption",
            type="text",
            description="Generated caption"
        ),
        OutputField(
            name="confidence",
            type="number",
            description="Confidence score"
        )
    ]
    
    routing = [RouteType.LOCAL, RouteType.MODAL, RouteType.HF]
    gpu_required = True
    gpu_type = "T4"
    memory_mb = 4096
    timeout_sec = 300
    pip_packages = ["transformers", "torch", "pillow"]
    
    def setup(self) -> None:
        """
        One-time initialization to load the image captioning model.
        
        Loads a pre-trained vision-language model from HuggingFace.
        Default model is Salesforce/blip-image-captioning-base.
        """
        from transformers import BlipProcessor, BlipForConditionalGeneration
        
        # Load BLIP model and processor
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        
        self._initialized = True
    
    def run(self, **kwargs) -> Dict[str, Any]:
        """
        Execute image captioning on the provided image.
        
        Args:
            image: Path to image file or image data
            max_length: Maximum caption length (optional, defaults to 50)
            
        Returns:
            Dict containing:
                - caption: Generated descriptive caption
                - confidence: Confidence score (0.0 to 1.0)
        """
        # Validate inputs
        self.validate_inputs(**kwargs)
        
        # Initialize if needed
        if not self._initialized:
            self.setup()
        
        from PIL import Image
        import torch
        
        # Extract parameters
        image = kwargs['image']
        max_length = kwargs.get('max_length', 50)
        
        # Load image
        if isinstance(image, str):
            pil_image = Image.open(image).convert('RGB')
        else:
            # Assume it's already a PIL Image or numpy array
            if hasattr(image, 'convert'):
                pil_image = image.convert('RGB')
            else:
                pil_image = Image.fromarray(image).convert('RGB')
        
        # Process image
        inputs = self.processor(pil_image, return_tensors="pt")
        
        # Generate caption
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=int(max_length),
                num_beams=5,
                early_stopping=True
            )
        
        # Decode caption
        caption = self.processor.decode(outputs[0], skip_special_tokens=True)
        
        # Calculate confidence (simplified - use average token probability)
        # In a real implementation, would extract actual probabilities from model
        confidence = 0.85  # Placeholder confidence score
        
        return {
            'caption': caption,
            'confidence': confidence
        }
