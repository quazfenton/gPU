"""
Text-to-Image Template for generating images from text descriptions.

This template uses diffusion models to generate high-quality images from
text prompts, enabling creative image generation and visualization tasks.
"""

from typing import Any, Dict
from templates.base import Template, InputField, OutputField, RouteType


class TextToImageTemplate(Template):
    """
    Generates images from text descriptions.
    
    Uses pre-trained diffusion models (e.g., Stable Diffusion, DALL-E) to
    generate images from text prompts. Supports various parameters for
    controlling image generation quality and style.
    
    Returns a generated image.
    """
    
    name = "text-to-image"
    category = "Multimodal"
    description = "Generate images from text prompts using diffusion models"
    version = "1.0.0"
    
    inputs = [
        InputField(
            name="prompt",
            type="text",
            description="Text description of desired image",
            required=True
        ),
        InputField(
            name="negative_prompt",
            type="text",
            description="What to avoid in the image",
            required=False,
            default=""
        ),
        InputField(
            name="width",
            type="number",
            description="Image width in pixels",
            required=False,
            default=512
        ),
        InputField(
            name="height",
            type="number",
            description="Image height in pixels",
            required=False,
            default=512
        ),
        InputField(
            name="num_inference_steps",
            type="number",
            description="Number of denoising steps",
            required=False,
            default=50
        )
    ]
    
    outputs = [
        OutputField(
            name="image",
            type="image",
            description="Generated image"
        )
    ]
    
    routing = [RouteType.MODAL, RouteType.HF]
    gpu_required = True
    gpu_type = "A10G"
    memory_mb = 16384
    timeout_sec = 600
    pip_packages = ["diffusers", "transformers", "torch", "accelerate"]
    
    def setup(self) -> None:
        """
        One-time initialization to load the text-to-image model.
        
        Loads a pre-trained diffusion model from HuggingFace.
        Default model is stabilityai/stable-diffusion-2-1.
        """
        from diffusers import StableDiffusionPipeline
        import torch
        
        # Load Stable Diffusion pipeline
        self.pipe = StableDiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-1",
            torch_dtype=torch.float16
        )
        
        # Move to GPU if available
        if torch.cuda.is_available():
            self.pipe = self.pipe.to("cuda")
        
        self._initialized = True
    
    def run(self, **kwargs) -> Dict[str, Any]:
        """
        Execute text-to-image generation on the provided prompt.
        
        Args:
            prompt: Text description of desired image
            negative_prompt: What to avoid in the image (optional)
            width: Image width in pixels (optional, defaults to 512)
            height: Image height in pixels (optional, defaults to 512)
            num_inference_steps: Number of denoising steps (optional, defaults to 50)
            
        Returns:
            Dict containing:
                - image: Generated image as PIL Image or numpy array
        """
        # Validate inputs
        self.validate_inputs(**kwargs)
        
        # Initialize if needed
        if not self._initialized:
            self.setup()
        
        # Extract parameters
        prompt = kwargs['prompt']
        negative_prompt = kwargs.get('negative_prompt', '')
        width = int(kwargs.get('width', 512))
        height = int(kwargs.get('height', 512))
        num_inference_steps = int(kwargs.get('num_inference_steps', 50))
        
        # Generate image
        result = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt if negative_prompt else None,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps
        )
        
        # Extract image
        image = result.images[0]
        
        return {
            'image': image
        }
