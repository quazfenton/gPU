"""
Image Generation Template for generating images using Stable Diffusion.

This template uses diffusion models to generate high-quality images from text
prompts with fine-grained control over the generation process including
guidance scale, inference steps, and image dimensions.
"""

from typing import Any, Dict
from templates.base import Template, InputField, OutputField, RouteType


class ImageGenerationTemplate(Template):
    """
    Generates images from text descriptions using Stable Diffusion.
    
    Uses pre-trained Stable Diffusion models to generate images from text
    prompts. Supports negative prompts, configurable inference steps,
    guidance scale, and output dimensions.
    
    Returns a generated image and the random seed used.
    """
    
    name = "image-generation"
    category = "Vision"
    description = "Generate images from text prompts using Stable Diffusion"
    version = "1.0.0"
    
    inputs = [
        InputField(
            name="prompt",
            type="text",
            description="Text description of the desired image",
            required=True
        ),
        InputField(
            name="negative_prompt",
            type="text",
            description="What to avoid in the generated image",
            required=False,
            default=""
        ),
        InputField(
            name="num_steps",
            type="number",
            description="Number of denoising inference steps",
            required=False,
            default=30
        ),
        InputField(
            name="guidance_scale",
            type="number",
            description="Classifier-free guidance scale (higher = more prompt adherence)",
            required=False,
            default=7.5
        ),
        InputField(
            name="width",
            type="number",
            description="Output image width in pixels",
            required=False,
            default=512
        ),
        InputField(
            name="height",
            type="number",
            description="Output image height in pixels",
            required=False,
            default=512
        )
    ]
    
    outputs = [
        OutputField(
            name="image",
            type="image",
            description="Generated image"
        ),
        OutputField(
            name="seed",
            type="number",
            description="Random seed used for generation (for reproducibility)"
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
        One-time initialization to load the Stable Diffusion pipeline.
        
        Loads a pre-trained Stable Diffusion model from HuggingFace.
        Default model is runwayml/stable-diffusion-v1-5.
        """
        from diffusers import StableDiffusionPipeline
        import torch
        
        # Load Stable Diffusion pipeline with float16 for efficiency
        self.pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16
        )
        
        # Move to GPU if available
        if torch.cuda.is_available():
            self.pipe = self.pipe.to("cuda")
        
        self._initialized = True
    
    def run(self, **kwargs) -> Dict[str, Any]:
        """
        Execute image generation from the provided text prompt.
        
        Args:
            prompt: Text description of the desired image
            negative_prompt: What to avoid in the image (optional)
            num_steps: Number of denoising steps (optional, defaults to 30)
            guidance_scale: Guidance scale (optional, defaults to 7.5)
            width: Image width in pixels (optional, defaults to 512)
            height: Image height in pixels (optional, defaults to 512)
            
        Returns:
            Dict containing:
                - image: Generated image as a PIL Image
                - seed: Random seed used for generation
        """
        # Validate inputs
        self.validate_inputs(**kwargs)
        
        # Initialize if needed
        if not self._initialized:
            self.setup()
        
        import torch
        
        # Extract parameters
        prompt = kwargs['prompt']
        negative_prompt = kwargs.get('negative_prompt', '')
        num_steps = int(kwargs.get('num_steps', 30))
        guidance_scale = float(kwargs.get('guidance_scale', 7.5))
        width = int(kwargs.get('width', 512))
        height = int(kwargs.get('height', 512))
        
        # Generate a random seed for reproducibility
        generator = torch.Generator(device=self.pipe.device)
        seed = generator.seed()
        generator = generator.manual_seed(seed)
        
        # Generate image
        result = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt if negative_prompt else None,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
            generator=generator
        )
        
        # Extract generated image
        image = result.images[0]
        
        return {
            'image': image,
            'seed': int(seed)
        }
