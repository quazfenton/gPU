"""
Text to Image — generate images from text prompts using Stable Diffusion.

Automates the popular Colab workflow:
  1. Enter text prompt
  2. Run Stable Diffusion (any variant)
  3. Get back AI-generated images
"""

import os
import tempfile
from typing import Any, Dict

from templates.base import Template, InputField, OutputField, RouteType


class TextToImageTemplate(Template):
    name = "text-to-image"
    category = "Vision"
    description = (
        "Generate stunning images from text prompts using Stable Diffusion. "
        "Create artwork, photorealistic images, anime, logos, and more. "
        "Supports negative prompts and various aspect ratios."
    )
    version = "1.0.0"

    inputs = [
        InputField(name="prompt", type="text", description="Text description of desired image", required=True),
        InputField(
            name="negative_prompt",
            type="text",
            description="Things to exclude from image",
            required=False,
            default="",
        ),
        InputField(
            name="model",
            type="text",
            description="Stable Diffusion model",
            required=False,
            default="runwayml/stable-diffusion-v1-5",
        ),
        InputField(
            name="num_images",
            type="number",
            description="Number of images to generate",
            required=False,
            default=1,
        ),
        InputField(
            name="width",
            type="number",
            description="Image width (multiple of 8)",
            required=False,
            default=512,
        ),
        InputField(
            name="height",
            type="number",
            description="Image height (multiple of 8)",
            required=False,
            default=512,
        ),
        InputField(
            name="guidance_scale",
            type="number",
            description="Prompt adherence (higher = stricter)",
            required=False,
            default=7.5,
        ),
        InputField(
            name="num_inference_steps",
            type="number",
            description="Generation steps (higher = better quality)",
            required=False,
            default=50,
        ),
        InputField(
            name="seed",
            type="number",
            description="Random seed for reproducibility (-1 for random)",
            required=False,
            default=-1,
        ),
    ]

    outputs = [
        OutputField(name="images", type="json", description="List of generated image paths"),
        OutputField(name="seed_used", type="number", description="Seed used for generation"),
    ]

    routing = [RouteType.LOCAL, RouteType.MODAL, RouteType.COLAB]
    gpu_required = True
    gpu_type = "T4"
    memory_mb = 8192
    timeout_sec = 600
    pip_packages = ["diffusers", "transformers", "torch", "accelerate", "scipy", "safetensors"]

    def setup(self):
        self._initialized = True

    def run(self, **kwargs) -> Dict[str, Any]:
        self.validate_inputs(**kwargs)

        from diffusers import StableDiffusionPipeline
        import torch

        prompt = kwargs["prompt"]
        negative_prompt = kwargs.get("negative_prompt", "")
        model_id = kwargs.get("model", "runwayml/stable-diffusion-v1-5")
        num_images = int(kwargs.get("num_images", 1))
        width = int(kwargs.get("width", 512))
        height = int(kwargs.get("height", 512))
        guidance_scale = float(kwargs.get("guidance_scale", 7.5))
        num_steps = int(kwargs.get("num_inference_steps", 50))
        seed = int(kwargs.get("seed", -1))

        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )
        if torch.cuda.is_available():
            pipe = pipe.to("cuda")

        generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu")
        if seed == -1:
            seed = generator.seed()
        generator = generator.manual_seed(seed)

        results = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_images,
            width=width,
            height=height,
            guidance_scale=guidance_scale,
            num_inference_steps=num_steps,
            generator=generator,
        )

        out_dir = tempfile.mkdtemp(prefix="sd_")
        image_paths = []
        for i, img in enumerate(results.images):
            out_path = os.path.join(out_dir, f"generated_{i}.png")
            img.save(out_path)
            image_paths.append(out_path)

        return {
            "images": image_paths,
            "seed_used": seed,
        }
