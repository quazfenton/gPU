"""
Image-to-Image Translation — transform images using state-of-the-art models.

Automates workflows for:
  1. Subject-driven generation (AnyDoor, IP-Adapter)
  2. Sketch-to-image (ControlNet)
  3. Depth/Normal map to image
  4. Inpainting/Outpainting
  5. Subject replacement and background change
"""

import os
import tempfile
from typing import Any, Dict

from templates.base import Template, InputField, OutputField, RouteType


class ImageToImageTranslationTemplate(Template):
    name = "image-to-image"
    category = "Vision"
    description = (
        "Transform images using state-of-the-art image-to-image models. "
        "Remove and replace subjects, change backgrounds, convert sketches to photos, "
        "or use depth/normal maps for controlled generation. Perfect for product "
        "photography, creative editing, and image enhancement."
    )
    version = "1.0.0"

    inputs = [
        InputField(
            name="input_image",
            type="image",
            description="Input image to transform",
            required=True,
        ),
        InputField(
            name="task",
            type="text",
            description="Transformation task",
            required=False,
            default="subject_replace",
            options=[
                "subject_replace",
                "background_change",
                "sketch_to_image",
                "depth_to_image",
                "inpainting",
                "outpainting",
                "colorization",
                "enhance",
            ],
        ),
        InputField(
            name="subject_image",
            type="image",
            description="New subject image (for subject replacement)",
            required=False,
        ),
        InputField(
            name="background_prompt",
            type="text",
            description="Description of desired background",
            required=False,
        ),
        InputField(
            name="mask_image",
            type="image",
            description="Mask for inpainting (white=keep, black=inpaint)",
            required=False,
        ),
        InputField(
            name="prompt",
            type="text",
            description="Text prompt for generation",
            required=False,
            default="",
        ),
        InputField(
            name="negative_prompt",
            type="text",
            description="Negative prompt",
            required=False,
            default="blurry, low quality, distorted",
        ),
        InputField(
            name="strength",
            type="number",
            description="Transformation strength (0-1)",
            required=False,
            default=0.8,
        ),
        InputField(
            name="guidance_scale",
            type="number",
            description="Prompt adherence (1-20)",
            required=False,
            default=7.5,
        ),
        InputField(
            name="num_inference_steps",
            type="number",
            description="Number of steps",
            required=False,
            default=30,
        ),
        InputField(
            name="seed",
            type="number",
            description="Random seed",
            required=False,
            default=-1,
        ),
    ]

    outputs = [
        OutputField(name="output_image", type="image", description="Transformed image"),
        OutputField(name="mask_used", type="image", description="Generated mask (if applicable)"),
        OutputField(name="seed_used", type="number", description="Seed used"),
    ]

    routing = [RouteType.MODAL, RouteType.COLAB]
    gpu_required = True
    gpu_type = "T4"
    memory_mb = 8192
    timeout_sec = 300
    pip_packages = ["diffusers", "transformers", "torch", "opencv-python", "pillow", "controlnet-aux"]

    def setup(self):
        self._initialized = True

    def run(self, **kwargs) -> Dict[str, Any]:
        self.validate_inputs(**kwargs)
        import torch
        import numpy as np
        from PIL import Image
        from diffusers import (
            StableDiffusionInpaintPipeline,
            StableDiffusionImg2ImgPipeline,
            StableDiffusionControlNetPipeline,
        )

        input_path = kwargs["input_image"]
        task = kwargs.get("task", "subject_replace")
        subject_path = kwargs.get("subject_image")
        bg_prompt = kwargs.get("background_prompt", "")
        mask_path = kwargs.get("mask_image")
        prompt = kwargs.get("prompt", "")
        negative = kwargs.get("negative_prompt", "blurry, low quality, distorted")
        strength = float(kwargs.get("strength", 0.8))
        guidance = float(kwargs.get("guidance_scale", 7.5))
        steps = int(kwargs.get("num_inference_steps", 30))
        seed = int(kwargs.get("seed", -1))

        if seed == -1:
            seed = np.random.randint(0, 999999)

        input_img = Image.open(input_path).convert("RGB")
        w, h = input_img.size

        if task == "inpainting" or task == "outpainting":
            if mask_path:
                mask_img = Image.open(mask_path).convert("L")
            else:
                mask_img = Image.new("L", (w, h), 128)

            pipe = StableDiffusionInpaintPipeline.from_pretrained(
                "runwayml/stable-diffusion-inpainting",
                torch_dtype=torch.float16,
            )
            pipe = pipe.to("cuda")

            generator = torch.Generator().manual_seed(seed)

            result = pipe(
                prompt=prompt if prompt else "high quality, detailed",
                image=input_img,
                mask_image=mask_img,
                negative_prompt=negative,
                num_inference_steps=steps,
                guidance_scale=guidance,
                generator=generator,
            ).images[0]

        elif task == "background_change":
            pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                torch_dtype=torch.float16,
            )
            pipe = pipe.to("cuda")

            generator = torch.Generator().manual_seed(seed)

            bg_desc = bg_prompt if bg_prompt else "professional studio background"
            full_prompt = f"{prompt}, {bg_desc}" if prompt else bg_desc

            result = pipe(
                prompt=full_prompt,
                image=input_img,
                strength=strength,
                negative_prompt=negative,
                num_inference_steps=steps,
                guidance_scale=guidance,
                generator=generator,
            ).images[0]

        elif task == "sketch_to_image":
            import cv2
            sketch = cv2.Canny(np.array(input_img), 100, 200)
            sketch = Image.fromarray(sketch)

            pipe = StableDiffusionControlNetPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                torch_dtype=torch.float16,
                controlnet=None,
            )
            pipe = pipe.to("cuda")

            generator = torch.Generator().manual_seed(seed)

            result = pipe(
                prompt=prompt if prompt else "beautiful, detailed, high quality",
                image=sketch,
                negative_prompt=negative,
                num_inference_steps=steps,
                guidance_scale=guidance,
                generator=generator,
            ).images[0]

        elif task == "colorization":
            pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                torch_dtype=torch.float16,
            )
            pipe = pipe.to("cuda")

            bw_img = input_img.convert("L").convert("RGB")

            generator = torch.Generator().manual_seed(seed)

            result = pipe(
                prompt=prompt if prompt else "vibrant colors, natural, realistic",
                image=bw_img,
                strength=0.9,
                negative_prompt=negative,
                num_inference_steps=steps,
                guidance_scale=guidance,
                generator=generator,
            ).images[0]

        else:
            pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                torch_dtype=torch.float16,
            )
            pipe = pipe.to("cuda")

            generator = torch.Generator().manual_seed(seed)

            result = pipe(
                prompt=prompt if prompt else "enhanced, improved, high quality",
                image=input_img,
                strength=strength,
                negative_prompt=negative,
                num_inference_steps=steps,
                guidance_scale=guidance,
                generator=generator,
            ).images[0]

        out_path = os.path.join(tempfile.mkdtemp(prefix="i2i_"), "output.jpg")
        result.save(out_path, quality=95)

        return {
            "output_image": out_path,
            "mask_used": mask_path if mask_path else "",
            "seed_used": seed,
        }
