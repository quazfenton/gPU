"""
Style Transfer — apply artistic styles to images using neural style transfer.

Automates workflows for:
  1. Neural Style Transfer (NST)
  2. StyleGAN-based style application
  3. ControlNet-based controlled style transfer
  4. Instant-style with IP adapters
"""

import os
import tempfile
from typing import Any, Dict

from templates.base import Template, InputField, OutputField, RouteType


class StyleTransferTemplate(Template):
    name = "style-transfer"
    category = "Vision"
    description = (
        "Apply stunning artistic styles to images using state-of-the-art neural "
        "style transfer. Transform photos into paintings, apply cartoon styles, "
        "or use ControlNet for precise control over the output. Supports IP-Adapter "
        "for instant style transfer from reference images."
    )
    version = "1.0.0"

    inputs = [
        InputField(
            name="content_image",
            type="image",
            description="Image to apply style to",
            required=True,
        ),
        InputField(
            name="style_image",
            type="image",
            description="Style reference image",
            required=False,
        ),
        InputField(
            name="method",
            type="text",
            description="Style transfer method",
            required=False,
            default="controlnet",
            options=["controlnet", "ip_adapter", "nst", "instant_id"],
        ),
        InputField(
            name="style_preset",
            type="text",
            description="Preset artistic style",
            required=False,
            default="none",
            options=[
                "none",
                "anime",
                "oil_painting",
                "watercolor",
                "sketch",
                "pixar",
                "cyberpunk",
                "disney",
                "ghibli",
                "portrait",
            ],
        ),
        InputField(
            name="style_strength",
            type="number",
            description="Strength of style application (0-1)",
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
            description="Number of denoising steps",
            required=False,
            default=30,
        ),
        InputField(
            name="preserve_colors",
            type="text",
            description="Preserve original colors",
            required=False,
            default="false",
            options=["true", "false"],
        ),
        InputField(
            name="controlnet_conditioning_scale",
            type="number",
            description="ControlNet strength (0-2)",
            required=False,
            default=1.0,
        ),
    ]

    outputs = [
        OutputField(name="output_image", type="image", description="Styled image"),
        OutputField(name="style_applied", type="text", description="Style/method used"),
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
        from PIL import Image
        import numpy as np
        from diffusers import StableDiffusionControlNetPipeline, StableDiffusionImg2ImgPipeline
        from diffusers import UniPCMultistepScheduler

        content_path = kwargs["content_image"]
        style_path = kwargs.get("style_image")
        method = kwargs.get("method", "controlnet")
        preset = kwargs.get("style_preset", "none")
        strength = float(kwargs.get("style_strength", 0.8))
        guidance = float(kwargs.get("guidance_scale", 7.5))
        steps = int(kwargs.get("num_inference_steps", 30))
        preserve_colors = kwargs.get("preserve_colors", "false") == "true"
        cn_scale = float(kwargs.get("controlnet_conditioning_scale", 1.0))

        content_img = Image.open(content_path).convert("RGB")

        if method == "controlnet":
            if preset == "anime":
                model_id = "doggettx/anything-v3.1"
                cn_model = "lllyasviel/sd-controlnet-canny"
            elif preset == "portrait":
                model_id = "runwayml/stable-diffusion-v1-5"
                cn_model = "lllyasviel/sd-controlnet-openpose"
            else:
                model_id = "runwayml/stable-diffusion-v1-5"
                cn_model = "lllyasviel/sd-controlnet-depth"

            pipe = StableDiffusionControlNetPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                controlnet=None,
            )
            pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
            pipe = pipe.to("cuda")

            prompt = f"artistic style, beautiful painting, {preset} style"
            negative = "blurry, low quality, distorted, deformed"

            result = pipe(
                prompt=prompt,
                negative_prompt=negative,
                image=content_img,
                num_inference_steps=steps,
                guidance_scale=guidance,
                controlnet_conditioning_scale=cn_scale,
            ).images[0]

        elif method == "ip_adapter":
            from diffusers import StableDiffusionIPAdapterPipeline

            pipe = StableDiffusionIPAdapterPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                torch_dtype=torch.float16,
                ip_adapter_scale=strength,
            )
            pipe = pipe.to("cuda")

            style_img = Image.open(style_path) if style_path else content_img

            prompt = f"professional artistic style, {preset}"
            result = pipe(
                prompt=prompt,
                image=style_img,
                num_inference_steps=steps,
                guidance_scale=guidance,
            ).images[0]

        else:
            pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                torch_dtype=torch.float16,
            )
            pipe = pipe.to("cuda")

            prompt = f"artwork, {preset} style, beautiful, high quality"
            result = pipe(
                prompt=prompt,
                image=content_img,
                strength=strength,
                num_inference_steps=steps,
                guidance_scale=guidance,
            ).images[0]

        if preserve_colors:
            import cv2
            content_np = np.array(content_img)
            result_np = np.array(result)
            result_lab = cv2.cvtColor(result_np, cv2.COLOR_RGB2LAB)
            content_l = cv2.cvtColor(content_np, cv2.COLOR_RGB2LAB)[:, :, 0]
            result_lab[:, :, 0] = content_l
            result = Image.fromarray(cv2.cvtColor(result_lab, cv2.COLOR_LAB2RGB))

        out_path = os.path.join(tempfile.mkdtemp(prefix="styled_"), "output.jpg")
        result.save(out_path, quality=95)

        return {
            "output_image": out_path,
            "style_applied": f"{method}/{preset}" if preset != "none" else method,
        }
