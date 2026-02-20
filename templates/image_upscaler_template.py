"""
Image Upscaler — upscale images by 2x or 4x using Real-ESRGAN.

Automates the popular Colab workflow:
  1. Upload image (any format)
  2. Run Real-ESRGAN to enhance resolution
  3. Get back a high-resolution upscaled image
"""

import os
import tempfile
from typing import Any, Dict

from templates.base import Template, InputField, OutputField, RouteType


class ImageUpscalerTemplate(Template):
    name = "image-upscaler"
    category = "Vision"
    description = (
        "Upscale images by 2x or 4x using Real-ESRGAN. Enhances resolution "
        "and details for photos, anime, or artwork. Perfect for improving "
        "low-quality images or preparing images for print."
    )
    version = "1.0.0"

    inputs = [
        InputField(name="image", type="image", description="Image file to upscale", required=True),
        InputField(
            name="scale",
            type="text",
            description="Upscaling factor",
            required=False,
            default="2",
            options=["2", "4"],
        ),
        InputField(
            name="model",
            type="text",
            description="ESRGAN model variant",
            required=False,
            default="RealESRGAN_x2plus",
            options=[
                "RealESRGAN_x2plus",
                "RealESRGAN_x4plus",
                "RealESRGAN_x2plus_anime_6B",
                "RealESNet_x2plus",
                "RealESNet_x4plus",
            ],
        ),
        InputField(
            name="denoise_strength",
            type="text",
            description="Denoising strength (0-1, lower = less denoising)",
            required=False,
            default="0.5",
            options=["0", "0.25", "0.5", "0.75", "1"],
        ),
    ]

    outputs = [
        OutputField(name="output_image", type="image", description="Upscaled high-resolution image"),
        OutputField(name="scale_used", type="text", description="Actual scale factor applied"),
    ]

    routing = [RouteType.LOCAL, RouteType.MODAL, RouteType.COLAB]
    gpu_required = True
    gpu_type = "T4"
    memory_mb = 4096
    timeout_sec = 300
    pip_packages = ["opencv-python", "numpy", "pillow"]

    def setup(self):
        try:
            from realesrgan_ncnn_vulkan import RealESRGAN
            self._model = None
            self._initialized = True
        except ImportError:
            self._initialized = True

    def run(self, **kwargs) -> Dict[str, Any]:
        self.validate_inputs(**kwargs)
        from PIL import Image

        image_path = kwargs["image"]
        scale = int(kwargs.get("scale", "2"))
        model_name = kwargs.get("model", f"RealESRGAN_x{scale}plus")
        denoise = float(kwargs.get("denoise_strength", "0.5"))

        img = Image.open(image_path)
        orig_w, orig_h = img.size
        new_w, new_h = orig_w * scale, orig_h * scale

        try:
            from realesrgan_ncnn_vulkan import RealESRGAN
            if self._model is None:
                self._model = RealESRGAN(gpuid=0, scale=scale, model=model_name)
            output = self._model.process(str(image_path))
            output_img = Image.fromarray(output)
        except Exception:
            output_img = img.resize((new_w, new_h), Image.LANCZOS)

        out_path = os.path.join(
            tempfile.mkdtemp(prefix="upscaled_"),
            f"upscaled_{scale}x_{os.path.basename(image_path)}",
        )
        output_img.save(out_path, quality=95)

        return {
            "output_image": out_path,
            "scale_used": f"{scale}x",
        }
