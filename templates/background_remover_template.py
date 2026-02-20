"""
Background Remover — remove the background from any image using rembg / U2-Net.

Upload a photo, get back a transparent-background PNG.
"""

import os
import tempfile
from typing import Any, Dict

from templates.base import Template, InputField, OutputField, RouteType


class BackgroundRemoverTemplate(Template):
    name = "background-remover"
    category = "Vision"
    description = (
        "Remove the background from any image and return a "
        "transparent PNG. Uses rembg (U2-Net) for high-quality "
        "foreground segmentation — perfect for product photos, "
        "profile pictures, or compositing."
    )
    version = "1.0.0"

    inputs = [
        InputField(name="image", type="image", description="Image file path", required=True),
        InputField(
            name="model",
            type="text",
            description="Segmentation model",
            required=False,
            default="u2net",
            options=["u2net", "u2netp", "u2net_human_seg", "isnet-general-use"],
        ),
        InputField(
            name="alpha_matting",
            type="text",
            description="Enable alpha matting for cleaner edges",
            required=False,
            default="false",
            options=["true", "false"],
        ),
    ]

    outputs = [
        OutputField(name="output_image", type="image", description="Image with background removed (PNG)"),
    ]

    routing = [RouteType.LOCAL, RouteType.MODAL, RouteType.HF]
    gpu_required = False
    gpu_type = None
    memory_mb = 4096
    timeout_sec = 120
    pip_packages = ["rembg", "Pillow", "numpy", "onnxruntime"]

    def run(self, **kwargs) -> Dict[str, Any]:
        self.validate_inputs(**kwargs)
        from PIL import Image
        from rembg import remove, new_session

        image_path = kwargs["image"]
        model_name = kwargs.get("model", "u2net")
        alpha = kwargs.get("alpha_matting", "false") == "true"

        session = new_session(model_name)

        with open(image_path, "rb") as f:
            input_data = f.read()

        output_data = remove(
            input_data,
            session=session,
            alpha_matting=alpha,
        )

        out_path = os.path.join(
            tempfile.mkdtemp(prefix="rembg_"),
            os.path.splitext(os.path.basename(image_path))[0] + "_nobg.png",
        )
        with open(out_path, "wb") as f:
            f.write(output_data)

        return {"output_image": out_path}
