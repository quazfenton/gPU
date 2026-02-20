"""
Image Segmenter — segment images into different components using SAM.

Automates the popular Colab workflow:
  1. Upload image
  2. Run Segment Anything Model (SAM)
  3. Get back segmented regions with masks
"""

import os
import tempfile
from typing import Any, Dict

from templates.base import Template, InputField, OutputField, RouteType


class ImageSegmenterTemplate(Template):
    name = "image-segmenter"
    category = "Vision"
    description = (
        "Segment images into different components using Meta's Segment Anything Model. "
        "Automatically identifies and masks all objects in an image. "
        "Great for image editing, object removal, and computer vision tasks."
    )
    version = "1.0.0"

    inputs = [
        InputField(name="image", type="image", description="Image to segment", required=True),
        InputField(
            name="mode",
            type="text",
            description="Segmentation mode",
            required=False,
            default="automatic",
            options=["automatic", "bbox", "point"],
        ),
        InputField(
            name="points_per_side",
            type="number",
            description="Points per side for automatic mask generation",
            required=False,
            default=32,
        ),
        InputField(
            name="min_area",
            type="number",
            description="Minimum mask area in pixels",
            required=False,
            default=100,
        ),
    ]

    outputs = [
        OutputField(name="masks", type="json", description="List of segmentation masks"),
        OutputField(name="num_masks", type="number", description="Number of masks generated"),
        OutputField(name="annotated_image", type="image", description="Image with masks overlaid"),
    ]

    routing = [RouteType.LOCAL, RouteType.MODAL, RouteType.COLAB]
    gpu_required = True
    gpu_type = "T4"
    memory_mb = 8192
    timeout_sec = 300
    pip_packages = ["segment-anything", "opencv-python", "numpy", "pillow"]

    def setup(self):
        from segment_anything import sam_model_registry, SamPredictor
        self._predictor = None
        self._initialized = True

    def run(self, **kwargs) -> Dict[str, Any]:
        self.validate_inputs(**kwargs)
        from segment_anything import sam_model_registry, SamPredictor
        import cv2
        import numpy as np
        from PIL import Image

        image_path = kwargs["image"]
        mode = kwargs.get("mode", "automatic")
        points_per_side = int(kwargs.get("points_per_side", 32))
        min_area = int(kwargs.get("min_area", 100))

        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self._predictor is None:
            sam = sam_model_registry["vit_b"](checkpoint=None)
            self._predictor = SamPredictor(sam)

        self._predictor.set_image(image_rgb)

        if mode == "automatic":
            from segment_anything import SamAutomaticMaskGenerator
            mask_generator = SamAutomaticMaskGenerator(
                self._predictor.model,
                points_per_side=points_per_side,
            )
            masks = mask_generator.generate(image_rgb)
        else:
            h, w = image_rgb.shape[:2]
            center_point = np.array([[w // 2, h // 2]])
            masks = self._predictor.predict(
                point_coords=center_point,
                point_labels=np.array([1]),
            )

        filtered_masks = [
            {"segmentation": m["segmentation"].tolist(), "area": int(m["area"])}
            for m in masks if m["area"] >= min_area
        ]

        vis_image = image.copy()
        for m in filtered_masks:
            mask = np.array(m["segmentation"])
            color = np.random.randint(0, 255, 3).tolist()
            vis_image[mask] = vis_image[mask] * 0.5 + np.array(color) * 0.5

        out_path = os.path.join(tempfile.mkdtemp(prefix="segmented_"), "segmented.jpg")
        Image.fromarray(vis_image).save(out_path)

        return {
            "masks": filtered_masks,
            "num_masks": len(filtered_masks),
            "annotated_image": out_path,
        }
