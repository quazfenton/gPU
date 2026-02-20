"""
Object Detector — detect and identify objects in images using YOLO.

Automates the popular Colab workflow:
  1. Upload image
  2. Run YOLOv8 object detection
  3. Get back bounding boxes, labels, and annotated image
"""

import os
import tempfile
from typing import Any, Dict, List

from templates.base import Template, InputField, OutputField, RouteType


class ObjectDetectorTemplate(Template):
    name = "object-detector"
    category = "Vision"
    description = (
        "Detect and identify objects in images using YOLOv8. "
        "Returns bounding boxes, class labels, confidence scores, "
        "and an annotated image. Supports 80+ COCO object classes."
    )
    version = "1.0.0"

    inputs = [
        InputField(name="image", type="image", description="Image file to analyze", required=True),
        InputField(
            name="model_size",
            type="text",
            description="YOLO model size (larger = more accurate but slower)",
            required=False,
            default="n",
            options=["n", "s", "m", "l", "x"],
        ),
        InputField(
            name="confidence_threshold",
            type="number",
            description="Minimum confidence for detections (0-1)",
            required=False,
            default=0.25,
        ),
        InputField(
            name="iou_threshold",
            type="number",
            description="IOU threshold for NMS (0-1)",
            required=False,
            default=0.45,
        ),
    ]

    outputs = [
        OutputField(name="detections", type="json", description="List of detected objects with bounding boxes"),
        OutputField(name="annotated_image", type="image", description="Image with bounding boxes drawn"),
        OutputField(name="num_objects", type="number", description="Total number of objects detected"),
    ]

    routing = [RouteType.LOCAL, RouteType.MODAL, RouteType.COLAB]
    gpu_required = True
    gpu_type = "T4"
    memory_mb = 4096
    timeout_sec = 180
    pip_packages = ["ultralytics", "opencv-python", "pillow"]

    def setup(self):
        self._models = {}
        self._initialized = True

    def _get_model(self, size: str):
        from ultralytics import YOLO
        if size not in self._models:
            self._models[size] = YOLO(f"yolov8{size}.pt")
        return self._models[size]

    def run(self, **kwargs) -> Dict[str, Any]:
        self.validate_inputs(**kwargs)

        image_path = kwargs["image"]
        model_size = kwargs.get("model_size", "n")
        conf = float(kwargs.get("confidence_threshold", 0.25))
        iou = float(kwargs.get("iou_threshold", 0.45))

        model = self._get_model(model_size)
        results = model.predict(
            image_path,
            conf=conf,
            iou=iou,
            verbose=False,
        )

        detections = []
        for r in results:
            for box in r.boxes:
                detections.append({
                    "class": model.names[int(box.cls[0])],
                    "confidence": float(box.conf[0]),
                    "bbox": {
                        "x1": float(box.xyxy[0][0]),
                        "y1": float(box.xyxy[0][1]),
                        "x2": float(box.xyxy[0][2]),
                        "y2": float(box.xyxy[0][3]),
                    },
                })

        annotated = results[0].plot()
        out_path = os.path.join(tempfile.mkdtemp(prefix="detected_"), "annotated.jpg")
        from PIL import Image
        Image.fromarray(annotated).save(out_path)

        return {
            "detections": detections,
            "annotated_image": out_path,
            "num_objects": len(detections),
        }
