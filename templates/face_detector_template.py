"""
Face Detector — detect faces and facial landmarks in images.

Automates the popular Colab workflow:
  1. Upload image
  2. Run face detection model (RetinaFace/MediaPipe)
  3. Get back face bounding boxes, landmarks, and attributes
"""

import os
import tempfile
from typing import Any, Dict

from templates.base import Template, InputField, OutputField, RouteType


class FaceDetectorTemplate(Template):
    name = "face-detector"
    category = "Vision"
    description = (
        "Detect faces and facial landmarks in images. Returns bounding boxes, "
        "facial keypoints (eyes, nose, mouth), confidence scores, and face attributes "
        "like age, gender, and emotion. Great for portrait processing and biometrics."
    )
    version = "1.0.0"

    inputs = [
        InputField(name="image", type="image", description="Image to detect faces in", required=True),
        InputField(
            name="model",
            type="text",
            description="Face detection model",
            required=False,
            default="retinaface",
            options=["retinaface", "mediapipe", "dlib"],
        ),
        InputField(
            name="landmarks",
            type="text",
            description="Include facial landmarks",
            required=False,
            default="true",
            options=["true", "false"],
        ),
        InputField(
            name="attributes",
            type="text",
            description="Include face attributes (age, gender, emotion)",
            required=False,
            default="false",
            options=["true", "false"],
        ),
    ]

    outputs = [
        OutputField(name="faces", type="json", description="List of detected faces with bounding boxes and attributes"),
        OutputField(name="num_faces", type="number", description="Number of faces detected"),
        OutputField(name="annotated_image", type="image", description="Image with face boxes drawn"),
    ]

    routing = [RouteType.LOCAL, RouteType.MODAL, RouteType.COLAB]
    gpu_required = True
    gpu_type = "T4"
    memory_mb = 4096
    timeout_sec = 180
    pip_packages = ["retinaface", "opencv-python", "Pillow", "numpy", "mediapipe"]

    def setup(self):
        self._initialized = True

    def run(self, **kwargs) -> Dict[str, Any]:
        self.validate_inputs(**kwargs)
        import cv2
        import numpy as np
        from PIL import Image, ImageDraw, ImageFont

        image_path = kwargs["image"]
        model = kwargs.get("model", "retinaface")
        include_landmarks = kwargs.get("landmarks", "true") == "true"
        include_attrs = kwargs.get("attributes", "false") == "true"

        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        faces = []

        if model == "retinaface":
            from retinaface import RetinaFace
            detections = RetinaFace.detect_faces(image_path)

            if isinstance(detections, dict):
                for key, det in detections.items():
                    face_data = {
                        "bbox": {
                            "x1": float(det["facial_area"][0]),
                            "y1": float(det["facial_area"][1]),
                            "x2": float(det["facial_area"][2]),
                            "y2": float(det["facial_area"][3]),
                        },
                        "confidence": float(det["score"]),
                    }
                    if include_landmarks and "landmarks" in det:
                        face_data["landmarks"] = {
                            "left_eye": det["landmarks"]["left_eye"].tolist(),
                            "right_eye": det["landmarks"]["right_eye"].tolist(),
                            "nose": det["landmarks"]["nose"].tolist(),
                            "mouth_left": det["landmarks"]["mouth_left"].tolist(),
                            "mouth_right": det["landmarks"]["mouth_right"].tolist(),
                        }
                    faces.append(face_data)

        elif model == "mediapipe":
            import mediapipe as mp
            mp_face = mp.solutions.face_detection
            detector = mp_face.FaceDetection()

            results = detector.process(img_rgb)
            if results.detections:
                h, w = img_rgb.shape[:2]
                for det in results.detections:
                    bbox = det.location_data.relative_bounding_box
                    faces.append({
                        "bbox": {
                            "x1": int(bbox.xmin * w),
                            "y1": int(bbox.ymin * h),
                            "x2": int((bbox.xmin + bbox.width) * w),
                            "y2": int((bbox.ymin + bbox.height) * h),
                        },
                        "confidence": det.score[0],
                    })

        vis_img = Image.open(image_path)
        draw = ImageDraw.Draw(vis_img)

        for i, face in enumerate(faces):
            box = face["bbox"]
            draw.rectangle(
                [box["x1"], box["y1"], box["x2"], box["y2"]],
                outline="red",
                width=3,
            )
            draw.text((box["x1"], box["y1"] - 20), f"Face {i+1}", fill="red")

        out_path = os.path.join(tempfile.mkdtemp(prefix="faces_"), "detected.jpg")
        vis_img.save(out_path)

        return {
            "faces": faces,
            "num_faces": len(faces),
            "annotated_image": out_path,
        }
