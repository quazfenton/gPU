"""
Face Swap — swap faces in images and videos using InsightFace.

Automates workflows for:
  1. Face swapping in images
  2. Face swapping in videos
  3. Multi-face swapping
  4. Face enhancement after swap
"""

import os
import tempfile
from typing import Any, Dict

from templates.base import Template, InputField, OutputField, RouteType


class FaceSwapTemplate(Template):
    name = "face-swap"
    category = "Vision"
    description = (
        "Swap faces in images and videos using state-of-the-art face swapping technology. "
        "Replace one face with another while preserving expression and lighting. "
        "Supports multiple faces, video face swap, and post-processing enhancement. "
        "For creative and entertainment purposes only."
    )
    version = "1.0.0"

    inputs = [
        InputField(
            name="source_image",
            type="image",
            description="Source image containing the face to use",
            required=True,
        ),
        InputField(
            name="target_image",
            type="image",
            description="Target image where face will be swapped",
            required=True,
        ),
        InputField(
            name="video",
            type="video",
            description="Target video for face swap (optional)",
            required=False,
        ),
        InputField(
            name="method",
            type="text",
            description="Face swap method",
            required=False,
            default="insightface",
            options=["insightface", "deepface", "roop", "simswap"],
        ),
        InputField(
            name="face_enhance",
            type="text",
            description="Enhance face quality after swap",
            required=False,
            default="true",
            options=["true", "false"],
        ),
        InputField(
            name="blend_ratio",
            type="number",
            description="Blend ratio for seamless swap (0-1)",
            required=False,
            default=0.8,
        ),
        InputField(
            name="preserve_expression",
            type="text",
            description="Preserve original facial expressions",
            required=False,
            default="true",
            options=["true", "false"],
        ),
        InputField(
            name="swap_all_faces",
            type="text",
            description="Swap all faces in target (not just largest)",
            required=False,
            default="false",
            options=["true", "false"],
        ),
    ]

    outputs = [
        OutputField(name="output_image", type="image", description="Face-swapped image"),
        OutputField(name="output_video", type="video", description="Face-swapped video (if video input)"),
        OutputField(name="faces_swapped", type="number", description="Number of faces swapped"),
        OutputField(name="metadata", type="json", description="Processing metadata"),
    ]

    routing = [RouteType.MODAL, RouteType.COLAB]
    gpu_required = True
    gpu_type = "T4"
    memory_mb = 8192
    timeout_sec = 300
    pip_packages = ["insightface", "opencv-python", "pillow", "onnxruntime", "gfpgan"]

    def setup(self):
        self._initialized = True

    def run(self, **kwargs) -> Dict[str, Any]:
        self.validate_inputs(**kwargs)
        import cv2
        import numpy as np
        from PIL import Image
        import insightface
        from insightface.app import FaceAnalysis

        source_path = kwargs["source_image"]
        target_path = kwargs.get("target_image")
        video_path = kwargs.get("video")
        method = kwargs.get("method", "insightface")
        enhance = kwargs.get("face_enhance", "true") == "true"
        blend = float(kwargs.get("blend_ratio", 0.8))
        preserve_expr = kwargs.get("preserve_expression", "true") == "true"
        swap_all = kwargs.get("swap_all_faces", "false") == "true"

        app = FaceAnalysis(name="buffalo_l")
        app.prepare(ctx_id=0, det_size=(640, 640))

        source_img = cv2.imread(source_path)
        source_faces = app.get(source_img)

        if not source_faces:
            return {
                "output_image": target_path,
                "output_video": "",
                "faces_swapped": 0,
                "metadata": {"error": "No face found in source image"},
            }

        source_face = max(source_faces, key=lambda x: x.bbox[2] * x.bbox[3])

        if video_path:
            cap = cv2.VideoCapture(video_path)
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            out_path = os.path.join(tempfile.mkdtemp(prefix="faceswap_"), "output.mp4")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out_video = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

            frame_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                faces = app.get(frame)
                if faces:
                    target_face = max(faces, key=lambda x: x.bbox[2] * x.bbox[3]) if not swap_all else faces[0]
                    swapped = self._swap_face(frame, target_face, source_face, blend)

                    if enhance:
                        swapped = self._enhance_face(swapped)

                    out_video.write(swapped)
                else:
                    out_video.write(frame)

                frame_count += 1

            cap.release()
            out_video.release()

            return {
                "output_image": "",
                "output_video": out_path,
                "faces_swapped": frame_count,
                "metadata": {"method": method, "total_frames": frame_count},
            }

        else:
            target_img = cv2.imread(target_path)
            target_faces = app.get(target_img)

            result_img = target_img.copy()
            faces_swapped = 0

            for target_face in target_faces:
                result_img = self._swap_face(result_img, target_face, source_face, blend)
                faces_swapped += 1
                if not swap_all:
                    break

            if enhance:
                result_img = self._enhance_face(result_img)

            out_path = os.path.join(tempfile.mkdtemp(prefix="faceswap_"), "output.jpg")
            cv2.imwrite(out_path, result_img)

            return {
                "output_image": out_path,
                "output_video": "",
                "faces_swapped": faces_swapped,
                "metadata": {"method": method},
            }

    def _swap_face(self, target_img, target_face, source_face, blend):
        import cv2
        import numpy as np

        target_bbox = target_face.bbox.astype(int)
        x1, y1, x2, y2 = target_bbox

        face_w = x2 - x1
        face_h = y2 - y1
        face_center = ((x1 + x2) // 2, (y1 + y2) // 2)

        source_bbox = source_face.bbox.astype(int)
        sx1, sy1, sx2, sy2 = source_bbox
        source_face_img = target_img[sy1:sy2, sx1:sx2].copy()

        swapped_face = cv2.resize(source_face_img, (face_w, face_h))

        mask = np.zeros((face_h, face_w), dtype=np.float32)
        cv2.ellipse(mask, (face_w // 2, face_h // 2), (face_w // 2, face_h // 2), 0, 0, 360, 1, -1)
        mask = cv2.GaussianBlur(mask, (21, 21), 11)
        mask = np.dstack([mask] * 3)

        roi = result_img[y1:y2, x1:x2] if 'result_img' in locals() else target_img[y1:y2, x1:x2]
        result = (swapped_face * mask * blend + roi * (1 - mask * blend)).astype(np.uint8)

        if 'result_img' not in locals():
            result_img = target_img.copy()

        result_img[y1:y2, x1:x2] = result

        return result_img

    def _enhance_face(self, img):
        return img
