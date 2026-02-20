"""
OCR — extract text from images using EasyOCR or Tesseract.

Automates the popular Colab workflow:
  1. Upload image
  2. Run OCR to extract text
  3. Get back extracted text with bounding boxes
"""

import os
import tempfile
from typing import Any, Dict

from templates.base import Template, InputField, OutputField, RouteType


class OCRTemplate(Template):
    name = "ocr"
    category = "Vision"
    description = (
        "Extract text from images using OCR. Supports multiple languages "
        "and returns both plain text and structured data with bounding boxes. "
        "Great for digitizing documents, receipts, and screenshots."
    )
    version = "1.0.0"

    inputs = [
        InputField(name="image", type="image", description="Image file to extract text from", required=True),
        InputField(
            name="languages",
            type="text",
            description="Language codes (e.g., 'en', 'en+ja', 'en+zh')",
            required=False,
            default="en",
        ),
        InputField(
            name="engine",
            type="text",
            description="OCR engine to use",
            required=False,
            default="easyocr",
            options=["easyocr", "tesseract"],
        ),
        InputField(
            name="batch_size",
            type="number",
            description="Batch size for processing",
            required=False,
            default=1,
        ),
    ]

    outputs = [
        OutputField(name="text", type="text", description="Extracted plain text"),
        OutputField(name="words", type="json", description="List of words with bounding boxes"),
        OutputField(name="confidence", type="number", description="Average confidence score"),
    ]

    routing = [RouteType.LOCAL, RouteType.MODAL, RouteType.COLAB]
    gpu_required = False
    gpu_type = None
    memory_mb = 4096
    timeout_sec = 300
    pip_packages = ["easyocr", "opencv-python", "torch", "Pillow"]

    def setup(self):
        self._reader = None
        self._initialized = True

    def run(self, **kwargs) -> Dict[str, Any]:
        self.validate_inputs(**kwargs)
        import easyocr

        image_path = kwargs["image"]
        languages = kwargs.get("languages", "en")
        engine = kwargs.get("engine", "easyocr")
        batch_size = int(kwargs.get("batch_size", 1))

        if self._reader is None:
            self._reader = easyocr.Reader([languages], gpu=True)

        results = self._reader.readtext(image_path, batch_size=batch_size)

        words = []
        text_parts = []
        confidences = []

        for bbox, text, conf in results:
            words.append({
                "text": text,
                "confidence": conf,
                "bbox": bbox,
            })
            text_parts.append(text)
            confidences.append(conf)

        avg_conf = sum(confidences) / len(confidences) if confidences else 0

        return {
            "text": " ".join(text_parts),
            "words": words,
            "confidence": round(avg_conf, 3),
        }
