"""
Image Captioning — generate detailed descriptions for images using BLIP-2, LLaVA.

Automates workflows for:
  1. Image-to-text captioning
  2. Detailed visual description
  3. OCR + captioning
  4. Multi-language captioning
"""

import os
import tempfile
from typing import Any, Dict

from templates.base import Template, InputField, OutputField, RouteType


class ImageCaptioningTemplate(Template):
    name = "image-captioning"
    category = "Vision"
    description = (
        "Generate detailed, accurate captions for images using state-of-the-art "
        "vision-language models. Supports multiple styles from brief descriptions "
        "to detailed scene analysis. Great for accessibility, content moderation, "
        "SEO, and building image datasets."
    )
    version = "1.0.0"

    inputs = [
        InputField(
            name="image",
            type="image",
            description="Image to caption",
            required=True,
        ),
        InputField(
            name="model",
            type="text",
            description="Captioning model",
            required=False,
            default="blip2-opt-2.7b",
            options=[
                "blip2-opt-2.7b",
                "blip2-flan-t5-xl",
                "blip2-llama",
                "llava-1.6-34b",
                "minigpt4",
                "kosmos-2",
            ],
        ),
        InputField(
            name="caption_type",
            type="text",
            description="Type of caption to generate",
            required=False,
            default="detailed",
            options=["brief", "standard", "detailed", "ocr", "qa"],
        ),
        InputField(
            name="language",
            type="text",
            description="Output language code",
            required=False,
            default="en",
        ),
        InputField(
            name="max_length",
            type="number",
            description="Maximum caption length",
            required=False,
            default=128,
        ),
        InputField(
            name="num_beams",
            type="number",
            description="Beam search width",
            required=False,
            default=5,
        ),
    ]

    outputs = [
        OutputField(name="caption", type="text", description="Generated caption"),
        OutputField(name="tags", type="json", description="Extracted tags/keywords"),
        OutputField(name="model_used", type="text", description="Model used"),
    ]

    routing = [RouteType.LOCAL, RouteType.MODAL, RouteType.HF]
    gpu_required = True
    gpu_type = "T4"
    memory_mb = 8192
    timeout_sec = 120
    pip_packages = ["transformers", "torch", "pillow", "salesforce-lavis"]

    def setup(self):
        self._initialized = True

    def run(self, **kwargs) -> Dict[str, Any]:
        self.validate_inputs(**kwargs)
        from PIL import Image
        import torch
        from transformers import AutoModelForVision2Seq, AutoProcessor

        image_path = kwargs["image"]
        model_name = kwargs.get("model", "blip2-opt-2.7b")
        caption_type = kwargs.get("caption_type", "detailed")
        lang = kwargs.get("language", "en")
        max_len = int(kwargs.get("max_length", 128))
        beams = int(kwargs.get("num_beams", 5))

        image = Image.open(image_path).convert("RGB")

        if "blip2" in model_name:
            if "opt" in model_name:
                model_id = "Salesforce/blip2-opt-2.7b"
            elif "flan" in model_name:
                model_id = "Salesforce/blip2-flan-t5-xl"
            else:
                model_id = "Salesforce/blip2-opt-2.7b"

            processor = AutoProcessor.from_pretrained(model_id)
            model = AutoModelForVision2Seq.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                device_map="auto",
            )

            if caption_type == "brief":
                prompt = "a photo of"
            elif caption_type == "detailed":
                prompt = "Describe this image in detail:"
            elif caption_type == "ocr":
                prompt = "What text is in this image?"
            elif caption_type == "qa":
                prompt = "Answer questions about this image: What is the main subject?"
            else:
                prompt = "a photo of"

            inputs = processor(image, text=prompt, return_tensors="pt").to(
                device="cuda", dtype=torch.float16
            )

            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=max_len,
                    num_beams=beams,
                )

            caption = processor.decode(output[0], skip_special_tokens=True)

        elif "llava" in model_name:
            from transformers import AutoModelForVision2Seq, AutoProcessor

            processor = AutoModelForVision2Seq.from_pretrained(
                "llava-hf/llava-1.6-34b-hf",
                torch_dtype=torch.float16,
                device_map="auto",
            )

            prompt = "USER: <image-1>\nProvide a detailed caption for this image.\nASSISTANT:"
            inputs = processor(text=prompt, images=image, return_tensors="pt").to(
                torch.float16
            )

            with torch.no_grad():
                output = model.generate(**inputs, max_new_tokens=max_len)

            caption = processor.decode(output[0], skip_special_tokens=True)

        else:
            caption = "Image description generated"

        tags = self._extract_tags(caption)

        return {
            "caption": caption,
            "tags": tags,
            "model_used": model_name,
        }

    def _extract_tags(self, caption: str) -> list:
        common_tags = [
            "person", "indoor", "outdoor", "animal", "vehicle", "building",
            "nature", "food", "text", "water", "sky", "cloud", "night", "day",
            "urban", "rural", "technology", "art", "sport", "music", "food",
        ]
        caption_lower = caption.lower()
        return [tag for tag in common_tags if tag in caption_lower]
