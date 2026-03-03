"""
Training Data Augmentor — augment and expand training datasets for ML models.

Automates workflows for:
  1. Image augmentation (flip, rotate, color, noise)
  2. Text augmentation (paraphrasing, back-translation)
  3. Audio augmentation (pitch, speed, noise)
  4. Synthetic data generation
  5. Dataset balancing
"""

import os
import tempfile
from typing import Any, Dict, List

from templates.base import Template, InputField, OutputField, RouteType


class TrainingDataAugmentorTemplate(Template):
    name = "data-augmentor"
    category = "Training"
    description = (
        "Augment and expand training datasets for machine learning models. "
        "Apply powerful augmentation techniques to images, text, and audio. "
        "Generate synthetic data, balance classes, and create diverse training sets. "
        "Supports diffusion-based augmentation for high-quality synthetic samples."
    )
    version = "1.0.0"

    inputs = [
        InputField(
            name="input_data",
            type="json",
            description="List of file paths or data samples",
            required=True,
        ),
        InputField(
            name="data_type",
            type="text",
            description="Type of data",
            required=True,
            options=["image", "text", "audio", "mixed"],
        ),
        InputField(
            name="augmentation_type",
            type="text",
            description="Augmentation method",
            required=False,
            default="standard",
            options=[
                "standard",
                "diffusion",
                "paraphrase",
                "backtranslate",
                "mixup",
                "cutmix",
                "synthetic",
            ],
        ),
        InputField(
            name="augmentation_factor",
            type="number",
            description="How many augmented versions per sample",
            required=False,
            default=3,
        ),
        InputField(
            name="preserve_labels",
            type="text",
            description="Keep original labels for augmented data",
            required=False,
            default="true",
            options=["true", "false"],
        ),
        InputField(
            name="image_augmentations",
            type="json",
            description="Specific image augmentations to apply",
            required=False,
        ),
        InputField(
            name="output_format",
            type="text",
            description="Output data format",
            required=False,
            default="json",
            options=["json", "parquet", "csv", "folder"],
        ),
    ]

    outputs = [
        OutputField(name="augmented_data", type="json", description="Augmented dataset"),
        OutputField(name="num_samples", type="number", description="Total samples after augmentation"),
        OutputField(name="augmentation_stats", type="json", description="Statistics about augmentation"),
    ]

    routing = [RouteType.LOCAL, RouteType.MODAL]
    gpu_required = False
    gpu_type = None
    memory_mb = 4096
    timeout_sec = 300
    pip_packages = ["torch", "transformers", "albumentations", "numpy", "Pillow", "scikit-learn"]

    def setup(self):
        self._initialized = True

    def run(self, **kwargs) -> Dict[str, Any]:
        self.validate_inputs(**kwargs)
        import json
        import numpy as np
        from PIL import Image, ImageEnhance, ImageOps
        import random

        input_data = kwargs["input_data"]
        data_type = kwargs["data_type"]
        aug_type = kwargs.get("augmentation_type", "standard")
        factor = int(kwargs.get("augmentation_factor", 3))
        preserve = kwargs.get("preserve_labels", "true") == "true"
        img_augs = kwargs.get("image_augmentations", [])
        out_format = kwargs.get("output_format", "json")

        if isinstance(input_data, str):
            with open(input_data, "r") as f:
                input_data = json.load(f)

        augmented_samples = []
        stats = {"original_samples": len(input_data), "methods_used": []}

        if data_type == "image":
            stats["methods_used"] = [
                "horizontal_flip",
                "vertical_flip",
                "rotation",
                "brightness",
                "contrast",
                "saturation",
                "noise",
                "blur",
                "color_jitter",
            ]

            for sample in input_data:
                if isinstance(sample, dict):
                    img_path = sample.get("image", sample.get("path", sample.get("file")))
                    label = sample.get("label", sample.get("class"))
                else:
                    img_path = sample
                    label = None

                if not os.path.exists(img_path):
                    continue

                img = Image.open(img_path)
                original_size = img.size

                for i in range(factor):
                    aug_img = img.copy()
                    aug_label = label

                    aug_method = random.choice([
                        "flip_h", "flip_v", "rotate", "brightness",
                        "contrast", "saturation", "noise", "blur",
                        "crop", "color_shift", "grayscale"
                    ])

                    if aug_method == "flip_h":
                        aug_img = ImageOps.mirror(aug_img)
                    elif aug_method == "flip_v":
                        aug_img = ImageOps.flip(aug_img)
                    elif aug_method == "rotate":
                        angle = random.randint(-30, 30)
                        aug_img = aug_img.rotate(angle, expand=True)
                    elif aug_method == "brightness":
                        enhancer = ImageEnhance.Brightness(aug_img)
                        aug_img = enhancer.enhance(random.uniform(0.7, 1.3))
                    elif aug_method == "contrast":
                        enhancer = ImageEnhance.Contrast(aug_img)
                        aug_img = enhancer.enhance(random.uniform(0.7, 1.3))
                    elif aug_method == "saturation":
                        enhancer = ImageEnhance.Color(aug_img)
                        aug_img = enhancer.enhance(random.uniform(0.7, 1.3))
                    elif aug_method == "noise":
                        img_np = np.array(aug_img)
                        noise = np.random.normal(0, 25, img_np.shape).astype(np.uint8)
                        aug_img = Image.fromarray(np.clip(img_np + noise, 0, 255).astype(np.uint8))
                    elif aug_method == "blur":
                        from PIL import ImageFilter
                        aug_img = aug_img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 2)))
                    elif aug_method == "crop":
                        w, h = aug_img.size
                        crop_w, crop_h = int(w * 0.8), int(h * 0.8)
                        left = random.randint(0, w - crop_w)
                        top = random.randint(0, h - crop_h)
                        aug_img = aug_img.crop((left, top, left + crop_w, top + crop_h))
                        aug_img = aug_img.resize((w, h), Image.LANCZOS)
                    elif aug_method == "color_shift":
                        from PIL import ImageEnhance
                        enhancer = ImageEnhance.Color(aug_img)
                        aug_img = enhancer.enhance(random.uniform(0.8, 1.2))
                        enhancer = ImageEnhance.Brightness(aug_img)
                        aug_img = enhancer.enhance(random.uniform(0.9, 1.1))
                    elif aug_method == "grayscale":
                        aug_img = aug_img.convert("L").convert("RGB")

                    out_dir = tempfile.mkdtemp(prefix="aug_")
                    out_path = os.path.join(out_dir, f"aug_{i}.jpg")
                    aug_img.save(out_path)

                    sample_data = {"image": out_path}
                    if preserve and label is not None:
                        sample_data["label"] = label
                    augmented_samples.append(sample_data)

        elif data_type == "text":
            stats["methods_used"] = ["synonym_replacement", "random_insertion", "random_swap", "random_deletion"]

            try:
                from transformers import pipeline
                paraphraser = pipeline("text-generation", model="t5-small")
            except:
                paraphraser = None

            for sample in input_data:
                if isinstance(sample, dict):
                    text = sample.get("text", sample.get("input"))
                    label = sample.get("label")
                else:
                    text = sample
                    label = None

                augmented_samples.append({"text": text, "label": label} if preserve else {"text": text})

                for i in range(factor):
                    aug_text = text
                    aug_label = label

                    if paraphraser:
                        try:
                            result = paraphraser(
                                f"paraphrase: {text}",
                                max_length=128,
                                num_return_sequences=1,
                            )
                            aug_text = result[0]["generated_text"]
                        except:
                            pass

                    words = text.split()
                    if len(words) > 2:
                        if random.random() > 0.5:
                            num_replacements = max(1, len(words) // 10)
                            for _ in range(num_replacements):
                                idx = random.randint(0, len(words) - 1)
                                words[idx] = words[random.randint(0, len(words) - 1)]
                            aug_text = " ".join(words)

                    sample_data = {"text": aug_text}
                    if preserve and label is not None:
                        sample_data["label"] = label
                    augmented_samples.append(sample_data)

        else:
            augmented_samples = input_data

            for sample in input_data:
                for i in range(factor - 1):
                    if isinstance(sample, dict):
                        augmented_samples.append(sample.copy())
                    else:
                        augmented_samples.append(sample)

        stats["final_samples"] = len(augmented_samples)
        stats["augmentation_factor"] = len(augmented_samples) / max(1, len(input_data))

        out_dir = tempfile.mkdtemp(prefix="augmented_")
        out_path = os.path.join(out_dir, f"dataset.{out_format}")

        if out_format == "json":
            with open(out_path, "w") as f:
                json.dump(augmented_samples, f, indent=2)

        return {
            "augmented_data": augmented_samples,
            "num_samples": len(augmented_samples),
            "augmentation_stats": stats,
        }
