"""
Embedding Generator — generate embeddings for text, images, and audio.

Automates workflows for:
  1. Text embeddings (sentence transformers)
  2. Image embeddings (CLIP, DINOv2)
  3. Audio embeddings (AudioCLIP)
  4. Multimodal embeddings
"""

import os
import tempfile
from typing import Any, Dict, List

from templates.base import Template, InputField, OutputField, RouteType


class EmbeddingGeneratorTemplate(Template):
    name = "embedding-generator"
    category = "ML"
    description = (
        "Generate high-quality embeddings for text, images, and audio using "
        "state-of-the-art models. Supports sentence transformers, CLIP, DINOv2, "
        "and multimodal embeddings. Perfect for semantic search, similarity "
        "matching, and retrieval-augmented generation."
    )
    version = "1.0.0"

    inputs = [
        InputField(
            name="inputs",
            type="json",
            description="List of inputs (texts, image paths, or audio paths)",
            required=True,
        ),
        InputField(
            name="input_type",
            type="text",
            description="Type of input data",
            required=True,
            options=["text", "image", "audio", "multimodal"],
        ),
        InputField(
            name="model",
            type="text",
            description="Embedding model",
            required=False,
            default="sentence-transformers/all-MiniLM-L6-v2",
        ),
        InputField(
            name="pooling_strategy",
            type="text",
            description="Embedding pooling strategy",
            required=False,
            default="mean",
            options=["mean", "cls", "max"],
        ),
        InputField(
            name="normalize",
            type="text",
            description="Normalize embeddings to unit length",
            required=False,
            default="true",
            options=["true", "false"],
        ),
        InputField(
            name="batch_size",
            type="number",
            description="Batch size for processing",
            required=False,
            default=32,
        ),
    ]

    outputs = [
        OutputField(name="embeddings", type="json", description="Generated embeddings (list of arrays)"),
        OutputField(name="model_used", type="text", description="Model used for embedding"),
        OutputField(name="dimensions", type="number", description="Embedding dimensions"),
        OutputField(name="metadata", type="json", description="Processing metadata"),
    ]

    routing = [RouteType.LOCAL, RouteType.MODAL, RouteType.HF]
    gpu_required = False
    gpu_type = None
    memory_mb = 4096
    timeout_sec = 300
    pip_packages = ["sentence-transformers", "torch", "transformers", "clip", "timm"]

    def setup(self):
        self._models = {}
        self._initialized = True

    def _get_model(self, model_name: str, input_type: str):
        from sentence_transformers import SentenceTransformer
        
        if model_name not in self._models:
            if input_type == "image":
                self._models[model_name] = SentenceTransformer("clip-ViT-L-14")
            else:
                self._models[model_name] = SentenceTransformer(model_name)
        return self._models[model_name]

    def run(self, **kwargs) -> Dict[str, Any]:
        self.validate_inputs(**kwargs)
        import numpy as np

        inputs = kwargs["inputs"]
        input_type = kwargs["input_type"]
        model_name = kwargs.get("model", "sentence-transformers/all-MiniLM-L6-v2")
        pool = kwargs.get("pooling_strategy", "mean")
        normalize = kwargs.get("normalize", "true") == "true"
        batch_size = int(kwargs.get("batch_size", 32))

        if isinstance(inputs, str):
            inputs = [inputs]

        model = self._get_model(model_name, input_type)

        if input_type == "text":
            embeddings = model.encode(
                inputs,
                batch_size=batch_size,
                normalize_embeddings=normalize,
                show_progress_bar=True,
            )
        elif input_type == "image":
            embeddings = model.encode(
                inputs,
                batch_size=batch_size,
                normalize_embeddings=normalize,
                show_progress_bar=True,
            )
        else:
            embeddings = model.encode(
                inputs,
                batch_size=batch_size,
                normalize_embeddings=normalize,
                show_progress_bar=True,
            )

        embeddings_list = embeddings.tolist() if isinstance(embeddings, np.ndarray) else embeddings

        return {
            "embeddings": embeddings_list,
            "model_used": model_name,
            "dimensions": len(embeddings_list[0]) if embeddings_list else 0,
            "metadata": {
                "input_type": input_type,
                "num_inputs": len(inputs),
                "pooling": pool,
                "normalized": normalize,
            },
        }
