"""
Music Generator — generate music from text prompts using MusicGen.

Automates the popular Colab workflow:
  1. Enter text description of desired music
  2. Run MusicGen model
  3. Get back generated audio track
"""

import os
import tempfile
from typing import Any, Dict

from templates.base import Template, InputField, OutputField, RouteType


class MusicGeneratorTemplate(Template):
    name = "music-generator"
    category = "Audio"
    description = (
        "Generate music from text prompts using Meta's MusicGen. "
        "Create royalty-free music for videos, podcasts, and projects. "
        "Describe the style, mood, instruments, and tempo."
    )
    version = "1.0.0"

    inputs = [
        InputField(name="prompt", type="text", description="Description of music to generate", required=True),
        InputField(
            name="model",
            type="text",
            description="MusicGen model size",
            required=False,
            default="medium",
            options=["small", "medium", "large", "melody"],
        ),
        InputField(
            name="duration",
            type="number",
            description="Duration in seconds",
            required=False,
            default=10,
        ),
        InputField(
            name="temperature",
            type="number",
            description="Generation temperature (0-1, lower = more deterministic)",
            required=False,
            default=1.0,
        ),
    ]

    outputs = [
        OutputField(name="audio", type="audio", description="Generated music audio"),
        OutputField(name="duration_sec", type="number", description="Actual audio duration"),
    ]

    routing = [RouteType.LOCAL, RouteType.MODAL, RouteType.COLAB]
    gpu_required = True
    gpu_type = "T4"
    memory_mb = 8192
    timeout_sec = 300
    pip_packages = ["audiocraft", "torch", "torchaudio"]

    def setup(self):
        self._model = None
        self._initialized = True

    def run(self, **kwargs) -> Dict[str, Any]:
        self.validate_inputs(**kwargs)
        import torch
        from audiocraft.models import MusicGen

        prompt = kwargs["prompt"]
        model_size = kwargs.get("model", "medium")
        duration = int(kwargs.get("duration", 10))
        temperature = float(kwargs.get("temperature", 1.0))

        if self._model is None:
            self._model = MusicGen.get_pretrained(f"facebook/musicgen-{model_size}")

        self._model.set_generation_params(
            duration=duration,
            temperature=temperature,
        )

        output = self._model.generate(
            [prompt],
            progress=False,
        )

        out_path = os.path.join(tempfile.mkdtemp(prefix="music_"), "generated.wav")

        import scipy.io.wavfile as sio
        sample_rate = 32000
        sio.write(out_path, sample_rate, output[0].cpu().numpy().T)

        return {
            "audio": out_path,
            "duration_sec": duration,
        }
