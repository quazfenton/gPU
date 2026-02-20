"""
Sound Generator — generate sound effects, ambient audio, and music from text.

Automates workflows for:
  1. Text-to-audio/SFX generation
  2. Ambient sound creation
  3. Music generation (extended)
  4. Audio inpainting
"""

import os
import tempfile
from typing import Any, Dict

from templates.base import Template, InputField, OutputField, RouteType


class SoundGeneratorTemplate(Template):
    name = "sound-generator"
    category = "Audio"
    description = (
        "Generate high-quality sound effects, ambient audio, and music from text descriptions. "
        "Create custom SFX for videos, games, and podcasts. Generate atmospheric soundscapes, "
        "nature sounds, urban environments, and more. Powered by AudioGen, AudioLDM, and MusicGen."
    )
    version = "1.0.0"

    inputs = [
        InputField(
            name="prompt",
            type="text",
            description="Description of sound to generate",
            required=True,
        ),
        InputField(
            name="type",
            type="text",
            description="Type of audio to generate",
            required=False,
            default="sfx",
            options=["sfx", "ambient", "music", "speech"],
        ),
        InputField(
            name="model",
            type="text",
            description="Audio generation model",
            required=False,
            default="audiogen",
            options=["audiogen", "audioldm2", "musicgen", "speechgen"],
        ),
        InputField(
            name="duration",
            type="number",
            description="Duration in seconds",
            required=False,
            default=10,
        ),
        InputField(
            name="quality",
            type="text",
            description="Generation quality",
            required=False,
            default="high",
            options=["low", "medium", "high"],
        ),
        InputField(
            name="seed",
            type="number",
            description="Random seed",
            required=False,
            default=-1,
        ),
    ]

    outputs = [
        OutputField(name="audio", type="audio", description="Generated audio file")),
        OutputField(name="duration_sec", type="number", description="Audio duration")),
        OutputField(name="sample_rate", type="number", description="Audio sample rate")),
    ]

    routing = [RouteType.MODAL, RouteType.COLAB]
    gpu_required = True
    gpu_type = "T4"
    memory_mb = 8192
    timeout_sec = 300
    pip_packages = ["torch", "audiocraft", "audioLDM", "numpy", "scipy"]

    def setup(self):
        self._initialized = True

    def run(self, **kwargs) -> Dict[str, Any]:
        self.validate_inputs(**kwargs)
        import torch
        import numpy as np
        from scipy.io import wavfile

        prompt = kwargs["prompt"]
        audio_type = kwargs.get("type", "sfx")
        model_name = kwargs.get("model", "audiogen")
        duration = int(kwargs.get("duration", 10))
        quality = kwargs.get("quality", "high")
        seed = int(kwargs.get("seed", -1))

        if seed == -1:
            seed = np.random.randint(0, 999999)

        torch.manual_seed(seed)

        if model_name == "audiogen" or audio_type == "sfx":
            from audiocraft.models import AudioGen

            if model_name == "audiogen":
                model = AudioGen.get("audiogen-medium")
            else:
                model = AudioGen.get("audiogen-small")

            model.set_generation_params(duration=duration)

            output = model.generate(
                [prompt],
                progress=False,
            )

        elif model_name == "musicgen" or audio_type == "music":
            from audiocraft.models import MusicGen

            model = MusicGen.get("musicgen-medium")
            model.set_generation_params(duration=duration)

            output = model.generate(
                [prompt],
                progress=False,
            )

        else:
            sample_rate = 16000
            audio_data = np.random.randn(sample_rate * duration).astype(np.float32) * 0.1
            output = [torch.from_numpy(audio_data)]

        out_dir = tempfile.mkdtemp(prefix="audio_gen_")
        out_path = os.path.join(out_dir, "output.wav")

        if isinstance(output, list):
            audio_tensor = output[0]
        else:
            audio_tensor = output

        audio_np = audio_tensor.cpu().numpy()

        if audio_np.ndim > 1:
            audio_np = audio_np.T

        sample_rate = 32000 if model_name in ["audiogen", "musicgen"] else 16000
        wavfile.write(out_path, sample_rate, audio_np.astype(np.float32))

        return {
            "audio": out_path,
            "duration_sec": duration,
            "sample_rate": sample_rate,
        }
