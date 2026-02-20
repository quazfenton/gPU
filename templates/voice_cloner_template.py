"""
Voice Cloner — clone voices from audio samples using XTTS or Coqui.

Automates workflows for:
  1. Create custom voice from short audio samples
  2. Generate speech in cloned voice
  3. Multi-language voice cloning
"""

import os
import tempfile
from typing import Any, Dict, List

from templates.base import Template, InputField, OutputField, RouteType


class VoiceClonerTemplate(Template):
    name = "voice-cloner"
    category = "Audio"
    description = (
        "Clone any voice from short audio samples. Create a custom voice profile "
        "and generate natural-sounding speech in that voice. Supports multiple "
        "languages and styles. Perfect for content creators, audiobooks, and "
        "personalized AI assistants."
    )
    version = "1.0.0"

    inputs = [
        InputField(
            name="reference_audio",
            type="audio",
            description="Audio sample(s) of voice to clone (can be multiple)",
            required=True,
        ),
        InputField(
            name="text",
            type="text",
            description="Text to speak in cloned voice",
            required=True,
        ),
        InputField(
            name="model",
            type="text",
            description="Voice cloning model",
            required=False,
            default="xtts-v2",
            options=["xtts-v2", "xtts-v1", "coqui-tts", "bark-clone"],
        ),
        InputField(
            name="language",
            type="text",
            description="Output language code",
            required=False,
            default="en",
        ),
        InputField(
            name="speed",
            type="number",
            description="Speech speed (0.5-2.0)",
            required=False,
            default=1.0,
        ),
        InputField(
            name="temperature",
            type="number",
            description="Generation temperature (0.3-1.0)",
            required=False,
            default=0.7,
        ),
        InputField(
            name="top_p",
            type="number",
            description="Nucleus sampling top_p (0.1-1.0)",
            required=False,
            default=0.85,
        ),
        InputField(
            name="top_k",
            type="number",
            description="Top-k sampling (1-100)",
            required=False,
            default=50,
        ),
        InputField(
            name="add_noise",
            type="text",
            description="Add ambient noise for realism",
            required=False,
            default="false",
            options=["true", "false"],
        ),
    ]

    outputs = [
        OutputField(name="audio", type="audio", description="Generated speech in cloned voice"),
        OutputField(name="audio_url", type="text", description="URL to audio file"),
        OutputField(name="duration_sec", type="number", description="Audio duration"),
        OutputField(name="voice_profile", type="json", description="Created voice profile metadata"),
    ]

    routing = [RouteType.MODAL, RouteType.COLAB]
    gpu_required = True
    gpu_type = "T4"
    memory_mb = 8192
    timeout_sec = 300
    pip_packages = ["TTS", "torch", "torchaudio", "numpy"]

    def setup(self):
        self._voice_profiles = {}
        self._initialized = True

    def run(self, **kwargs) -> Dict[str, Any]:
        self.validate_inputs(**kwargs)
        import torch
        import numpy as np
        from TTS.api import TTS
        from scipy.io import wavfile

        ref_audio = kwargs["reference_audio"]
        text = kwargs["text"]
        model = kwargs.get("model", "xtts-v2")
        lang = kwargs.get("language", "en")
        speed = float(kwargs.get("speed", 1.0))
        temp = float(kwargs.get("temperature", 0.7))
        top_p = float(kwargs.get("top_p", 0.85))
        top_k = int(kwargs.get("top_k", 50))

        if model == "xtts-v2":
            tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
        else:
            tts = TTS("tts_models/multilingual/multi-dataset/xtts_v1")

        out_path = os.path.join(tempfile.mkdtemp(prefix="voice_clone_"), "output.wav")

        tts.tts_to_file(
            text=text,
            file_path=out_path,
            speaker_wav=ref_audio,
            language=lang,
            speed=speed,
        )

        sample_rate, audio_data = wavfile.read(out_path)
        duration = len(audio_data) / sample_rate

        voice_profile = {
            "reference_audio": ref_audio,
            "model": model,
            "language": lang,
            "speed": speed,
            "temperature": temp,
        }

        return {
            "audio": out_path,
            "audio_url": "",
            "duration_sec": round(duration, 2),
            "voice_profile": voice_profile,
        }
