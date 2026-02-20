"""
Text to Speech — convert text to natural-sounding speech using Coqui TTS.

Automates the popular Colab workflow:
  1. Enter text or upload document
  2. Select voice/model
  3. Get back generated audio file
"""

import os
import tempfile
from typing import Any, Dict

from templates.base import Template, InputField, OutputField, RouteType


class TextToSpeechTemplate(Template):
    name = "text-to-speech"
    category = "Audio"
    description = (
        "Convert text to natural-sounding speech using Coqui TTS. "
        "Choose from multiple languages and voice models. Perfect for "
        "audiobooks, voiceovers, accessibility content, and more."
    )
    version = "1.0.0"

    inputs = [
        InputField(name="text", type="text", description="Text to convert to speech", required=True),
        InputField(
            name="model",
            type="text",
            description="TTS model",
            required=False,
            default="tts_models/multilingual/multi-dataset/xtts_v2",
        ),
        InputField(
            name="language",
            type="text",
            description="Language code",
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
            name="pitch",
            type="number",
            description="Voice pitch adjustment (-12 to 12)",
            required=False,
            default=0,
        ),
    ]

    outputs = [
        OutputField(name="audio", type="audio", description="Generated speech audio"),
        OutputField(name="duration_sec", type="number", description="Audio duration in seconds"),
    ]

    routing = [RouteType.LOCAL, RouteType.MODAL, RouteType.COLAB]
    gpu_required = True
    gpu_type = "T4"
    memory_mb = 4096
    timeout_sec = 300
    pip_packages = ["TTS", "torch", "torchaudio"]

    def setup(self):
        self._model = None
        self._initialized = True

    def run(self, **kwargs) -> Dict[str, Any]:
        self.validate_inputs(**kwargs)
        from TTS.api import TTS

        text = kwargs["text"]
        model_name = kwargs.get("model", "tts_models/multilingual/multi-dataset/xtts_v2")
        speed = float(kwargs.get("speed", 1.0))
        pitch = float(kwargs.get("pitch", 0))

        if self._model is None:
            self._model = TTS(model_name)

        out_path = os.path.join(tempfile.mkdtemp(prefix="tts_"), "output.wav")

        self._model.tts_to_file(
            text=text,
            file_path=out_path,
            speed=speed,
            pitch=pitch if pitch != 0 else None,
        )

        from pathlib import Path
        duration = Path(out_path).stat().st_size / 20000

        return {
            "audio": out_path,
            "duration_sec": round(duration, 2),
        }
