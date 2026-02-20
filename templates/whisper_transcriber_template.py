"""
Whisper Transcriber — transcribe or translate any audio/video to text + SRT subtitles.

Automates the popular Colab workflow:
  1. Upload audio/video
  2. Run OpenAI Whisper (any size) with language detection
  3. Get back plain text + timestamped SRT subtitle file
"""

import json
import os
import tempfile
from typing import Any, Dict

from templates.base import Template, InputField, OutputField, RouteType


class WhisperTranscriberTemplate(Template):
    name = "whisper-transcriber"
    category = "Audio"
    description = (
        "Transcribe any audio or video file to text with timestamps "
        "using OpenAI Whisper. Outputs plain text and SRT subtitles. "
        "Supports 99 languages with automatic language detection."
    )
    version = "1.0.0"

    inputs = [
        InputField(name="audio_file", type="audio", description="Audio/video file to transcribe", required=True),
        InputField(
            name="model_size",
            type="text",
            description="Whisper model size (larger = more accurate but slower)",
            required=False,
            default="base",
            options=["tiny", "base", "small", "medium", "large-v3"],
        ),
        InputField(
            name="language",
            type="text",
            description="Language code (e.g. 'en', 'es', 'ja') or 'auto' to detect",
            required=False,
            default="auto",
        ),
        InputField(
            name="task",
            type="text",
            description="'transcribe' keeps original language, 'translate' translates to English",
            required=False,
            default="transcribe",
            options=["transcribe", "translate"],
        ),
    ]

    outputs = [
        OutputField(name="text", type="text", description="Full transcription text"),
        OutputField(name="srt_file", type="file", description="SRT subtitle file path"),
        OutputField(name="segments", type="json", description="Timestamped segments"),
        OutputField(name="detected_language", type="text", description="Detected language code"),
    ]

    routing = [RouteType.LOCAL, RouteType.MODAL, RouteType.COLAB]
    gpu_required = True
    gpu_type = "T4"
    memory_mb = 8192
    timeout_sec = 900
    pip_packages = ["openai-whisper", "torch", "torchaudio"]

    def setup(self):
        import whisper
        self._models = {}
        self._initialized = True

    def _get_model(self, size: str):
        import whisper
        if size not in self._models:
            self._models[size] = whisper.load_model(size)
        return self._models[size]

    def run(self, **kwargs) -> Dict[str, Any]:
        self.validate_inputs(**kwargs)
        if not self._initialized:
            self.setup()

        audio_file = kwargs["audio_file"]
        model_size = kwargs.get("model_size", "base")
        language = kwargs.get("language", "auto")
        task = kwargs.get("task", "transcribe")

        model = self._get_model(model_size)

        opts = {"task": task}
        if language and language != "auto":
            opts["language"] = language

        result = model.transcribe(audio_file, **opts)

        # Build SRT file
        srt_lines = []
        for i, seg in enumerate(result.get("segments", []), 1):
            start = self._format_ts(seg["start"])
            end = self._format_ts(seg["end"])
            srt_lines.append(f"{i}\n{start} --> {end}\n{seg['text'].strip()}\n")

        srt_content = "\n".join(srt_lines)
        out_dir = tempfile.mkdtemp(prefix="whisper_")
        srt_path = os.path.join(out_dir, os.path.splitext(os.path.basename(audio_file))[0] + ".srt")
        with open(srt_path, "w", encoding="utf-8") as f:
            f.write(srt_content)

        segments = [
            {"start": s["start"], "end": s["end"], "text": s["text"].strip()}
            for s in result.get("segments", [])
        ]

        return {
            "text": result["text"],
            "srt_file": srt_path,
            "segments": segments,
            "detected_language": result.get("language", "unknown"),
        }

    @staticmethod
    def _format_ts(seconds: float) -> str:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        ms = int((seconds - int(seconds)) * 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
