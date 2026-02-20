"""
Vocal Separator — split any song into vocals + instrumentals using Demucs.

Automates the popular Colab workflow:
  1. Upload audio file
  2. Run facebook/demucs (htdemucs model) to separate stems
  3. Download separated vocals.wav and no_vocals.wav as a zip

pip install: demucs torch torchaudio
"""

import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from typing import Any, Dict

from templates.base import Template, InputField, OutputField, RouteType


class VocalSeparatorTemplate(Template):
    name = "vocal-separator"
    category = "Audio"
    description = (
        "Separate vocals from instrumentals in any audio file using "
        "Facebook's Demucs (htdemucs). Upload a song, get back isolated "
        "vocals and accompaniment tracks."
    )
    version = "1.0.0"

    inputs = [
        InputField(
            name="audio_file",
            type="audio",
            description="Audio file path (mp3, wav, flac, ogg)",
            required=True,
        ),
        InputField(
            name="model",
            type="text",
            description="Demucs model variant",
            required=False,
            default="htdemucs",
            options=["htdemucs", "htdemucs_ft", "mdx_extra", "mdx_extra_q"],
        ),
        InputField(
            name="two_stems",
            type="text",
            description="If set, only separate into two stems (vocals or drums)",
            required=False,
            default="vocals",
            options=["vocals", "drums", "bass", "other", ""],
        ),
    ]

    outputs = [
        OutputField(name="output_zip", type="file", description="Zip of separated stems"),
        OutputField(name="stems", type="json", description="List of stem file paths"),
    ]

    routing = [RouteType.LOCAL, RouteType.MODAL, RouteType.COLAB]
    gpu_required = True
    gpu_type = "T4"
    memory_mb = 8192
    timeout_sec = 900
    pip_packages = ["demucs", "torch", "torchaudio"]

    def run(self, **kwargs) -> Dict[str, Any]:
        self.validate_inputs(**kwargs)

        audio_file = kwargs["audio_file"]
        model = kwargs.get("model", "htdemucs")
        two_stems = kwargs.get("two_stems", "vocals")

        work_dir = tempfile.mkdtemp(prefix="demucs_")
        out_dir = os.path.join(work_dir, "separated")
        os.makedirs(out_dir, exist_ok=True)

        # Build demucs command
        cmd = [
            sys.executable, "-m", "demucs",
            "--out", out_dir,
            "-n", model,
        ]
        if two_stems:
            cmd += ["--two-stems", two_stems]
        cmd.append(audio_file)

        subprocess.run(cmd, check=True, capture_output=True, text=True)

        # Collect output stems
        stems_dir = os.path.join(out_dir, model)
        stem_files = []
        for root, _, files in os.walk(stems_dir):
            for f in files:
                if f.endswith((".wav", ".mp3", ".flac")):
                    stem_files.append(os.path.join(root, f))

        # Zip them up
        zip_path = os.path.join(work_dir, "separated_stems.zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for fp in stem_files:
                zf.write(fp, os.path.relpath(fp, stems_dir))

        return {
            "output_zip": zip_path,
            "stems": stem_files,
        }
