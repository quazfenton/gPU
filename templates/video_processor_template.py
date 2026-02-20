"""
Video Processor — process videos with trimming, effects, and transformations.

Automates the popular Colab workflow:
  1. Upload video
  2. Apply transformations (trim, resize, speed, effects)
  3. Get back processed video
"""

import os
import tempfile
from typing import Any, Dict

from templates.base import Template, InputField, OutputField, RouteType


class VideoProcessorTemplate(Template):
    name = "video-processor"
    category = "Video"
    description = (
        "Process videos with trimming, speed changes, resizing, and effects. "
        "Convert formats, extract audio, create GIFs, and more. "
        "Supports MP4, AVI, MOV, and other common formats."
    )
    version = "1.0.0"

    inputs = [
        InputField(name="video", type="video", description="Video file to process", required=True),
        InputField(
            name="start_time",
            type="number",
            description="Start time in seconds",
            required=False,
            default=0,
        ),
        InputField(
            name="duration",
            type="number",
            description="Duration in seconds (-1 for full video)",
            required=False,
            default=-1,
        ),
        InputField(
            name="speed",
            type="number",
            description="Playback speed multiplier (0.5-4.0)",
            required=False,
            default=1.0,
        ),
        InputField(
            name="width",
            type="number",
            description="Output width (auto if not set)",
            required=False,
            default=0,
        ),
        InputField(
            name="height",
            type="number",
            description="Output height (auto if not set)",
            required=False,
            default=0,
        ),
        InputField(
            name="extract_audio",
            type="text",
            description="Extract audio as separate file",
            required=False,
            default="false",
            options=["true", "false"],
        ),
        InputField(
            name="output_format",
            type="text",
            description="Output video format",
            required=False,
            default="mp4",
            options=["mp4", "avi", "mov", "gif"],
        ),
    ]

    outputs = [
        OutputField(name="output_video", type="video", description="Processed video file"),
        OutputField(name="output_audio", type="audio", description="Extracted audio (if requested)"),
        OutputField(name="duration_sec", type="number", description="Output video duration"),
    ]

    routing = [RouteType.LOCAL, RouteType.MODAL, RouteType.COLAB]
    gpu_required = False
    gpu_type = None
    memory_mb = 4096
    timeout_sec = 600
    pip_packages = ["moviepy", "numpy", "pillow"]

    def setup(self):
        self._initialized = True

    def run(self, **kwargs) -> Dict[str, Any]:
        self.validate_inputs(**kwargs)
        from moviepy.editor import VideoFileClip

        video_path = kwargs["video"]
        start = float(kwargs.get("start_time", 0))
        duration = float(kwargs.get("duration", -1))
        speed = float(kwargs.get("speed", 1.0))
        width = int(kwargs.get("width", 0))
        height = int(kwargs.get("height", 0))
        extract_audio = kwargs.get("extract_audio", "false") == "true"
        output_format = kwargs.get("output_format", "mp4")

        clip = VideoFileClip(video_path)

        if start > 0:
            clip = clip.subclip(start)

        if duration > 0:
            clip = clip.subclip(0, min(duration, clip.duration))

        if speed != 1.0:
            clip = clip.fx(lambda c: c.set_fps(c.fps * speed))
            clip = clip.time_mirror()

        if width > 0 or height > 0:
            w, h = clip.size
            new_w = width if width > 0 else w
            new_h = height if height > 0 else h
            clip = clip.resize((new_w, new_h))

        out_dir = tempfile.mkdtemp(prefix="video_")
        out_path = os.path.join(out_dir, f"output.{output_format}")
        clip.write_videofile(out_path, codec="libx264", audio_codec="aac", verbose=False, logger=None)

        response = {
            "output_video": out_path,
            "duration_sec": clip.duration,
        }

        if extract_audio:
            audio_path = os.path.join(out_dir, "audio.mp3")
            clip.audio.write_audiofile(audio_path, verbose=False, logger=None)
            response["output_audio"] = audio_path

        clip.close()
        return response
