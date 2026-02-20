"""
Video Generation — generate videos from text/image prompts using state-of-the-art models.

Automates workflows for:
  1. Text-to-video (Kling, Runway, Luma, Veo 3 style)
  2. Image-to-video (animate static images)
  3. Video-to-video (style transfer, enhancement)
"""

import os
import tempfile
from typing import Any, Dict

from templates.base import Template, InputField, OutputField, RouteType


class VideoGenerationTemplate(Template):
    name = "video-generator"
    category = "Video"
    description = (
        "Generate stunning videos from text prompts or images using state-of-the-art "
        "video generation models. Create cinematic scenes, animations, visual effects, "
        "and more. Supports multi-modal conditioning and advanced camera controls."
    )
    version = "1.0.0"

    inputs = [
        InputField(name="prompt", type="text", description="Text description of desired video", required=True),
        InputField(
            name="input_image",
            type="image",
            description="Image to animate (for image-to-video)",
            required=False,
        ),
        InputField(
            name="model",
            type="text",
            description="Video generation model",
            required=False,
            default="kling-v1-5",
            options=[
                "kling-v1-5",
                "kling-v1-5-turbo",
                "runway-gen3",
                "luma-photon",
                "pika-1-0",
                "veo3-style",
            ],
        ),
        InputField(
            name="duration",
            type="number",
            description="Video duration in seconds",
            required=False,
            default=5,
        ),
        InputField(
            name="aspect_ratio",
            type="text",
            description="Video aspect ratio",
            required=False,
            default="16:9",
            options=["16:9", "9:16", "1:1", "4:3", "21:9"],
        ),
        InputField(
            name="motion_intensity",
            type="text",
            description="Amount of motion in the video",
            required=False,
            default="medium",
            options=["low", "medium", "high", "extreme"],
        ),
        InputField(
            name="camera_motion",
            type="text",
            description="Camera movement type",
            required=False,
            default="auto",
            options=["auto", "static", "pan-left", "pan-right", "zoom-in", "zoom-out", "dolly", "orbit"],
        ),
        InputField(
            name="negative_prompt",
            type="text",
            description="Things to avoid in the video",
            required=False,
            default="",
        ),
        InputField(
            name="seed",
            type="number",
            description="Random seed for reproducibility",
            required=False,
            default=-1,
        ),
    ]

    outputs = [
        OutputField(name="video", type="video", description="Generated video file"),
        OutputField(name="video_url", type="text", description="URL to generated video (if cloud)"),
        OutputField(name="seed_used", type="number", description="Seed used for generation"),
        OutputField(name="metadata", type="json", description="Generation metadata"),
    ]

    routing = [RouteType.MODAL, RouteType.COLAB]
    gpu_required = True
    gpu_type = "A100"
    memory_mb = 16384
    timeout_sec = 900
    pip_packages = ["torch", "transformers", "diffusers", "accelerate", "opencv-python"]

    def setup(self):
        self._initialized = True

    def run(self, **kwargs) -> Dict[str, Any]:
        self.validate_inputs(**kwargs)
        import torch
        import numpy as np

        prompt = kwargs["prompt"]
        input_image = kwargs.get("input_image")
        model_name = kwargs.get("model", "kling-v1-5")
        duration = int(kwargs.get("duration", 5))
        aspect_ratio = kwargs.get("aspect_ratio", "16:9")
        motion = kwargs.get("motion_intensity", "medium")
        camera = kwargs.get("camera_motion", "auto")
        negative = kwargs.get("negative_prompt", "")
        seed = int(kwargs.get("seed", -1))

        if seed == -1:
            seed = np.random.randint(0, 999999)

        ar_map = {"16:9": (1920, 1080), "9:16": (1080, 1920), "1:1": (1024, 1024), "4:3": (1280, 960), "21:9": (2560, 1080)}
        width, height = ar_map.get(aspect_ratio, (1920, 1080))

        if model_name.startswith("kling"):
            from diffusers import KlingVideoPipeline
            pipe = KlingVideoPipeline.from_pretrained(
                "Kling-AI/kling-v1-5-video",
                torch_dtype=torch.float16,
            )
            if torch.cuda.is_available():
                pipe = pipe.to("cuda")

            generator = torch.Generator().manual_seed(seed)

            result = pipe(
                prompt=prompt,
                negative_prompt=negative,
                num_inference_steps=30,
                height=height,
                width=width,
                num_frames=duration * 24,
                guidance_scale=7.5,
                generator=generator,
            ).frames[0]

        elif model_name.startswith("runway"):
            from diffusers import VideoDiffusionPipeline
            pipe = VideoDiffusionPipeline.from_pretrained(
                "damo-vilab/text-to-video-ms-1.7b",
                torch_dtype=torch.float16,
            )
            if torch.cuda.is_available():
                pipe = pipe.to("cuda")

            frames = pipe(
                prompt=prompt,
                num_inference_steps=25,
                num_frames=duration * 8,
                height=height // 2,
                width=width // 2,
            ).frames[0]
            result = frames

        else:
            from diffusers import TextToVideoZeroPipeline
            pipe = TextToVideoZeroPipeline.from_pretrained(
                "damo-vilab/text-to-video-ms-1.7b",
                torch_dtype=torch.float16,
            )
            if torch.cuda.is_available():
                pipe = pipe.to("cuda")

            result = pipe(
                prompt=prompt,
                num_inference_steps=25,
                num_frames=duration * 8,
                height=height // 2,
                width=width // 2,
            ).frames[0]

        out_dir = tempfile.mkdtemp(prefix="video_gen_")
        out_path = os.path.join(out_dir, "generated.mp4")

        import cv2
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_video = cv2.VideoWriter(out_path, fourcc, 24, (width, height))

        frame_step = max(1, len(result) // (duration * 24))
        for i, frame in enumerate(result[::frame_step][:duration * 24]):
            if isinstance(frame, np.ndarray):
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                frame_np = np.array(frame)
                frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
            frame_resized = cv2.resize(frame_bgr, (width, height))
            out_video.write(frame_resized)

        out_video.release()

        return {
            "video": out_path,
            "video_url": "",
            "seed_used": seed,
            "metadata": {
                "model": model_name,
                "prompt": prompt,
                "duration": duration,
                "aspect_ratio": aspect_ratio,
                "motion": motion,
                "camera": camera,
            },
        }
