"""
Text to Speech Template for generating speech audio from text.

This template uses the Coqui TTS library to synthesize natural-sounding
speech from text input. Supports multiple languages, voice selection,
and adjustable speech speed.
"""

from typing import Any, Dict
from templates.base import Template, InputField, OutputField, RouteType


class TextToSpeechTemplate(Template):
    """
    Synthesizes speech audio from text input.
    
    Uses the Coqui TTS library with pre-trained models to generate
    natural-sounding speech. Supports language selection, voice
    customization, and speed adjustment.
    """
    
    name = "text-to-speech"
    category = "Audio"
    description = "Synthesize natural-sounding speech audio from text"
    version = "1.0.0"
    
    inputs = [
        InputField(
            name="text",
            type="text",
            description="Text to convert to speech",
            required=True
        ),
        InputField(
            name="language",
            type="text",
            description="Language code (e.g., 'en', 'es', 'fr', 'de')",
            required=False,
            default="en"
        ),
        InputField(
            name="speed",
            type="number",
            description="Speech speed multiplier (0.5 to 2.0)",
            required=False,
            default=1.0
        ),
        InputField(
            name="voice",
            type="text",
            description="Voice ID or name to use",
            required=False,
            default="default"
        )
    ]
    
    outputs = [
        OutputField(
            name="audio",
            type="audio",
            description="Path to generated audio file"
        ),
        OutputField(
            name="duration_seconds",
            type="number",
            description="Duration of the generated audio in seconds"
        )
    ]
    
    routing = [RouteType.LOCAL, RouteType.MODAL]
    gpu_required = True
    gpu_type = "T4"
    memory_mb = 4096
    timeout_sec = 300
    pip_packages = ["TTS", "torch", "numpy"]
    
    def setup(self) -> None:
        """
        One-time initialization to load the TTS model.
        
        Downloads and loads a pre-trained TTS model. The model is cached
        for subsequent uses.
        """
        from TTS.api import TTS
        import torch
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Load a multi-speaker, multi-lingual TTS model
        self.tts = TTS(
            model_name="tts_models/multilingual/multi-dataset/xtts_v2",
            progress_bar=False
        ).to(device)
        
        self._initialized = True
    
    def run(self, **kwargs) -> Dict[str, Any]:
        """
        Execute text-to-speech synthesis on the provided text.
        
        Args:
            text: Text to convert to speech
            language: Language code (optional, defaults to 'en')
            speed: Speech speed multiplier (optional, defaults to 1.0)
            voice: Voice ID or name (optional, defaults to 'default')
            
        Returns:
            Dict containing:
                - audio: Path to generated audio file
                - duration_seconds: Duration of the generated audio
        """
        # Validate inputs
        self.validate_inputs(**kwargs)
        
        import tempfile
        import numpy as np
        
        # Initialize if needed
        if not self._initialized:
            self.setup()
        
        # Extract parameters
        text = kwargs['text']
        language = kwargs.get('language', 'en')
        speed = kwargs.get('speed', 1.0)
        voice = kwargs.get('voice', 'default')
        
        # Create temporary file for output
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            output_path = tmp_file.name
        
        # Generate speech
        self.tts.tts_to_file(
            text=text,
            file_path=output_path,
            language=language,
            speed=speed
        )
        
        # Calculate duration from the generated audio file
        import wave
        with wave.open(output_path, 'r') as wav_file:
            frames = wav_file.getnframes()
            sample_rate = wav_file.getframerate()
            duration_seconds = float(frames) / float(sample_rate)
        
        return {
            'audio': output_path,
            'duration_seconds': duration_seconds
        }
