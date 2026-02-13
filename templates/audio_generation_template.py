"""
Audio Generation Template for text-to-speech synthesis.

This template uses TTS (Text-to-Speech) models to generate speech audio from text.
Supports multiple voices and adjustable speech speed.
"""

from typing import Any, Dict
from templates.base import Template, InputField, OutputField, RouteType


class AudioGenerationTemplate(Template):
    """
    Generates audio from text using text-to-speech models.
    
    Uses TTS models to convert text into natural-sounding speech audio.
    Supports voice selection and speed adjustment for customized output.
    """
    
    name = "audio-generation"
    category = "Audio"
    description = "Generate speech audio from text using TTS models"
    version = "1.0.0"
    
    inputs = [
        InputField(
            name="text",
            type="text",
            description="Text to convert to speech",
            required=True
        ),
        InputField(
            name="voice",
            type="text",
            description="Voice ID or style",
            required=False,
            default="default"
        ),
        InputField(
            name="speed",
            type="number",
            description="Speech speed multiplier (0.5 to 2.0)",
            required=False,
            default=1.0
        )
    ]
    
    outputs = [
        OutputField(
            name="audio",
            type="audio",
            description="Generated audio file"
        )
    ]
    
    routing = [RouteType.LOCAL, RouteType.MODAL]
    gpu_required = True
    gpu_type = "T4"
    memory_mb = 2048
    timeout_sec = 300
    pip_packages = ["TTS", "torch"]
    
    def setup(self) -> None:
        """
        One-time initialization to load the TTS model.
        
        Downloads and loads the TTS model. The model is cached
        for subsequent uses.
        """
        from TTS.api import TTS
        
        # Load default TTS model (Tacotron2 or similar)
        self.tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False)
        self._initialized = True
    
    def run(self, **kwargs) -> Dict[str, Any]:
        """
        Execute audio generation from the provided text.
        
        Args:
            text: Text to convert to speech
            voice: Voice ID or style (optional, defaults to 'default')
            speed: Speech speed multiplier (optional, defaults to 1.0)
            
        Returns:
            Dict containing:
                - audio: Path to generated audio file or audio data
        """
        # Validate inputs
        self.validate_inputs(**kwargs)
        
        # Initialize if needed
        if not self._initialized:
            self.setup()
        
        # Extract parameters
        text = kwargs['text']
        voice = kwargs.get('voice', 'default')
        speed = kwargs.get('speed', 1.0)
        
        # Validate speed parameter
        if not (0.5 <= speed <= 2.0):
            raise ValueError(f"Speed must be between 0.5 and 2.0, got {speed}")
        
        # Generate audio
        import tempfile
        import os
        
        # Create temporary file for output
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            output_path = tmp_file.name
        
        # Generate speech
        self.tts.tts_to_file(
            text=text,
            file_path=output_path,
            speed=speed
        )
        
        return {
            'audio': output_path
        }
