"""
Speech Recognition Template for audio transcription.

This template uses OpenAI's Whisper model to transcribe audio files to text.
Supports multiple languages and model sizes for different accuracy/speed tradeoffs.
"""

from typing import Any, Dict
from templates.base import Template, InputField, OutputField, RouteType


class SpeechRecognitionTemplate(Template):
    """
    Transcribes audio to text using speech recognition models.
    
    Uses OpenAI's Whisper model for high-quality speech recognition across
    multiple languages. Supports various model sizes from tiny (fast) to 
    large (most accurate).
    """
    
    name = "speech-recognition"
    category = "Audio"
    description = "Transcribe audio files to text using Whisper or similar models"
    version = "1.0.0"
    
    inputs = [
        InputField(
            name="audio",
            type="audio",
            description="Audio file to transcribe (wav, mp3, flac)",
            required=True
        ),
        InputField(
            name="language",
            type="text",
            description="Language code (e.g., 'en', 'es', 'fr')",
            required=False,
            default="en"
        ),
        InputField(
            name="model_size",
            type="text",
            description="Model size (tiny, base, small, medium, large)",
            required=False,
            default="base",
            options=["tiny", "base", "small", "medium", "large"]
        )
    ]
    
    outputs = [
        OutputField(
            name="text",
            type="text",
            description="Transcribed text"
        ),
        OutputField(
            name="segments",
            type="json",
            description="Timestamped segments with text"
        )
    ]
    
    routing = [RouteType.LOCAL, RouteType.MODAL, RouteType.HF]
    gpu_required = True
    gpu_type = "T4"
    memory_mb = 4096
    timeout_sec = 600
    pip_packages = ["openai-whisper", "torch", "torchaudio"]
    
    def setup(self) -> None:
        """
        One-time initialization to load the Whisper model.
        
        Downloads and loads the specified Whisper model. The model is cached
        for subsequent uses.
        """
        import whisper
        
        # Default to base model if not specified
        model_size = getattr(self, '_model_size', 'base')
        self.model = whisper.load_model(model_size)
        self._initialized = True
    
    def run(self, **kwargs) -> Dict[str, Any]:
        """
        Execute speech recognition on the provided audio.
        
        Args:
            audio: Path to audio file or audio data
            language: Language code (optional, defaults to 'en')
            model_size: Model size to use (optional, defaults to 'base')
            
        Returns:
            Dict containing:
                - text: Full transcribed text
                - segments: List of timestamped segments with text
        """
        # Validate inputs
        self.validate_inputs(**kwargs)
        
        # Initialize if needed
        if not self._initialized:
            # Store model size for setup
            self._model_size = kwargs.get('model_size', 'base')
            self.setup()
        
        # Extract parameters
        audio = kwargs['audio']
        language = kwargs.get('language', 'en')
        
        # Transcribe audio
        result = self.model.transcribe(
            audio,
            language=language if language != 'auto' else None
        )
        
        # Extract text and segments
        text = result['text']
        segments = [
            {
                'start': seg['start'],
                'end': seg['end'],
                'text': seg['text']
            }
            for seg in result.get('segments', [])
        ]
        
        return {
            'text': text,
            'segments': segments
        }
