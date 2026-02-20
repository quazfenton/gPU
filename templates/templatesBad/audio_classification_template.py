"""
Audio Classification Template for classifying audio content.

This template uses HuggingFace's transformers audio classification pipeline
to classify audio into categories such as genre, emotion, or environmental
sounds with confidence scores.
"""

from typing import Any, Dict
from templates.base import Template, InputField, OutputField, RouteType


class AudioClassificationTemplate(Template):
    """
    Classifies audio content into predefined categories.
    
    Uses pre-trained transformer models for audio classification tasks such as
    music genre detection, emotion recognition, environmental sound classification,
    and speaker identification. Returns top-k predictions with confidence scores.
    """
    
    name = "audio-classification"
    category = "Audio"
    description = "Classify audio content (genre, emotion, environmental sounds)"
    version = "1.0.0"
    
    inputs = [
        InputField(
            name="audio",
            type="audio",
            description="Audio file to classify (wav, mp3, flac)",
            required=True
        ),
        InputField(
            name="top_k",
            type="number",
            description="Number of top predictions to return",
            required=False,
            default=5
        )
    ]
    
    outputs = [
        OutputField(
            name="classifications",
            type="json",
            description="List of classification results with label and score"
        ),
        OutputField(
            name="top_label",
            type="text",
            description="Top predicted label"
        ),
        OutputField(
            name="top_score",
            type="number",
            description="Confidence score of the top prediction"
        )
    ]
    
    routing = [RouteType.LOCAL, RouteType.MODAL, RouteType.HF]
    gpu_required = True
    gpu_type = "T4"
    memory_mb = 4096
    timeout_sec = 120
    pip_packages = ["transformers", "torch", "torchaudio", "librosa"]
    
    def setup(self) -> None:
        """
        One-time initialization to load the audio classification model.
        
        Loads a pre-trained audio classification pipeline from HuggingFace.
        Default model is MIT/ast-finetuned-audioset-10-10-0.4593 (Audio
        Spectrogram Transformer).
        """
        from transformers import pipeline
        
        self.classifier = pipeline(
            "audio-classification",
            model="MIT/ast-finetuned-audioset-10-10-0.4593"
        )
        
        self._initialized = True
    
    def run(self, **kwargs) -> Dict[str, Any]:
        """
        Execute audio classification on the provided audio.
        
        Args:
            audio: Path to audio file or audio data
            top_k: Number of top predictions to return (optional, defaults to 5)
            
        Returns:
            Dict containing:
                - classifications: List of {label, score} dicts
                - top_label: The highest confidence label
                - top_score: The highest confidence score
        """
        # Validate inputs
        self.validate_inputs(**kwargs)
        
        # Initialize if needed
        if not self._initialized:
            self.setup()
        
        import librosa
        
        # Extract parameters
        audio = kwargs['audio']
        top_k = int(kwargs.get('top_k', 5))
        
        # Load audio file
        if isinstance(audio, str):
            waveform, sample_rate = librosa.load(audio, sr=16000)
        else:
            waveform = audio
            sample_rate = 16000
        
        # Run classification
        results = self.classifier(
            waveform,
            top_k=top_k
        )
        
        # Process results
        classifications = [
            {
                'label': result['label'],
                'score': float(result['score'])
            }
            for result in results
        ]
        
        # Extract top prediction
        top_label = classifications[0]['label'] if classifications else ""
        top_score = float(classifications[0]['score']) if classifications else 0.0
        
        return {
            'classifications': classifications,
            'top_label': top_label,
            'top_score': top_score
        }
