"""
Music Processing Template for audio analysis.

This template uses librosa and essentia to analyze music audio files.
Supports tempo detection, key detection, beat tracking, and comprehensive analysis.
"""

from typing import Any, Dict
from templates.base import Template, InputField, OutputField, RouteType


class MusicProcessingTemplate(Template):
    """
    Processes and analyzes music audio files.
    
    Uses librosa and essentia for music information retrieval tasks including
    tempo detection, key detection, beat tracking, and spectral analysis.
    """
    
    name = "music-processing"
    category = "Audio"
    description = "Analyze and process music audio (tempo, key, beats)"
    version = "1.0.0"
    
    inputs = [
        InputField(
            name="audio",
            type="audio",
            description="Music audio file to process",
            required=True
        ),
        InputField(
            name="analysis_type",
            type="text",
            description="Type of analysis (tempo, key, beats, all)",
            required=False,
            default="all",
            options=["tempo", "key", "beats", "all"]
        )
    ]
    
    outputs = [
        OutputField(
            name="analysis",
            type="json",
            description="Music analysis results"
        ),
        OutputField(
            name="processed_audio",
            type="audio",
            description="Processed audio file (if applicable)"
        )
    ]
    
    routing = [RouteType.LOCAL, RouteType.MODAL]
    gpu_required = False
    memory_mb = 1024
    timeout_sec = 300
    pip_packages = ["librosa", "essentia"]
    
    def setup(self) -> None:
        """
        One-time initialization to load music processing libraries.
        
        Imports librosa and essentia for music analysis. These libraries
        are loaded on-demand to avoid unnecessary imports.
        """
        import librosa
        import essentia.standard as es
        
        self.librosa = librosa
        self.essentia = es
        self._initialized = True
    
    def run(self, **kwargs) -> Dict[str, Any]:
        """
        Execute music processing on the provided audio.
        
        Args:
            audio: Path to audio file or audio data
            analysis_type: Type of analysis to perform (tempo, key, beats, all)
            
        Returns:
            Dict containing:
                - analysis: Dictionary with analysis results
                - processed_audio: Path to processed audio (if applicable)
        """
        # Validate inputs
        self.validate_inputs(**kwargs)
        
        # Initialize if needed
        if not self._initialized:
            self.setup()
        
        # Extract parameters
        audio_path = kwargs['audio']
        analysis_type = kwargs.get('analysis_type', 'all')
        
        # Load audio file
        y, sr = self.librosa.load(audio_path, sr=None)
        
        # Initialize results
        analysis = {}
        
        # Perform requested analysis
        if analysis_type in ['tempo', 'all']:
            tempo, beats = self.librosa.beat.beat_track(y=y, sr=sr)
            analysis['tempo'] = float(tempo)
        
        if analysis_type in ['beats', 'all']:
            tempo, beat_frames = self.librosa.beat.beat_track(y=y, sr=sr)
            beat_times = self.librosa.frames_to_time(beat_frames, sr=sr)
            analysis['beats'] = beat_times.tolist()
            analysis['beat_count'] = len(beat_times)
        
        if analysis_type in ['key', 'all']:
            # Use essentia for key detection
            audio_essentia = self.essentia.MonoLoader(filename=audio_path)()
            key_extractor = self.essentia.KeyExtractor()
            key, scale, strength = key_extractor(audio_essentia)
            analysis['key'] = key
            analysis['scale'] = scale
            analysis['key_strength'] = float(strength)
        
        if analysis_type == 'all':
            # Add spectral features
            spectral_centroids = self.librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            analysis['spectral_centroid_mean'] = float(spectral_centroids.mean())
            analysis['spectral_centroid_std'] = float(spectral_centroids.std())
            
            # Add zero crossing rate
            zcr = self.librosa.feature.zero_crossing_rate(y)[0]
            analysis['zero_crossing_rate_mean'] = float(zcr.mean())
            
            # Add duration
            analysis['duration_seconds'] = float(len(y) / sr)
        
        return {
            'analysis': analysis,
            'processed_audio': audio_path  # Return original for now
        }
