"""
Unit tests for Audio Processing Templates.

Tests cover:
- SpeechRecognitionTemplate instantiation and metadata
- AudioGenerationTemplate instantiation and metadata
- Input validation
- Setup and initialization
- Run method with valid inputs
- Error handling
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from templates.speech_recognition_template import SpeechRecognitionTemplate
from templates.audio_generation_template import AudioGenerationTemplate
from templates.base import RouteType


class TestSpeechRecognitionTemplate:
    """Test suite for SpeechRecognitionTemplate."""
    
    def test_template_metadata(self):
        """Test that template has correct metadata."""
        template = SpeechRecognitionTemplate()
        
        assert template.name == "speech-recognition"
        assert template.category == "Audio"
        assert template.description == "Transcribe audio files to text using Whisper or similar models"
        assert template.version == "1.0.0"
    
    def test_input_definitions(self):
        """Test that template has correct input definitions."""
        template = SpeechRecognitionTemplate()
        
        assert len(template.inputs) == 3
        
        # Check audio input
        audio_input = next(i for i in template.inputs if i.name == "audio")
        assert audio_input.type == "audio"
        assert audio_input.required is True
        assert audio_input.description == "Audio file to transcribe (wav, mp3, flac)"
        
        # Check language input
        language_input = next(i for i in template.inputs if i.name == "language")
        assert language_input.type == "text"
        assert language_input.required is False
        assert language_input.default == "en"
        
        # Check model_size input
        model_input = next(i for i in template.inputs if i.name == "model_size")
        assert model_input.type == "text"
        assert model_input.required is False
        assert model_input.default == "base"
        assert model_input.options == ["tiny", "base", "small", "medium", "large"]
    
    def test_output_definitions(self):
        """Test that template has correct output definitions."""
        template = SpeechRecognitionTemplate()
        
        assert len(template.outputs) == 2
        
        # Check text output
        text_output = next(o for o in template.outputs if o.name == "text")
        assert text_output.type == "text"
        assert text_output.description == "Transcribed text"
        
        # Check segments output
        segments_output = next(o for o in template.outputs if o.name == "segments")
        assert segments_output.type == "json"
        assert segments_output.description == "Timestamped segments with text"
    
    def test_resource_requirements(self):
        """Test that template has correct resource requirements."""
        template = SpeechRecognitionTemplate()
        
        assert template.gpu_required is True
        assert template.gpu_type == "T4"
        assert template.memory_mb == 4096
        assert template.timeout_sec == 600
    
    def test_routing_support(self):
        """Test that template supports correct backends."""
        template = SpeechRecognitionTemplate()
        
        assert RouteType.LOCAL in template.routing
        assert RouteType.MODAL in template.routing
        assert RouteType.HF in template.routing
    
    def test_pip_packages(self):
        """Test that template declares correct dependencies."""
        template = SpeechRecognitionTemplate()
        
        assert "openai-whisper" in template.pip_packages
        assert "torch" in template.pip_packages
        assert "torchaudio" in template.pip_packages
    
    def test_validate_inputs_valid(self):
        """Test input validation with valid inputs."""
        template = SpeechRecognitionTemplate()
        
        # Should not raise with required audio input
        result = template.validate_inputs(audio="test.wav")
        assert result is True
        
        # Should not raise with all inputs
        result = template.validate_inputs(
            audio="test.wav",
            language="en",
            model_size="base"
        )
        assert result is True
    
    def test_validate_inputs_missing_required(self):
        """Test input validation fails when required input is missing."""
        template = SpeechRecognitionTemplate()
        
        # Should raise ValueError when audio is missing
        with pytest.raises(ValueError, match="Missing required input: audio"):
            template.validate_inputs(language="en")
    
    def test_setup(self):
        """Test setup method loads Whisper model."""
        template = SpeechRecognitionTemplate()
        template._model_size = 'base'
        
        # Mock whisper module at import time
        with patch.dict('sys.modules', {'whisper': MagicMock()}):
            import sys
            mock_whisper = sys.modules['whisper']
            mock_model = Mock()
            mock_whisper.load_model.return_value = mock_model
            
            template.setup()
            
            mock_whisper.load_model.assert_called_once_with('base')
            assert template.model == mock_model
            assert template._initialized is True
    
    def test_run_with_valid_inputs(self):
        """Test run method with valid inputs."""
        template = SpeechRecognitionTemplate()
        
        # Mock the Whisper model at import time
        with patch.dict('sys.modules', {'whisper': MagicMock()}):
            import sys
            mock_whisper = sys.modules['whisper']
            mock_model = Mock()
            mock_model.transcribe.return_value = {
                'text': 'Hello world',
                'segments': [
                    {'start': 0.0, 'end': 1.5, 'text': 'Hello'},
                    {'start': 1.5, 'end': 2.5, 'text': 'world'}
                ]
            }
            mock_whisper.load_model.return_value = mock_model
            
            # Run the template
            result = template.run(audio="test.wav", language="en", model_size="base")
            
            # Verify results
            assert result['text'] == 'Hello world'
            assert len(result['segments']) == 2
            assert result['segments'][0]['text'] == 'Hello'
            assert result['segments'][0]['start'] == 0.0
            assert result['segments'][0]['end'] == 1.5
            assert result['segments'][1]['text'] == 'world'
    
    def test_run_with_auto_language(self):
        """Test run method with auto language detection."""
        template = SpeechRecognitionTemplate()
        
        # Mock the Whisper model at import time
        with patch.dict('sys.modules', {'whisper': MagicMock()}):
            import sys
            mock_whisper = sys.modules['whisper']
            mock_model = Mock()
            mock_model.transcribe.return_value = {
                'text': 'Bonjour',
                'segments': [
                    {'start': 0.0, 'end': 1.0, 'text': 'Bonjour'}
                ]
            }
            mock_whisper.load_model.return_value = mock_model
            
            # Run with auto language
            result = template.run(audio="test.wav", language="auto")
            
            # Verify transcribe was called with language=None for auto-detection
            mock_model.transcribe.assert_called_once_with("test.wav", language=None)
            assert result['text'] == 'Bonjour'
    
    def test_run_initializes_if_needed(self):
        """Test that run calls setup if not initialized."""
        template = SpeechRecognitionTemplate()
        
        # Mock the Whisper model at import time
        with patch.dict('sys.modules', {'whisper': MagicMock()}):
            import sys
            mock_whisper = sys.modules['whisper']
            mock_model = Mock()
            mock_model.transcribe.return_value = {
                'text': 'Test',
                'segments': []
            }
            mock_whisper.load_model.return_value = mock_model
            
            # Ensure not initialized
            assert template._initialized is False
            
            # Run should call setup
            template.run(audio="test.wav")
            
            # Verify setup was called
            mock_whisper.load_model.assert_called_once()
            assert template._initialized is True
    
    def test_run_with_missing_audio(self):
        """Test run method fails with missing audio input."""
        template = SpeechRecognitionTemplate()
        
        # Should raise ValueError for missing required input
        with pytest.raises(ValueError, match="Missing required input: audio"):
            template.run(language="en")
    
    def test_to_dict_serialization(self):
        """Test that template can be serialized to dict."""
        template = SpeechRecognitionTemplate()
        
        metadata = template.to_dict()
        
        assert metadata['name'] == "speech-recognition"
        assert metadata['category'] == "Audio"
        assert metadata['gpu_required'] is True
        assert metadata['gpu_type'] == "T4"
        assert metadata['memory_mb'] == 4096
        assert metadata['timeout_sec'] == 600
        assert len(metadata['inputs']) == 3
        assert len(metadata['outputs']) == 2
        assert 'openai-whisper' in metadata['pip_packages']


class TestAudioGenerationTemplate:
    """Test suite for AudioGenerationTemplate."""
    
    def test_template_metadata(self):
        """Test that template has correct metadata."""
        template = AudioGenerationTemplate()
        
        assert template.name == "audio-generation"
        assert template.category == "Audio"
        assert template.description == "Generate speech audio from text using TTS models"
        assert template.version == "1.0.0"
    
    def test_input_definitions(self):
        """Test that template has correct input definitions."""
        template = AudioGenerationTemplate()
        
        assert len(template.inputs) == 3
        
        # Check text input
        text_input = next(i for i in template.inputs if i.name == "text")
        assert text_input.type == "text"
        assert text_input.required is True
        assert text_input.description == "Text to convert to speech"
        
        # Check voice input
        voice_input = next(i for i in template.inputs if i.name == "voice")
        assert voice_input.type == "text"
        assert voice_input.required is False
        assert voice_input.default == "default"
        
        # Check speed input
        speed_input = next(i for i in template.inputs if i.name == "speed")
        assert speed_input.type == "number"
        assert speed_input.required is False
        assert speed_input.default == 1.0
    
    def test_output_definitions(self):
        """Test that template has correct output definitions."""
        template = AudioGenerationTemplate()
        
        assert len(template.outputs) == 1
        
        # Check audio output
        audio_output = next(o for o in template.outputs if o.name == "audio")
        assert audio_output.type == "audio"
        assert audio_output.description == "Generated audio file"
    
    def test_resource_requirements(self):
        """Test that template has correct resource requirements."""
        template = AudioGenerationTemplate()
        
        assert template.gpu_required is True
        assert template.gpu_type == "T4"
        assert template.memory_mb == 2048
        assert template.timeout_sec == 300
    
    def test_routing_support(self):
        """Test that template supports correct backends."""
        template = AudioGenerationTemplate()
        
        assert RouteType.LOCAL in template.routing
        assert RouteType.MODAL in template.routing
        assert RouteType.HF not in template.routing
    
    def test_pip_packages(self):
        """Test that template declares correct dependencies."""
        template = AudioGenerationTemplate()
        
        assert "TTS" in template.pip_packages
        assert "torch" in template.pip_packages
    
    def test_validate_inputs_valid(self):
        """Test input validation with valid inputs."""
        template = AudioGenerationTemplate()
        
        # Should not raise with required text input
        result = template.validate_inputs(text="Hello world")
        assert result is True
        
        # Should not raise with all inputs
        result = template.validate_inputs(
            text="Hello world",
            voice="default",
            speed=1.0
        )
        assert result is True
    
    def test_validate_inputs_missing_required(self):
        """Test input validation fails when required input is missing."""
        template = AudioGenerationTemplate()
        
        # Should raise ValueError when text is missing
        with pytest.raises(ValueError, match="Missing required input: text"):
            template.validate_inputs(voice="default")
    
    def test_setup(self):
        """Test setup method loads TTS model."""
        template = AudioGenerationTemplate()
        
        # Mock TTS module at import time
        with patch.dict('sys.modules', {'TTS': MagicMock(), 'TTS.api': MagicMock()}):
            import sys
            mock_tts_module = MagicMock()
            mock_tts_class = Mock()
            mock_tts_instance = Mock()
            mock_tts_class.return_value = mock_tts_instance
            mock_tts_module.TTS = mock_tts_class
            sys.modules['TTS.api'] = mock_tts_module
            
            template.setup()
            
            mock_tts_class.assert_called_once_with(
                model_name="tts_models/en/ljspeech/tacotron2-DDC",
                progress_bar=False
            )
            assert template.tts == mock_tts_instance
            assert template._initialized is True
    
    def test_run_with_valid_inputs(self):
        """Test run method with valid inputs."""
        template = AudioGenerationTemplate()
        
        # Mock the TTS module at import time
        with patch.dict('sys.modules', {'TTS': MagicMock(), 'TTS.api': MagicMock()}):
            import sys
            mock_tts_module = MagicMock()
            mock_tts_class = Mock()
            mock_tts_instance = Mock()
            mock_tts_class.return_value = mock_tts_instance
            mock_tts_module.TTS = mock_tts_class
            sys.modules['TTS.api'] = mock_tts_module
            
            # Mock tempfile
            with patch('tempfile.NamedTemporaryFile') as mock_temp:
                mock_file = Mock()
                mock_file.name = '/tmp/test_audio.wav'
                mock_temp.return_value.__enter__.return_value = mock_file
                
                # Run the template
                result = template.run(text="Hello world", voice="default", speed=1.0)
                
                # Verify results
                assert 'audio' in result
                assert result['audio'] == '/tmp/test_audio.wav'
                
                # Verify TTS was called correctly
                mock_tts_instance.tts_to_file.assert_called_once_with(
                    text="Hello world",
                    file_path='/tmp/test_audio.wav',
                    speed=1.0
                )
    
    def test_run_with_custom_speed(self):
        """Test run method with custom speed parameter."""
        template = AudioGenerationTemplate()
        
        # Mock the TTS module at import time
        with patch.dict('sys.modules', {'TTS': MagicMock(), 'TTS.api': MagicMock()}):
            import sys
            mock_tts_module = MagicMock()
            mock_tts_class = Mock()
            mock_tts_instance = Mock()
            mock_tts_class.return_value = mock_tts_instance
            mock_tts_module.TTS = mock_tts_class
            sys.modules['TTS.api'] = mock_tts_module
            
            # Mock tempfile
            with patch('tempfile.NamedTemporaryFile') as mock_temp:
                mock_file = Mock()
                mock_file.name = '/tmp/test_audio.wav'
                mock_temp.return_value.__enter__.return_value = mock_file
                
                # Run with custom speed
                result = template.run(text="Fast speech", speed=1.5)
                
                # Verify speed was passed correctly
                mock_tts_instance.tts_to_file.assert_called_once()
                call_args = mock_tts_instance.tts_to_file.call_args
                assert call_args[1]['speed'] == 1.5
    
    def test_run_with_invalid_speed(self):
        """Test run method fails with invalid speed parameter."""
        template = AudioGenerationTemplate()
        
        # Mock the TTS module at import time
        with patch.dict('sys.modules', {'TTS': MagicMock(), 'TTS.api': MagicMock()}):
            import sys
            mock_tts_module = MagicMock()
            mock_tts_class = Mock()
            mock_tts_instance = Mock()
            mock_tts_class.return_value = mock_tts_instance
            mock_tts_module.TTS = mock_tts_class
            sys.modules['TTS.api'] = mock_tts_module
            
            # Should raise ValueError for speed < 0.5
            with pytest.raises(ValueError, match="Speed must be between 0.5 and 2.0"):
                template.run(text="Test", speed=0.3)
            
            # Should raise ValueError for speed > 2.0
            with pytest.raises(ValueError, match="Speed must be between 0.5 and 2.0"):
                template.run(text="Test", speed=2.5)
    
    def test_run_initializes_if_needed(self):
        """Test that run calls setup if not initialized."""
        template = AudioGenerationTemplate()
        
        # Mock the TTS module at import time
        with patch.dict('sys.modules', {'TTS': MagicMock(), 'TTS.api': MagicMock()}):
            import sys
            mock_tts_module = MagicMock()
            mock_tts_class = Mock()
            mock_tts_instance = Mock()
            mock_tts_class.return_value = mock_tts_instance
            mock_tts_module.TTS = mock_tts_class
            sys.modules['TTS.api'] = mock_tts_module
            
            # Mock tempfile
            with patch('tempfile.NamedTemporaryFile') as mock_temp:
                mock_file = Mock()
                mock_file.name = '/tmp/test_audio.wav'
                mock_temp.return_value.__enter__.return_value = mock_file
                
                # Ensure not initialized
                assert template._initialized is False
                
                # Run should call setup
                template.run(text="Test")
                
                # Verify setup was called
                mock_tts_class.assert_called_once()
                assert template._initialized is True
    
    def test_run_with_missing_text(self):
        """Test run method fails with missing text input."""
        template = AudioGenerationTemplate()
        
        # Should raise ValueError for missing required input
        with pytest.raises(ValueError, match="Missing required input: text"):
            template.run(voice="default")
    
    def test_to_dict_serialization(self):
        """Test that template can be serialized to dict."""
        template = AudioGenerationTemplate()
        
        metadata = template.to_dict()
        
        assert metadata['name'] == "audio-generation"
        assert metadata['category'] == "Audio"
        assert metadata['gpu_required'] is True
        assert metadata['gpu_type'] == "T4"
        assert metadata['memory_mb'] == 2048
        assert metadata['timeout_sec'] == 300
        assert len(metadata['inputs']) == 3
        assert len(metadata['outputs']) == 1
        assert 'TTS' in metadata['pip_packages']



class TestMusicProcessingTemplate:
    """Test suite for MusicProcessingTemplate."""
    
    def test_template_metadata(self):
        """Test that template has correct metadata."""
        from templates.music_processing_template import MusicProcessingTemplate
        template = MusicProcessingTemplate()
        
        assert template.name == "music-processing"
        assert template.category == "Audio"
        assert template.description == "Analyze and process music audio (tempo, key, beats)"
        assert template.version == "1.0.0"
    
    def test_input_definitions(self):
        """Test that template has correct input definitions."""
        from templates.music_processing_template import MusicProcessingTemplate
        template = MusicProcessingTemplate()
        
        assert len(template.inputs) == 2
        
        # Check audio input
        audio_input = next(i for i in template.inputs if i.name == "audio")
        assert audio_input.type == "audio"
        assert audio_input.required is True
        assert audio_input.description == "Music audio file to process"
        
        # Check analysis_type input
        analysis_input = next(i for i in template.inputs if i.name == "analysis_type")
        assert analysis_input.type == "text"
        assert analysis_input.required is False
        assert analysis_input.default == "all"
        assert analysis_input.options == ["tempo", "key", "beats", "all"]
    
    def test_output_definitions(self):
        """Test that template has correct output definitions."""
        from templates.music_processing_template import MusicProcessingTemplate
        template = MusicProcessingTemplate()
        
        assert len(template.outputs) == 2
        
        # Check analysis output
        analysis_output = next(o for o in template.outputs if o.name == "analysis")
        assert analysis_output.type == "json"
        assert analysis_output.description == "Music analysis results"
        
        # Check processed_audio output
        audio_output = next(o for o in template.outputs if o.name == "processed_audio")
        assert audio_output.type == "audio"
        assert audio_output.description == "Processed audio file (if applicable)"
    
    def test_resource_requirements(self):
        """Test that template has correct resource requirements."""
        from templates.music_processing_template import MusicProcessingTemplate
        template = MusicProcessingTemplate()
        
        assert template.gpu_required is False
        assert template.gpu_type is None
        assert template.memory_mb == 1024
        assert template.timeout_sec == 300
    
    def test_routing_support(self):
        """Test that template supports correct backends."""
        from templates.music_processing_template import MusicProcessingTemplate
        template = MusicProcessingTemplate()
        
        assert RouteType.LOCAL in template.routing
        assert RouteType.MODAL in template.routing
        assert RouteType.HF not in template.routing
    
    def test_pip_packages(self):
        """Test that template declares correct dependencies."""
        from templates.music_processing_template import MusicProcessingTemplate
        template = MusicProcessingTemplate()
        
        assert "librosa" in template.pip_packages
        assert "essentia" in template.pip_packages
    
    def test_validate_inputs_valid(self):
        """Test input validation with valid inputs."""
        from templates.music_processing_template import MusicProcessingTemplate
        template = MusicProcessingTemplate()
        
        # Should not raise with required audio input
        result = template.validate_inputs(audio="test.wav")
        assert result is True
        
        # Should not raise with all inputs
        result = template.validate_inputs(
            audio="test.wav",
            analysis_type="tempo"
        )
        assert result is True
    
    def test_validate_inputs_missing_required(self):
        """Test input validation fails when required input is missing."""
        from templates.music_processing_template import MusicProcessingTemplate
        template = MusicProcessingTemplate()
        
        # Should raise ValueError when audio is missing
        with pytest.raises(ValueError, match="Missing required input: audio"):
            template.validate_inputs(analysis_type="tempo")
    
    def test_setup(self):
        """Test setup method loads librosa and essentia."""
        from templates.music_processing_template import MusicProcessingTemplate
        template = MusicProcessingTemplate()
        
        # Mock librosa and essentia modules at import time
        mock_essentia = MagicMock()
        mock_essentia_std = MagicMock()
        mock_essentia.standard = mock_essentia_std
        
        with patch.dict('sys.modules', {
            'librosa': MagicMock(),
            'librosa.beat': MagicMock(),
            'librosa.feature': MagicMock(),
            'essentia': mock_essentia,
            'essentia.standard': mock_essentia_std
        }):
            import sys
            mock_librosa = sys.modules['librosa']
            
            template.setup()
            
            assert template.librosa == mock_librosa
            assert template.essentia == mock_essentia_std
            assert template._initialized is True
    
    def test_run_with_tempo_analysis(self):
        """Test run method with tempo analysis."""
        from templates.music_processing_template import MusicProcessingTemplate
        template = MusicProcessingTemplate()
        
        # Mock the librosa and essentia modules
        with patch.dict('sys.modules', {
            'librosa': MagicMock(),
            'librosa.beat': MagicMock(),
            'librosa.feature': MagicMock(),
            'essentia': MagicMock(),
            'essentia.standard': MagicMock()
        }):
            import sys
            mock_librosa = sys.modules['librosa']
            mock_librosa.load.return_value = (Mock(), 22050)  # y, sr
            mock_librosa.beat.beat_track.return_value = (120.0, [0, 10, 20])
            
            # Run the template
            result = template.run(audio="test.wav", analysis_type="tempo")
            
            # Verify results
            assert 'analysis' in result
            assert 'tempo' in result['analysis']
            assert result['analysis']['tempo'] == 120.0
            assert 'processed_audio' in result
    
    def test_run_with_beats_analysis(self):
        """Test run method with beats analysis."""
        from templates.music_processing_template import MusicProcessingTemplate
        template = MusicProcessingTemplate()
        
        # Mock the librosa and essentia modules
        with patch.dict('sys.modules', {
            'librosa': MagicMock(),
            'librosa.beat': MagicMock(),
            'librosa.feature': MagicMock(),
            'essentia': MagicMock(),
            'essentia.standard': MagicMock(),
            'numpy': MagicMock()
        }):
            import sys
            mock_librosa = sys.modules['librosa']
            
            # Create mock arrays
            mock_beat_frames = Mock()
            mock_beat_frames.tolist = Mock(return_value=[0, 10, 20])
            mock_beat_times = Mock()
            mock_beat_times.tolist = Mock(return_value=[0.0, 0.5, 1.0])
            mock_beat_times.__len__ = Mock(return_value=3)
            
            mock_librosa.load.return_value = (Mock(), 22050)
            mock_librosa.beat.beat_track.return_value = (120.0, mock_beat_frames)
            mock_librosa.frames_to_time.return_value = mock_beat_times
            
            # Run the template
            result = template.run(audio="test.wav", analysis_type="beats")
            
            # Verify results
            assert 'analysis' in result
            assert 'beats' in result['analysis']
            assert 'beat_count' in result['analysis']
            assert result['analysis']['beat_count'] == 3
    
    def test_run_with_key_analysis(self):
        """Test run method with key analysis."""
        from templates.music_processing_template import MusicProcessingTemplate
        template = MusicProcessingTemplate()
        
        # Mock the librosa and essentia modules
        mock_essentia = MagicMock()
        mock_essentia_std = MagicMock()
        mock_essentia.standard = mock_essentia_std
        
        with patch.dict('sys.modules', {
            'librosa': MagicMock(),
            'librosa.beat': MagicMock(),
            'librosa.feature': MagicMock(),
            'essentia': mock_essentia,
            'essentia.standard': mock_essentia_std,
            'numpy': MagicMock()
        }):
            import sys
            mock_librosa = sys.modules['librosa']
            
            # Create mock audio data
            mock_audio = Mock()
            mock_librosa.load.return_value = (mock_audio, 22050)
            
            # Mock essentia components
            mock_loader = Mock()
            mock_loader.return_value = mock_audio
            mock_essentia_std.MonoLoader.return_value = mock_loader
            
            mock_key_extractor = Mock()
            mock_key_extractor.return_value = ("C", "major", 0.85)
            mock_essentia_std.KeyExtractor.return_value = mock_key_extractor
            
            # Run the template
            result = template.run(audio="test.wav", analysis_type="key")
            
            # Verify results
            assert 'analysis' in result
            assert 'key' in result['analysis']
            assert result['analysis']['key'] == "C"
            assert result['analysis']['scale'] == "major"
            assert result['analysis']['key_strength'] == 0.85
    
    def test_run_with_all_analysis(self):
        """Test run method with comprehensive analysis."""
        from templates.music_processing_template import MusicProcessingTemplate
        template = MusicProcessingTemplate()
        
        # Mock the librosa and essentia modules
        mock_essentia = MagicMock()
        mock_essentia_std = MagicMock()
        mock_essentia.standard = mock_essentia_std
        
        with patch.dict('sys.modules', {
            'librosa': MagicMock(),
            'librosa.beat': MagicMock(),
            'librosa.feature': MagicMock(),
            'essentia': mock_essentia,
            'essentia.standard': mock_essentia_std,
            'numpy': MagicMock()
        }):
            import sys
            mock_librosa = sys.modules['librosa']
            
            # Mock audio data
            mock_audio = Mock()
            mock_audio.__len__ = Mock(return_value=3000)
            mock_librosa.load.return_value = (mock_audio, 22050)
            
            # Mock beat tracking
            mock_beat_frames = Mock()
            mock_beat_frames.tolist = Mock(return_value=[0, 10, 20])
            mock_beat_times = Mock()
            mock_beat_times.tolist = Mock(return_value=[0.0, 0.5, 1.0])
            mock_beat_times.__len__ = Mock(return_value=3)
            mock_librosa.beat.beat_track.return_value = (120.0, mock_beat_frames)
            mock_librosa.frames_to_time.return_value = mock_beat_times
            
            # Mock spectral features
            mock_spectral = Mock()
            mock_spectral.mean = Mock(return_value=1100.0)
            mock_spectral.std = Mock(return_value=50.0)
            mock_librosa.feature.spectral_centroid.return_value = [mock_spectral]
            
            mock_zcr = Mock()
            mock_zcr.mean = Mock(return_value=0.15)
            mock_librosa.feature.zero_crossing_rate.return_value = [mock_zcr]
            
            # Mock essentia components
            mock_loader = Mock()
            mock_loader.return_value = mock_audio
            mock_essentia_std.MonoLoader.return_value = mock_loader
            
            mock_key_extractor = Mock()
            mock_key_extractor.return_value = ("C", "major", 0.85)
            mock_essentia_std.KeyExtractor.return_value = mock_key_extractor
            
            # Run the template
            result = template.run(audio="test.wav", analysis_type="all")
            
            # Verify comprehensive results
            assert 'analysis' in result
            assert 'tempo' in result['analysis']
            assert 'beats' in result['analysis']
            assert 'beat_count' in result['analysis']
            assert 'key' in result['analysis']
            assert 'scale' in result['analysis']
            assert 'key_strength' in result['analysis']
            assert 'spectral_centroid_mean' in result['analysis']
            assert 'spectral_centroid_std' in result['analysis']
            assert 'zero_crossing_rate_mean' in result['analysis']
            assert 'duration_seconds' in result['analysis']
    
    def test_run_initializes_if_needed(self):
        """Test that run calls setup if not initialized."""
        from templates.music_processing_template import MusicProcessingTemplate
        template = MusicProcessingTemplate()
        
        # Mock the librosa and essentia modules
        with patch.dict('sys.modules', {
            'librosa': MagicMock(),
            'librosa.beat': MagicMock(),
            'librosa.feature': MagicMock(),
            'essentia': MagicMock(),
            'essentia.standard': MagicMock(),
            'numpy': MagicMock()
        }):
            import sys
            mock_librosa = sys.modules['librosa']
            mock_audio = Mock()
            mock_librosa.load.return_value = (mock_audio, 22050)
            mock_librosa.beat.beat_track.return_value = (120.0, Mock())
            
            # Ensure not initialized
            assert template._initialized is False
            
            # Run should call setup
            template.run(audio="test.wav", analysis_type="tempo")
            
            # Verify setup was called
            assert template._initialized is True
    
    def test_run_with_missing_audio(self):
        """Test run method fails with missing audio input."""
        from templates.music_processing_template import MusicProcessingTemplate
        template = MusicProcessingTemplate()
        
        # Should raise ValueError for missing required input
        with pytest.raises(ValueError, match="Missing required input: audio"):
            template.run(analysis_type="tempo")
    
    def test_to_dict_serialization(self):
        """Test that template can be serialized to dict."""
        from templates.music_processing_template import MusicProcessingTemplate
        template = MusicProcessingTemplate()
        
        metadata = template.to_dict()
        
        assert metadata['name'] == "music-processing"
        assert metadata['category'] == "Audio"
        assert metadata['gpu_required'] is False
        assert metadata['gpu_type'] is None
        assert metadata['memory_mb'] == 1024
        assert metadata['timeout_sec'] == 300
        assert len(metadata['inputs']) == 2
        assert len(metadata['outputs']) == 2
        assert 'librosa' in metadata['pip_packages']
        assert 'essentia' in metadata['pip_packages']
