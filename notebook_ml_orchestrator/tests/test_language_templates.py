"""
Unit tests for Language Processing Templates.

Tests cover:
- NERTemplate instantiation and metadata
- SentimentAnalysisTemplate instantiation and metadata
- TranslationTemplate instantiation and metadata
- SummarizationTemplate instantiation and metadata
- Input validation
- Setup and initialization
- Run method with valid inputs
- Error handling
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from templates.ner_template import NERTemplate
from templates.sentiment_analysis_template import SentimentAnalysisTemplate
from templates.translation_template import TranslationTemplate
from templates.summarization_template import SummarizationTemplate
from templates.base import RouteType


class TestNERTemplate:
    """Test suite for NERTemplate."""
    
    def test_template_metadata(self):
        """Test that template has correct metadata."""
        template = NERTemplate()
        
        assert template.name == "named-entity-recognition"
        assert template.category == "Language"
        assert template.description == "Extract named entities (people, places, organizations) from text"
        assert template.version == "1.0.0"
    
    def test_input_definitions(self):
        """Test that template has correct input definitions."""
        template = NERTemplate()
        
        assert len(template.inputs) == 2
        
        # Check text input
        text_input = next(i for i in template.inputs if i.name == "text")
        assert text_input.type == "text"
        assert text_input.required is True
        assert text_input.description == "Text to analyze"
        
        # Check model input
        model_input = next(i for i in template.inputs if i.name == "model")
        assert model_input.type == "text"
        assert model_input.required is False
        assert model_input.default == "en_core_web_sm"
    
    def test_output_definitions(self):
        """Test that template has correct output definitions."""
        template = NERTemplate()
        
        assert len(template.outputs) == 1
        
        # Check entities output
        entities_output = next(o for o in template.outputs if o.name == "entities")
        assert entities_output.type == "json"
        assert entities_output.description == "List of entities with types and positions"
    
    def test_resource_requirements(self):
        """Test that template has correct resource requirements."""
        template = NERTemplate()
        
        assert template.gpu_required is False
        assert template.gpu_type is None
        assert template.memory_mb == 1024
        assert template.timeout_sec == 60
    
    def test_routing_support(self):
        """Test that template supports correct backends."""
        template = NERTemplate()
        
        assert RouteType.LOCAL in template.routing
        assert RouteType.MODAL in template.routing
        assert RouteType.HF in template.routing
    
    def test_pip_packages(self):
        """Test that template declares correct dependencies."""
        template = NERTemplate()
        
        assert "spacy" in template.pip_packages
        assert "transformers" in template.pip_packages
    
    def test_validate_inputs_valid(self):
        """Test input validation with valid inputs."""
        template = NERTemplate()
        
        # Should not raise with required text input
        result = template.validate_inputs(text="John lives in New York")
        assert result is True
        
        # Should not raise with all inputs
        result = template.validate_inputs(
            text="John lives in New York",
            model="en_core_web_sm"
        )
        assert result is True
    
    def test_validate_inputs_missing_required(self):
        """Test input validation fails when required input is missing."""
        template = NERTemplate()
        
        # Should raise ValueError when text is missing
        with pytest.raises(ValueError, match="Missing required input: text"):
            template.validate_inputs(model="en_core_web_sm")
    
    def test_setup_with_spacy(self):
        """Test setup method loads spaCy model."""
        template = NERTemplate()
        template._model_name = 'en_core_web_sm'
        
        # Mock spacy module
        with patch.dict('sys.modules', {'spacy': MagicMock()}):
            import sys
            mock_spacy = sys.modules['spacy']
            mock_nlp = Mock()
            mock_spacy.load.return_value = mock_nlp
            
            template.setup()
            
            mock_spacy.load.assert_called_once_with('en_core_web_sm')
            assert template.nlp == mock_nlp
            assert template.model_type == 'spacy'
            assert template._initialized is True
    
    def test_setup_with_transformers(self):
        """Test setup method falls back to transformers when spaCy model not found."""
        template = NERTemplate()
        template._model_name = 'custom-model'
        
        # Mock spacy to raise OSError and transformers to succeed
        with patch.dict('sys.modules', {'spacy': MagicMock(), 'transformers': MagicMock()}):
            import sys
            mock_spacy = sys.modules['spacy']
            mock_spacy.load.side_effect = OSError("Model not found")
            
            mock_transformers = sys.modules['transformers']
            mock_pipeline = Mock()
            mock_transformers.pipeline.return_value = mock_pipeline
            
            template.setup()
            
            mock_transformers.pipeline.assert_called_once_with(
                "ner", model='custom-model', aggregation_strategy="simple"
            )
            assert template.nlp == mock_pipeline
            assert template.model_type == 'transformers'
            assert template._initialized is True
    
    def test_run_with_spacy(self):
        """Test run method with spaCy model."""
        template = NERTemplate()
        
        # Mock spacy
        with patch.dict('sys.modules', {'spacy': MagicMock()}):
            import sys
            mock_spacy = sys.modules['spacy']
            
            # Create mock entities
            mock_ent1 = Mock()
            mock_ent1.text = "John"
            mock_ent1.label_ = "PERSON"
            mock_ent1.start_char = 0
            mock_ent1.end_char = 4
            
            mock_ent2 = Mock()
            mock_ent2.text = "New York"
            mock_ent2.label_ = "GPE"
            mock_ent2.start_char = 14
            mock_ent2.end_char = 22
            
            mock_doc = Mock()
            mock_doc.ents = [mock_ent1, mock_ent2]
            
            mock_nlp = Mock()
            mock_nlp.return_value = mock_doc
            mock_spacy.load.return_value = mock_nlp
            
            # Run the template
            result = template.run(text="John lives in New York")
            
            # Verify results
            assert 'entities' in result
            assert len(result['entities']) == 2
            assert result['entities'][0]['text'] == "John"
            assert result['entities'][0]['label'] == "PERSON"
            assert result['entities'][1]['text'] == "New York"
            assert result['entities'][1]['label'] == "GPE"
    
    def test_run_with_transformers(self):
        """Test run method with transformers model."""
        template = NERTemplate()
        
        # Mock spacy to fail and transformers to succeed
        with patch.dict('sys.modules', {'spacy': MagicMock(), 'transformers': MagicMock()}):
            import sys
            mock_spacy = sys.modules['spacy']
            mock_spacy.load.side_effect = OSError("Model not found")
            
            mock_transformers = sys.modules['transformers']
            mock_pipeline = Mock()
            mock_pipeline.return_value = [
                {'word': 'John', 'entity_group': 'PER', 'start': 0, 'end': 4, 'score': 0.99},
                {'word': 'New York', 'entity_group': 'LOC', 'start': 14, 'end': 22, 'score': 0.95}
            ]
            mock_transformers.pipeline.return_value = mock_pipeline
            
            # Run the template
            result = template.run(text="John lives in New York", model="custom-model")
            
            # Verify results
            assert 'entities' in result
            assert len(result['entities']) == 2
            assert result['entities'][0]['text'] == "John"
            assert result['entities'][0]['label'] == "PER"
            assert result['entities'][0]['score'] == 0.99
    
    def test_run_with_empty_text(self):
        """Test run method with empty text."""
        template = NERTemplate()
        
        # Mock spacy
        with patch.dict('sys.modules', {'spacy': MagicMock()}):
            import sys
            mock_spacy = sys.modules['spacy']
            
            mock_doc = Mock()
            mock_doc.ents = []
            
            mock_nlp = Mock()
            mock_nlp.return_value = mock_doc
            mock_spacy.load.return_value = mock_nlp
            
            # Run with empty text
            result = template.run(text="")
            
            # Should return empty entities list
            assert 'entities' in result
            assert len(result['entities']) == 0


class TestSentimentAnalysisTemplate:
    """Test suite for SentimentAnalysisTemplate."""
    
    def test_template_metadata(self):
        """Test that template has correct metadata."""
        template = SentimentAnalysisTemplate()
        
        assert template.name == "sentiment-analysis"
        assert template.category == "Language"
        assert template.description == "Analyze sentiment (positive, negative, neutral) of text"
        assert template.version == "1.0.0"
    
    def test_input_definitions(self):
        """Test that template has correct input definitions."""
        template = SentimentAnalysisTemplate()
        
        assert len(template.inputs) == 1
        
        # Check text input
        text_input = next(i for i in template.inputs if i.name == "text")
        assert text_input.type == "text"
        assert text_input.required is True
        assert text_input.description == "Text to analyze"
    
    def test_output_definitions(self):
        """Test that template has correct output definitions."""
        template = SentimentAnalysisTemplate()
        
        assert len(template.outputs) == 1
        
        # Check sentiment output
        sentiment_output = next(o for o in template.outputs if o.name == "sentiment")
        assert sentiment_output.type == "json"
        assert sentiment_output.description == "Sentiment scores (positive, negative, neutral)"
    
    def test_resource_requirements(self):
        """Test that template has correct resource requirements."""
        template = SentimentAnalysisTemplate()
        
        assert template.gpu_required is False
        assert template.gpu_type is None
        assert template.memory_mb == 512
        assert template.timeout_sec == 30
    
    def test_routing_support(self):
        """Test that template supports correct backends."""
        template = SentimentAnalysisTemplate()
        
        assert RouteType.LOCAL in template.routing
        assert RouteType.MODAL in template.routing
        assert RouteType.HF in template.routing
    
    def test_pip_packages(self):
        """Test that template declares correct dependencies."""
        template = SentimentAnalysisTemplate()
        
        assert "transformers" in template.pip_packages
        assert "torch" in template.pip_packages
    
    def test_validate_inputs_valid(self):
        """Test input validation with valid inputs."""
        template = SentimentAnalysisTemplate()
        
        # Should not raise with required text input
        result = template.validate_inputs(text="This is great!")
        assert result is True
    
    def test_validate_inputs_missing_required(self):
        """Test input validation fails when required input is missing."""
        template = SentimentAnalysisTemplate()
        
        # Should raise ValueError when text is missing
        with pytest.raises(ValueError, match="Missing required input: text"):
            template.validate_inputs()
    
    def test_setup(self):
        """Test setup method loads sentiment analysis pipeline."""
        template = SentimentAnalysisTemplate()
        
        # Mock transformers module
        with patch.dict('sys.modules', {'transformers': MagicMock()}):
            import sys
            mock_transformers = sys.modules['transformers']
            mock_pipeline = Mock()
            mock_transformers.pipeline.return_value = mock_pipeline
            
            template.setup()
            
            mock_transformers.pipeline.assert_called_once_with(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english"
            )
            assert template.sentiment_pipeline == mock_pipeline
            assert template._initialized is True
    
    def test_run_with_positive_sentiment(self):
        """Test run method with positive sentiment text."""
        template = SentimentAnalysisTemplate()
        
        # Mock transformers
        with patch.dict('sys.modules', {'transformers': MagicMock()}):
            import sys
            mock_transformers = sys.modules['transformers']
            mock_pipeline = Mock()
            mock_pipeline.return_value = [{'label': 'POSITIVE', 'score': 0.9998}]
            mock_transformers.pipeline.return_value = mock_pipeline
            
            # Run the template
            result = template.run(text="This is amazing!")
            
            # Verify results
            assert 'sentiment' in result
            assert result['sentiment']['label'] == 'POSITIVE'
            assert result['sentiment']['score'] == 0.9998
    
    def test_run_with_negative_sentiment(self):
        """Test run method with negative sentiment text."""
        template = SentimentAnalysisTemplate()
        
        # Mock transformers
        with patch.dict('sys.modules', {'transformers': MagicMock()}):
            import sys
            mock_transformers = sys.modules['transformers']
            mock_pipeline = Mock()
            mock_pipeline.return_value = [{'label': 'NEGATIVE', 'score': 0.9995}]
            mock_transformers.pipeline.return_value = mock_pipeline
            
            # Run the template
            result = template.run(text="This is terrible!")
            
            # Verify results
            assert 'sentiment' in result
            assert result['sentiment']['label'] == 'NEGATIVE'
            assert result['sentiment']['score'] == 0.9995
    
    def test_run_with_empty_result(self):
        """Test run method when pipeline returns empty result."""
        template = SentimentAnalysisTemplate()
        
        # Mock transformers
        with patch.dict('sys.modules', {'transformers': MagicMock()}):
            import sys
            mock_transformers = sys.modules['transformers']
            mock_pipeline = Mock()
            mock_pipeline.return_value = []
            mock_transformers.pipeline.return_value = mock_pipeline
            
            # Run the template
            result = template.run(text="")
            
            # Should return neutral sentiment
            assert 'sentiment' in result
            assert result['sentiment']['label'] == 'NEUTRAL'
            assert result['sentiment']['score'] == 0.0


class TestTranslationTemplate:
    """Test suite for TranslationTemplate."""
    
    def test_template_metadata(self):
        """Test that template has correct metadata."""
        template = TranslationTemplate()
        
        assert template.name == "translation"
        assert template.category == "Language"
        assert template.description == "Translate text between languages"
        assert template.version == "1.0.0"
    
    def test_input_definitions(self):
        """Test that template has correct input definitions."""
        template = TranslationTemplate()
        
        assert len(template.inputs) == 3
        
        # Check text input
        text_input = next(i for i in template.inputs if i.name == "text")
        assert text_input.type == "text"
        assert text_input.required is True
        assert text_input.description == "Text to translate"
        
        # Check source_language input
        source_input = next(i for i in template.inputs if i.name == "source_language")
        assert source_input.type == "text"
        assert source_input.required is False
        assert source_input.default == "auto"
        
        # Check target_language input
        target_input = next(i for i in template.inputs if i.name == "target_language")
        assert target_input.type == "text"
        assert target_input.required is True
    
    def test_output_definitions(self):
        """Test that template has correct output definitions."""
        template = TranslationTemplate()
        
        assert len(template.outputs) == 2
        
        # Check translated_text output
        text_output = next(o for o in template.outputs if o.name == "translated_text")
        assert text_output.type == "text"
        assert text_output.description == "Translated text"
        
        # Check detected_language output
        lang_output = next(o for o in template.outputs if o.name == "detected_language")
        assert lang_output.type == "text"
        assert lang_output.description == "Detected source language (if auto)"
    
    def test_resource_requirements(self):
        """Test that template has correct resource requirements."""
        template = TranslationTemplate()
        
        assert template.gpu_required is True
        assert template.gpu_type == "T4"
        assert template.memory_mb == 2048
        assert template.timeout_sec == 120
    
    def test_routing_support(self):
        """Test that template supports correct backends."""
        template = TranslationTemplate()
        
        assert RouteType.LOCAL in template.routing
        assert RouteType.MODAL in template.routing
        assert RouteType.HF in template.routing
    
    def test_pip_packages(self):
        """Test that template declares correct dependencies."""
        template = TranslationTemplate()
        
        assert "transformers" in template.pip_packages
        assert "torch" in template.pip_packages
        assert "sentencepiece" in template.pip_packages
    
    def test_validate_inputs_valid(self):
        """Test input validation with valid inputs."""
        template = TranslationTemplate()
        
        # Should not raise with required inputs
        result = template.validate_inputs(text="Hello", target_language="es")
        assert result is True
        
        # Should not raise with all inputs
        result = template.validate_inputs(
            text="Hello",
            source_language="en",
            target_language="es"
        )
        assert result is True
    
    def test_validate_inputs_missing_required(self):
        """Test input validation fails when required input is missing."""
        template = TranslationTemplate()
        
        # Should raise ValueError when text is missing
        with pytest.raises(ValueError, match="Missing required input: text"):
            template.validate_inputs(target_language="es")
        
        # Should raise ValueError when target_language is missing
        with pytest.raises(ValueError, match="Missing required input: target_language"):
            template.validate_inputs(text="Hello")
    
    def test_setup(self):
        """Test setup method loads translation pipeline."""
        template = TranslationTemplate()
        template._source_language = 'en'
        template._target_language = 'es'
        
        # Mock transformers module
        with patch.dict('sys.modules', {'transformers': MagicMock()}):
            import sys
            mock_transformers = sys.modules['transformers']
            mock_pipeline = Mock()
            mock_transformers.pipeline.return_value = mock_pipeline
            
            template.setup()
            
            mock_transformers.pipeline.assert_called_once_with(
                "translation", model="Helsinki-NLP/opus-mt-en-es"
            )
            assert template.translator == mock_pipeline
            assert template._initialized is True
    
    def test_run_with_explicit_languages(self):
        """Test run method with explicit source and target languages."""
        template = TranslationTemplate()
        
        # Mock transformers
        with patch.dict('sys.modules', {'transformers': MagicMock()}):
            import sys
            mock_transformers = sys.modules['transformers']
            mock_pipeline = Mock()
            mock_pipeline.return_value = [{'translation_text': 'Hola mundo'}]
            mock_transformers.pipeline.return_value = mock_pipeline
            
            # Run the template
            result = template.run(
                text="Hello world",
                source_language="en",
                target_language="es"
            )
            
            # Verify results
            assert 'translated_text' in result
            assert result['translated_text'] == 'Hola mundo'
            assert result['detected_language'] == 'en'
    
    def test_run_with_auto_detection(self):
        """Test run method with automatic language detection."""
        template = TranslationTemplate()
        
        # Mock transformers
        with patch.dict('sys.modules', {'transformers': MagicMock()}):
            import sys
            mock_transformers = sys.modules['transformers']
            mock_pipeline = Mock()
            mock_pipeline.return_value = [{'translation_text': 'Hola'}]
            mock_transformers.pipeline.return_value = mock_pipeline
            
            # Run the template
            result = template.run(
                text="Hello",
                source_language="auto",
                target_language="es"
            )
            
            # Verify results
            assert 'translated_text' in result
            assert result['translated_text'] == 'Hola'
            assert 'detected_language' in result
            # Auto-detection defaults to 'en' in the simple implementation
            assert result['detected_language'] == 'en'
    
    def test_run_with_empty_result(self):
        """Test run method when translation fails."""
        template = TranslationTemplate()
        
        # Mock transformers
        with patch.dict('sys.modules', {'transformers': MagicMock()}):
            import sys
            mock_transformers = sys.modules['transformers']
            mock_pipeline = Mock()
            mock_pipeline.return_value = []
            mock_transformers.pipeline.return_value = mock_pipeline
            
            # Run the template
            result = template.run(text="Hello", target_language="es")
            
            # Should fallback to original text
            assert 'translated_text' in result
            assert result['translated_text'] == 'Hello'


class TestSummarizationTemplate:
    """Test suite for SummarizationTemplate."""
    
    def test_template_metadata(self):
        """Test that template has correct metadata."""
        template = SummarizationTemplate()
        
        assert template.name == "summarization"
        assert template.category == "Language"
        assert template.description == "Summarize long text into concise form"
        assert template.version == "1.0.0"
    
    def test_input_definitions(self):
        """Test that template has correct input definitions."""
        template = SummarizationTemplate()
        
        assert len(template.inputs) == 3
        
        # Check text input
        text_input = next(i for i in template.inputs if i.name == "text")
        assert text_input.type == "text"
        assert text_input.required is True
        assert text_input.description == "Text to summarize"
        
        # Check max_length input
        max_input = next(i for i in template.inputs if i.name == "max_length")
        assert max_input.type == "number"
        assert max_input.required is False
        assert max_input.default == 150
        
        # Check min_length input
        min_input = next(i for i in template.inputs if i.name == "min_length")
        assert min_input.type == "number"
        assert min_input.required is False
        assert min_input.default == 50
    
    def test_output_definitions(self):
        """Test that template has correct output definitions."""
        template = SummarizationTemplate()
        
        assert len(template.outputs) == 1
        
        # Check summary output
        summary_output = next(o for o in template.outputs if o.name == "summary")
        assert summary_output.type == "text"
        assert summary_output.description == "Summarized text"
    
    def test_resource_requirements(self):
        """Test that template has correct resource requirements."""
        template = SummarizationTemplate()
        
        assert template.gpu_required is True
        assert template.gpu_type == "T4"
        assert template.memory_mb == 2048
        assert template.timeout_sec == 120
    
    def test_routing_support(self):
        """Test that template supports correct backends."""
        template = SummarizationTemplate()
        
        assert RouteType.LOCAL in template.routing
        assert RouteType.MODAL in template.routing
        assert RouteType.HF in template.routing
    
    def test_pip_packages(self):
        """Test that template declares correct dependencies."""
        template = SummarizationTemplate()
        
        assert "transformers" in template.pip_packages
        assert "torch" in template.pip_packages
    
    def test_validate_inputs_valid(self):
        """Test input validation with valid inputs."""
        template = SummarizationTemplate()
        
        # Should not raise with required text input
        result = template.validate_inputs(text="Long text to summarize...")
        assert result is True
        
        # Should not raise with all inputs
        result = template.validate_inputs(
            text="Long text to summarize...",
            max_length=100,
            min_length=30
        )
        assert result is True
    
    def test_validate_inputs_missing_required(self):
        """Test input validation fails when required input is missing."""
        template = SummarizationTemplate()
        
        # Should raise ValueError when text is missing
        with pytest.raises(ValueError, match="Missing required input: text"):
            template.validate_inputs(max_length=100)
    
    def test_setup(self):
        """Test setup method loads summarization pipeline."""
        template = SummarizationTemplate()
        
        # Mock transformers module
        with patch.dict('sys.modules', {'transformers': MagicMock()}):
            import sys
            mock_transformers = sys.modules['transformers']
            mock_pipeline = Mock()
            mock_transformers.pipeline.return_value = mock_pipeline
            
            template.setup()
            
            mock_transformers.pipeline.assert_called_once_with(
                "summarization",
                model="facebook/bart-large-cnn"
            )
            assert template.summarizer == mock_pipeline
            assert template._initialized is True
    
    def test_run_with_valid_inputs(self):
        """Test run method with valid inputs."""
        template = SummarizationTemplate()
        
        # Mock transformers
        with patch.dict('sys.modules', {'transformers': MagicMock()}):
            import sys
            mock_transformers = sys.modules['transformers']
            mock_pipeline = Mock()
            mock_pipeline.return_value = [{'summary_text': 'This is a summary.'}]
            mock_transformers.pipeline.return_value = mock_pipeline
            
            # Run the template
            long_text = "This is a very long article about machine learning. " * 10
            result = template.run(text=long_text, max_length=150, min_length=50)
            
            # Verify results
            assert 'summary' in result
            assert result['summary'] == 'This is a summary.'
            
            # Verify summarizer was called with correct parameters
            mock_pipeline.assert_called_once()
            call_args = mock_pipeline.call_args
            assert call_args[0][0] == long_text
            assert call_args[1]['max_length'] == 150
            assert call_args[1]['min_length'] == 50
            assert call_args[1]['do_sample'] is False
    
    def test_run_with_invalid_length_params(self):
        """Test run method adjusts min_length when it exceeds max_length."""
        template = SummarizationTemplate()
        
        # Mock transformers
        with patch.dict('sys.modules', {'transformers': MagicMock()}):
            import sys
            mock_transformers = sys.modules['transformers']
            mock_pipeline = Mock()
            mock_pipeline.return_value = [{'summary_text': 'Summary.'}]
            mock_transformers.pipeline.return_value = mock_pipeline
            
            # Run with min_length >= max_length
            result = template.run(text="Test text", max_length=50, min_length=60)
            
            # Verify min_length was adjusted
            call_args = mock_pipeline.call_args
            assert call_args[1]['max_length'] == 50
            assert call_args[1]['min_length'] == 40  # max_length - 10
    
    def test_run_with_empty_result(self):
        """Test run method when summarization fails."""
        template = SummarizationTemplate()
        
        # Mock transformers
        with patch.dict('sys.modules', {'transformers': MagicMock()}):
            import sys
            mock_transformers = sys.modules['transformers']
            mock_pipeline = Mock()
            mock_pipeline.return_value = []
            mock_transformers.pipeline.return_value = mock_pipeline
            
            # Run the template with text that has enough words
            text = "This is a longer text with many words that should be summarized but the summarization fails."
            result = template.run(text=text, min_length=10)
            
            # Should fallback to first N words
            assert 'summary' in result
            assert result['summary'].endswith('...')
            # The fallback returns first min_length words
            words_in_summary = result['summary'].replace('...', '').split()
            assert len(words_in_summary) == 10
    
    def test_run_with_custom_lengths(self):
        """Test run method with custom length parameters."""
        template = SummarizationTemplate()
        
        # Mock transformers
        with patch.dict('sys.modules', {'transformers': MagicMock()}):
            import sys
            mock_transformers = sys.modules['transformers']
            mock_pipeline = Mock()
            mock_pipeline.return_value = [{'summary_text': 'Custom summary.'}]
            mock_transformers.pipeline.return_value = mock_pipeline
            
            # Run with custom lengths
            result = template.run(text="Test text", max_length=200, min_length=100)
            
            # Verify custom lengths were used
            call_args = mock_pipeline.call_args
            assert call_args[1]['max_length'] == 200
            assert call_args[1]['min_length'] == 100
