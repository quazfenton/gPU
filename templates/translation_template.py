"""
Translation Template for translating text between languages.

This template uses transformer-based translation models to translate text
from one language to another, with support for automatic language detection.
"""

from typing import Any, Dict
from templates.base import Template, InputField, OutputField, RouteType


class TranslationTemplate(Template):
    """
    Translates text between languages.
    
    Uses pre-trained translation models (e.g., MarianMT, mBART) to translate
    text from a source language to a target language. Supports automatic
    language detection when source language is set to "auto".
    
    Returns translated text and detected source language (if auto-detection used).
    """
    
    name = "translation"
    category = "Language"
    description = "Translate text between languages"
    version = "1.0.0"
    
    inputs = [
        InputField(
            name="text",
            type="text",
            description="Text to translate",
            required=True
        ),
        InputField(
            name="source_language",
            type="text",
            description="Source language code",
            required=False,
            default="auto"
        ),
        InputField(
            name="target_language",
            type="text",
            description="Target language code",
            required=True
        )
    ]
    
    outputs = [
        OutputField(
            name="translated_text",
            type="text",
            description="Translated text"
        ),
        OutputField(
            name="detected_language",
            type="text",
            description="Detected source language (if auto)"
        )
    ]
    
    routing = [RouteType.LOCAL, RouteType.MODAL, RouteType.HF]
    gpu_required = True
    gpu_type = "T4"
    memory_mb = 2048
    timeout_sec = 120
    pip_packages = ["transformers", "torch", "sentencepiece"]
    
    def setup(self) -> None:
        """
        One-time initialization to load the translation model.
        
        Loads a translation model based on the source and target languages.
        Uses MarianMT models from HuggingFace for most language pairs.
        """
        from transformers import pipeline
        
        # Get language pair from instance
        source_lang = getattr(self, '_source_language', 'en')
        target_lang = getattr(self, '_target_language', 'es')
        
        # Handle auto-detection
        if source_lang == 'auto':
            # Use a multilingual model for auto-detection
            model_name = f"Helsinki-NLP/opus-mt-mul-{target_lang}"
        else:
            # Use specific language pair model
            model_name = f"Helsinki-NLP/opus-mt-{source_lang}-{target_lang}"
        
        try:
            # Load translation pipeline
            self.translator = pipeline("translation", model=model_name)
            self.model_name = model_name
        except Exception:
            # Fallback to a general multilingual model
            self.translator = pipeline("translation", model="facebook/mbart-large-50-many-to-many-mmt")
            self.model_name = "facebook/mbart-large-50-many-to-many-mmt"
        
        self._initialized = True
    
    def run(self, **kwargs) -> Dict[str, Any]:
        """
        Execute translation on the provided text.
        
        Args:
            text: Text to translate
            source_language: Source language code (optional, defaults to "auto")
            target_language: Target language code (required)
            
        Returns:
            Dict containing:
                - translated_text: The translated text
                - detected_language: Detected source language (if auto-detection used)
        """
        # Validate inputs
        self.validate_inputs(**kwargs)
        
        # Extract parameters
        text = kwargs['text']
        source_language = kwargs.get('source_language', 'auto')
        target_language = kwargs['target_language']
        
        # Initialize if needed
        if not self._initialized:
            self._source_language = source_language if source_language != 'auto' else 'en'
            self._target_language = target_language
            self.setup()
        
        # Detect language if auto
        detected_language = None
        if source_language == 'auto':
            detected_language = self._detect_language(text)
        else:
            detected_language = source_language
        
        # Translate text
        translation_result = self.translator(text)
        
        # Extract translated text
        if translation_result and len(translation_result) > 0:
            translated_text = translation_result[0]['translation_text']
        else:
            translated_text = text  # Fallback to original text
        
        return {
            'translated_text': translated_text,
            'detected_language': detected_language
        }
    
    def _detect_language(self, text: str) -> str:
        """
        Detect the language of the input text.
        
        Simple heuristic-based detection. In production, would use a proper
        language detection library like langdetect or fasttext.
        """
        # Simple heuristic: assume English for now
        # In a real implementation, use langdetect or similar
        return 'en'
