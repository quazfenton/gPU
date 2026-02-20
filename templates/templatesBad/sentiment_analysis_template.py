"""
Sentiment Analysis Template for analyzing text sentiment.

This template uses transformer-based models to analyze the sentiment of text,
classifying it as positive, negative, or neutral with confidence scores.
"""

from typing import Any, Dict
from templates.base import Template, InputField, OutputField, RouteType


class SentimentAnalysisTemplate(Template):
    """
    Analyzes sentiment of text.
    
    Uses pre-trained transformer models to classify text sentiment into:
    - Positive
    - Negative
    - Neutral (depending on model)
    
    Returns sentiment labels with confidence scores.
    """
    
    name = "sentiment-analysis"
    category = "Language"
    description = "Analyze sentiment (positive, negative, neutral) of text"
    version = "1.0.0"
    
    inputs = [
        InputField(
            name="text",
            type="text",
            description="Text to analyze",
            required=True
        )
    ]
    
    outputs = [
        OutputField(
            name="sentiment",
            type="json",
            description="Sentiment scores (positive, negative, neutral)"
        )
    ]
    
    routing = [RouteType.LOCAL, RouteType.MODAL, RouteType.HF]
    gpu_required = False
    gpu_type = None
    memory_mb = 512
    timeout_sec = 30
    pip_packages = ["transformers", "torch"]
    
    def setup(self) -> None:
        """
        One-time initialization to load the sentiment analysis model.
        
        Loads a pre-trained sentiment analysis model from HuggingFace.
        Default model is distilbert-base-uncased-finetuned-sst-2-english.
        """
        from transformers import pipeline
        
        # Load sentiment analysis pipeline
        self.sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english"
        )
        
        self._initialized = True
    
    def run(self, **kwargs) -> Dict[str, Any]:
        """
        Execute sentiment analysis on the provided text.
        
        Args:
            text: Text to analyze for sentiment
            
        Returns:
            Dict containing:
                - sentiment: Dictionary with label and score
        """
        # Validate inputs
        self.validate_inputs(**kwargs)
        
        # Initialize if needed
        if not self._initialized:
            self.setup()
        
        # Extract parameters
        text = kwargs['text']
        
        # Analyze sentiment
        results = self.sentiment_pipeline(text)
        
        # Extract result (pipeline returns a list)
        if results:
            result = results[0]
            sentiment = {
                'label': result['label'],
                'score': float(result['score'])
            }
        else:
            sentiment = {
                'label': 'NEUTRAL',
                'score': 0.0
            }
        
        return {
            'sentiment': sentiment
        }
