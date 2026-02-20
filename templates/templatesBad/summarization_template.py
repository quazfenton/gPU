"""
Summarization Template for condensing long text into shorter summaries.

This template uses transformer-based summarization models to generate
concise summaries of longer text documents while preserving key information.
"""

from typing import Any, Dict
from templates.base import Template, InputField, OutputField, RouteType


class SummarizationTemplate(Template):
    """
    Summarizes long text into shorter form.
    
    Uses pre-trained summarization models (e.g., BART, T5, Pegasus) to
    generate abstractive summaries of text. Supports configurable summary
    length constraints.
    
    Returns a concise summary of the input text.
    """
    
    name = "summarization"
    category = "Language"
    description = "Summarize long text into concise form"
    version = "1.0.0"
    
    inputs = [
        InputField(
            name="text",
            type="text",
            description="Text to summarize",
            required=True
        ),
        InputField(
            name="max_length",
            type="number",
            description="Maximum length of summary",
            required=False,
            default=150
        ),
        InputField(
            name="min_length",
            type="number",
            description="Minimum length of summary",
            required=False,
            default=50
        )
    ]
    
    outputs = [
        OutputField(
            name="summary",
            type="text",
            description="Summarized text"
        )
    ]
    
    routing = [RouteType.LOCAL, RouteType.MODAL, RouteType.HF]
    gpu_required = True
    gpu_type = "T4"
    memory_mb = 2048
    timeout_sec = 120
    pip_packages = ["transformers", "torch"]
    
    def setup(self) -> None:
        """
        One-time initialization to load the summarization model.
        
        Loads a pre-trained summarization model from HuggingFace.
        Default model is facebook/bart-large-cnn which is optimized for
        news article summarization.
        """
        from transformers import pipeline
        
        # Load summarization pipeline
        self.summarizer = pipeline(
            "summarization",
            model="facebook/bart-large-cnn"
        )
        
        self._initialized = True
    
    def run(self, **kwargs) -> Dict[str, Any]:
        """
        Execute summarization on the provided text.
        
        Args:
            text: Text to summarize
            max_length: Maximum length of summary (optional, defaults to 150)
            min_length: Minimum length of summary (optional, defaults to 50)
            
        Returns:
            Dict containing:
                - summary: The summarized text
        """
        # Validate inputs
        self.validate_inputs(**kwargs)
        
        # Initialize if needed
        if not self._initialized:
            self.setup()
        
        # Extract parameters
        text = kwargs['text']
        max_length = kwargs.get('max_length', 150)
        min_length = kwargs.get('min_length', 50)
        
        # Ensure min_length < max_length
        if min_length >= max_length:
            min_length = max(1, max_length - 10)
        
        # Generate summary
        summary_result = self.summarizer(
            text,
            max_length=int(max_length),
            min_length=int(min_length),
            do_sample=False
        )
        
        # Extract summary text
        if summary_result and len(summary_result) > 0:
            summary = summary_result[0]['summary_text']
        else:
            # Fallback: return first N words if summarization fails
            words = text.split()
            summary = ' '.join(words[:min_length]) + '...'
        
        return {
            'summary': summary
        }
