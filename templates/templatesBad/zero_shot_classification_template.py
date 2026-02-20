"""
Zero-Shot Classification Template for classifying text without training.

This template uses transformer-based zero-shot classification pipelines to
classify text into arbitrary categories without requiring task-specific training.
"""

from typing import Any, Dict
from templates.base import Template, InputField, OutputField, RouteType


class ZeroShotClassificationTemplate(Template):
    """
    Classifies text into arbitrary categories without training.
    
    Uses a zero-shot classification pipeline from HuggingFace transformers
    to classify text against user-provided candidate labels. Supports both
    single-label and multi-label classification.
    """
    
    name = "zero-shot-classification"
    category = "Language"
    description = "Classify text into arbitrary categories using zero-shot learning"
    version = "1.0.0"
    
    inputs = [
        InputField(
            name="text",
            type="text",
            description="Text to classify",
            required=True
        ),
        InputField(
            name="candidate_labels",
            type="json",
            description="List of candidate label strings to classify against",
            required=True
        ),
        InputField(
            name="multi_label",
            type="text",
            description="Whether to allow multiple labels per text",
            required=False,
            default="false",
            options=["true", "false"]
        )
    ]
    
    outputs = [
        OutputField(
            name="labels",
            type="json",
            description="Labels sorted by confidence score"
        ),
        OutputField(
            name="scores",
            type="json",
            description="Confidence scores corresponding to labels"
        ),
        OutputField(
            name="best_label",
            type="text",
            description="Highest confidence label"
        ),
        OutputField(
            name="best_score",
            type="number",
            description="Confidence score for the best label"
        )
    ]
    
    routing = [RouteType.LOCAL, RouteType.MODAL, RouteType.HF]
    gpu_required = False
    gpu_type = None
    memory_mb = 2048
    timeout_sec = 60
    pip_packages = ["transformers", "torch"]
    
    def setup(self) -> None:
        """
        One-time initialization to load the zero-shot classification pipeline.
        
        Downloads and loads a pre-trained NLI-based model for zero-shot
        classification. The model is cached for subsequent uses.
        """
        from transformers import pipeline
        
        self.classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli"
        )
        self._initialized = True
    
    def run(self, **kwargs) -> Dict[str, Any]:
        """
        Execute zero-shot classification on the provided text.
        
        Args:
            text: Text to classify
            candidate_labels: List of candidate label strings
            multi_label: Whether to allow multiple labels (optional, defaults to 'false')
            
        Returns:
            Dict containing:
                - labels: Labels sorted by confidence
                - scores: Corresponding confidence scores
                - best_label: Highest confidence label
                - best_score: Confidence score for the best label
        """
        # Validate inputs
        self.validate_inputs(**kwargs)
        
        # Initialize if needed
        if not self._initialized:
            self.setup()
        
        # Extract parameters
        text = kwargs['text']
        candidate_labels = kwargs['candidate_labels']
        multi_label = kwargs.get('multi_label', 'false') == 'true'
        
        # Run classification
        result = self.classifier(
            text,
            candidate_labels=candidate_labels,
            multi_label=multi_label
        )
        
        # Extract results
        labels = result['labels']
        scores = [float(s) for s in result['scores']]
        
        return {
            'labels': labels,
            'scores': scores,
            'best_label': labels[0],
            'best_score': scores[0]
        }
