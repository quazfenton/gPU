"""
Document Question Answering Template for extracting answers from documents.

This template uses transformer-based question answering models to extract
precise answers from document text given a natural language question,
returning the answer span along with confidence and position information.
"""

from typing import Any, Dict
from templates.base import Template, InputField, OutputField, RouteType


class DocumentQATemplate(Template):
    """
    Extracts answers to questions from document text.
    
    Uses pre-trained extractive question answering models to find and extract
    answer spans from provided context documents. The model identifies the
    most relevant passage in the document that answers the given question.
    
    Returns the answer text, confidence score, and character positions.
    """
    
    name = "document-qa"
    category = "Multimodal"
    description = "Answer questions by extracting information from document text"
    version = "1.0.0"
    
    inputs = [
        InputField(
            name="context",
            type="text",
            description="The document text to search for answers",
            required=True
        ),
        InputField(
            name="question",
            type="text",
            description="The question to answer based on the document",
            required=True
        )
    ]
    
    outputs = [
        OutputField(
            name="answer",
            type="text",
            description="Extracted answer from the document"
        ),
        OutputField(
            name="confidence",
            type="number",
            description="Confidence score of the answer (0.0 to 1.0)"
        ),
        OutputField(
            name="start_position",
            type="number",
            description="Start character position of the answer in the context"
        ),
        OutputField(
            name="end_position",
            type="number",
            description="End character position of the answer in the context"
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
        One-time initialization to load the question answering model.
        
        Loads a pre-trained extractive QA model from HuggingFace.
        Default model is distilbert-base-cased-distilled-squad.
        """
        from transformers import pipeline
        
        # Load question-answering pipeline
        self.qa_pipeline = pipeline(
            "question-answering",
            model="distilbert-base-cased-distilled-squad"
        )
        
        self._initialized = True
    
    def run(self, **kwargs) -> Dict[str, Any]:
        """
        Execute document question answering on the provided context and question.
        
        Args:
            context: The document text to search for answers
            question: The question to answer based on the document
            
        Returns:
            Dict containing:
                - answer: Extracted answer text
                - confidence: Confidence score (0.0 to 1.0)
                - start_position: Start character index of the answer in context
                - end_position: End character index of the answer in context
        """
        # Validate inputs
        self.validate_inputs(**kwargs)
        
        # Initialize if needed
        if not self._initialized:
            self.setup()
        
        # Extract parameters
        context = kwargs['context']
        question = kwargs['question']
        
        # Run question answering
        result = self.qa_pipeline(
            question=question,
            context=context
        )
        
        return {
            'answer': result['answer'],
            'confidence': float(result['score']),
            'start_position': int(result['start']),
            'end_position': int(result['end'])
        }
