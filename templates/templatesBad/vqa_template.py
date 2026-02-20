"""
Visual Question Answering (VQA) Template for answering questions about images.

This template uses vision-language models to answer natural language questions
about image content, enabling interactive image understanding and analysis.
"""

from typing import Any, Dict
from templates.base import Template, InputField, OutputField, RouteType


class VQATemplate(Template):
    """
    Answers questions about images.
    
    Uses pre-trained vision-language models (e.g., BLIP-VQA, ViLT) to answer
    natural language questions about image content. The model analyzes both
    the image and question to generate relevant answers.
    
    Returns an answer and confidence score.
    """
    
    name = "visual-question-answering"
    category = "Multimodal"
    description = "Answer questions about image content"
    version = "1.0.0"
    
    inputs = [
        InputField(
            name="image",
            type="image",
            description="Image to analyze",
            required=True
        ),
        InputField(
            name="question",
            type="text",
            description="Question about the image",
            required=True
        )
    ]
    
    outputs = [
        OutputField(
            name="answer",
            type="text",
            description="Answer to the question"
        ),
        OutputField(
            name="confidence",
            type="number",
            description="Confidence score"
        )
    ]
    
    routing = [RouteType.LOCAL, RouteType.MODAL, RouteType.HF]
    gpu_required = True
    gpu_type = "T4"
    memory_mb = 4096
    timeout_sec = 300
    pip_packages = ["transformers", "torch", "pillow"]
    
    def setup(self) -> None:
        """
        One-time initialization to load the VQA model.
        
        Loads a pre-trained vision-language model from HuggingFace.
        Default model is Salesforce/blip-vqa-base.
        """
        from transformers import BlipProcessor, BlipForQuestionAnswering
        
        # Load BLIP VQA model and processor
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
        self.model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
        
        self._initialized = True
    
    def run(self, **kwargs) -> Dict[str, Any]:
        """
        Execute visual question answering on the provided image and question.
        
        Args:
            image: Path to image file or image data
            question: Natural language question about the image
            
        Returns:
            Dict containing:
                - answer: Generated answer to the question
                - confidence: Confidence score (0.0 to 1.0)
        """
        # Validate inputs
        self.validate_inputs(**kwargs)
        
        # Initialize if needed
        if not self._initialized:
            self.setup()
        
        from PIL import Image
        import torch
        
        # Extract parameters
        image = kwargs['image']
        question = kwargs['question']
        
        # Load image
        if isinstance(image, str):
            pil_image = Image.open(image).convert('RGB')
        else:
            # Assume it's already a PIL Image or numpy array
            if hasattr(image, 'convert'):
                pil_image = image.convert('RGB')
            else:
                pil_image = Image.fromarray(image).convert('RGB')
        
        # Process image and question
        inputs = self.processor(pil_image, question, return_tensors="pt")
        
        # Generate answer
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_length=50)
        
        # Decode answer
        answer = self.processor.decode(outputs[0], skip_special_tokens=True)
        
        # Calculate confidence (simplified - use average token probability)
        # In a real implementation, would extract actual probabilities from model
        confidence = 0.80  # Placeholder confidence score
        
        return {
            'answer': answer,
            'confidence': confidence
        }
