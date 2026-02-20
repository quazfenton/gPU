"""
Named Entity Recognition (NER) Template for extracting entities from text.

This template uses NLP models (spaCy or transformers) to identify and extract
named entities such as people, places, organizations, dates, and more from text.
"""

from typing import Any, Dict, List
from templates.base import Template, InputField, OutputField, RouteType


class NERTemplate(Template):
    """
    Extracts named entities from text.
    
    Identifies and extracts named entities including:
    - PERSON: People names
    - ORG: Organizations
    - GPE: Geopolitical entities (countries, cities, states)
    - DATE: Dates and time expressions
    - MONEY: Monetary values
    - And more depending on the model
    
    Returns a list of entities with their types, text, and positions.
    """
    
    name = "named-entity-recognition"
    category = "Language"
    description = "Extract named entities (people, places, organizations) from text"
    version = "1.0.0"
    
    inputs = [
        InputField(
            name="text",
            type="text",
            description="Text to analyze",
            required=True
        ),
        InputField(
            name="model",
            type="text",
            description="NER model to use",
            required=False,
            default="en_core_web_sm"
        )
    ]
    
    outputs = [
        OutputField(
            name="entities",
            type="json",
            description="List of entities with types and positions"
        )
    ]
    
    routing = [RouteType.LOCAL, RouteType.MODAL, RouteType.HF]
    gpu_required = False
    gpu_type = None
    memory_mb = 1024
    timeout_sec = 60
    pip_packages = ["spacy", "transformers"]
    
    def setup(self) -> None:
        """
        One-time initialization to load the NER model.
        
        Loads the specified spaCy model. If using a transformer-based model,
        loads from HuggingFace transformers library.
        """
        import spacy
        
        # Get model name from instance or use default
        model_name = getattr(self, '_model_name', 'en_core_web_sm')
        
        try:
            # Try to load spaCy model
            self.nlp = spacy.load(model_name)
            self.model_type = 'spacy'
        except OSError:
            # If spaCy model not found, try transformers
            from transformers import pipeline
            self.nlp = pipeline("ner", model=model_name, aggregation_strategy="simple")
            self.model_type = 'transformers'
        
        self._initialized = True
    
    def run(self, **kwargs) -> Dict[str, Any]:
        """
        Execute named entity recognition on the provided text.
        
        Args:
            text: Text to analyze for named entities
            model: Model name to use (optional, defaults to 'en_core_web_sm')
            
        Returns:
            Dict containing:
                - entities: List of extracted entities with type, text, start, and end positions
        """
        # Validate inputs
        self.validate_inputs(**kwargs)
        
        # Initialize if needed
        if not self._initialized:
            self._model_name = kwargs.get('model', 'en_core_web_sm')
            self.setup()
        
        # Extract parameters
        text = kwargs['text']
        
        # Process text based on model type
        if self.model_type == 'spacy':
            entities = self._extract_entities_spacy(text)
        else:
            entities = self._extract_entities_transformers(text)
        
        return {
            'entities': entities
        }
    
    def _extract_entities_spacy(self, text: str) -> List[Dict[str, Any]]:
        """Extract entities using spaCy model."""
        doc = self.nlp(text)
        
        entities = []
        for ent in doc.ents:
            entities.append({
                'text': ent.text,
                'label': ent.label_,
                'start': ent.start_char,
                'end': ent.end_char
            })
        
        return entities
    
    def _extract_entities_transformers(self, text: str) -> List[Dict[str, Any]]:
        """Extract entities using transformers pipeline."""
        results = self.nlp(text)
        
        entities = []
        for entity in results:
            entities.append({
                'text': entity['word'],
                'label': entity['entity_group'],
                'start': entity['start'],
                'end': entity['end'],
                'score': float(entity['score'])
            })
        
        return entities
