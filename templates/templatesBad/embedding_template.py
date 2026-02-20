"""
Text Embedding Template for generating dense vector embeddings.

This template uses sentence-transformers to generate dense vector embeddings
from text input. Useful for semantic search, clustering, and similarity tasks.
"""

from typing import Any, Dict
from templates.base import Template, InputField, OutputField, RouteType


class TextEmbeddingTemplate(Template):
    """
    Generates dense vector embeddings from text.
    
    Uses sentence-transformers models to encode text into high-dimensional
    vectors suitable for semantic search, clustering, and similarity
    comparison tasks.
    """
    
    name = "text-embedding"
    category = "Language"
    description = "Generate dense vector embeddings from text using sentence-transformers"
    version = "1.0.0"
    
    inputs = [
        InputField(
            name="text",
            type="text",
            description="Text to embed",
            required=True
        ),
        InputField(
            name="model_name",
            type="text",
            description="Sentence-transformers model name",
            required=False,
            default="all-MiniLM-L6-v2"
        )
    ]
    
    outputs = [
        OutputField(
            name="embedding",
            type="json",
            description="Dense vector embedding as list of floats"
        ),
        OutputField(
            name="dimensions",
            type="number",
            description="Number of dimensions in the embedding vector"
        )
    ]
    
    routing = [RouteType.LOCAL, RouteType.MODAL, RouteType.HF]
    gpu_required = False
    gpu_type = None
    memory_mb = 2048
    timeout_sec = 60
    pip_packages = ["sentence-transformers", "torch"]
    
    def setup(self) -> None:
        """
        One-time initialization to load the sentence-transformers model.
        
        Downloads and loads the specified embedding model. The model is cached
        for subsequent uses.
        """
        from sentence_transformers import SentenceTransformer
        
        model_name = getattr(self, '_model_name', 'all-MiniLM-L6-v2')
        self.model = SentenceTransformer(model_name)
        self._initialized = True
    
    def run(self, **kwargs) -> Dict[str, Any]:
        """
        Execute text embedding on the provided text.
        
        Args:
            text: Text to generate embeddings for
            model_name: Sentence-transformers model name (optional, defaults to 'all-MiniLM-L6-v2')
            
        Returns:
            Dict containing:
                - embedding: List of floats representing the dense vector
                - dimensions: Number of dimensions in the embedding
        """
        # Validate inputs
        self.validate_inputs(**kwargs)
        
        # Initialize if needed
        if not self._initialized:
            self._model_name = kwargs.get('model_name', 'all-MiniLM-L6-v2')
            self.setup()
        
        # Extract parameters
        text = kwargs['text']
        
        # Generate embedding
        embedding = self.model.encode(text)
        
        # Convert numpy array to list of floats
        embedding_list = [float(x) for x in embedding.tolist()]
        
        return {
            'embedding': embedding_list,
            'dimensions': len(embedding_list)
        }
