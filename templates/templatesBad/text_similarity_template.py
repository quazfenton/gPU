"""
Text Similarity Template for computing semantic similarity between texts.

This template uses sentence-transformers to encode text into dense vector
embeddings and compute cosine similarity, enabling semantic comparison
of text pairs for tasks like duplicate detection and semantic search.
"""

from typing import Any, Dict
from templates.base import Template, InputField, OutputField, RouteType


class TextSimilarityTemplate(Template):
    """
    Computes semantic similarity between two texts.
    
    Uses sentence-transformers to generate dense embeddings for input texts
    and computes cosine similarity between them. Returns a similarity score
    between 0.0 (completely dissimilar) and 1.0 (identical meaning) along
    with a human-readable interpretation.
    """
    
    name = "text-similarity"
    category = "Language"
    description = "Compute semantic similarity between two text passages"
    version = "1.0.0"
    
    inputs = [
        InputField(
            name="text_a",
            type="text",
            description="First text for comparison",
            required=True
        ),
        InputField(
            name="text_b",
            type="text",
            description="Second text for comparison",
            required=True
        )
    ]
    
    outputs = [
        OutputField(
            name="similarity_score",
            type="number",
            description="Cosine similarity score between 0.0 and 1.0"
        ),
        OutputField(
            name="interpretation",
            type="text",
            description="Human-readable interpretation of the similarity score"
        )
    ]
    
    routing = [RouteType.LOCAL, RouteType.MODAL, RouteType.HF]
    gpu_required = False
    gpu_type = None
    memory_mb = 2048
    timeout_sec = 60
    pip_packages = ["sentence-transformers", "torch", "numpy"]
    
    def setup(self) -> None:
        """
        One-time initialization to load the sentence-transformers model.
        
        Loads a pre-trained sentence-transformers model for generating
        text embeddings. Default model is all-MiniLM-L6-v2, which provides
        a good balance of speed and quality.
        """
        from sentence_transformers import SentenceTransformer
        
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        self._initialized = True
    
    def _interpret_score(self, score: float) -> str:
        """Generate a human-readable interpretation of the similarity score."""
        if score >= 0.9:
            return "Very high similarity - texts are nearly identical in meaning"
        elif score >= 0.7:
            return "High similarity - texts convey closely related meanings"
        elif score >= 0.5:
            return "Moderate similarity - texts share some common themes"
        elif score >= 0.3:
            return "Low similarity - texts have limited overlap in meaning"
        else:
            return "Very low similarity - texts are largely unrelated"
    
    def run(self, **kwargs) -> Dict[str, Any]:
        """
        Compute semantic similarity between two texts.
        
        Args:
            text_a: First text for comparison
            text_b: Second text for comparison
            
        Returns:
            Dict containing:
                - similarity_score: Cosine similarity (0.0 to 1.0)
                - interpretation: Human-readable description of similarity
        """
        # Validate inputs
        self.validate_inputs(**kwargs)
        
        # Initialize if needed
        if not self._initialized:
            self.setup()
        
        import numpy as np
        
        # Extract parameters
        text_a = kwargs['text_a']
        text_b = kwargs['text_b']
        
        # Generate embeddings
        embeddings = self.model.encode([text_a, text_b])
        
        # Compute cosine similarity
        embedding_a = embeddings[0]
        embedding_b = embeddings[1]
        
        cosine_sim = float(np.dot(embedding_a, embedding_b) / (
            np.linalg.norm(embedding_a) * np.linalg.norm(embedding_b)
        ))
        
        # Clamp to [0, 1] range
        similarity_score = float(max(0.0, min(1.0, cosine_sim)))
        
        # Generate interpretation
        interpretation = self._interpret_score(similarity_score)
        
        return {
            'similarity_score': similarity_score,
            'interpretation': interpretation
        }
