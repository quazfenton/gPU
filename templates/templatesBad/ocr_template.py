"""
Optical Character Recognition Template for extracting text from images.

This template uses EasyOCR to detect and extract text from images,
supporting multiple languages with confidence scores and bounding box
coordinates for each detected text region.
"""

from typing import Any, Dict
from templates.base import Template, InputField, OutputField, RouteType


class OCRTemplate(Template):
    """
    Extracts text from images using optical character recognition.
    
    Uses EasyOCR for robust text detection and recognition across multiple
    languages. Returns extracted text, per-region confidence scores, and
    bounding box coordinates for each detected text region.
    """
    
    name = "ocr"
    category = "Vision"
    description = "Extract text from images using optical character recognition"
    version = "1.0.0"
    
    inputs = [
        InputField(
            name="image",
            type="image",
            description="Image to extract text from",
            required=True
        ),
        InputField(
            name="language",
            type="text",
            description="Language code for OCR (e.g., 'en', 'fr', 'de')",
            required=False,
            default="en"
        )
    ]
    
    outputs = [
        OutputField(
            name="text",
            type="text",
            description="Full extracted text from the image"
        ),
        OutputField(
            name="confidence",
            type="number",
            description="Average confidence score across all detected regions"
        ),
        OutputField(
            name="bounding_boxes",
            type="json",
            description="List of detected regions with text, confidence, and coordinates"
        )
    ]
    
    routing = [RouteType.LOCAL, RouteType.MODAL, RouteType.HF]
    gpu_required = False
    gpu_type = None
    memory_mb = 2048
    timeout_sec = 120
    pip_packages = ["easyocr", "Pillow", "numpy"]
    
    def setup(self) -> None:
        """
        One-time initialization to load the EasyOCR reader.
        
        Creates an EasyOCR reader instance with the default language.
        The reader is cached for subsequent uses.
        """
        import easyocr
        
        self._readers = {}
        # Pre-load English reader
        self._readers['en'] = easyocr.Reader(['en'], gpu=False)
        
        self._initialized = True
    
    def _get_reader(self, language: str):
        """Get or create an EasyOCR reader for the specified language."""
        import easyocr
        
        if language not in self._readers:
            self._readers[language] = easyocr.Reader([language], gpu=False)
        return self._readers[language]
    
    def run(self, **kwargs) -> Dict[str, Any]:
        """
        Execute OCR on the provided image.
        
        Args:
            image: Path to image file or image data
            language: Language code for OCR (optional, defaults to 'en')
            
        Returns:
            Dict containing:
                - text: Full extracted text
                - confidence: Average confidence score (0.0 to 1.0)
                - bounding_boxes: List of detected regions with coordinates
        """
        # Validate inputs
        self.validate_inputs(**kwargs)
        
        # Initialize if needed
        if not self._initialized:
            self.setup()
        
        import numpy as np
        from PIL import Image
        
        # Extract parameters
        image = kwargs['image']
        language = kwargs.get('language', 'en')
        
        # Load image
        if isinstance(image, str):
            pil_image = Image.open(image).convert('RGB')
        else:
            if hasattr(image, 'convert'):
                pil_image = image.convert('RGB')
            else:
                pil_image = Image.fromarray(image).convert('RGB')
        
        image_array = np.array(pil_image)
        
        # Get reader for the specified language
        reader = self._get_reader(language)
        
        # Run OCR
        results = reader.readtext(image_array)
        
        # Process results
        bounding_boxes = []
        text_parts = []
        confidences = []
        
        for bbox, text_val, conf in results:
            # bbox is a list of 4 corner points [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
            bounding_boxes.append({
                'text': text_val,
                'confidence': float(conf),
                'bbox': [[int(coord) for coord in point] for point in bbox]
            })
            text_parts.append(text_val)
            confidences.append(float(conf))
        
        # Calculate average confidence
        avg_confidence = float(np.mean(confidences)) if confidences else 0.0
        
        # Join all text
        full_text = ' '.join(text_parts)
        
        return {
            'text': full_text,
            'confidence': avg_confidence,
            'bounding_boxes': bounding_boxes
        }
