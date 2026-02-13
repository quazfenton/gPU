"""
Object Detection Template for detecting objects in images.

This template uses YOLO (You Only Look Once) models to detect objects in images
with bounding boxes and confidence scores. Supports multiple YOLO model sizes
for different accuracy/speed tradeoffs.
"""

from typing import Any, Dict
from templates.base import Template, InputField, OutputField, RouteType


class ObjectDetectionTemplate(Template):
    """
    Detects objects in images with bounding boxes.
    
    Uses YOLO models for real-time object detection. Returns detected objects
    with bounding boxes, class labels, and confidence scores. Can also return
    an annotated image with bounding boxes drawn.
    """
    
    name = "object-detection"
    category = "Vision"
    description = "Detect objects in images using YOLO or similar models"
    version = "1.0.0"
    
    inputs = [
        InputField(
            name="image",
            type="image",
            description="Image file to analyze",
            required=True
        ),
        InputField(
            name="confidence_threshold",
            type="number",
            description="Minimum confidence score (0.0 to 1.0)",
            required=False,
            default=0.5
        ),
        InputField(
            name="model",
            type="text",
            description="Detection model to use",
            required=False,
            default="yolov8n",
            options=["yolov8n", "yolov8s", "yolov8m", "yolov8l"]
        )
    ]
    
    outputs = [
        OutputField(
            name="detections",
            type="json",
            description="List of detected objects with bounding boxes and confidence"
        ),
        OutputField(
            name="annotated_image",
            type="image",
            description="Image with bounding boxes drawn"
        )
    ]
    
    routing = [RouteType.LOCAL, RouteType.MODAL, RouteType.HF]
    gpu_required = True
    gpu_type = "T4"
    memory_mb = 4096
    timeout_sec = 300
    pip_packages = ["ultralytics", "torch", "opencv-python"]
    
    def setup(self) -> None:
        """
        One-time initialization to load the YOLO model.
        
        Downloads and loads the specified YOLO model. The model is cached
        for subsequent uses.
        """
        from ultralytics import YOLO
        
        # Default to yolov8n model if not specified
        model_name = getattr(self, '_model_name', 'yolov8n')
        self.model = YOLO(f'{model_name}.pt')
        self._initialized = True
    
    def run(self, **kwargs) -> Dict[str, Any]:
        """
        Execute object detection on the provided image.
        
        Args:
            image: Path to image file or image data
            confidence_threshold: Minimum confidence score (optional, defaults to 0.5)
            model: Model name to use (optional, defaults to 'yolov8n')
            
        Returns:
            Dict containing:
                - detections: List of detected objects with bounding boxes, labels, and confidence
                - annotated_image: Image with bounding boxes drawn
        """
        # Validate inputs
        self.validate_inputs(**kwargs)
        
        import cv2
        import numpy as np
        
        # Initialize if needed
        if not self._initialized:
            # Store model name for setup
            self._model_name = kwargs.get('model', 'yolov8n')
            self.setup()
        
        # Extract parameters
        image = kwargs['image']
        confidence_threshold = kwargs.get('confidence_threshold', 0.5)
        
        # Load image if it's a path
        if isinstance(image, str):
            img = cv2.imread(image)
        else:
            img = image
        
        # Run detection
        results = self.model(img, conf=confidence_threshold)
        
        # Extract detections
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                
                # Get confidence and class
                confidence = float(box.conf[0].cpu().numpy())
                class_id = int(box.cls[0].cpu().numpy())
                class_name = result.names[class_id]
                
                detections.append({
                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                    'confidence': confidence,
                    'class_id': class_id,
                    'class_name': class_name
                })
        
        # Get annotated image
        annotated_img = results[0].plot() if results else img
        
        return {
            'detections': detections,
            'annotated_image': annotated_img
        }
