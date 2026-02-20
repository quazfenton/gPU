"""
Face Detection Template for detecting and analyzing faces in images.

This template uses facenet-pytorch (MTCNN) to detect faces in images,
returning bounding boxes, facial landmarks, and an annotated image.
"""

from typing import Any, Dict
from templates.base import Template, InputField, OutputField, RouteType


class FaceDetectionTemplate(Template):
    """
    Detects faces in images with bounding boxes and landmarks.
    
    Uses MTCNN from facenet-pytorch for robust face detection. Returns
    bounding box coordinates for each detected face and optionally
    facial landmark positions. Can also produce an annotated image
    with detected faces highlighted.
    """
    
    name = "face-detection"
    category = "Vision"
    description = "Detect faces in images with bounding boxes and optional landmarks"
    version = "1.0.0"
    
    inputs = [
        InputField(
            name="image",
            type="image",
            description="Image file to analyze for faces",
            required=True
        ),
        InputField(
            name="return_landmarks",
            type="text",
            description="Whether to return facial landmarks",
            required=False,
            default="false",
            options=["true", "false"]
        )
    ]
    
    outputs = [
        OutputField(
            name="faces",
            type="json",
            description="List of detected face bounding boxes and optional landmarks"
        ),
        OutputField(
            name="count",
            type="number",
            description="Number of faces detected"
        ),
        OutputField(
            name="annotated_image",
            type="image",
            description="Image with detected faces highlighted"
        )
    ]
    
    routing = [RouteType.LOCAL, RouteType.MODAL]
    gpu_required = True
    gpu_type = "T4"
    memory_mb = 4096
    timeout_sec = 120
    pip_packages = ["facenet-pytorch", "torch", "Pillow"]
    
    def setup(self) -> None:
        """
        One-time initialization to load the MTCNN face detection model.
        
        Downloads and loads the MTCNN model from facenet-pytorch.
        The model is cached for subsequent uses.
        """
        from facenet_pytorch import MTCNN
        import torch
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.detector = MTCNN(keep_all=True, device=device)
        self._initialized = True
    
    def run(self, **kwargs) -> Dict[str, Any]:
        """
        Execute face detection on the provided image.
        
        Args:
            image: Path to image file or image data
            return_landmarks: Whether to return facial landmarks (optional, defaults to 'false')
            
        Returns:
            Dict containing:
                - faces: List of dicts with bounding boxes and optional landmarks
                - count: Number of detected faces
                - annotated_image: Image with face bounding boxes drawn
        """
        # Validate inputs
        self.validate_inputs(**kwargs)
        
        from PIL import Image, ImageDraw
        import numpy as np
        
        # Initialize if needed
        if not self._initialized:
            self.setup()
        
        # Extract parameters
        image = kwargs['image']
        return_landmarks = kwargs.get('return_landmarks', 'false') == 'true'
        
        # Load image if it's a path
        if isinstance(image, str):
            img = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            img = Image.fromarray(image)
        else:
            img = image
        
        # Detect faces
        boxes, probs, landmarks = self.detector.detect(img, landmarks=True)
        
        # Build face list
        faces = []
        if boxes is not None:
            for i, (box, prob) in enumerate(zip(boxes, probs)):
                face_info = {
                    'bbox': [float(x) for x in box.tolist()],
                    'confidence': float(prob)
                }
                if return_landmarks and landmarks is not None:
                    face_info['landmarks'] = {
                        'left_eye': [float(x) for x in landmarks[i][0].tolist()],
                        'right_eye': [float(x) for x in landmarks[i][1].tolist()],
                        'nose': [float(x) for x in landmarks[i][2].tolist()],
                        'mouth_left': [float(x) for x in landmarks[i][3].tolist()],
                        'mouth_right': [float(x) for x in landmarks[i][4].tolist()]
                    }
                faces.append(face_info)
        
        # Create annotated image
        annotated = img.copy()
        draw = ImageDraw.Draw(annotated)
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = [float(x) for x in box.tolist()]
                draw.rectangle([x1, y1, x2, y2], outline='green', width=2)
        
        return {
            'faces': faces,
            'count': len(faces),
            'annotated_image': annotated
        }
