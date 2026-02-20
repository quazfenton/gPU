"""
Video Processing Template for frame-level video analysis.

This template processes video files frame by frame, supporting various processing
types including object tracking, scene detection, and frame extraction. Designed
for video analysis tasks that require temporal understanding.
"""

from typing import Any, Dict, List
from templates.base import Template, InputField, OutputField, RouteType


class VideoProcessingTemplate(Template):
    """
    Processes video files frame by frame.
    
    Supports multiple processing types:
    - object_tracking: Track objects across video frames
    - scene_detection: Detect scene changes in video
    - frame_extraction: Extract frames at specified intervals
    
    Returns per-frame analysis results and optionally a processed video file.
    """
    
    name = "video-processing"
    category = "Vision"
    description = "Process video files with frame-level analysis"
    version = "1.0.0"
    
    inputs = [
        InputField(
            name="video",
            type="video",
            description="Video file to process",
            required=True
        ),
        InputField(
            name="processing_type",
            type="text",
            description="Type of processing (object_tracking, scene_detection, frame_extraction)",
            required=True,
            options=["object_tracking", "scene_detection", "frame_extraction"]
        ),
        InputField(
            name="frame_rate",
            type="number",
            description="Frames per second to process",
            required=False,
            default=1.0
        )
    ]
    
    outputs = [
        OutputField(
            name="results",
            type="json",
            description="Processing results per frame"
        ),
        OutputField(
            name="processed_video",
            type="video",
            description="Processed video file (if applicable)"
        )
    ]
    
    routing = [RouteType.LOCAL, RouteType.MODAL]
    gpu_required = True
    gpu_type = "A10G"
    memory_mb = 8192
    timeout_sec = 1800
    pip_packages = ["opencv-python", "torch", "torchvision"]
    
    def setup(self) -> None:
        """
        One-time initialization to load video processing models.
        
        Loads necessary models based on the processing type. For object tracking,
        loads a tracking model. For scene detection, loads scene detection models.
        """
        import cv2
        
        # Store OpenCV for later use
        self.cv2 = cv2
        
        # Initialize tracking model if needed
        processing_type = getattr(self, '_processing_type', None)
        if processing_type == 'object_tracking':
            # Initialize tracker (using CSRT tracker as default)
            self.tracker_type = cv2.TrackerCSRT_create
        
        self._initialized = True
    
    def run(self, **kwargs) -> Dict[str, Any]:
        """
        Execute video processing on the provided video file.
        
        Args:
            video: Path to video file
            processing_type: Type of processing to perform
            frame_rate: Frames per second to process (optional, defaults to 1.0)
            
        Returns:
            Dict containing:
                - results: List of processing results per frame
                - processed_video: Path to processed video (if applicable)
        """
        # Validate inputs
        self.validate_inputs(**kwargs)
        
        import cv2
        import numpy as np
        
        # Initialize if needed
        if not self._initialized:
            self._processing_type = kwargs['processing_type']
            self.setup()
        
        # Extract parameters
        video_path = kwargs['video']
        processing_type = kwargs['processing_type']
        frame_rate = kwargs.get('frame_rate', 1.0)
        
        # Open video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Calculate frame skip based on desired frame rate
        frame_skip = max(1, int(fps / frame_rate))
        
        # Process video based on type
        if processing_type == 'frame_extraction':
            results = self._extract_frames(cap, frame_skip, total_frames)
        elif processing_type == 'scene_detection':
            results = self._detect_scenes(cap, frame_skip, total_frames)
        elif processing_type == 'object_tracking':
            results = self._track_objects(cap, frame_skip, total_frames)
        else:
            raise ValueError(f"Unknown processing type: {processing_type}")
        
        cap.release()
        
        return {
            'results': results,
            'processed_video': None  # Could save processed video here if needed
        }
    
    def _extract_frames(self, cap, frame_skip: int, total_frames: int) -> List[Dict[str, Any]]:
        """Extract frames at specified intervals."""
        import cv2
        
        results = []
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % frame_skip == 0:
                # Store frame information
                results.append({
                    'frame_number': frame_idx,
                    'timestamp': frame_idx / cap.get(cv2.CAP_PROP_FPS),
                    'extracted': True
                })
            
            frame_idx += 1
        
        return results
    
    def _detect_scenes(self, cap, frame_skip: int, total_frames: int) -> List[Dict[str, Any]]:
        """Detect scene changes in video."""
        import cv2
        import numpy as np
        
        results = []
        frame_idx = 0
        prev_frame = None
        scene_threshold = 30.0  # Threshold for scene change detection
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % frame_skip == 0:
                # Convert to grayscale for comparison
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                if prev_frame is not None:
                    # Calculate frame difference
                    diff = cv2.absdiff(prev_frame, gray)
                    mean_diff = np.mean(diff)
                    
                    # Detect scene change
                    is_scene_change = mean_diff > scene_threshold
                    
                    results.append({
                        'frame_number': frame_idx,
                        'timestamp': frame_idx / cap.get(cv2.CAP_PROP_FPS),
                        'scene_change': is_scene_change,
                        'difference_score': float(mean_diff)
                    })
                
                prev_frame = gray
            
            frame_idx += 1
        
        return results
    
    def _track_objects(self, cap, frame_skip: int, total_frames: int) -> List[Dict[str, Any]]:
        """Track objects across video frames."""
        import cv2
        
        results = []
        frame_idx = 0
        tracker = None
        tracking_initialized = False
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % frame_skip == 0:
                if not tracking_initialized:
                    # Initialize tracker with first frame
                    # For simplicity, track the center region
                    h, w = frame.shape[:2]
                    bbox = (w // 4, h // 4, w // 2, h // 2)
                    
                    tracker = self.tracker_type()
                    tracker.init(frame, bbox)
                    tracking_initialized = True
                    
                    results.append({
                        'frame_number': frame_idx,
                        'timestamp': frame_idx / cap.get(cv2.CAP_PROP_FPS),
                        'tracking_status': 'initialized',
                        'bbox': list(bbox)
                    })
                else:
                    # Update tracker
                    success, bbox = tracker.update(frame)
                    
                    results.append({
                        'frame_number': frame_idx,
                        'timestamp': frame_idx / cap.get(cv2.CAP_PROP_FPS),
                        'tracking_status': 'success' if success else 'lost',
                        'bbox': [float(x) for x in bbox] if success else None
                    })
            
            frame_idx += 1
        
        return results
