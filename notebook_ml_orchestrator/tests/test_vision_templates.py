"""
Unit tests for Vision Processing Templates.

Tests cover:
- ObjectDetectionTemplate instantiation and metadata
- Input validation
- Setup and initialization
- Run method with valid inputs
- Error handling
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from templates.object_detection_template import ObjectDetectionTemplate
from templates.base import RouteType


class TestObjectDetectionTemplate:
    """Test suite for ObjectDetectionTemplate."""
    
    def test_template_metadata(self):
        """Test that template has correct metadata."""
        template = ObjectDetectionTemplate()
        
        assert template.name == "object-detection"
        assert template.category == "Vision"
        assert template.description == "Detect objects in images using YOLO or similar models"
        assert template.version == "1.0.0"
    
    def test_input_definitions(self):
        """Test that template has correct input definitions."""
        template = ObjectDetectionTemplate()
        
        assert len(template.inputs) == 3
        
        # Check image input
        image_input = next(i for i in template.inputs if i.name == "image")
        assert image_input.type == "image"
        assert image_input.required is True
        assert image_input.description == "Image file to analyze"
        
        # Check confidence_threshold input
        conf_input = next(i for i in template.inputs if i.name == "confidence_threshold")
        assert conf_input.type == "number"
        assert conf_input.required is False
        assert conf_input.default == 0.5
        
        # Check model input
        model_input = next(i for i in template.inputs if i.name == "model")
        assert model_input.type == "text"
        assert model_input.required is False
        assert model_input.default == "yolov8n"
        assert model_input.options == ["yolov8n", "yolov8s", "yolov8m", "yolov8l"]
    
    def test_output_definitions(self):
        """Test that template has correct output definitions."""
        template = ObjectDetectionTemplate()
        
        assert len(template.outputs) == 2
        
        # Check detections output
        detections_output = next(o for o in template.outputs if o.name == "detections")
        assert detections_output.type == "json"
        assert detections_output.description == "List of detected objects with bounding boxes and confidence"
        
        # Check annotated_image output
        image_output = next(o for o in template.outputs if o.name == "annotated_image")
        assert image_output.type == "image"
        assert image_output.description == "Image with bounding boxes drawn"
    
    def test_resource_requirements(self):
        """Test that template has correct resource requirements."""
        template = ObjectDetectionTemplate()
        
        assert template.gpu_required is True
        assert template.gpu_type == "T4"
        assert template.memory_mb == 4096
        assert template.timeout_sec == 300
    
    def test_routing_support(self):
        """Test that template supports correct backends."""
        template = ObjectDetectionTemplate()
        
        assert RouteType.LOCAL in template.routing
        assert RouteType.MODAL in template.routing
        assert RouteType.HF in template.routing
    
    def test_pip_packages(self):
        """Test that template declares correct dependencies."""
        template = ObjectDetectionTemplate()
        
        assert "ultralytics" in template.pip_packages
        assert "torch" in template.pip_packages
        assert "opencv-python" in template.pip_packages
    
    def test_validate_inputs_valid(self):
        """Test input validation with valid inputs."""
        template = ObjectDetectionTemplate()
        
        # Should not raise with required image input
        result = template.validate_inputs(image="test.jpg")
        assert result is True
        
        # Should not raise with all inputs
        result = template.validate_inputs(
            image="test.jpg",
            confidence_threshold=0.7,
            model="yolov8s"
        )
        assert result is True
    
    def test_validate_inputs_missing_required(self):
        """Test input validation fails when required input is missing."""
        template = ObjectDetectionTemplate()
        
        # Should raise ValueError when image is missing
        with pytest.raises(ValueError, match="Missing required input: image"):
            template.validate_inputs(confidence_threshold=0.5)
    
    def test_setup(self):
        """Test setup method loads YOLO model."""
        template = ObjectDetectionTemplate()
        template._model_name = 'yolov8n'
        
        # Mock ultralytics module at import time
        with patch.dict('sys.modules', {'ultralytics': MagicMock()}):
            import sys
            mock_ultralytics = sys.modules['ultralytics']
            mock_yolo_class = Mock()
            mock_model = Mock()
            mock_yolo_class.return_value = mock_model
            mock_ultralytics.YOLO = mock_yolo_class
            
            template.setup()
            
            mock_yolo_class.assert_called_once_with('yolov8n.pt')
            assert template.model == mock_model
            assert template._initialized is True
    
    def test_run_with_valid_inputs(self):
        """Test run method with valid inputs."""
        template = ObjectDetectionTemplate()
        
        # Create mock image
        mock_image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Mock the YOLO model and cv2
        with patch.dict('sys.modules', {'ultralytics': MagicMock(), 'cv2': MagicMock()}):
            import sys
            mock_ultralytics = sys.modules['ultralytics']
            mock_cv2 = sys.modules['cv2']
            
            # Mock YOLO model
            mock_yolo_class = Mock()
            mock_model = Mock()
            mock_ultralytics.YOLO = mock_yolo_class
            mock_yolo_class.return_value = mock_model
            
            # Mock detection results
            mock_box = Mock()
            mock_box.xyxy = [Mock()]
            mock_box.xyxy[0].cpu.return_value.numpy.return_value = np.array([10, 20, 100, 200])
            mock_box.conf = [Mock()]
            mock_box.conf[0].cpu.return_value.numpy.return_value = 0.85
            mock_box.cls = [Mock()]
            mock_box.cls[0].cpu.return_value.numpy.return_value = 0
            
            mock_result = Mock()
            mock_result.boxes = [mock_box]
            mock_result.names = {0: 'person'}
            mock_result.plot.return_value = mock_image
            
            mock_model.return_value = [mock_result]
            
            # Mock cv2.imread
            mock_cv2.imread.return_value = mock_image
            
            # Run the template
            result = template.run(image="test.jpg", confidence_threshold=0.5, model="yolov8n")
            
            # Verify results
            assert 'detections' in result
            assert 'annotated_image' in result
            assert len(result['detections']) == 1
            
            detection = result['detections'][0]
            assert detection['class_name'] == 'person'
            assert detection['confidence'] == 0.85
            assert detection['class_id'] == 0
            assert len(detection['bbox']) == 4
    
    def test_run_with_image_array(self):
        """Test run method with numpy array input."""
        template = ObjectDetectionTemplate()
        
        # Create mock image
        mock_image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Mock the YOLO model
        with patch.dict('sys.modules', {'ultralytics': MagicMock(), 'cv2': MagicMock()}):
            import sys
            mock_ultralytics = sys.modules['ultralytics']
            
            # Mock YOLO model
            mock_yolo_class = Mock()
            mock_model = Mock()
            mock_ultralytics.YOLO = mock_yolo_class
            mock_yolo_class.return_value = mock_model
            
            # Mock empty detection results
            mock_result = Mock()
            mock_result.boxes = []
            mock_result.plot.return_value = mock_image
            
            mock_model.return_value = [mock_result]
            
            # Run with numpy array (not a path)
            result = template.run(image=mock_image)
            
            # Verify model was called with the array directly
            mock_model.assert_called_once()
            call_args = mock_model.call_args[0]
            assert isinstance(call_args[0], np.ndarray)
    
    def test_run_with_custom_confidence(self):
        """Test run method with custom confidence threshold."""
        template = ObjectDetectionTemplate()
        
        # Create mock image
        mock_image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Mock the YOLO model and cv2
        with patch.dict('sys.modules', {'ultralytics': MagicMock(), 'cv2': MagicMock()}):
            import sys
            mock_ultralytics = sys.modules['ultralytics']
            mock_cv2 = sys.modules['cv2']
            
            # Mock YOLO model
            mock_yolo_class = Mock()
            mock_model = Mock()
            mock_ultralytics.YOLO = mock_yolo_class
            mock_yolo_class.return_value = mock_model
            
            # Mock empty detection results
            mock_result = Mock()
            mock_result.boxes = []
            mock_result.plot.return_value = mock_image
            
            mock_model.return_value = [mock_result]
            mock_cv2.imread.return_value = mock_image
            
            # Run with custom confidence
            result = template.run(image="test.jpg", confidence_threshold=0.8)
            
            # Verify model was called with correct confidence
            mock_model.assert_called_once()
            call_kwargs = mock_model.call_args[1]
            assert call_kwargs['conf'] == 0.8
    
    def test_run_initializes_if_needed(self):
        """Test that run calls setup if not initialized."""
        template = ObjectDetectionTemplate()
        
        # Create mock image
        mock_image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Mock the YOLO model and cv2
        with patch.dict('sys.modules', {'ultralytics': MagicMock(), 'cv2': MagicMock()}):
            import sys
            mock_ultralytics = sys.modules['ultralytics']
            mock_cv2 = sys.modules['cv2']
            
            # Mock YOLO model
            mock_yolo_class = Mock()
            mock_model = Mock()
            mock_ultralytics.YOLO = mock_yolo_class
            mock_yolo_class.return_value = mock_model
            
            # Mock empty detection results
            mock_result = Mock()
            mock_result.boxes = []
            mock_result.plot.return_value = mock_image
            
            mock_model.return_value = [mock_result]
            mock_cv2.imread.return_value = mock_image
            
            # Ensure not initialized
            assert template._initialized is False
            
            # Run should call setup
            template.run(image="test.jpg")
            
            # Verify setup was called
            mock_yolo_class.assert_called_once()
            assert template._initialized is True
    
    def test_run_with_missing_image(self):
        """Test run method fails with missing image input."""
        template = ObjectDetectionTemplate()
        
        # Should raise ValueError for missing required input
        with pytest.raises(ValueError, match="Missing required input: image"):
            template.run(confidence_threshold=0.5)
    
    def test_run_with_multiple_detections(self):
        """Test run method with multiple detected objects."""
        template = ObjectDetectionTemplate()
        
        # Create mock image
        mock_image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Mock the YOLO model and cv2
        with patch.dict('sys.modules', {'ultralytics': MagicMock(), 'cv2': MagicMock()}):
            import sys
            mock_ultralytics = sys.modules['ultralytics']
            mock_cv2 = sys.modules['cv2']
            
            # Mock YOLO model
            mock_yolo_class = Mock()
            mock_model = Mock()
            mock_ultralytics.YOLO = mock_yolo_class
            mock_yolo_class.return_value = mock_model
            
            # Mock multiple detection results
            mock_box1 = Mock()
            mock_box1.xyxy = [Mock()]
            mock_box1.xyxy[0].cpu.return_value.numpy.return_value = np.array([10, 20, 100, 200])
            mock_box1.conf = [Mock()]
            mock_box1.conf[0].cpu.return_value.numpy.return_value = 0.85
            mock_box1.cls = [Mock()]
            mock_box1.cls[0].cpu.return_value.numpy.return_value = 0
            
            mock_box2 = Mock()
            mock_box2.xyxy = [Mock()]
            mock_box2.xyxy[0].cpu.return_value.numpy.return_value = np.array([200, 150, 350, 400])
            mock_box2.conf = [Mock()]
            mock_box2.conf[0].cpu.return_value.numpy.return_value = 0.92
            mock_box2.cls = [Mock()]
            mock_box2.cls[0].cpu.return_value.numpy.return_value = 1
            
            mock_result = Mock()
            mock_result.boxes = [mock_box1, mock_box2]
            mock_result.names = {0: 'person', 1: 'car'}
            mock_result.plot.return_value = mock_image
            
            mock_model.return_value = [mock_result]
            mock_cv2.imread.return_value = mock_image
            
            # Run the template
            result = template.run(image="test.jpg")
            
            # Verify results
            assert len(result['detections']) == 2
            
            # Check first detection
            assert result['detections'][0]['class_name'] == 'person'
            assert result['detections'][0]['confidence'] == 0.85
            
            # Check second detection
            assert result['detections'][1]['class_name'] == 'car'
            assert result['detections'][1]['confidence'] == 0.92
    
    def test_run_with_no_detections(self):
        """Test run method when no objects are detected."""
        template = ObjectDetectionTemplate()
        
        # Create mock image
        mock_image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Mock the YOLO model and cv2
        with patch.dict('sys.modules', {'ultralytics': MagicMock(), 'cv2': MagicMock()}):
            import sys
            mock_ultralytics = sys.modules['ultralytics']
            mock_cv2 = sys.modules['cv2']
            
            # Mock YOLO model
            mock_yolo_class = Mock()
            mock_model = Mock()
            mock_ultralytics.YOLO = mock_yolo_class
            mock_yolo_class.return_value = mock_model
            
            # Mock empty detection results
            mock_result = Mock()
            mock_result.boxes = []
            mock_result.plot.return_value = mock_image
            
            mock_model.return_value = [mock_result]
            mock_cv2.imread.return_value = mock_image
            
            # Run the template
            result = template.run(image="test.jpg")
            
            # Verify empty detections
            assert result['detections'] == []
            assert 'annotated_image' in result
    
    def test_to_dict_serialization(self):
        """Test that template can be serialized to dict."""
        template = ObjectDetectionTemplate()
        
        metadata = template.to_dict()
        
        assert metadata['name'] == "object-detection"
        assert metadata['category'] == "Vision"
        assert metadata['gpu_required'] is True
        assert metadata['gpu_type'] == "T4"
        assert metadata['memory_mb'] == 4096
        assert metadata['timeout_sec'] == 300
        assert len(metadata['inputs']) == 3
        assert len(metadata['outputs']) == 2
        assert 'ultralytics' in metadata['pip_packages']


class TestImageSegmentationTemplate:
    """Test suite for ImageSegmentationTemplate."""
    
    def test_template_metadata(self):
        """Test that template has correct metadata."""
        from templates.image_segmentation_template import ImageSegmentationTemplate
        template = ImageSegmentationTemplate()
        
        assert template.name == "image-segmentation"
        assert template.category == "Vision"
        assert template.description == "Segment images into regions using segmentation models"
        assert template.version == "1.0.0"
    
    def test_input_definitions(self):
        """Test that template has correct input definitions."""
        from templates.image_segmentation_template import ImageSegmentationTemplate
        template = ImageSegmentationTemplate()
        
        assert len(template.inputs) == 2
        
        # Check image input
        image_input = next(i for i in template.inputs if i.name == "image")
        assert image_input.type == "image"
        assert image_input.required is True
        assert image_input.description == "Image file to segment"
        
        # Check segmentation_type input
        seg_type_input = next(i for i in template.inputs if i.name == "segmentation_type")
        assert seg_type_input.type == "text"
        assert seg_type_input.required is False
        assert seg_type_input.default == "semantic"
        assert seg_type_input.options == ["semantic", "instance"]
    
    def test_output_definitions(self):
        """Test that template has correct output definitions."""
        from templates.image_segmentation_template import ImageSegmentationTemplate
        template = ImageSegmentationTemplate()
        
        assert len(template.outputs) == 2
        
        # Check mask output
        mask_output = next(o for o in template.outputs if o.name == "mask")
        assert mask_output.type == "image"
        assert mask_output.description == "Segmentation mask"
        
        # Check segments output
        segments_output = next(o for o in template.outputs if o.name == "segments")
        assert segments_output.type == "json"
        assert segments_output.description == "Segment information with labels and areas"
    
    def test_resource_requirements(self):
        """Test that template has correct resource requirements."""
        from templates.image_segmentation_template import ImageSegmentationTemplate
        template = ImageSegmentationTemplate()
        
        assert template.gpu_required is True
        assert template.gpu_type == "T4"
        assert template.memory_mb == 6144
        assert template.timeout_sec == 300
    
    def test_routing_support(self):
        """Test that template supports correct backends."""
        from templates.image_segmentation_template import ImageSegmentationTemplate
        template = ImageSegmentationTemplate()
        
        assert RouteType.LOCAL in template.routing
        assert RouteType.MODAL in template.routing
    
    def test_pip_packages(self):
        """Test that template declares correct dependencies."""
        from templates.image_segmentation_template import ImageSegmentationTemplate
        template = ImageSegmentationTemplate()
        
        assert "transformers" in template.pip_packages
        assert "torch" in template.pip_packages
        assert "pillow" in template.pip_packages
    
    def test_validate_inputs_valid(self):
        """Test input validation with valid inputs."""
        from templates.image_segmentation_template import ImageSegmentationTemplate
        template = ImageSegmentationTemplate()
        
        # Should not raise with required image input
        result = template.validate_inputs(image="test.jpg")
        assert result is True
        
        # Should not raise with all inputs
        result = template.validate_inputs(
            image="test.jpg",
            segmentation_type="semantic"
        )
        assert result is True
    
    def test_validate_inputs_missing_required(self):
        """Test input validation fails when required input is missing."""
        from templates.image_segmentation_template import ImageSegmentationTemplate
        template = ImageSegmentationTemplate()
        
        # Should raise ValueError when image is missing
        with pytest.raises(ValueError, match="Missing required input: image"):
            template.validate_inputs(segmentation_type="semantic")
    
    def test_setup(self):
        """Test setup method loads segmentation model."""
        from templates.image_segmentation_template import ImageSegmentationTemplate
        template = ImageSegmentationTemplate()
        
        # Mock transformers module
        with patch.dict('sys.modules', {'transformers': MagicMock()}):
            import sys
            mock_transformers = sys.modules['transformers']
            mock_processor = Mock()
            mock_model = Mock()
            mock_transformers.AutoImageProcessor.from_pretrained.return_value = mock_processor
            mock_transformers.AutoModelForSemanticSegmentation.from_pretrained.return_value = mock_model
            
            template.setup()
            
            mock_transformers.AutoImageProcessor.from_pretrained.assert_called_once()
            mock_transformers.AutoModelForSemanticSegmentation.from_pretrained.assert_called_once()
            assert template.processor == mock_processor
            assert template.model == mock_model
            assert template._initialized is True
    
    def test_run_with_valid_inputs(self):
        """Test run method with valid inputs."""
        from templates.image_segmentation_template import ImageSegmentationTemplate
        template = ImageSegmentationTemplate()
        
        # Create mock image
        mock_image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Mock the transformers, PIL, and torch
        with patch.dict('sys.modules', {'transformers': MagicMock(), 'PIL': MagicMock(), 'PIL.Image': MagicMock(), 'torch': MagicMock()}):
            import sys
            mock_transformers = sys.modules['transformers']
            mock_pil = sys.modules['PIL']
            mock_image_module = sys.modules['PIL.Image']
            mock_torch = sys.modules['torch']
            
            # Mock processor and model
            mock_processor = Mock()
            mock_model = Mock()
            mock_transformers.AutoImageProcessor.from_pretrained.return_value = mock_processor
            mock_transformers.AutoModelForSemanticSegmentation.from_pretrained.return_value = mock_model
            
            # Mock processor output
            mock_processor.return_value = {'pixel_values': Mock()}
            
            # Mock model output
            mock_logits = Mock()
            mock_logits.argmax.return_value = [Mock()]
            mock_logits.argmax.return_value[0].cpu.return_value.numpy.return_value = np.zeros((480, 640), dtype=np.int32)
            
            mock_output = Mock()
            mock_output.logits = mock_logits
            mock_model.return_value = mock_output
            mock_model.config.id2label = {0: 'background'}
            
            # Mock torch.no_grad
            mock_torch.no_grad.return_value.__enter__ = Mock()
            mock_torch.no_grad.return_value.__exit__ = Mock()
            
            # Mock PIL Image
            mock_pil_image = Mock()
            mock_image_module.open.return_value = mock_pil_image
            mock_pil_image.convert.return_value = mock_pil_image
            mock_image_module.fromarray.return_value = Mock()
            
            # Run the template
            result = template.run(image="test.jpg", segmentation_type="semantic")
            
            # Verify results
            assert 'mask' in result
            assert 'segments' in result
            assert isinstance(result['segments'], list)
    
    def test_run_with_numpy_array(self):
        """Test run method with numpy array input."""
        from templates.image_segmentation_template import ImageSegmentationTemplate
        template = ImageSegmentationTemplate()
        
        # Create mock image
        mock_image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Mock the transformers, PIL, and torch
        with patch.dict('sys.modules', {'transformers': MagicMock(), 'PIL': MagicMock(), 'PIL.Image': MagicMock(), 'torch': MagicMock()}):
            import sys
            mock_transformers = sys.modules['transformers']
            mock_pil = sys.modules['PIL']
            mock_image_module = sys.modules['PIL.Image']
            mock_torch = sys.modules['torch']
            
            # Mock processor and model
            mock_processor = Mock()
            mock_model = Mock()
            mock_transformers.AutoImageProcessor.from_pretrained.return_value = mock_processor
            mock_transformers.AutoModelForSemanticSegmentation.from_pretrained.return_value = mock_model
            
            # Mock processor output
            mock_processor.return_value = {'pixel_values': Mock()}
            
            # Mock model output
            mock_logits = Mock()
            mock_logits.argmax.return_value = [Mock()]
            mock_logits.argmax.return_value[0].cpu.return_value.numpy.return_value = np.zeros((480, 640), dtype=np.int32)
            
            mock_output = Mock()
            mock_output.logits = mock_logits
            mock_model.return_value = mock_output
            mock_model.config.id2label = {0: 'background'}
            
            # Mock torch.no_grad
            mock_torch.no_grad.return_value.__enter__ = Mock()
            mock_torch.no_grad.return_value.__exit__ = Mock()
            
            # Mock PIL Image
            mock_pil_image = Mock()
            mock_image_module.fromarray.return_value = mock_pil_image
            
            # Run with numpy array
            result = template.run(image=mock_image)
            
            # Verify model was called
            mock_model.assert_called_once()
    
    def test_run_initializes_if_needed(self):
        """Test that run calls setup if not initialized."""
        from templates.image_segmentation_template import ImageSegmentationTemplate
        template = ImageSegmentationTemplate()
        
        # Create mock image
        mock_image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Mock the transformers, PIL, and torch
        with patch.dict('sys.modules', {'transformers': MagicMock(), 'PIL': MagicMock(), 'PIL.Image': MagicMock(), 'torch': MagicMock()}):
            import sys
            mock_transformers = sys.modules['transformers']
            mock_pil = sys.modules['PIL']
            mock_image_module = sys.modules['PIL.Image']
            mock_torch = sys.modules['torch']
            
            # Mock processor and model
            mock_processor = Mock()
            mock_model = Mock()
            mock_transformers.AutoImageProcessor.from_pretrained.return_value = mock_processor
            mock_transformers.AutoModelForSemanticSegmentation.from_pretrained.return_value = mock_model
            
            # Mock processor output
            mock_processor.return_value = {'pixel_values': Mock()}
            
            # Mock model output
            mock_logits = Mock()
            mock_logits.argmax.return_value = [Mock()]
            mock_logits.argmax.return_value[0].cpu.return_value.numpy.return_value = np.zeros((480, 640), dtype=np.int32)
            
            mock_output = Mock()
            mock_output.logits = mock_logits
            mock_model.return_value = mock_output
            mock_model.config.id2label = {0: 'background'}
            
            # Mock torch.no_grad
            mock_torch.no_grad.return_value.__enter__ = Mock()
            mock_torch.no_grad.return_value.__exit__ = Mock()
            
            # Mock PIL Image
            mock_pil_image = Mock()
            mock_image_module.fromarray.return_value = mock_pil_image
            
            # Ensure not initialized
            assert template._initialized is False
            
            # Run should call setup
            template.run(image=mock_image)
            
            # Verify setup was called
            mock_transformers.AutoImageProcessor.from_pretrained.assert_called_once()
            assert template._initialized is True
    
    def test_run_with_missing_image(self):
        """Test run method fails with missing image input."""
        from templates.image_segmentation_template import ImageSegmentationTemplate
        template = ImageSegmentationTemplate()
        
        # Should raise ValueError for missing required input
        with pytest.raises(ValueError, match="Missing required input: image"):
            template.run(segmentation_type="semantic")
    
    def test_to_dict_serialization(self):
        """Test that template can be serialized to dict."""
        from templates.image_segmentation_template import ImageSegmentationTemplate
        template = ImageSegmentationTemplate()
        
        metadata = template.to_dict()
        
        assert metadata['name'] == "image-segmentation"
        assert metadata['category'] == "Vision"
        assert metadata['gpu_required'] is True
        assert metadata['gpu_type'] == "T4"
        assert metadata['memory_mb'] == 6144
        assert metadata['timeout_sec'] == 300
        assert len(metadata['inputs']) == 2
        assert len(metadata['outputs']) == 2
        assert 'transformers' in metadata['pip_packages']


class TestVideoProcessingTemplate:
    """Test suite for VideoProcessingTemplate."""
    
    def test_template_metadata(self):
        """Test that template has correct metadata."""
        from templates.video_processing_template import VideoProcessingTemplate
        template = VideoProcessingTemplate()
        
        assert template.name == "video-processing"
        assert template.category == "Vision"
        assert template.description == "Process video files with frame-level analysis"
        assert template.version == "1.0.0"
    
    def test_input_definitions(self):
        """Test that template has correct input definitions."""
        from templates.video_processing_template import VideoProcessingTemplate
        template = VideoProcessingTemplate()
        
        assert len(template.inputs) == 3
        
        # Check video input
        video_input = next(i for i in template.inputs if i.name == "video")
        assert video_input.type == "video"
        assert video_input.required is True
        assert video_input.description == "Video file to process"
        
        # Check processing_type input
        proc_type_input = next(i for i in template.inputs if i.name == "processing_type")
        assert proc_type_input.type == "text"
        assert proc_type_input.required is True
        assert proc_type_input.options == ["object_tracking", "scene_detection", "frame_extraction"]
        
        # Check frame_rate input
        frame_rate_input = next(i for i in template.inputs if i.name == "frame_rate")
        assert frame_rate_input.type == "number"
        assert frame_rate_input.required is False
        assert frame_rate_input.default == 1.0
    
    def test_output_definitions(self):
        """Test that template has correct output definitions."""
        from templates.video_processing_template import VideoProcessingTemplate
        template = VideoProcessingTemplate()
        
        assert len(template.outputs) == 2
        
        # Check results output
        results_output = next(o for o in template.outputs if o.name == "results")
        assert results_output.type == "json"
        assert results_output.description == "Processing results per frame"
        
        # Check processed_video output
        video_output = next(o for o in template.outputs if o.name == "processed_video")
        assert video_output.type == "video"
        assert video_output.description == "Processed video file (if applicable)"
    
    def test_resource_requirements(self):
        """Test that template has correct resource requirements."""
        from templates.video_processing_template import VideoProcessingTemplate
        template = VideoProcessingTemplate()
        
        assert template.gpu_required is True
        assert template.gpu_type == "A10G"
        assert template.memory_mb == 8192
        assert template.timeout_sec == 1800
    
    def test_routing_support(self):
        """Test that template supports correct backends."""
        from templates.video_processing_template import VideoProcessingTemplate
        template = VideoProcessingTemplate()
        
        assert RouteType.LOCAL in template.routing
        assert RouteType.MODAL in template.routing
    
    def test_pip_packages(self):
        """Test that template declares correct dependencies."""
        from templates.video_processing_template import VideoProcessingTemplate
        template = VideoProcessingTemplate()
        
        assert "opencv-python" in template.pip_packages
        assert "torch" in template.pip_packages
        assert "torchvision" in template.pip_packages
    
    def test_validate_inputs_valid(self):
        """Test input validation with valid inputs."""
        from templates.video_processing_template import VideoProcessingTemplate
        template = VideoProcessingTemplate()
        
        # Should not raise with required inputs
        result = template.validate_inputs(video="test.mp4", processing_type="frame_extraction")
        assert result is True
        
        # Should not raise with all inputs
        result = template.validate_inputs(
            video="test.mp4",
            processing_type="scene_detection",
            frame_rate=2.0
        )
        assert result is True
    
    def test_validate_inputs_missing_required(self):
        """Test input validation fails when required input is missing."""
        from templates.video_processing_template import VideoProcessingTemplate
        template = VideoProcessingTemplate()
        
        # Should raise ValueError when video is missing
        with pytest.raises(ValueError, match="Missing required input: video"):
            template.validate_inputs(processing_type="frame_extraction")
        
        # Should raise ValueError when processing_type is missing
        with pytest.raises(ValueError, match="Missing required input: processing_type"):
            template.validate_inputs(video="test.mp4")
    
    def test_setup(self):
        """Test setup method initializes correctly."""
        from templates.video_processing_template import VideoProcessingTemplate
        template = VideoProcessingTemplate()
        
        # Mock cv2 module
        with patch.dict('sys.modules', {'cv2': MagicMock()}):
            import sys
            mock_cv2 = sys.modules['cv2']
            mock_tracker = Mock()
            mock_cv2.TrackerCSRT_create = mock_tracker
            
            template._processing_type = 'object_tracking'
            template.setup()
            
            assert template.cv2 == mock_cv2
            assert template.tracker_type == mock_tracker
            assert template._initialized is True
    
    def test_run_frame_extraction(self):
        """Test run method with frame extraction."""
        from templates.video_processing_template import VideoProcessingTemplate
        template = VideoProcessingTemplate()
        
        # Mock cv2 module
        with patch.dict('sys.modules', {'cv2': MagicMock()}):
            import sys
            mock_cv2 = sys.modules['cv2']
            
            # Mock video capture
            mock_cap = Mock()
            mock_cap.isOpened.return_value = True
            mock_cap.get.side_effect = lambda prop: {
                mock_cv2.CAP_PROP_FPS: 30.0,
                mock_cv2.CAP_PROP_FRAME_COUNT: 90,
                mock_cv2.CAP_PROP_FRAME_WIDTH: 640,
                mock_cv2.CAP_PROP_FRAME_HEIGHT: 480
            }.get(prop, 0)
            
            # Mock frame reading - return 3 frames then stop
            frame_count = [0]
            def read_side_effect():
                frame_count[0] += 1
                if frame_count[0] <= 90:
                    return True, np.zeros((480, 640, 3), dtype=np.uint8)
                return False, None
            
            mock_cap.read.side_effect = read_side_effect
            mock_cv2.VideoCapture.return_value = mock_cap
            
            # Run frame extraction
            result = template.run(video="test.mp4", processing_type="frame_extraction", frame_rate=1.0)
            
            # Verify results
            assert 'results' in result
            assert 'processed_video' in result
            assert isinstance(result['results'], list)
            assert len(result['results']) == 3  # 90 frames / 30 fps * 1 fps = 3 frames
            
            # Check first result
            assert result['results'][0]['frame_number'] == 0
            assert result['results'][0]['extracted'] is True
    
    def test_run_scene_detection(self):
        """Test run method with scene detection."""
        from templates.video_processing_template import VideoProcessingTemplate
        template = VideoProcessingTemplate()
        
        # Mock cv2 module
        with patch.dict('sys.modules', {'cv2': MagicMock()}):
            import sys
            mock_cv2 = sys.modules['cv2']
            
            # Mock video capture
            mock_cap = Mock()
            mock_cap.isOpened.return_value = True
            mock_cap.get.side_effect = lambda prop: {
                mock_cv2.CAP_PROP_FPS: 30.0,
                mock_cv2.CAP_PROP_FRAME_COUNT: 60,
                mock_cv2.CAP_PROP_FRAME_WIDTH: 640,
                mock_cv2.CAP_PROP_FRAME_HEIGHT: 480
            }.get(prop, 0)
            
            # Mock frame reading
            frame_count = [0]
            def read_side_effect():
                frame_count[0] += 1
                if frame_count[0] <= 60:
                    return True, np.zeros((480, 640, 3), dtype=np.uint8)
                return False, None
            
            mock_cap.read.side_effect = read_side_effect
            mock_cv2.VideoCapture.return_value = mock_cap
            
            # Mock cv2 functions
            mock_cv2.cvtColor.return_value = np.zeros((480, 640), dtype=np.uint8)
            mock_cv2.absdiff.return_value = np.zeros((480, 640), dtype=np.uint8)
            mock_cv2.COLOR_BGR2GRAY = 6
            
            # Run scene detection
            result = template.run(video="test.mp4", processing_type="scene_detection", frame_rate=1.0)
            
            # Verify results
            assert 'results' in result
            assert isinstance(result['results'], list)
            # First frame has no previous frame to compare, so we get 1 less result
            assert len(result['results']) >= 1
    
    def test_run_object_tracking(self):
        """Test run method with object tracking."""
        from templates.video_processing_template import VideoProcessingTemplate
        template = VideoProcessingTemplate()
        
        # Mock cv2 module
        with patch.dict('sys.modules', {'cv2': MagicMock()}):
            import sys
            mock_cv2 = sys.modules['cv2']
            
            # Mock video capture
            mock_cap = Mock()
            mock_cap.isOpened.return_value = True
            mock_cap.get.side_effect = lambda prop: {
                mock_cv2.CAP_PROP_FPS: 30.0,
                mock_cv2.CAP_PROP_FRAME_COUNT: 60,
                mock_cv2.CAP_PROP_FRAME_WIDTH: 640,
                mock_cv2.CAP_PROP_FRAME_HEIGHT: 480
            }.get(prop, 0)
            
            # Mock frame reading
            frame_count = [0]
            def read_side_effect():
                frame_count[0] += 1
                if frame_count[0] <= 60:
                    return True, np.zeros((480, 640, 3), dtype=np.uint8)
                return False, None
            
            mock_cap.read.side_effect = read_side_effect
            mock_cv2.VideoCapture.return_value = mock_cap
            
            # Mock tracker
            mock_tracker_instance = Mock()
            mock_tracker_instance.init.return_value = True
            mock_tracker_instance.update.return_value = (True, (100, 100, 200, 200))
            mock_tracker = Mock(return_value=mock_tracker_instance)
            mock_cv2.TrackerCSRT_create = mock_tracker
            
            # Run object tracking
            result = template.run(video="test.mp4", processing_type="object_tracking", frame_rate=1.0)
            
            # Verify results
            assert 'results' in result
            assert isinstance(result['results'], list)
            assert len(result['results']) >= 1
            
            # Check first result (initialization)
            assert result['results'][0]['tracking_status'] == 'initialized'
            assert 'bbox' in result['results'][0]
    
    def test_run_with_invalid_video(self):
        """Test run method with invalid video file."""
        from templates.video_processing_template import VideoProcessingTemplate
        template = VideoProcessingTemplate()
        
        # Mock cv2 module
        with patch.dict('sys.modules', {'cv2': MagicMock()}):
            import sys
            mock_cv2 = sys.modules['cv2']
            
            # Mock video capture that fails to open
            mock_cap = Mock()
            mock_cap.isOpened.return_value = False
            mock_cv2.VideoCapture.return_value = mock_cap
            
            # Should raise ValueError for invalid video
            with pytest.raises(ValueError, match="Could not open video file"):
                template.run(video="invalid.mp4", processing_type="frame_extraction")
    
    def test_run_with_invalid_processing_type(self):
        """Test run method with invalid processing type."""
        from templates.video_processing_template import VideoProcessingTemplate
        template = VideoProcessingTemplate()
        
        # Mock cv2 module
        with patch.dict('sys.modules', {'cv2': MagicMock()}):
            import sys
            mock_cv2 = sys.modules['cv2']
            
            # Mock video capture
            mock_cap = Mock()
            mock_cap.isOpened.return_value = True
            mock_cap.get.side_effect = lambda prop: {
                mock_cv2.CAP_PROP_FPS: 30.0,
                mock_cv2.CAP_PROP_FRAME_COUNT: 60,
                mock_cv2.CAP_PROP_FRAME_WIDTH: 640,
                mock_cv2.CAP_PROP_FRAME_HEIGHT: 480
            }.get(prop, 0)
            mock_cv2.VideoCapture.return_value = mock_cap
            
            # Should raise ValueError for invalid processing type
            with pytest.raises(ValueError, match="Unknown processing type"):
                template.run(video="test.mp4", processing_type="invalid_type")
    
    def test_run_initializes_if_needed(self):
        """Test that run calls setup if not initialized."""
        from templates.video_processing_template import VideoProcessingTemplate
        template = VideoProcessingTemplate()
        
        # Mock cv2 module
        with patch.dict('sys.modules', {'cv2': MagicMock()}):
            import sys
            mock_cv2 = sys.modules['cv2']
            
            # Mock video capture
            mock_cap = Mock()
            mock_cap.isOpened.return_value = True
            mock_cap.get.side_effect = lambda prop: {
                mock_cv2.CAP_PROP_FPS: 30.0,
                mock_cv2.CAP_PROP_FRAME_COUNT: 30,
                mock_cv2.CAP_PROP_FRAME_WIDTH: 640,
                mock_cv2.CAP_PROP_FRAME_HEIGHT: 480
            }.get(prop, 0)
            
            # Mock frame reading
            frame_count = [0]
            def read_side_effect():
                frame_count[0] += 1
                if frame_count[0] <= 30:
                    return True, np.zeros((480, 640, 3), dtype=np.uint8)
                return False, None
            
            mock_cap.read.side_effect = read_side_effect
            mock_cv2.VideoCapture.return_value = mock_cap
            
            # Ensure not initialized
            assert template._initialized is False
            
            # Run should call setup
            template.run(video="test.mp4", processing_type="frame_extraction")
            
            # Verify setup was called
            assert template._initialized is True
    
    def test_to_dict_serialization(self):
        """Test that template can be serialized to dict."""
        from templates.video_processing_template import VideoProcessingTemplate
        template = VideoProcessingTemplate()
        
        metadata = template.to_dict()
        
        assert metadata['name'] == "video-processing"
        assert metadata['category'] == "Vision"
        assert metadata['gpu_required'] is True
        assert metadata['gpu_type'] == "A10G"
        assert metadata['memory_mb'] == 8192
        assert metadata['timeout_sec'] == 1800
        assert len(metadata['inputs']) == 3
        assert len(metadata['outputs']) == 2
        assert 'opencv-python' in metadata['pip_packages']
