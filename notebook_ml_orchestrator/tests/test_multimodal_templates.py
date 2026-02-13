"""
Unit tests for Multimodal Pipeline Templates.

Tests cover:
- ImageCaptioningTemplate instantiation and metadata
- VQATemplate instantiation and metadata
- TextToImageTemplate instantiation and metadata
- Input validation
- Setup and initialization
- Run method with valid inputs
- Error handling
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from templates.image_captioning_template import ImageCaptioningTemplate
from templates.vqa_template import VQATemplate
from templates.text_to_image_template import TextToImageTemplate
from templates.base import RouteType


class TestImageCaptioningTemplate:
    """Test suite for ImageCaptioningTemplate."""
    
    def test_template_metadata(self):
        """Test that template has correct metadata."""
        template = ImageCaptioningTemplate()
        
        assert template.name == "image-captioning"
        assert template.category == "Multimodal"
        assert template.description == "Generate descriptive text captions for images"
        assert template.version == "1.0.0"
    
    def test_input_definitions(self):
        """Test that template has correct input definitions."""
        template = ImageCaptioningTemplate()
        
        assert len(template.inputs) == 2
        
        # Check image input
        image_input = next(i for i in template.inputs if i.name == "image")
        assert image_input.type == "image"
        assert image_input.required is True
        assert image_input.description == "Image to caption"
        
        # Check max_length input
        max_length_input = next(i for i in template.inputs if i.name == "max_length")
        assert max_length_input.type == "number"
        assert max_length_input.required is False
        assert max_length_input.default == 50
    
    def test_output_definitions(self):
        """Test that template has correct output definitions."""
        template = ImageCaptioningTemplate()
        
        assert len(template.outputs) == 2
        
        # Check caption output
        caption_output = next(o for o in template.outputs if o.name == "caption")
        assert caption_output.type == "text"
        assert caption_output.description == "Generated caption"
        
        # Check confidence output
        confidence_output = next(o for o in template.outputs if o.name == "confidence")
        assert confidence_output.type == "number"
        assert confidence_output.description == "Confidence score"
    
    def test_resource_requirements(self):
        """Test that template has correct resource requirements."""
        template = ImageCaptioningTemplate()
        
        assert template.gpu_required is True
        assert template.gpu_type == "T4"
        assert template.memory_mb == 4096
        assert template.timeout_sec == 300
    
    def test_routing_support(self):
        """Test that template supports correct backends."""
        template = ImageCaptioningTemplate()
        
        assert RouteType.LOCAL in template.routing
        assert RouteType.MODAL in template.routing
        assert RouteType.HF in template.routing
    
    def test_pip_packages(self):
        """Test that template declares correct dependencies."""
        template = ImageCaptioningTemplate()
        
        assert "transformers" in template.pip_packages
        assert "torch" in template.pip_packages
        assert "pillow" in template.pip_packages
    
    def test_validate_inputs_valid(self):
        """Test input validation with valid inputs."""
        template = ImageCaptioningTemplate()
        
        # Should not raise with required image input
        result = template.validate_inputs(image="test.jpg")
        assert result is True
        
        # Should not raise with all inputs
        result = template.validate_inputs(
            image="test.jpg",
            max_length=50
        )
        assert result is True
    
    def test_validate_inputs_missing_required(self):
        """Test input validation fails when required input is missing."""
        template = ImageCaptioningTemplate()
        
        # Should raise ValueError when image is missing
        with pytest.raises(ValueError, match="Missing required input: image"):
            template.validate_inputs(max_length=50)
    
    def test_setup(self):
        """Test setup method loads image captioning model."""
        template = ImageCaptioningTemplate()
        
        # Mock transformers module
        with patch.dict('sys.modules', {'transformers': MagicMock()}):
            import sys
            mock_transformers = sys.modules['transformers']
            mock_processor = Mock()
            mock_model = Mock()
            mock_transformers.BlipProcessor.from_pretrained.return_value = mock_processor
            mock_transformers.BlipForConditionalGeneration.from_pretrained.return_value = mock_model
            
            template.setup()
            
            mock_transformers.BlipProcessor.from_pretrained.assert_called_once_with(
                "Salesforce/blip-image-captioning-base"
            )
            mock_transformers.BlipForConditionalGeneration.from_pretrained.assert_called_once_with(
                "Salesforce/blip-image-captioning-base"
            )
            assert template.processor == mock_processor
            assert template.model == mock_model
            assert template._initialized is True
    
    def test_run_with_image_path(self):
        """Test run method with image file path."""
        template = ImageCaptioningTemplate()
        
        # Mock transformers and PIL
        with patch.dict('sys.modules', {'transformers': MagicMock(), 'PIL': MagicMock(), 'torch': MagicMock()}):
            import sys
            mock_transformers = sys.modules['transformers']
            mock_pil = sys.modules['PIL']
            mock_torch = sys.modules['torch']
            
            # Setup mocks
            mock_image = Mock()
            mock_image.convert.return_value = mock_image
            mock_pil.Image.open.return_value = mock_image
            
            mock_processor = Mock()
            mock_processor.return_value = {'pixel_values': Mock()}
            mock_processor.decode.return_value = "A cat sitting on a couch"
            
            mock_model = Mock()
            mock_outputs = MagicMock()
            mock_outputs.__getitem__.return_value = Mock()
            mock_model.generate.return_value = mock_outputs
            
            mock_transformers.BlipProcessor.from_pretrained.return_value = mock_processor
            mock_transformers.BlipForConditionalGeneration.from_pretrained.return_value = mock_model
            
            mock_torch.no_grad.return_value.__enter__ = Mock()
            mock_torch.no_grad.return_value.__exit__ = Mock()
            
            # Run the template
            result = template.run(image="test.jpg", max_length=50)
            
            # Verify results
            assert 'caption' in result
            assert result['caption'] == "A cat sitting on a couch"
            assert 'confidence' in result
            assert 0.0 <= result['confidence'] <= 1.0
    
    def test_run_with_pil_image(self):
        """Test run method with PIL Image object."""
        template = ImageCaptioningTemplate()
        
        # Mock transformers and PIL
        with patch.dict('sys.modules', {'transformers': MagicMock(), 'PIL': MagicMock(), 'torch': MagicMock()}):
            import sys
            mock_transformers = sys.modules['transformers']
            mock_torch = sys.modules['torch']
            
            # Setup mocks
            mock_image = Mock()
            mock_image.convert.return_value = mock_image
            
            mock_processor = Mock()
            mock_processor.return_value = {'pixel_values': Mock()}
            mock_processor.decode.return_value = "A dog playing in the park"
            
            mock_model = Mock()
            mock_outputs = MagicMock()
            mock_outputs.__getitem__.return_value = Mock()
            mock_model.generate.return_value = mock_outputs
            
            mock_transformers.BlipProcessor.from_pretrained.return_value = mock_processor
            mock_transformers.BlipForConditionalGeneration.from_pretrained.return_value = mock_model
            
            mock_torch.no_grad.return_value.__enter__ = Mock()
            mock_torch.no_grad.return_value.__exit__ = Mock()
            
            # Run with PIL Image
            result = template.run(image=mock_image)
            
            # Verify results
            assert 'caption' in result
            assert result['caption'] == "A dog playing in the park"


class TestVQATemplate:
    """Test suite for VQATemplate."""
    
    def test_template_metadata(self):
        """Test that template has correct metadata."""
        template = VQATemplate()
        
        assert template.name == "visual-question-answering"
        assert template.category == "Multimodal"
        assert template.description == "Answer questions about image content"
        assert template.version == "1.0.0"
    
    def test_input_definitions(self):
        """Test that template has correct input definitions."""
        template = VQATemplate()
        
        assert len(template.inputs) == 2
        
        # Check image input
        image_input = next(i for i in template.inputs if i.name == "image")
        assert image_input.type == "image"
        assert image_input.required is True
        assert image_input.description == "Image to analyze"
        
        # Check question input
        question_input = next(i for i in template.inputs if i.name == "question")
        assert question_input.type == "text"
        assert question_input.required is True
        assert question_input.description == "Question about the image"
    
    def test_output_definitions(self):
        """Test that template has correct output definitions."""
        template = VQATemplate()
        
        assert len(template.outputs) == 2
        
        # Check answer output
        answer_output = next(o for o in template.outputs if o.name == "answer")
        assert answer_output.type == "text"
        assert answer_output.description == "Answer to the question"
        
        # Check confidence output
        confidence_output = next(o for o in template.outputs if o.name == "confidence")
        assert confidence_output.type == "number"
        assert confidence_output.description == "Confidence score"
    
    def test_resource_requirements(self):
        """Test that template has correct resource requirements."""
        template = VQATemplate()
        
        assert template.gpu_required is True
        assert template.gpu_type == "T4"
        assert template.memory_mb == 4096
        assert template.timeout_sec == 300
    
    def test_routing_support(self):
        """Test that template supports correct backends."""
        template = VQATemplate()
        
        assert RouteType.LOCAL in template.routing
        assert RouteType.MODAL in template.routing
        assert RouteType.HF in template.routing
    
    def test_pip_packages(self):
        """Test that template declares correct dependencies."""
        template = VQATemplate()
        
        assert "transformers" in template.pip_packages
        assert "torch" in template.pip_packages
        assert "pillow" in template.pip_packages
    
    def test_validate_inputs_valid(self):
        """Test input validation with valid inputs."""
        template = VQATemplate()
        
        # Should not raise with required inputs
        result = template.validate_inputs(image="test.jpg", question="What is in the image?")
        assert result is True
    
    def test_validate_inputs_missing_required(self):
        """Test input validation fails when required input is missing."""
        template = VQATemplate()
        
        # Should raise ValueError when image is missing
        with pytest.raises(ValueError, match="Missing required input: image"):
            template.validate_inputs(question="What is in the image?")
        
        # Should raise ValueError when question is missing
        with pytest.raises(ValueError, match="Missing required input: question"):
            template.validate_inputs(image="test.jpg")
    
    def test_setup(self):
        """Test setup method loads VQA model."""
        template = VQATemplate()
        
        # Mock transformers module
        with patch.dict('sys.modules', {'transformers': MagicMock()}):
            import sys
            mock_transformers = sys.modules['transformers']
            mock_processor = Mock()
            mock_model = Mock()
            mock_transformers.BlipProcessor.from_pretrained.return_value = mock_processor
            mock_transformers.BlipForQuestionAnswering.from_pretrained.return_value = mock_model
            
            template.setup()
            
            mock_transformers.BlipProcessor.from_pretrained.assert_called_once_with(
                "Salesforce/blip-vqa-base"
            )
            mock_transformers.BlipForQuestionAnswering.from_pretrained.assert_called_once_with(
                "Salesforce/blip-vqa-base"
            )
            assert template.processor == mock_processor
            assert template.model == mock_model
            assert template._initialized is True
    
    def test_run_with_valid_inputs(self):
        """Test run method with valid image and question."""
        template = VQATemplate()
        
        # Mock transformers and PIL
        with patch.dict('sys.modules', {'transformers': MagicMock(), 'PIL': MagicMock(), 'torch': MagicMock()}):
            import sys
            mock_transformers = sys.modules['transformers']
            mock_pil = sys.modules['PIL']
            mock_torch = sys.modules['torch']
            
            # Setup mocks
            mock_image = Mock()
            mock_image.convert.return_value = mock_image
            mock_pil.Image.open.return_value = mock_image
            
            mock_processor = Mock()
            mock_processor.return_value = {'pixel_values': Mock(), 'input_ids': Mock()}
            mock_processor.decode.return_value = "A cat"
            
            mock_model = Mock()
            mock_outputs = MagicMock()
            mock_outputs.__getitem__.return_value = Mock()
            mock_model.generate.return_value = mock_outputs
            
            mock_transformers.BlipProcessor.from_pretrained.return_value = mock_processor
            mock_transformers.BlipForQuestionAnswering.from_pretrained.return_value = mock_model
            
            mock_torch.no_grad.return_value.__enter__ = Mock()
            mock_torch.no_grad.return_value.__exit__ = Mock()
            
            # Run the template
            result = template.run(image="test.jpg", question="What animal is this?")
            
            # Verify results
            assert 'answer' in result
            assert result['answer'] == "A cat"
            assert 'confidence' in result
            assert 0.0 <= result['confidence'] <= 1.0
    
    def test_run_with_different_questions(self):
        """Test run method with different types of questions."""
        template = VQATemplate()
        
        # Mock transformers and PIL
        with patch.dict('sys.modules', {'transformers': MagicMock(), 'PIL': MagicMock(), 'torch': MagicMock()}):
            import sys
            mock_transformers = sys.modules['transformers']
            mock_pil = sys.modules['PIL']
            mock_torch = sys.modules['torch']
            
            # Setup mocks
            mock_image = Mock()
            mock_image.convert.return_value = mock_image
            mock_pil.Image.open.return_value = mock_image
            
            mock_processor = Mock()
            mock_processor.return_value = {'pixel_values': Mock(), 'input_ids': Mock()}
            mock_processor.decode.return_value = "Blue"
            
            mock_model = Mock()
            mock_outputs = MagicMock()
            mock_outputs.__getitem__.return_value = Mock()
            mock_model.generate.return_value = mock_outputs
            
            mock_transformers.BlipProcessor.from_pretrained.return_value = mock_processor
            mock_transformers.BlipForQuestionAnswering.from_pretrained.return_value = mock_model
            
            mock_torch.no_grad.return_value.__enter__ = Mock()
            mock_torch.no_grad.return_value.__exit__ = Mock()
            
            # Run with color question
            result = template.run(image="test.jpg", question="What color is the sky?")
            
            # Verify results
            assert 'answer' in result
            assert result['answer'] == "Blue"


class TestTextToImageTemplate:
    """Test suite for TextToImageTemplate."""
    
    def test_template_metadata(self):
        """Test that template has correct metadata."""
        template = TextToImageTemplate()
        
        assert template.name == "text-to-image"
        assert template.category == "Multimodal"
        assert template.description == "Generate images from text prompts using diffusion models"
        assert template.version == "1.0.0"
    
    def test_input_definitions(self):
        """Test that template has correct input definitions."""
        template = TextToImageTemplate()
        
        assert len(template.inputs) == 5
        
        # Check prompt input
        prompt_input = next(i for i in template.inputs if i.name == "prompt")
        assert prompt_input.type == "text"
        assert prompt_input.required is True
        assert prompt_input.description == "Text description of desired image"
        
        # Check negative_prompt input
        neg_prompt_input = next(i for i in template.inputs if i.name == "negative_prompt")
        assert neg_prompt_input.type == "text"
        assert neg_prompt_input.required is False
        assert neg_prompt_input.default == ""
        
        # Check width input
        width_input = next(i for i in template.inputs if i.name == "width")
        assert width_input.type == "number"
        assert width_input.required is False
        assert width_input.default == 512
        
        # Check height input
        height_input = next(i for i in template.inputs if i.name == "height")
        assert height_input.type == "number"
        assert height_input.required is False
        assert height_input.default == 512
        
        # Check num_inference_steps input
        steps_input = next(i for i in template.inputs if i.name == "num_inference_steps")
        assert steps_input.type == "number"
        assert steps_input.required is False
        assert steps_input.default == 50
    
    def test_output_definitions(self):
        """Test that template has correct output definitions."""
        template = TextToImageTemplate()
        
        assert len(template.outputs) == 1
        
        # Check image output
        image_output = next(o for o in template.outputs if o.name == "image")
        assert image_output.type == "image"
        assert image_output.description == "Generated image"
    
    def test_resource_requirements(self):
        """Test that template has correct resource requirements."""
        template = TextToImageTemplate()
        
        assert template.gpu_required is True
        assert template.gpu_type == "A10G"
        assert template.memory_mb == 16384
        assert template.timeout_sec == 600
    
    def test_routing_support(self):
        """Test that template supports correct backends."""
        template = TextToImageTemplate()
        
        assert RouteType.MODAL in template.routing
        assert RouteType.HF in template.routing
        # Should NOT support LOCAL due to high resource requirements
        assert RouteType.LOCAL not in template.routing
    
    def test_pip_packages(self):
        """Test that template declares correct dependencies."""
        template = TextToImageTemplate()
        
        assert "diffusers" in template.pip_packages
        assert "transformers" in template.pip_packages
        assert "torch" in template.pip_packages
        assert "accelerate" in template.pip_packages
    
    def test_validate_inputs_valid(self):
        """Test input validation with valid inputs."""
        template = TextToImageTemplate()
        
        # Should not raise with required prompt input
        result = template.validate_inputs(prompt="A beautiful sunset")
        assert result is True
        
        # Should not raise with all inputs
        result = template.validate_inputs(
            prompt="A beautiful sunset",
            negative_prompt="blurry, low quality",
            width=768,
            height=768,
            num_inference_steps=30
        )
        assert result is True
    
    def test_validate_inputs_missing_required(self):
        """Test input validation fails when required input is missing."""
        template = TextToImageTemplate()
        
        # Should raise ValueError when prompt is missing
        with pytest.raises(ValueError, match="Missing required input: prompt"):
            template.validate_inputs(width=512)
    
    def test_setup(self):
        """Test setup method loads text-to-image model."""
        template = TextToImageTemplate()
        
        # Mock diffusers and torch modules
        with patch.dict('sys.modules', {'diffusers': MagicMock(), 'torch': MagicMock()}):
            import sys
            mock_diffusers = sys.modules['diffusers']
            mock_torch = sys.modules['torch']
            
            mock_pipeline = Mock()
            mock_pipeline.to.return_value = mock_pipeline
            mock_diffusers.StableDiffusionPipeline.from_pretrained.return_value = mock_pipeline
            
            mock_torch.float16 = Mock()
            mock_torch.cuda.is_available.return_value = True
            
            template.setup()
            
            mock_diffusers.StableDiffusionPipeline.from_pretrained.assert_called_once()
            assert template.pipe == mock_pipeline
            assert template._initialized is True
    
    def test_run_with_basic_prompt(self):
        """Test run method with basic prompt."""
        template = TextToImageTemplate()
        
        # Mock diffusers and torch modules
        with patch.dict('sys.modules', {'diffusers': MagicMock(), 'torch': MagicMock()}):
            import sys
            mock_diffusers = sys.modules['diffusers']
            mock_torch = sys.modules['torch']
            
            # Setup mocks
            mock_image = Mock()
            mock_result = Mock()
            mock_result.images = [mock_image]
            
            mock_pipeline = Mock()
            mock_pipeline.to.return_value = mock_pipeline
            mock_pipeline.return_value = mock_result
            mock_diffusers.StableDiffusionPipeline.from_pretrained.return_value = mock_pipeline
            
            mock_torch.float16 = Mock()
            mock_torch.cuda.is_available.return_value = False
            
            # Run the template
            result = template.run(prompt="A beautiful sunset over the ocean")
            
            # Verify results
            assert 'image' in result
            assert result['image'] == mock_image
            
            # Verify pipeline was called with correct parameters
            mock_pipeline.assert_called_once()
            call_args = mock_pipeline.call_args
            assert call_args[1]['prompt'] == "A beautiful sunset over the ocean"
            assert call_args[1]['width'] == 512
            assert call_args[1]['height'] == 512
            assert call_args[1]['num_inference_steps'] == 50
    
    def test_run_with_negative_prompt(self):
        """Test run method with negative prompt."""
        template = TextToImageTemplate()
        
        # Mock diffusers and torch modules
        with patch.dict('sys.modules', {'diffusers': MagicMock(), 'torch': MagicMock()}):
            import sys
            mock_diffusers = sys.modules['diffusers']
            mock_torch = sys.modules['torch']
            
            # Setup mocks
            mock_image = Mock()
            mock_result = Mock()
            mock_result.images = [mock_image]
            
            mock_pipeline = Mock()
            mock_pipeline.to.return_value = mock_pipeline
            mock_pipeline.return_value = mock_result
            mock_diffusers.StableDiffusionPipeline.from_pretrained.return_value = mock_pipeline
            
            mock_torch.float16 = Mock()
            mock_torch.cuda.is_available.return_value = False
            
            # Run with negative prompt
            result = template.run(
                prompt="A cat",
                negative_prompt="blurry, low quality"
            )
            
            # Verify negative prompt was passed
            call_args = mock_pipeline.call_args
            assert call_args[1]['negative_prompt'] == "blurry, low quality"
    
    def test_run_with_custom_dimensions(self):
        """Test run method with custom width and height."""
        template = TextToImageTemplate()
        
        # Mock diffusers and torch modules
        with patch.dict('sys.modules', {'diffusers': MagicMock(), 'torch': MagicMock()}):
            import sys
            mock_diffusers = sys.modules['diffusers']
            mock_torch = sys.modules['torch']
            
            # Setup mocks
            mock_image = Mock()
            mock_result = Mock()
            mock_result.images = [mock_image]
            
            mock_pipeline = Mock()
            mock_pipeline.to.return_value = mock_pipeline
            mock_pipeline.return_value = mock_result
            mock_diffusers.StableDiffusionPipeline.from_pretrained.return_value = mock_pipeline
            
            mock_torch.float16 = Mock()
            mock_torch.cuda.is_available.return_value = False
            
            # Run with custom dimensions
            result = template.run(
                prompt="A landscape",
                width=768,
                height=512
            )
            
            # Verify custom dimensions were used
            call_args = mock_pipeline.call_args
            assert call_args[1]['width'] == 768
            assert call_args[1]['height'] == 512
    
    def test_run_with_custom_steps(self):
        """Test run method with custom inference steps."""
        template = TextToImageTemplate()
        
        # Mock diffusers and torch modules
        with patch.dict('sys.modules', {'diffusers': MagicMock(), 'torch': MagicMock()}):
            import sys
            mock_diffusers = sys.modules['diffusers']
            mock_torch = sys.modules['torch']
            
            # Setup mocks
            mock_image = Mock()
            mock_result = Mock()
            mock_result.images = [mock_image]
            
            mock_pipeline = Mock()
            mock_pipeline.to.return_value = mock_pipeline
            mock_pipeline.return_value = mock_result
            mock_diffusers.StableDiffusionPipeline.from_pretrained.return_value = mock_pipeline
            
            mock_torch.float16 = Mock()
            mock_torch.cuda.is_available.return_value = False
            
            # Run with custom steps
            result = template.run(
                prompt="A portrait",
                num_inference_steps=30
            )
            
            # Verify custom steps were used
            call_args = mock_pipeline.call_args
            assert call_args[1]['num_inference_steps'] == 30
