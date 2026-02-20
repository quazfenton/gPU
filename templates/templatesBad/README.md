# Template Library

The Notebook ML Orchestrator Template Library provides a comprehensive collection of ML templates across four major categories: Audio Processing, Vision Processing, Language Processing, and Multimodal Pipelines. Each template is production-ready, fully tested, and integrates seamlessly with the orchestrator's job queue, backend router, and workflow engine.

## Table of Contents

- [Quick Start](#quick-start)
- [Complete Template Catalog](#complete-template-catalog)
- [Template Categories](#template-categories)
  - [Audio Processing](#audio-processing-templates)
  - [Vision Processing](#vision-processing-templates)
  - [Language Processing](#language-processing-templates)
  - [Multimodal Pipelines](#multimodal-pipeline-templates)
- [Template Discovery and Registration](#template-discovery-and-registration)
- [Using Templates](#using-templates)
- [Backend Support](#backend-support)
- [Creating Custom Templates](#creating-custom-templates)
- [Testing](#testing-templates)

## Quick Start

### Basic Usage with Orchestrator

```python
from notebook_ml_orchestrator.orchestrator import Orchestrator

# Initialize orchestrator (automatically discovers templates)
orchestrator = Orchestrator(templates_dir="templates")

# Submit a job using a template
job_id = orchestrator.submit_job(
    template_name="speech-recognition",
    inputs={
        "audio": "/path/to/audio.wav",
        "language": "en",
        "model_size": "base"
    },
    user_id="user-123"
)

print(f"Job submitted: {job_id}")
```

### Direct Template Usage

```python
from notebook_ml_orchestrator.core.template_registry import TemplateRegistry

# Initialize registry
registry = TemplateRegistry(templates_dir="templates")
registry.discover_templates()

# Get and execute a template
template = registry.get_template('speech-recognition')
result = template.execute(
    audio="/path/to/audio.wav",
    language="en",
    model_size="base"
)

print(result['text'])  # Transcribed text
print(result['segments'])  # Timestamped segments
```

## Complete Template Catalog

| Template Name | Category | Description | GPU | Memory | Timeout | Backends |
|--------------|----------|-------------|-----|--------|---------|----------|
| `speech-recognition` | Audio | Transcribe audio to text using Whisper | T4 | 4GB | 10min | Local, Modal, HF |
| `audio-generation` | Audio | Generate speech from text using TTS | T4 | 2GB | 5min | Local, Modal |
| `music-processing` | Audio | Analyze music (tempo, key, beats) | No | 1GB | 5min | Local, Modal |
| `object-detection` | Vision | Detect objects with YOLO | T4 | 4GB | 5min | Local, Modal, HF |
| `image-segmentation` | Vision | Semantic/instance segmentation | T4 | 6GB | 5min | Local, Modal |
| `video-processing` | Vision | Process video frames | A10G | 8GB | 30min | Local, Modal |
| `named-entity-recognition` | Language | Extract named entities (NER) | No | 1GB | 1min | Local, Modal, HF |
| `sentiment-analysis` | Language | Analyze text sentiment | No | 512MB | 30s | Local, Modal, HF |
| `translation` | Language | Translate between languages | T4 | 2GB | 2min | Local, Modal, HF |
| `summarization` | Language | Summarize long text | T4 | 2GB | 2min | Local, Modal, HF |
| `image-captioning` | Multimodal | Generate image captions | T4 | 4GB | 5min | Local, Modal, HF |
| `visual-question-answering` | Multimodal | Answer questions about images | T4 | 4GB | 5min | Local, Modal, HF |
| `text-to-image` | Multimodal | Generate images from text prompts | A10G | 16GB | 10min | Modal, HF |

## Template Categories

### Audio Processing Templates

Audio templates handle speech recognition, audio generation, and music analysis tasks.

| Template Name | Description | GPU Required | Memory | Timeout | Supported Backends |
|--------------|-------------|--------------|--------|---------|-------------------|
| `speech-recognition` | Transcribe audio to text using Whisper or similar models | Yes (T4) | 4GB | 10min | Local, Modal, HF |
| `audio-generation` | Generate speech from text using TTS models | Yes (T4) | 2GB | 5min | Local, Modal |
| `music-processing` | Analyze music audio (tempo, key, beats) | No | 1GB | 5min | Local, Modal |

#### Speech Recognition Template

Transcribes audio files to text with optional language detection and timestamped segments.

**Inputs:**
- `audio` (audio, required): Audio file to transcribe (wav, mp3, flac)
- `language` (text, optional): Language code (e.g., 'en', 'es', 'fr'). Default: 'en'
- `model_size` (text, optional): Model size (tiny, base, small, medium, large). Default: 'base'

**Outputs:**
- `text` (text): Transcribed text
- `segments` (json): Timestamped segments with text

**Example:**
```python
template = registry.get_template('speech-recognition')
result = template.execute(
    audio="meeting.wav",
    language="en",
    model_size="base"
)
print(result['text'])
print(result['segments'])  # [{'start': 0.0, 'end': 2.5, 'text': 'Hello world'}]
```

#### Audio Generation Template

Generates speech audio from text using text-to-speech models.

**Inputs:**
- `text` (text, required): Text to convert to speech
- `voice` (text, optional): Voice ID or style. Default: 'default'
- `speed` (number, optional): Speech speed multiplier (0.5 to 2.0). Default: 1.0

**Outputs:**
- `audio` (audio): Generated audio file

**Example:**
```python
template = registry.get_template('audio-generation')
result = template.execute(
    text="Hello, this is a test of text to speech.",
    voice="default",
    speed=1.0
)
# result['audio'] contains the generated audio file
```

#### Music Processing Template

Analyzes and processes music audio files.

**Inputs:**
- `audio` (audio, required): Music audio file to process
- `analysis_type` (text, optional): Type of analysis (tempo, key, beats, all). Default: 'all'

**Outputs:**
- `analysis` (json): Music analysis results
- `processed_audio` (audio): Processed audio file (if applicable)

**Example:**
```python
template = registry.get_template('music-processing')
result = template.execute(
    audio="song.mp3",
    analysis_type="all"
)
print(result['analysis'])  # {'tempo': 120, 'key': 'C major', 'beats': [...]}
```

### Vision Processing Templates

Vision templates handle image and video processing tasks including object detection, segmentation, and video analysis.

| Template Name | Description | GPU Required | Memory | Timeout | Supported Backends |
|--------------|-------------|--------------|--------|---------|-------------------|
| `object-detection` | Detect objects with bounding boxes using YOLO | Yes (T4) | 4GB | 5min | Local, Modal, HF |
| `image-segmentation` | Segment images into regions (semantic/instance) | Yes (T4) | 6GB | 5min | Local, Modal |
| `video-processing` | Process video frames with various algorithms | Yes (A10G) | 8GB | 30min | Local, Modal |

#### Object Detection Template

Detects objects in images with bounding boxes and confidence scores.

**Inputs:**
- `image` (image, required): Image file to analyze
- `confidence_threshold` (number, optional): Minimum confidence score (0.0 to 1.0). Default: 0.5
- `model` (text, optional): Detection model (yolov8n, yolov8s, yolov8m, yolov8l). Default: 'yolov8n'

**Outputs:**
- `detections` (json): List of detected objects with bounding boxes and confidence
- `annotated_image` (image): Image with bounding boxes drawn

**Example:**
```python
template = registry.get_template('object-detection')
result = template.execute(
    image="photo.jpg",
    confidence_threshold=0.5,
    model="yolov8n"
)
print(result['detections'])  
# [{'class': 'person', 'confidence': 0.95, 'bbox': [x, y, w, h]}, ...]
```

#### Image Segmentation Template

Performs semantic or instance segmentation on images.

**Inputs:**
- `image` (image, required): Image file to segment
- `segmentation_type` (text, optional): Type (semantic, instance). Default: 'semantic'

**Outputs:**
- `mask` (image): Segmentation mask
- `segments` (json): Segment information with labels and areas

**Example:**
```python
template = registry.get_template('image-segmentation')
result = template.execute(
    image="scene.jpg",
    segmentation_type="semantic"
)
# result['mask'] contains the segmentation mask
# result['segments'] contains segment metadata
```

#### Video Processing Template

Processes video files frame by frame with various algorithms.

**Inputs:**
- `video` (video, required): Video file to process
- `processing_type` (text, required): Type (object_tracking, scene_detection, frame_extraction)
- `frame_rate` (number, optional): Frames per second to process. Default: 1.0

**Outputs:**
- `results` (json): Processing results per frame
- `processed_video` (video): Processed video file (if applicable)

**Example:**
```python
template = registry.get_template('video-processing')
result = template.execute(
    video="clip.mp4",
    processing_type="scene_detection",
    frame_rate=1.0
)
print(result['results'])  # Scene detection results
```

### Language Processing Templates

Language templates handle NLP tasks including entity recognition, sentiment analysis, translation, and summarization.

| Template Name | Description | GPU Required | Memory | Timeout | Supported Backends |
|--------------|-------------|--------------|--------|---------|-------------------|
| `named-entity-recognition` | Extract named entities (people, places, organizations) | No | 1GB | 1min | Local, Modal, HF |
| `sentiment-analysis` | Analyze sentiment (positive, negative, neutral) | No | 512MB | 30s | Local, Modal, HF |
| `translation` | Translate text between languages | Yes (T4) | 2GB | 2min | Local, Modal, HF |
| `summarization` | Summarize long text into concise form | Yes (T4) | 2GB | 2min | Local, Modal, HF |

#### Named Entity Recognition Template

Extracts named entities from text.

**Inputs:**
- `text` (text, required): Text to analyze
- `model` (text, optional): NER model to use. Default: 'en_core_web_sm'

**Outputs:**
- `entities` (json): List of entities with types and positions

**Example:**
```python
template = registry.get_template('named-entity-recognition')
result = template.execute(
    text="Apple Inc. was founded by Steve Jobs in Cupertino, California."
)
print(result['entities'])
# [{'text': 'Apple Inc.', 'type': 'ORG', 'start': 0, 'end': 10}, ...]
```

#### Sentiment Analysis Template

Analyzes sentiment of text.

**Inputs:**
- `text` (text, required): Text to analyze

**Outputs:**
- `sentiment` (json): Sentiment scores (positive, negative, neutral)

**Example:**
```python
template = registry.get_template('sentiment-analysis')
result = template.execute(
    text="This product is amazing! I love it."
)
print(result['sentiment'])  
# {'positive': 0.95, 'negative': 0.02, 'neutral': 0.03}
```

#### Translation Template

Translates text between languages.

**Inputs:**
- `text` (text, required): Text to translate
- `source_language` (text, optional): Source language code. Default: 'auto'
- `target_language` (text, required): Target language code

**Outputs:**
- `translated_text` (text): Translated text
- `detected_language` (text): Detected source language (if auto)

**Example:**
```python
template = registry.get_template('translation')
result = template.execute(
    text="Hello, how are you?",
    source_language="en",
    target_language="es"
)
print(result['translated_text'])  # "Hola, ¿cómo estás?"
```

#### Summarization Template

Summarizes long text into shorter form.

**Inputs:**
- `text` (text, required): Text to summarize
- `max_length` (number, optional): Maximum length of summary. Default: 150
- `min_length` (number, optional): Minimum length of summary. Default: 50

**Outputs:**
- `summary` (text): Summarized text

**Example:**
```python
template = registry.get_template('summarization')
result = template.execute(
    text="Long article text here...",
    max_length=150,
    min_length=50
)
print(result['summary'])
```

### Multimodal Pipeline Templates

Multimodal templates process multiple data types (image + text, video + audio) for tasks like captioning and generation.

| Template Name | Description | GPU Required | Memory | Timeout | Supported Backends |
|--------------|-------------|--------------|--------|---------|-------------------|
| `image-captioning` | Generate descriptive captions for images | Yes (T4) | 4GB | 5min | Local, Modal, HF |
| `visual-question-answering` | Answer questions about image content | Yes (T4) | 4GB | 5min | Local, Modal, HF |
| `text-to-image` | Generate images from text prompts using diffusion | Yes (A10G) | 16GB | 10min | Modal, HF |

#### Image Captioning Template

Generates descriptive text captions for images.

**Inputs:**
- `image` (image, required): Image to caption
- `max_length` (number, optional): Maximum caption length. Default: 50

**Outputs:**
- `caption` (text): Generated caption
- `confidence` (number): Confidence score

**Example:**
```python
template = registry.get_template('image-captioning')
result = template.execute(
    image="vacation.jpg",
    max_length=50
)
print(result['caption'])  # "A beautiful sunset over the ocean"
print(result['confidence'])  # 0.92
```

#### Visual Question Answering Template

Answers questions about images.

**Inputs:**
- `image` (image, required): Image to analyze
- `question` (text, required): Question about the image

**Outputs:**
- `answer` (text): Answer to the question
- `confidence` (number): Confidence score

**Example:**
```python
template = registry.get_template('visual-question-answering')
result = template.execute(
    image="photo.jpg",
    question="What color is the car?"
)
print(result['answer'])  # "red"
print(result['confidence'])  # 0.88
```

#### Text-to-Image Template

Generates images from text descriptions using diffusion models.

**Inputs:**
- `prompt` (text, required): Text description of desired image
- `negative_prompt` (text, optional): What to avoid in the image. Default: ''
- `width` (number, optional): Image width in pixels. Default: 512
- `height` (number, optional): Image height in pixels. Default: 512
- `num_inference_steps` (number, optional): Number of denoising steps. Default: 50

**Outputs:**
- `image` (image): Generated image

**Example:**
```python
template = registry.get_template('text-to-image')
result = template.execute(
    prompt="A serene mountain landscape at sunset",
    negative_prompt="blurry, low quality",
    width=512,
    height=512,
    num_inference_steps=50
)
# result['image'] contains the generated image
```

## Template Discovery and Registration

Templates are automatically discovered and registered when the orchestrator starts:

```python
from notebook_ml_orchestrator.orchestrator import Orchestrator

# Initialize orchestrator (discovers templates automatically)
orchestrator = Orchestrator(templates_dir="templates")

# List all templates
templates = orchestrator.list_templates()
for template in templates:
    print(f"{template.name} ({template.category}): {template.description}")

# List templates by category
audio_templates = orchestrator.list_templates(category="Audio")
print(f"Found {len(audio_templates)} audio templates")

# Get template metadata
metadata = orchestrator.get_template_metadata("speech-recognition")
print(metadata)
```

### Manual Template Registry Usage

You can also use the Template Registry directly:

```python
from notebook_ml_orchestrator.core.template_registry import TemplateRegistry

# Initialize registry
registry = TemplateRegistry(templates_dir="templates")

# Discover templates
count = registry.discover_templates()
print(f"Discovered {count} templates")

# Get a specific template
template = registry.get_template('speech-recognition')

# Execute directly
result = template.execute(audio="file.wav", language="en")
```

## Using Templates

### Method 1: Through Orchestrator (Recommended)

The orchestrator provides automatic input validation, resource estimation, and backend routing:

```python
from notebook_ml_orchestrator.orchestrator import Orchestrator

orchestrator = Orchestrator()

# Submit a job
job_id = orchestrator.submit_job(
    template_name="object-detection",
    inputs={
        "image": "photo.jpg",
        "confidence_threshold": 0.7
    },
    user_id="user-123",
    routing_strategy="cost-optimized"
)

print(f"Job submitted: {job_id}")
```

### Method 2: Direct Template Execution

For quick testing or local execution:

```python
from notebook_ml_orchestrator.core.template_registry import TemplateRegistry

registry = TemplateRegistry()
registry.discover_templates()

template = registry.get_template('sentiment-analysis')
result = template.execute(text="I love this product!")

print(result['sentiment'])
```

### Method 3: With Job Queue

For asynchronous execution with job tracking:

```python
from notebook_ml_orchestrator.core.job_queue import JobQueueManager
from notebook_ml_orchestrator.core.interfaces import Job

job_queue = JobQueueManager()

job = Job(
    template_name="speech-recognition",
    inputs={"audio": "meeting.wav", "language": "en"},
    user_id="user-123"
)

job_id = job_queue.submit_job(job)

# Check job status
job = job_queue.get_job(job_id)
print(f"Status: {job.status}")
```

## Template Metadata

Each template provides comprehensive metadata:

```python
template = registry.get_template('speech-recognition')
metadata = registry.get_template_metadata('speech-recognition')

print(metadata)
# {
#     'name': 'speech-recognition',
#     'category': 'Audio',
#     'description': 'Transcribe audio files to text',
#     'version': '1.0.0',
#     'inputs': [
#         {'name': 'audio', 'type': 'audio', 'required': True, ...},
#         {'name': 'language', 'type': 'text', 'required': False, ...}
#     ],
#     'outputs': [
#         {'name': 'text', 'type': 'text', ...},
#         {'name': 'segments', 'type': 'json', ...}
#     ],
#     'routing': ['local', 'modal', 'hf'],
#     'gpu_required': True,
#     'gpu_type': 'T4',
#     'memory_mb': 4096,
#     'timeout_sec': 600,
#     'pip_packages': ['openai-whisper', 'torch', 'torchaudio']
# }
```

## Backend Support

Templates declare which backends they support:

- **LOCAL**: Run on local machine
- **MODAL**: Run on Modal cloud infrastructure
- **HF**: Run on HuggingFace Spaces/Inference API
- **COLAB**: Run on Google Colab

The backend router automatically selects the best backend based on:
- Template requirements (GPU, memory, timeout)
- Backend availability and health
- Cost optimization
- Load balancing

## Creating Custom Templates

To create a custom template:

1. Create a new file in the `templates/` directory
2. Inherit from the `Template` base class
3. Define template metadata and resource requirements
4. Implement the `run()` method

```python
from templates.base import Template, InputField, OutputField, RouteType

class MyCustomTemplate(Template):
    name = "my-custom-template"
    category = "Custom"
    description = "My custom ML template"
    version = "1.0.0"
    
    inputs = [
        InputField(name="input_data", type="text", required=True)
    ]
    
    outputs = [
        OutputField(name="result", type="text")
    ]
    
    routing = [RouteType.LOCAL, RouteType.MODAL]
    gpu_required = False
    memory_mb = 512
    timeout_sec = 60
    pip_packages = []
    
    def run(self, **kwargs) -> Dict[str, Any]:
        input_data = kwargs['input_data']
        # Your processing logic here
        return {"result": f"Processed: {input_data}"}
```

The template will be automatically discovered on next startup.

## Error Handling

Templates provide comprehensive error handling:

```python
try:
    result = template.execute(audio="file.wav")
except ValueError as e:
    # Input validation error
    print(f"Invalid input: {e}")
except RuntimeError as e:
    # Execution error
    print(f"Execution failed: {e}")
```

## Integration with Job Queue

Templates integrate seamlessly with the job queue:

```python
from notebook_ml_orchestrator.core.job_queue import JobQueue
from notebook_ml_orchestrator.core.interfaces import Job

job_queue = JobQueue()

# Submit template job
job = Job(
    template_name="speech-recognition",
    inputs={"audio": "meeting.wav", "language": "en"}
)

job_queue.submit_job(job)
```

## Integration with Workflows

Templates can be chained in workflows:

```python
from notebook_ml_orchestrator.core.workflow_engine import WorkflowEngine
from notebook_ml_orchestrator.core.models import WorkflowDefinition

workflow_def = WorkflowDefinition(
    steps=[
        {
            "name": "transcribe",
            "template": "speech-recognition",
            "inputs": {"audio": "meeting.wav"}
        },
        {
            "name": "analyze",
            "template": "sentiment-analysis",
            "inputs": {"text": "${transcribe.text}"}  # Use output from previous step
        }
    ]
)

engine = WorkflowEngine()
workflow = engine.create_workflow(workflow_def)
execution = engine.execute_workflow(workflow.id, {})
```

## Testing Templates

All templates include comprehensive tests:

```bash
# Run all template tests
pytest notebook_ml_orchestrator/tests/test_*_templates.py

# Run property-based tests
pytest notebook_ml_orchestrator/tests/test_*_templates_properties.py

# Run specific template tests
pytest notebook_ml_orchestrator/tests/test_audio_templates.py -v
```

## Contributing

To contribute a new template:

1. Create the template class in `templates/`
2. Add unit tests in `notebook_ml_orchestrator/tests/`
3. Add property-based tests for correctness properties
4. Update this README with template documentation
5. Submit a pull request

## License

See the main project LICENSE file.
