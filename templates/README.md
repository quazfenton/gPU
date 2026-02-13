# Template Library

The Notebook ML Orchestrator Template Library provides a comprehensive collection of ML templates across four major categories: Audio Processing, Vision Processing, Language Processing, and Multimodal Pipelines.

## Quick Start

### Using a Template

```python
from templates import discover_templates

# Discover all available templates
templates = discover_templates()

# Get a specific template
speech_template = templates['speech-recognition']

# Execute the template
result = speech_template.execute(
    audio="/path/to/audio.wav",
    language="en",
    model_size="base"
)

print(result['text'])  # Transcribed text
```

### Template Discovery and Registration

Templates are automatically discovered at startup:

```python
from notebook_ml_orchestrator.core.template_registry import TemplateRegistry

# Initialize registry
registry = TemplateRegistry(templates_dir="templates")

# Discover templates
count = registry.discover_templates()
print(f"Discovered {count} templates")

# List templates by category
audio_templates = registry.list_templates(category="Audio")
for template in audio_templates:
    print(f"- {template.name}: {template.description}")
```

## Available Templates

### Audio Processing Templates

| Template Name | Description | GPU Required | Memory | Timeout |
|--------------|-------------|--------------|--------|---------|
| `speech-recognition` | Transcribe audio to text using Whisper | Yes (T4) | 4GB | 10min |
| `audio-generation` | Generate speech from text using TTS | Yes (T4) | 2GB | 5min |
| `music-processing` | Analyze music (tempo, key, beats) | No | 1GB | 5min |

**Example: Speech Recognition**
```python
template = registry.get_template('speech-recognition')
result = template.execute(
    audio="meeting.wav",
    language="en",
    model_size="base"
)
print(result['text'])
print(result['segments'])  # Timestamped segments
```

### Vision Processing Templates

| Template Name | Description | GPU Required | Memory | Timeout |
|--------------|-------------|--------------|--------|---------|
| `object-detection` | Detect objects with bounding boxes | Yes (T4) | 4GB | 5min |
| `image-segmentation` | Segment images into regions | Yes (T4) | 6GB | 5min |
| `video-processing` | Process video frames | Yes (A10G) | 8GB | 30min |

**Example: Object Detection**
```python
template = registry.get_template('object-detection')
result = template.execute(
    image="photo.jpg",
    confidence_threshold=0.5,
    model="yolov8n"
)
print(result['detections'])  # List of detected objects
```

### Language Processing Templates

| Template Name | Description | GPU Required | Memory | Timeout |
|--------------|-------------|--------------|--------|---------|
| `named-entity-recognition` | Extract entities from text | No | 1GB | 1min |
| `sentiment-analysis` | Analyze sentiment | No | 512MB | 30s |
| `translation` | Translate between languages | Yes (T4) | 2GB | 2min |
| `summarization` | Summarize long text | Yes (T4) | 2GB | 2min |

**Example: Sentiment Analysis**
```python
template = registry.get_template('sentiment-analysis')
result = template.execute(
    text="This product is amazing! I love it."
)
print(result['sentiment'])  # {'positive': 0.95, 'negative': 0.02, 'neutral': 0.03}
```

### Multimodal Pipeline Templates

| Template Name | Description | GPU Required | Memory | Timeout |
|--------------|-------------|--------------|--------|---------|
| `image-captioning` | Generate captions for images | Yes (T4) | 4GB | 5min |
| `visual-question-answering` | Answer questions about images | Yes (T4) | 4GB | 5min |
| `text-to-image` | Generate images from text | Yes (A10G) | 16GB | 10min |

**Example: Image Captioning**
```python
template = registry.get_template('image-captioning')
result = template.execute(
    image="vacation.jpg",
    max_length=50
)
print(result['caption'])  # "A beautiful sunset over the ocean"
print(result['confidence'])  # 0.92
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
