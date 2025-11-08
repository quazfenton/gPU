# App Library - Serverless Function Templates

Reusable Modal serverless function templates for common use cases.

## Quick Start

```bash
# List available apps
python3 runna.py app-list

# View app code
python3 runna.py app-show image-classifier

# Deploy app
python3 runna.py app-deploy image-classifier

# Add your own app
python3 runna.py app-add my-app --file my_app.py --description "My app" --gpu A10G
```

## Built-in Apps

### image-classifier
**GPU:** T4 | **Tags:** ml, vision, classification

Image classification using ResNet50. Accepts base64-encoded images.

```bash
python3 runna.py app-deploy image-classifier
```

**Usage:**
```bash
curl -X POST https://your-app.modal.run \
  -H "Content-Type: application/json" \
  -d '{"image": "base64_encoded_image_data"}'
```

### text-generator
**GPU:** T4 | **Tags:** ml, nlp, generation

Text generation using GPT-2. Generate text from prompts.

```bash
python3 runna.py app-deploy text-generator
```

**Usage:**
```bash
curl -X POST https://your-app.modal.run \
  -d '{"prompt": "Once upon a time", "max_length": 100}'
```

### web-scraper
**GPU:** None | **Tags:** scraping, web

Web scraper using BeautifulSoup. Extract title, links, and text.

```bash
python3 runna.py app-deploy web-scraper
```

**Usage:**
```bash
curl -X POST https://your-app.modal.run \
  -d '{"url": "https://example.com"}'
```

### batch-processor
**GPU:** None | **Tags:** data, batch

Batch data processor using pandas and numpy.

```bash
python3 runna.py app-deploy batch-processor
```

**Usage:**
```bash
curl -X POST https://your-app.modal.run \
  -d '{"items": [{"id": 1, "values": [1,2,3]}, {"id": 2, "values": [4,5,6]}]}'
```

### scheduled-task
**GPU:** None | **Tags:** cron, scheduled

Hourly scheduled task example with status endpoint.

```bash
python3 runna.py app-deploy scheduled-task
```

## Commands

### List Apps
```bash
python3 runna.py app-list
```

Shows all apps with GPU requirements and deployment status.

### Show App Code
```bash
python3 runna.py app-show <name>
```

Display app source code and metadata.

### Deploy App
```bash
# Basic deployment
python3 runna.py app-deploy <name>

# Override GPU
python3 runna.py app-deploy <name> --gpu A100

# With secrets
python3 runna.py app-deploy <name> --secrets api-key

# Custom timeout
python3 runna.py app-deploy <name> --timeout 600
```

### Add App
```bash
python3 runna.py app-add <name> \
  --file app.py \
  --description "My app" \
  --tags ml vision \
  --gpu T4
```

### Update App
```bash
python3 runna.py app-update <name> --file updated_app.py
```

### Delete App
```bash
python3 runna.py app-delete <name>
```

## Creating Custom Apps

### Basic Template

```python
import modal

app = modal.App("my-app")
image = modal.Image.debian_slim().pip_install("requests")

@app.function(image=image)
@modal.web_endpoint(method="POST")
def handler(data: dict):
    # Your logic here
    return {"result": "success"}
```

### With GPU

```python
import modal

app = modal.App("gpu-app")
image = modal.Image.debian_slim().pip_install("torch")

@app.function(image=image, gpu="A10G")
@modal.web_endpoint(method="POST")
def process(data: dict):
    import torch
    # GPU processing
    return {"gpu": torch.cuda.is_available()}
```

### With Secrets

```python
import modal

app = modal.App("secure-app")
image = modal.Image.debian_slim().pip_install("requests")

@app.function(
    image=image,
    secrets=[modal.Secret.from_name("api-key")]
)
@modal.web_endpoint(method="POST")
def call_api(data: dict):
    import os
    api_key = os.environ["API_KEY"]
    # Use api_key
    return {"status": "ok"}
```

### Scheduled Function

```python
import modal

app = modal.App("scheduled-app")
image = modal.Image.debian_slim()

@app.function(
    image=image,
    schedule=modal.Cron("0 0 * * *")  # Daily at midnight
)
def daily_task():
    print("Running daily task")
    return {"status": "completed"}
```

## App Library Structure

```
apps/
├── library.json              # App index
├── image_classifier.py       # Template apps
├── text_generator.py
├── web_scraper.py
├── batch_processor.py
└── scheduled_task.py
```

## Integration with Endpoints

Apps deployed from the library are automatically registered:

```bash
# Deploy and save endpoint
python3 runna.py app-deploy image-classifier

# List endpoints
python3 runna.py endpoints

# Call endpoint
python3 runna.py call image-classifier --json '{"image": "..."}'
```

## Use Cases

### ML Model Serving
- Image classification
- Text generation
- Object detection
- Sentiment analysis

### Data Processing
- Batch processing
- ETL pipelines
- Data validation
- Format conversion

### Web Services
- Web scraping
- API proxies
- Webhooks
- File processing

### Automation
- Scheduled tasks
- Monitoring
- Alerts
- Backups

## Best Practices

1. **Use appropriate GPU** - T4 for inference, A100 for training
2. **Cache models** - Use volumes for model storage
3. **Handle errors** - Return proper error responses
4. **Set timeouts** - Avoid runaway functions
5. **Use secrets** - Never hardcode credentials
6. **Tag apps** - Organize with meaningful tags

## Examples

### Deploy Image Classifier
```bash
python3 runna.py app-deploy image-classifier
```

### Create Custom App
```bash
# Create app file
cat > my_sentiment.py << 'EOF'
import modal

app = modal.App("sentiment")
image = modal.Image.debian_slim().pip_install("transformers", "torch")

@app.function(image=image, gpu="T4")
@modal.web_endpoint(method="POST")
def analyze(data: dict):
    from transformers import pipeline
    classifier = pipeline("sentiment-analysis")
    result = classifier(data["text"])[0]
    return result
EOF

# Add to library
python3 runna.py app-add sentiment \
  --file my_sentiment.py \
  --description "Sentiment analysis" \
  --tags ml nlp \
  --gpu T4

# Deploy
python3 runna.py app-deploy sentiment
```

### Update Existing App
```bash
# Modify code
vim apps/image_classifier.py

# Update in library
python3 runna.py app-update image-classifier --file apps/image_classifier.py

# Redeploy
python3 runna.py app-deploy image-classifier
```

## Troubleshooting

### App not found
```bash
python3 runna.py app-list  # Check available apps
```

### Deployment fails
```bash
modal app logs <app-name>  # Check logs
```

### Import errors
Check dependencies in app's `image.pip_install()` call.

## Resources

- [Modal Documentation](https://modal.com/docs)
- [Modal Examples](https://modal.com/docs/examples)
- [MODAL_GUIDE.md](MODAL_GUIDE.md) - Deployment guide
