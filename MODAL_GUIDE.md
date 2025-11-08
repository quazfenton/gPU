# Modal.com Deployment Guide

Complete guide for deploying Kaggle notebooks to Modal.com with GPU support, secrets, and volumes.

## Quick Start

```bash
# Install Modal
pip install modal

# Authenticate
modal token new

# Deploy basic notebook
python3 runna.py deploy-modal ./notebook-dir

# Deploy with GPU
python3 runna.py deploy-modal ./notebook-dir --gpu A10G

# Deploy with secrets
python3 runna.py deploy-modal ./notebook-dir --secrets api-key db-credentials
```

## Features

### GPU Support

Modal supports multiple GPU types:

```bash
# T4 GPU (cost-effective)
python3 runna.py deploy-modal ./notebook --gpu T4

# A10G GPU (balanced)
python3 runna.py deploy-modal ./notebook --gpu A10G

# A100 GPU (high-performance)
python3 runna.py deploy-modal ./notebook --gpu A100
```

### Secrets Management

Store sensitive data securely:

```bash
# Create secret via Modal CLI
modal secret create api-key API_KEY=your_key_here

# Deploy with secret
python3 runna.py deploy-modal ./notebook --secrets api-key

# Access in notebook code:
import os
api_key = os.environ["API_KEY"]
```

### Timeout Configuration

```bash
# Set custom timeout (default: 300s)
python3 runna.py deploy-modal ./notebook --timeout 600
```

### Save Endpoint

```bash
# Save endpoint for later use
python3 runna.py deploy-modal ./notebook --save-name my-model

# Call saved endpoint
python3 runna.py call my-model --json '{"features": [1, 2, 3]}'
```

## Notebook Structure

### Basic Prediction Function

```python
# Cell 1: Imports
import numpy as np
import pandas as pd

# Cell 2: Define prediction function
def process_request(data):
    """Main function called by Modal endpoint."""
    features = data.get('features', [])
    result = np.mean(features)
    return {'result': result}

# Cell 3: Test locally
if __name__ == '__main__':
    test_data = {'features': [1, 2, 3, 4, 5]}
    print(process_request(test_data))
```

### ML Model Deployment

```python
# Cell 1: Imports
import torch
import torch.nn as nn
import numpy as np

# Cell 2: Define model
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 1)
    
    def forward(self, x):
        return self.fc(x)

# Cell 3: Load model
model = SimpleModel()
model.load_state_dict(torch.load('/cache/model.pth'))
model.eval()

# Cell 4: Prediction function
def process_request(data):
    features = torch.tensor(data['features'], dtype=torch.float32)
    with torch.no_grad():
        prediction = model(features)
    return {'prediction': prediction.item()}
```

### Using Secrets

```python
# Cell 1: Imports
import os
import requests

# Cell 2: Access secrets
API_KEY = os.environ.get('API_KEY')
DB_URL = os.environ.get('DB_URL')

# Cell 3: Use in function
def process_request(data):
    headers = {'Authorization': f'Bearer {API_KEY}'}
    response = requests.get('https://api.example.com', headers=headers)
    return response.json()
```

## Advanced Usage

### Custom Image with Dependencies

Edit the generated `modal_app.py`:

```python
image = modal.Image.debian_slim().pip_install(
    "torch",
    "transformers",
    "scikit-learn",
    "pandas",
    "numpy"
).apt_install("ffmpeg")  # System packages
```

### Volume for Model Caching

```python
# Models are automatically cached in /cache
# Access in your notebook:
import os
model_path = '/cache/my_model.pth'
if os.path.exists(model_path):
    model = torch.load(model_path)
else:
    # Download and save
    model = download_model()
    torch.save(model, model_path)
```

### Scheduled Functions

Edit `modal_app.py` to add scheduled tasks:

```python
@app.function(
    image=image,
    schedule=modal.Cron("0 0 * * *")  # Daily at midnight
)
def daily_task():
    """Runs daily."""
    print("Running scheduled task")
    # Your code here
```

## Deployment Workflow

### 1. Prepare Notebook

```bash
# Test locally first
python3 runna.py serve-local ./notebook --run --port 8080

# Test endpoint
curl -X POST http://localhost:8080 \
  -H "Content-Type: application/json" \
  -d '{"features": [1, 2, 3]}'
```

### 2. Deploy to Modal

```bash
# Deploy with all options
python3 runna.py deploy-modal ./notebook \
  --gpu A10G \
  --secrets api-key \
  --timeout 600 \
  --save-name production-model
```

### 3. Test Deployment

```bash
# Health check
curl https://your-app--health.modal.run

# Prediction
curl -X POST https://your-app--predict.modal.run \
  -H "Content-Type: application/json" \
  -d '{"features": [1, 2, 3]}'
```

### 4. Monitor and Update

```bash
# View logs
modal app logs kaggle-notebook

# Update deployment
python3 runna.py deploy-modal ./notebook --gpu A10G
```

## Cost Optimization

### GPU Selection

- **T4**: $0.60/hour - Good for inference
- **A10G**: $1.10/hour - Balanced performance
- **A100**: $4.00/hour - High-performance training

### Tips

1. Use CPU for simple tasks
2. Cache models in volumes
3. Set appropriate timeouts
4. Use T4 for inference, A100 for training

## Troubleshooting

### Import Errors

```python
# Add to notebook
import sys
sys.path.append('/cache')
```

### GPU Not Available

```python
# Check GPU in notebook
import torch
print(f"GPU available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")
```

### Timeout Issues

```bash
# Increase timeout
python3 runna.py deploy-modal ./notebook --timeout 900
```

### Secret Not Found

```bash
# List secrets
modal secret list

# Create secret
modal secret create my-secret KEY=value
```

## Examples

### Image Classification

```python
import torch
from torchvision import models, transforms
from PIL import Image
import base64
import io

model = models.resnet50(pretrained=True)
model.eval()

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])

def process_request(data):
    # Decode base64 image
    img_data = base64.b64decode(data['image'])
    img = Image.open(io.BytesIO(img_data))
    
    # Predict
    img_tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        output = model(img_tensor)
    
    return {'prediction': output.argmax().item()}
```

### Text Generation

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

def process_request(data):
    prompt = data.get('prompt', '')
    inputs = tokenizer(prompt, return_tensors='pt')
    outputs = model.generate(**inputs, max_length=100)
    text = tokenizer.decode(outputs[0])
    return {'generated_text': text}
```

### Batch Processing

```python
import numpy as np

def process_request(data):
    batch = data.get('batch', [])
    results = []
    
    for item in batch:
        # Process each item
        result = np.mean(item['features'])
        results.append({'id': item['id'], 'result': result})
    
    return {'results': results}
```

## Comparison with Other Platforms

| Feature | Modal | GCP Functions | AWS Lambda |
|---------|-------|---------------|------------|
| GPU Support | ✅ T4/A10G/A100 | ❌ | ❌ |
| Max Timeout | ∞ | 60m | 15m |
| Cold Start | ~1-2s | ~2-5s | ~1-3s |
| Package Size | 10GB+ | 500MB | 250MB |
| Python Version | Latest | 3.7-3.11 | 3.8-3.12 |
| Pricing | Pay-per-second | Pay-per-invocation | Pay-per-invocation |

## Best Practices

1. **Use GPU only when needed** - CPU is cheaper
2. **Cache models** - Use volumes for persistence
3. **Set appropriate timeouts** - Avoid unnecessary costs
4. **Use secrets** - Never hardcode credentials
5. **Test locally first** - Use `serve-local` command
6. **Monitor usage** - Check Modal dashboard regularly

## Resources

- [Modal Documentation](https://modal.com/docs)
- [Modal Examples](https://modal.com/docs/examples)
- [Modal Pricing](https://modal.com/pricing)
- [Modal Discord](https://discord.gg/modal)

## Support

For issues with Modal deployment:
1. Check Modal logs: `modal app logs kaggle-notebook`
2. Review generated `modal_app.py` in `modal-deploy/` directory
3. Test locally: `modal serve modal-deploy/modal_app.py`
4. Open issue on GitHub with logs
