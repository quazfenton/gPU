# Complete Workflow Example

End-to-end example of using the app library with Modal deployment.

## Scenario: Deploy a Sentiment Analysis API

### Step 1: Browse Available Apps

```bash
python3 runna.py app-list
```

Output:
```
Name                 GPU        Deployed        Description
--------------------------------------------------------------------------------
image-classifier     T4         ✗               Image classification with ResNet50
text-generator       T4         ✗               Text generation with GPT-2
web-scraper          None       ✗               Web scraper with BeautifulSoup
batch-processor      None       ✗               Batch data processor with pandas
scheduled-task       None       ✗               Hourly scheduled task example
```

### Step 2: Create Custom Sentiment App

```bash
cat > sentiment_app.py << 'EOF'
import modal

app = modal.App("sentiment-analyzer")
image = modal.Image.debian_slim().pip_install("transformers", "torch")

@app.function(image=image, gpu="T4", timeout=300)
@modal.web_endpoint(method="POST")
def analyze(data: dict):
    from transformers import pipeline
    
    text = data.get("text", "")
    if not text:
        return {"error": "No text provided"}, 400
    
    classifier = pipeline("sentiment-analysis")
    result = classifier(text)[0]
    
    return {
        "text": text,
        "sentiment": result["label"],
        "confidence": result["score"]
    }

@app.function(image=image)
@modal.web_endpoint(method="GET")
def health():
    return {"status": "ok", "service": "sentiment-analyzer"}
EOF
```

### Step 3: Add to Library

```bash
python3 runna.py app-add sentiment-analyzer \
  --file sentiment_app.py \
  --description "Sentiment analysis with transformers" \
  --tags ml nlp sentiment \
  --gpu T4
```

Output:
```
Added app: sentiment-analyzer
```

### Step 4: View App Details

```bash
python3 runna.py app-show sentiment-analyzer
```

### Step 5: Deploy App

```bash
python3 runna.py app-deploy sentiment-analyzer
```

Output:
```
Modal SDK found
Deploying to Modal.com...
Modal deployment successful!
Modal endpoint URL: https://username--sentiment-analyzer-analyze.modal.run
App deployed: https://username--sentiment-analyzer-analyze.modal.run
```

### Step 6: Test Deployment

```bash
curl -X POST https://username--sentiment-analyzer-analyze.modal.run \
  -H "Content-Type: application/json" \
  -d '{"text": "I love this product! It works great."}'
```

Response:
```json
{
  "text": "I love this product! It works great.",
  "sentiment": "POSITIVE",
  "confidence": 0.9998
}
```

### Step 7: Update App (Add Batch Support)

```bash
cat > sentiment_app_v2.py << 'EOF'
import modal

app = modal.App("sentiment-analyzer")
image = modal.Image.debian_slim().pip_install("transformers", "torch")

@app.function(image=image, gpu="T4", timeout=300)
@modal.web_endpoint(method="POST")
def analyze(data: dict):
    from transformers import pipeline
    
    text = data.get("text", "")
    if not text:
        return {"error": "No text provided"}, 400
    
    classifier = pipeline("sentiment-analysis")
    result = classifier(text)[0]
    
    return {
        "text": text,
        "sentiment": result["label"],
        "confidence": result["score"]
    }

@app.function(image=image, gpu="T4", timeout=600)
@modal.web_endpoint(method="POST")
def analyze_batch(data: dict):
    from transformers import pipeline
    
    texts = data.get("texts", [])
    if not texts:
        return {"error": "No texts provided"}, 400
    
    classifier = pipeline("sentiment-analysis")
    results = classifier(texts)
    
    return {
        "results": [
            {
                "text": text,
                "sentiment": result["label"],
                "confidence": result["score"]
            }
            for text, result in zip(texts, results)
        ]
    }

@app.function(image=image)
@modal.web_endpoint(method="GET")
def health():
    return {"status": "ok", "service": "sentiment-analyzer", "version": "2.0"}
EOF

# Update in library
python3 runna.py app-update sentiment-analyzer --file sentiment_app_v2.py

# Redeploy
python3 runna.py app-deploy sentiment-analyzer
```

### Step 8: Test Batch Endpoint

```bash
curl -X POST https://username--sentiment-analyzer-analyze-batch.modal.run \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [
      "This is amazing!",
      "I hate this.",
      "It is okay."
    ]
  }'
```

Response:
```json
{
  "results": [
    {"text": "This is amazing!", "sentiment": "POSITIVE", "confidence": 0.9998},
    {"text": "I hate this.", "sentiment": "NEGATIVE", "confidence": 0.9995},
    {"text": "It is okay.", "sentiment": "NEUTRAL", "confidence": 0.8234}
  ]
}
```

## More Examples

### Deploy Image Classifier

```bash
# Use built-in template
python3 runna.py app-deploy image-classifier

# Test with image
python3 -c "
import base64, requests

with open('test.jpg', 'rb') as f:
    img_b64 = base64.b64encode(f.read()).decode()

response = requests.post(
    'https://your-app.modal.run',
    json={'image': img_b64}
)
print(response.json())
"
```

### Deploy Scheduled Task

```bash
# Deploy hourly task
python3 runna.py app-deploy scheduled-task

# Check status
curl https://your-app--status.modal.run
```

### Create Web Scraper

```bash
# Deploy scraper
python3 runna.py app-deploy web-scraper

# Scrape website
curl -X POST https://your-app.modal.run \
  -d '{"url": "https://news.ycombinator.com"}'
```

## Integration with Endpoints

Apps are automatically registered as endpoints:

```bash
# List all endpoints
python3 runna.py endpoints

# Call by name
python3 runna.py call sentiment-analyzer \
  --json '{"text": "Great product!"}'
```

## Best Practices

1. **Start with templates** - Use built-in apps as starting points
2. **Test locally first** - Use `modal serve` for local testing
3. **Version your apps** - Update descriptions with version info
4. **Use appropriate GPU** - T4 for inference, A100 for training
5. **Add health checks** - Include status endpoints
6. **Handle errors** - Return proper error codes
7. **Document usage** - Add examples in descriptions

## Troubleshooting

### App deployment fails

```bash
# Check logs
modal app logs sentiment-analyzer

# Test locally
cd /tmp/app_sentiment-analyzer
modal serve app.py
```

### Import errors

Ensure all dependencies are in `image.pip_install()`:

```python
image = modal.Image.debian_slim().pip_install(
    "transformers",
    "torch",
    "numpy"
)
```

### GPU not available

```python
# Check GPU in function
import torch
print(f"GPU: {torch.cuda.is_available()}")
```

## Summary

The app library provides:
- ✅ Pre-built templates for common use cases
- ✅ Easy deployment with `app-deploy`
- ✅ Version control with `app-update`
- ✅ Integration with endpoint registry
- ✅ GPU support configuration
- ✅ Secrets management

See [APP_LIBRARY.md](APP_LIBRARY.md) for complete documentation.
