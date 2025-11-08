# App Library Quick Reference

## Commands

```bash
# List
python3 runna.py app-list

# Show
python3 runna.py app-show <name>

# Deploy
python3 runna.py app-deploy <name>
python3 runna.py app-deploy <name> --gpu A100
python3 runna.py app-deploy <name> --secrets api-key

# Add
python3 runna.py app-add <name> --file app.py --gpu T4

# Update
python3 runna.py app-update <name> --file app.py

# Delete
python3 runna.py app-delete <name>
```

## Built-in Apps

| App | GPU | Use Case |
|-----|-----|----------|
| image-classifier | T4 | Image classification |
| text-generator | T4 | Text generation |
| web-scraper | None | Web scraping |
| batch-processor | None | Batch processing |
| scheduled-task | None | Cron jobs |

## Template

```python
import modal

app = modal.App("my-app")
image = modal.Image.debian_slim().pip_install("deps")

@app.function(image=image, gpu="T4")
@modal.web_endpoint(method="POST")
def handler(data: dict):
    return {"result": "ok"}
```

## GPU Types

- T4: $0.60/hr - Inference
- A10G: $1.10/hr - Balanced
- A100: $4.00/hr - Training
