# Modal.com Quick Start

## Setup (One-time)

```bash
pip install modal
modal token new
```

## Deploy Commands

```bash
# Basic
python3 runna.py deploy-modal ./notebook

# With GPU
python3 runna.py deploy-modal ./notebook --gpu A10G

# With secrets
python3 runna.py deploy-modal ./notebook --secrets api-key

# Full options
python3 runna.py deploy-modal ./notebook \
  --gpu A10G \
  --secrets api-key db-creds \
  --timeout 600 \
  --save-name my-model
```

## GPU Types

- `T4` - $0.60/hr - Inference
- `A10G` - $1.10/hr - Balanced
- `A100` - $4.00/hr - Training

## Secrets

```bash
# Create
modal secret create my-secret KEY=value

# Use in notebook
import os
key = os.environ['KEY']
```

## Test

```bash
# Local
python3 runna.py serve-local ./notebook --run

# Call endpoint
curl -X POST https://your-app.modal.run \
  -H "Content-Type: application/json" \
  -d '{"features": [1,2,3]}'
```

## Notebook Template

```python
import numpy as np

def process_request(data):
    features = data['features']
    result = np.mean(features)
    return {'result': result}
```

## Troubleshooting

```bash
# View logs
modal app logs kaggle-notebook

# Test locally
cd modal-deploy
modal serve modal_app.py
```

## More Info

See [MODAL_GUIDE.md](MODAL_GUIDE.md) for complete documentation.
