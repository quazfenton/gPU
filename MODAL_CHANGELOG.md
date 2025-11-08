# Modal.com Enhancement Changelog

## What's New

Enhanced Modal.com deployment support with GPU, secrets, volumes, and advanced features based on https://modal.com/docs/guide.

## New Features

### 1. GPU Support
- Deploy with T4, A10G, or A100 GPUs
- Command: `--gpu A10G`
- Automatic GPU detection in notebooks
- Cost-optimized GPU selection

### 2. Secrets Management
- Secure credential storage
- Command: `--secrets api-key db-creds`
- Access via environment variables
- Multiple secrets support

### 3. Custom Timeouts
- Configure function timeouts
- Command: `--timeout 600`
- Default: 300 seconds
- No hard limits (unlike Lambda/GCP)

### 4. Persistent Volumes
- Automatic model caching
- Mounted at `/cache`
- Shared across invocations
- Reduces cold start times

### 5. Enhanced App Template
- Modern Modal SDK usage
- Better error handling
- Health check endpoint
- Local testing support

## Updated Files

### Core Changes
- `runna.py`: Enhanced Modal deployment functions
  - `create_modal_app()`: Added GPU/secrets/timeout support
  - `package_for_modal()`: Pass options to app creation
  - `deploy_to_modal()`: Support new options
  - `cmd_deploy_modal()`: Parse new arguments
  - Argument parser: Added `--gpu`, `--secrets`, `--timeout`

### New Files
- `modal_deploy.py`: Standalone Modal deployment template
- `MODAL_GUIDE.md`: Complete deployment guide
- `MODAL_QUICKSTART.md`: Quick reference
- `MODAL_CHANGELOG.md`: This file
- `examples/modal_example.py`: Example notebook

### Updated Files
- `requirements.txt`: Added `modal>=0.63.0`
- `README.md`: Updated Modal section with new features

## Usage Examples

### Basic Deployment
```bash
python3 runna.py deploy-modal ./notebook
```

### GPU Deployment
```bash
python3 runna.py deploy-modal ./notebook --gpu A10G
```

### With Secrets
```bash
# Create secret first
modal secret create api-key API_KEY=your_key

# Deploy with secret
python3 runna.py deploy-modal ./notebook --secrets api-key
```

### Full Options
```bash
python3 runna.py deploy-modal ./notebook \
  --gpu A10G \
  --secrets api-key db-creds \
  --timeout 600 \
  --save-name production-model
```

## Notebook Requirements

Your notebook should define a `process_request(data)` function:

```python
def process_request(data):
    """Main function called by Modal endpoint."""
    features = data.get('features', [])
    # Your processing logic
    return {'result': result}
```

## GPU Access in Notebooks

```python
import torch

def process_request(data):
    # Check GPU availability
    if torch.cuda.is_available():
        device = torch.device('cuda')
        # Use GPU
    else:
        device = torch.device('cpu')
    
    # Your code here
    return {'result': result}
```

## Secrets Access

```python
import os

def process_request(data):
    api_key = os.environ.get('API_KEY')
    # Use secret
    return {'result': result}
```

## Volume Usage

```python
import os
import torch

def process_request(data):
    model_path = '/cache/model.pth'
    
    # Check if model is cached
    if os.path.exists(model_path):
        model = torch.load(model_path)
    else:
        # Download and cache
        model = download_model()
        torch.save(model, model_path)
    
    # Use model
    return {'result': result}
```

## Testing

### Local Testing
```bash
# Package and test locally
python3 runna.py serve-local ./notebook --run --port 8080

# Test endpoint
curl -X POST http://localhost:8080 \
  -H "Content-Type: application/json" \
  -d '{"features": [1, 2, 3]}'
```

### Modal Local Testing
```bash
cd modal-deploy
modal serve modal_app.py
```

### View Logs
```bash
modal app logs kaggle-notebook
```

## Cost Optimization

### GPU Pricing
- T4: $0.60/hour - Best for inference
- A10G: $1.10/hour - Balanced performance
- A100: $4.00/hour - High-performance training

### Tips
1. Use CPU for simple tasks (no `--gpu` flag)
2. Cache models in volumes to reduce cold starts
3. Set appropriate timeouts to avoid unnecessary charges
4. Use T4 for inference, A100 only for training

## Migration from Old Modal Deployment

Old command:
```bash
python3 runna.py deploy-modal ./notebook --save-name my-model
```

New command (same, but with options):
```bash
python3 runna.py deploy-modal ./notebook \
  --gpu A10G \
  --secrets api-key \
  --timeout 600 \
  --save-name my-model
```

## Breaking Changes

None - all new features are optional and backward compatible.

## Documentation

- [MODAL_GUIDE.md](MODAL_GUIDE.md) - Complete guide
- [MODAL_QUICKSTART.md](MODAL_QUICKSTART.md) - Quick reference
- [Modal Docs](https://modal.com/docs) - Official documentation

## Support

For issues:
1. Check logs: `modal app logs kaggle-notebook`
2. Review generated `modal_app.py`
3. Test locally: `modal serve modal-deploy/modal_app.py`
4. See [MODAL_GUIDE.md](MODAL_GUIDE.md) troubleshooting section

## Future Enhancements

Potential additions:
- [ ] Custom image specifications
- [ ] Multiple endpoints per app
- [ ] Batch processing support
- [ ] Streaming responses
- [ ] WebSocket support
- [ ] Custom domain support

## Version

- **Version**: 2.1.0
- **Date**: 2025-11-08
- **Status**: ✅ Production Ready
